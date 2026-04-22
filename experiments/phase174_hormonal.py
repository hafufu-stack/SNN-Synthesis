"""
Phase 174: Hormonal Broadcast  -  Breaking the NCA Light Speed Wall

P159 proved NCA info propagation is bounded by 1 cell/step (3x3 receptive field).
This phase adds a 'Hormone Channel'  -  Global Average Pooling of one hidden channel
broadcast to all cells each step  -  enabling non-local communication.

Test: Does hormone-augmented NCA improve PA on real ARC tasks that require
global spatial understanding (symmetry, long-range dependencies)?

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
PAD_SIZE = 32
N_COLORS = 11
EMB_DIM = 64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, FoundationEncoder, FoundationLatentNCA,
    load_arc_training, prepare_arc_meta_dataset, grid_to_tensor, tensor_to_grid
)


class HormonalLatentNCA(nn.Module):
    """Latent NCA with Hormone Channel for non-local communication.
    
    Each step:
    1. Compute NCA update as normal (local 3x3 convolution)
    2. Global Average Pool one hidden channel -> scalar 'hormone'
    3. Broadcast hormone to all cells as additional input
    """
    def __init__(self, in_ch=11, hidden_ch=64, latent_ch=32, embed_dim=64, n_hormone=4):
        super().__init__()
        self.n_hormone = n_hormone  # number of hormone channels
        # Encoder: input grid -> latent (same as Foundation)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1))
        # NCA step: latent + embed + hormone -> latent
        self.nca = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim + n_hormone, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1))
        # Decoder: latent -> output
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, hidden_ch, 1), nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, 1))
        # Hormone extractor: select channels and pool
        self.hormone_proj = nn.Conv2d(latent_ch, n_hormone, 1)

    def forward(self, x, task_embed, n_steps=5):
        H, W = x.shape[-2:]
        state = self.encoder(x)
        emb = task_embed.view(-1, EMB_DIM, 1, 1).expand(-1, -1, H, W)

        for step in range(n_steps):
            # Extract hormone: global average pool of projected channels
            hormone_map = self.hormone_proj(state)  # (B, n_hormone, H, W)
            hormone = hormone_map.mean(dim=(-2, -1), keepdim=True)  # (B, n_hormone, 1, 1)
            hormone_broadcast = hormone.expand(-1, -1, H, W)  # broadcast to all cells

            # NCA input: state + embedding + hormone
            inp = torch.cat([state, emb, hormone_broadcast], dim=1)
            delta = self.nca(inp)
            state = state + delta * 0.1  # residual

        return self.decoder(state)


class BaselineLatentNCA(nn.Module):
    """Baseline NCA without hormone (same architecture, no hormone channels)."""
    def __init__(self, in_ch=11, hidden_ch=64, latent_ch=32, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1))
        self.nca = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, hidden_ch, 1), nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, 1))

    def forward(self, x, task_embed, n_steps=5):
        H, W = x.shape[-2:]
        state = self.encoder(x)
        emb = task_embed.view(-1, EMB_DIM, 1, 1).expand(-1, -1, H, W)
        for step in range(n_steps):
            inp = torch.cat([state, emb], dim=1)
            delta = self.nca(inp)
            state = state + delta * 0.1
        return self.decoder(state)


def evaluate_model(model, encoder, tasks, n_steps=5, label="Model"):
    """Evaluate PA/EM on ARC tasks."""
    model.eval(); encoder.eval()
    pas, ems = [], []
    with torch.no_grad():
        for item in tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            logits = model(ti, emb, n_steps=n_steps)
            pred = logits[0, :10].argmax(dim=0)
            gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            pas.append(pa); ems.append(em)
    return np.mean(pas), np.mean(ems)


def train_model(model, encoder, tasks, n_epochs=30, lr=1e-3, label="Model"):
    """Train NCA model on ARC tasks."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        random.shuffle(tasks)
        for item in tasks[:30]:  # mini-batch of 30 tasks
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            with torch.no_grad():
                emb = encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
            logits = model(ti, emb, n_steps=5)
            loss = F.cross_entropy(logits[:, :10], gt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  [{label}] Epoch {epoch+1}/{n_epochs}: loss={total_loss/30:.4f}")
    return model


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 174: Hormonal Broadcast  -  Breaking the Light Speed Wall")
    print(f"  Hormone Channel for non-local NCA communication")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load Foundation encoder (for task embeddings)
    print("\n[Step 1] Loading Foundation encoder...")
    foundation = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    ckpt_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    foundation.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    foundation.eval()
    encoder = foundation.task_encoder

    # Load ARC data
    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train_tasks = all_tasks[:100]
    test_tasks = all_tasks[100:150]
    print(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Train Baseline NCA
    print("\n[Step 3] Training Baseline NCA (no hormone)...")
    baseline = BaselineLatentNCA(hidden_ch=64, latent_ch=32).to(DEVICE)
    baseline = train_model(baseline, encoder, train_tasks, n_epochs=30, label="Baseline")
    base_pa, base_em = evaluate_model(baseline, encoder, test_tasks, label="Baseline")
    print(f"  Baseline: PA={base_pa*100:.2f}%, EM={base_em*100:.1f}%")

    # Train Hormonal NCA
    print("\n[Step 4] Training Hormonal NCA (with hormone broadcast)...")
    hormonal = HormonalLatentNCA(hidden_ch=64, latent_ch=32, n_hormone=4).to(DEVICE)
    hormonal = train_model(hormonal, encoder, train_tasks, n_epochs=30, label="Hormonal")
    horm_pa, horm_em = evaluate_model(hormonal, encoder, test_tasks, label="Hormonal")
    print(f"  Hormonal: PA={horm_pa*100:.2f}%, EM={horm_em*100:.1f}%")

    # Compare at different step counts (more steps = more time for hormone to propagate)
    print("\n[Step 5] Step-count comparison...")
    step_results = {}
    for n_steps in [3, 5, 8, 12]:
        b_pa, b_em = evaluate_model(baseline, encoder, test_tasks, n_steps=n_steps)
        h_pa, h_em = evaluate_model(hormonal, encoder, test_tasks, n_steps=n_steps)
        step_results[n_steps] = {
            'baseline_pa': b_pa, 'hormonal_pa': h_pa,
            'improvement': h_pa - b_pa
        }
        print(f"  Steps={n_steps}: Baseline PA={b_pa*100:.1f}%, Hormonal PA={h_pa*100:.1f}% "
              f"(Δ={((h_pa-b_pa)*100):+.1f}pp)")

    n_base_p = sum(p.numel() for p in baseline.parameters())
    n_horm_p = sum(p.numel() for p in hormonal.parameters())

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 174 Complete ({elapsed:.0f}s)")
    print(f"  Baseline: PA={base_pa*100:.2f}% ({n_base_p:,} params)")
    print(f"  Hormonal: PA={horm_pa*100:.2f}% ({n_horm_p:,} params)")
    print(f"  Hormone advantage: {(horm_pa-base_pa)*100:+.2f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase174_hormonal.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 174: Hormonal Broadcast',
            'timestamp': datetime.now().isoformat(),
            'baseline': {'pa': base_pa, 'em': base_em, 'params': n_base_p},
            'hormonal': {'pa': horm_pa, 'em': horm_em, 'params': n_horm_p},
            'improvement_pp': (horm_pa - base_pa) * 100,
            'step_comparison': step_results,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Bar: PA comparison
        methods = ['Baseline\n(Local only)', 'Hormonal\n(+Broadcast)']
        pas = [base_pa*100, horm_pa*100]
        colors = ['#95a5a6', '#e74c3c']
        bars = axes[0].bar(methods, pas, color=colors, alpha=0.85, edgecolor='black')
        for bar, pa in zip(bars, pas):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{pa:.1f}%', ha='center', fontweight='bold', fontsize=11)
        axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title('PA Comparison', fontweight='bold')

        # Step comparison
        steps = sorted(step_results.keys())
        base_line = [step_results[s]['baseline_pa']*100 for s in steps]
        horm_line = [step_results[s]['hormonal_pa']*100 for s in steps]
        axes[1].plot(steps, base_line, 'o-', color='#95a5a6', label='Baseline', linewidth=2)
        axes[1].plot(steps, horm_line, 's-', color='#e74c3c', label='Hormonal', linewidth=2)
        axes[1].set_xlabel('NCA Steps'); axes[1].set_ylabel('PA (%)')
        axes[1].set_title('PA vs Steps', fontweight='bold'); axes[1].legend()

        # Improvement per step
        improvements = [step_results[s]['improvement']*100 for s in steps]
        axes[2].bar([str(s) for s in steps], improvements,
                   color=['#2ecc71' if v > 0 else '#e74c3c' for v in improvements],
                   alpha=0.85, edgecolor='black')
        axes[2].set_xlabel('NCA Steps'); axes[2].set_ylabel('Δ PA (pp)')
        axes[2].set_title('Hormone Advantage', fontweight='bold')
        axes[2].axhline(0, color='black', linewidth=0.5)

        fig.suptitle('Phase 174: Hormonal Broadcast (Non-Local NCA Communication)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase174_hormonal.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {'baseline_pa': base_pa, 'hormonal_pa': horm_pa}

if __name__ == '__main__':
    main()
