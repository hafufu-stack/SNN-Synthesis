"""
Phase 176: Annealed Crystallization  -  Simulated Annealing for Exact Match

VQ (hard discretization) kills gradients (P132-135). Soft Crystallization
(entropy minimization) helps (P149) but doesn't force exact outputs.

This phase implements thermodynamic crystallization via Simulated Annealing:
- Temperature τ starts high (soft, liquid) → decays to near-zero (hard, crystal)
- Output softmax(logits/τ) smoothly transitions from uniform to one-hot
- No VQ needed; the annealing schedule handles crystallization

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
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


class AnnealedLatentNCA(nn.Module):
    """Latent NCA with temperature-annealed output crystallization.
    
    During forward pass:
    - Step 1..T: compute NCA updates normally
    - Temperature schedule: τ(t) = τ_max * (τ_min/τ_max)^(t/T)
    - At each step, inject temperature-scaled feedback into state
    """
    def __init__(self, in_ch=11, hidden_ch=64, latent_ch=32, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1))
        # NCA: latent + embed + crystallization_feedback
        self.nca = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim + in_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, hidden_ch, 1), nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, 1))

    def forward(self, x, task_embed, n_steps=8, tau_max=2.0, tau_min=0.01):
        """Forward with simulated annealing temperature schedule."""
        H, W = x.shape[-2:]
        B = x.shape[0]
        state = self.encoder(x)
        emb = task_embed.view(-1, EMB_DIM, 1, 1).expand(-1, -1, H, W)

        entropy_history = []
        confidence_history = []

        for step in range(n_steps):
            # Exponential cooling schedule
            progress = step / max(n_steps - 1, 1)
            tau = tau_max * (tau_min / tau_max) ** progress

            # Decode current state with temperature
            logits = self.decoder(state)
            tempered = F.softmax(logits[:, :10] / tau, dim=1)  # (B, 10, H, W)

            # Record stats
            entropy = -(tempered * (tempered + 1e-8).log()).sum(dim=1).mean().item()
            confidence = tempered.max(dim=1)[0].mean().item()
            entropy_history.append(entropy)
            confidence_history.append(confidence)

            # Crystal feedback: feed tempered distribution back as input
            # This creates a self-reinforcing loop: confident outputs → sharper state
            crystal_feedback = F.pad(tempered, (0, 0, 0, 0, 0, max(0, 11 - 10)),
                                     value=0)[:, :11]

            inp = torch.cat([state, emb, crystal_feedback], dim=1)
            delta = self.nca(inp)
            state = state + delta * 0.1

        # Final output at minimum temperature (fully crystallized)
        final_logits = self.decoder(state)
        return final_logits, {
            'entropy_history': entropy_history,
            'confidence_history': confidence_history,
            'final_tau': tau_min
        }


class StandardLatentNCA(nn.Module):
    """Standard NCA without annealing (for comparison)."""
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

    def forward(self, x, task_embed, n_steps=8):
        H, W = x.shape[-2:]
        state = self.encoder(x)
        emb = task_embed.view(-1, EMB_DIM, 1, 1).expand(-1, -1, H, W)
        for step in range(n_steps):
            inp = torch.cat([state, emb], dim=1)
            delta = self.nca(inp)
            state = state + delta * 0.1
        return self.decoder(state)


def train_annealed(model, encoder, tasks, n_epochs=30, lr=1e-3):
    """Train AnnealedLatentNCA."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        random.shuffle(tasks)
        for item in tasks[:30]:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            with torch.no_grad():
                emb = encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
            logits, _ = model(ti, emb, n_steps=8, tau_max=2.0, tau_min=0.05)
            loss = F.cross_entropy(logits[:, :10], gt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={total_loss/30:.4f}")
    return model


def train_standard(model, encoder, tasks, n_epochs=30, lr=1e-3):
    """Train StandardLatentNCA."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        random.shuffle(tasks)
        for item in tasks[:30]:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            with torch.no_grad():
                emb = encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
            logits = model(ti, emb, n_steps=8)
            loss = F.cross_entropy(logits[:, :10], gt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={total_loss/30:.4f}")
    return model


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 176: Annealed Crystallization  -  Thermodynamic Exact Match")
    print(f"  Simulated Annealing: τ=2.0 → 0.01 over NCA steps")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load encoder
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

    # Train Standard NCA
    print("\n[Step 3] Training Standard NCA (no annealing)...")
    standard = StandardLatentNCA(hidden_ch=64, latent_ch=32).to(DEVICE)
    standard = train_standard(standard, encoder, train_tasks, n_epochs=30)

    # Train Annealed NCA
    print("\n[Step 4] Training Annealed NCA...")
    annealed = AnnealedLatentNCA(hidden_ch=64, latent_ch=32).to(DEVICE)
    annealed = train_annealed(annealed, encoder, train_tasks, n_epochs=30)

    # Evaluate
    print("\n[Step 5] Evaluation...")
    std_pas, std_ems = [], []
    ann_pas, ann_ems = [], []
    all_entropy_hist = []
    all_confidence_hist = []

    # Test different annealing schedules
    anneal_configs = [
        {'tau_max': 1.0, 'tau_min': 0.1, 'label': 'mild'},
        {'tau_max': 2.0, 'tau_min': 0.01, 'label': 'standard'},
        {'tau_max': 5.0, 'tau_min': 0.001, 'label': 'aggressive'},
    ]

    config_results = {}
    for config in anneal_configs:
        c_pas, c_ems = [], []
        c_entropy, c_conf = [], []
        for item in test_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            with torch.no_grad():
                emb = encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            logits, stats = annealed(ti, emb, n_steps=8,
                                      tau_max=config['tau_max'],
                                      tau_min=config['tau_min'])
            pred = logits[0, :10].argmax(dim=0)
            gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            c_pas.append(pa); c_ems.append(em)
            c_entropy.append(stats['entropy_history'])
            c_conf.append(stats['confidence_history'])

        avg_pa = np.mean(c_pas); avg_em = np.mean(c_ems)
        config_results[config['label']] = {
            'pa': avg_pa, 'em': avg_em,
            'tau_max': config['tau_max'], 'tau_min': config['tau_min'],
            'avg_entropy_traj': np.mean(c_entropy, axis=0).tolist(),
            'avg_confidence_traj': np.mean(c_conf, axis=0).tolist()
        }
        print(f"  Annealed ({config['label']}): PA={avg_pa*100:.2f}%, EM={avg_em*100:.1f}%")

    # Standard baseline
    for item in test_tasks:
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do = [d.to(DEVICE) for d in item['demo_outputs']]
        with torch.no_grad():
            emb = encoder(di, do)
        ti = item['test_input'].unsqueeze(0).to(DEVICE)
        logits = standard(ti, emb, n_steps=8)
        pred = logits[0, :10].argmax(dim=0)
        gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
        oh, ow = item['out_h'], item['out_w']
        pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
        em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
        std_pas.append(pa); std_ems.append(em)

    std_pa = np.mean(std_pas); std_em = np.mean(std_ems)
    print(f"  Standard (no anneal): PA={std_pa*100:.2f}%, EM={std_em*100:.1f}%")

    best_config = max(config_results.items(), key=lambda x: x[1]['pa'])
    best_pa = best_config[1]['pa']
    improvement = best_pa - std_pa

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 176 Complete ({elapsed:.0f}s)")
    print(f"  Standard:     PA={std_pa*100:.2f}%, EM={std_em*100:.1f}%")
    print(f"  Best Anneal ({best_config[0]}): PA={best_pa*100:.2f}%, EM={best_config[1]['em']*100:.1f}%")
    print(f"  Improvement: {improvement*100:+.2f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase176_annealed.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 176: Annealed Crystallization',
            'timestamp': datetime.now().isoformat(),
            'standard': {'pa': std_pa, 'em': std_em},
            'annealed_configs': config_results,
            'best_config': best_config[0],
            'improvement_pp': improvement * 100,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # PA comparison
        labels = ['Standard'] + [f'Anneal\n({k})' for k in config_results]
        pa_vals = [std_pa*100] + [v['pa']*100 for v in config_results.values()]
        colors = ['#95a5a6'] + ['#e74c3c', '#2ecc71', '#3498db'][:len(config_results)]
        bars = axes[0].bar(labels, pa_vals, color=colors, alpha=0.85, edgecolor='black')
        for bar, pa in zip(bars, pa_vals):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{pa:.1f}%', ha='center', fontweight='bold', fontsize=9)
        axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title('PA: Standard vs Annealed', fontweight='bold')

        # Entropy trajectory
        for label, data in config_results.items():
            if 'avg_entropy_traj' in data:
                axes[1].plot(data['avg_entropy_traj'], 'o-', label=label, linewidth=1.5)
        axes[1].set_xlabel('NCA Step'); axes[1].set_ylabel('Entropy')
        axes[1].set_title('Entropy Over Steps (Crystallization)', fontweight='bold')
        axes[1].legend()

        # Confidence trajectory
        for label, data in config_results.items():
            if 'avg_confidence_traj' in data:
                axes[2].plot(data['avg_confidence_traj'], 's-', label=label, linewidth=1.5)
        axes[2].set_xlabel('NCA Step'); axes[2].set_ylabel('Max Confidence')
        axes[2].set_title('Confidence Over Steps', fontweight='bold')
        axes[2].legend()

        fig.suptitle('Phase 176: Annealed Crystallization (Simulated Annealing for NCA)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase176_annealed.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {'standard_pa': std_pa, 'best_anneal_pa': best_pa}

if __name__ == '__main__':
    main()
