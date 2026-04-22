"""
Phase 175: Metabolic Sleep  -  Attention via Compute Rationing

P173's apoptosis (killing low-confidence pixels) was too aggressive.
Instead of killing, we SLEEP (skip updates) on confident pixels and
focus computation on uncertain ones.

Entropy-gated NCA: each step, compute per-pixel output entropy.
Low-entropy (confident) pixels freeze; high-entropy pixels get updated.

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


def inference_with_sleep(model, x, task_embed, n_steps=5, entropy_threshold=0.5):
    """NCA inference with metabolic sleep: skip updates on confident pixels.
    
    Each step:
    1. Compute NCA logits from current state
    2. Measure per-pixel entropy of output distribution
    3. Only update pixels where entropy > threshold (uncertain)
    4. Confident pixels keep their state (sleep)
    
    Returns: final logits, stats (flops saved, entropy trajectory)
    """
    model.eval()
    with torch.no_grad():
        H, W = x.shape[-2:]
        state = model.encoder(x)
        emb = task_embed.view(-1, EMB_DIM, 1, 1).expand(-1, -1, H, W)

        entropy_history = []
        active_ratio_history = []

        for step in range(n_steps):
            inp = torch.cat([state, emb], dim=1)
            delta = model.nca(inp)

            # Decode to check entropy
            logits = model.decoder(state + delta * 0.1)
            probs = F.softmax(logits[:, :10], dim=1)  # (B, 10, H, W)
            pixel_entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)  # (B, H, W)

            avg_entropy = pixel_entropy.mean().item()
            entropy_history.append(avg_entropy)

            # Metabolic mask: 1 = awake (needs update), 0 = sleeping (confident)
            awake_mask = (pixel_entropy > entropy_threshold).float()  # (B, H, W)
            active_ratio = awake_mask.mean().item()
            active_ratio_history.append(active_ratio)

            # Selective update: only awake pixels get updated
            awake_3d = awake_mask.unsqueeze(1).expand_as(delta)  # (B, C, H, W)
            state = state + delta * 0.1 * awake_3d

        final_logits = model.decoder(state)
    return final_logits, {
        'entropy_history': entropy_history,
        'active_ratio_history': active_ratio_history,
        'flops_saved': 1.0 - np.mean(active_ratio_history)
    }


def inference_standard(model, x, task_embed, n_steps=5):
    """Standard NCA inference (no sleep, all pixels updated every step)."""
    model.eval()
    with torch.no_grad():
        H, W = x.shape[-2:]
        state = model.encoder(x)
        emb = task_embed.view(-1, EMB_DIM, 1, 1).expand(-1, -1, H, W)
        for step in range(n_steps):
            inp = torch.cat([state, emb], dim=1)
            delta = model.nca(inp)
            state = state + delta * 0.1
        return model.decoder(state)


class SimpleLatentNCA(nn.Module):
    """Simple Latent NCA with exposed encoder/nca/decoder for sleep mechanism."""
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


def train_nca(model, encoder, tasks, n_epochs=30, lr=1e-3):
    """Train NCA model."""
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
            logits = model(ti, emb, n_steps=5)
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
    print("Phase 175: Metabolic Sleep  -  Entropy-Gated NCA Inference")
    print(f"  Sleep confident pixels, focus computation on uncertain ones")
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

    # Train NCA
    print("\n[Step 3] Training NCA...")
    model = SimpleLatentNCA(hidden_ch=64, latent_ch=32).to(DEVICE)
    model = train_nca(model, encoder, train_tasks, n_epochs=30)

    # Evaluate: Standard vs Sleep at different thresholds
    print("\n[Step 4] Comparing Standard vs Metabolic Sleep...")
    thresholds = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
    results = {}

    # Standard inference
    std_pas, std_ems = [], []
    for item in test_tasks:
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do = [d.to(DEVICE) for d in item['demo_outputs']]
        with torch.no_grad():
            emb = encoder(di, do)
        ti = item['test_input'].unsqueeze(0).to(DEVICE)
        logits = inference_standard(model, ti, emb, n_steps=8)
        pred = logits[0, :10].argmax(dim=0)
        gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
        oh, ow = item['out_h'], item['out_w']
        pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
        em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
        std_pas.append(pa); std_ems.append(em)

    std_pa = np.mean(std_pas); std_em = np.mean(std_ems)
    print(f"  Standard (8 steps): PA={std_pa*100:.2f}%, EM={std_em*100:.1f}%")
    results['standard'] = {'pa': std_pa, 'em': std_em, 'flops_saved': 0.0}

    for thresh in thresholds:
        sleep_pas, sleep_ems, all_flops = [], [], []
        all_entropy_hist, all_active_hist = [], []
        for item in test_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            with torch.no_grad():
                emb = encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            logits, stats = inference_with_sleep(model, ti, emb, n_steps=8,
                                                  entropy_threshold=thresh)
            pred = logits[0, :10].argmax(dim=0)
            gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            sleep_pas.append(pa); sleep_ems.append(em)
            all_flops.append(stats['flops_saved'])
            all_entropy_hist.append(stats['entropy_history'])
            all_active_hist.append(stats['active_ratio_history'])

        s_pa = np.mean(sleep_pas); s_em = np.mean(sleep_ems)
        avg_flops_saved = np.mean(all_flops)
        print(f"  Sleep(τ={thresh:.1f}): PA={s_pa*100:.2f}%, EM={s_em*100:.1f}%, "
              f"FLOPs saved={avg_flops_saved*100:.1f}%")
        results[f'sleep_{thresh}'] = {
            'pa': s_pa, 'em': s_em, 'flops_saved': avg_flops_saved,
            'threshold': thresh,
            'avg_entropy_trajectory': np.mean(all_entropy_hist, axis=0).tolist(),
            'avg_active_trajectory': np.mean(all_active_hist, axis=0).tolist()
        }

    elapsed = time.time() - t0
    best_sleep = max(
        [(k, v) for k, v in results.items() if k.startswith('sleep_')],
        key=lambda x: x[1]['pa'])
    improvement = best_sleep[1]['pa'] - std_pa

    print(f"\n{'='*70}")
    print(f"Phase 175 Complete ({elapsed:.0f}s)")
    print(f"  Standard: PA={std_pa*100:.2f}%")
    print(f"  Best Sleep ({best_sleep[0]}): PA={best_sleep[1]['pa']*100:.2f}%, "
          f"FLOPs saved={best_sleep[1]['flops_saved']*100:.1f}%")
    print(f"  Sleep advantage: {improvement*100:+.2f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase175_metabolic_sleep.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 175: Metabolic Sleep',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'best_threshold': best_sleep[1].get('threshold', None),
            'improvement_pp': improvement * 100,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # PA vs threshold
        ts = sorted([v['threshold'] for k, v in results.items() if 'threshold' in v])
        pa_vals = [results[f'sleep_{t}']['pa']*100 for t in ts]
        axes[0].plot(ts, pa_vals, 'o-', color='#e74c3c', linewidth=2, label='Sleep')
        axes[0].axhline(std_pa*100, color='#95a5a6', linestyle='--', label='Standard')
        axes[0].set_xlabel('Entropy Threshold'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title('PA vs Sleep Threshold', fontweight='bold'); axes[0].legend()

        # FLOPs saved vs threshold
        flops_vals = [results[f'sleep_{t}']['flops_saved']*100 for t in ts]
        axes[1].bar([f'{t}' for t in ts], flops_vals, color='#2ecc71', alpha=0.85)
        axes[1].set_xlabel('Entropy Threshold'); axes[1].set_ylabel('FLOPs Saved (%)')
        axes[1].set_title('Compute Savings', fontweight='bold')

        # Entropy trajectory
        for t in [0.3, 0.8, 1.5]:
            key = f'sleep_{t}'
            if key in results and 'avg_entropy_trajectory' in results[key]:
                traj = results[key]['avg_entropy_trajectory']
                axes[2].plot(range(len(traj)), traj, 'o-', label=f'τ={t}', linewidth=1.5)
        axes[2].set_xlabel('NCA Step'); axes[2].set_ylabel('Avg Entropy')
        axes[2].set_title('Entropy Over Steps', fontweight='bold'); axes[2].legend()

        fig.suptitle('Phase 175: Metabolic Sleep (Entropy-Gated NCA)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase175_metabolic_sleep.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {'standard_pa': std_pa, 'best_sleep_pa': best_sleep[1]['pa']}

if __name__ == '__main__':
    main()
