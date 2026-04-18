"""
Phase 130: VQ-Temporal NBS (Discrete Stochastic Resonance)

Problem: In VQ space, continuous Gaussian noise gets absorbed
by quantization. Need DISCRETE noise for exploration.

Solution: Gumbel Noise on VQ distance logits enables
"tunnel jumps" between discrete attractors while maintaining
pixel-perfect crystallized outputs.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase122_meta_nca import TASK_RULES, generate_meta_dataset


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes=32, dim=16, commitment=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.commitment = commitment
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1/n_codes, 1/n_codes)

    def forward(self, z, gumbel_scale=0.0):
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)

        dists = (z_flat ** 2).sum(1, keepdim=True) + \
                (self.codebook.weight ** 2).sum(1) - \
                2 * z_flat @ self.codebook.weight.t()

        if gumbel_scale > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(dists) + 1e-20) + 1e-20)
            dists = dists - gumbel_scale * gumbel

        indices = dists.argmin(dim=1)
        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        commit_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment * commit_loss
        z_q_st = z + (z_q - z).detach()
        usage = len(indices.unique()) / self.n_codes
        return z_q_st, vq_loss, usage, indices.view(B, H, W)


class VQNCA(nn.Module):
    """VQ-NCA for single-task training."""
    def __init__(self, hidden_ch=16, n_codes=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_ch, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1), nn.Sigmoid())
        self.vq = VectorQuantizer(n_codes, hidden_ch)
        self.decoder = nn.Conv2d(hidden_ch, 1, 1)

    def forward(self, x, n_steps=5, gumbel_scale=0.0, return_all=False):
        state = self.stem(x)
        vq_total = 0; usage_total = 0
        all_indices = []

        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau(state)
            state = beta * state + (1 - beta) * delta
            state, vq_loss, usage, indices = self.vq(state, gumbel_scale=gumbel_scale)
            vq_total += vq_loss; usage_total += usage
            all_indices.append(indices)

        output = torch.sigmoid(self.decoder(state))
        if return_all:
            return output, vq_total/n_steps, usage_total/n_steps, all_indices
        return output, vq_total/n_steps, usage_total/n_steps


def vq_temporal_nbs(model, x, K=11, n_steps=5, gumbel_scale=0.5):
    """
    Temporal NBS with Gumbel noise on VQ logits.
    Each beam explores different discrete trajectories.
    Majority vote on final output.
    """
    votes = torch.zeros(1, 1, x.shape[2], x.shape[3], device=DEVICE)

    for k in range(K):
        pred, _, _ = model(x, n_steps=n_steps, gumbel_scale=gumbel_scale)
        votes += (pred > 0.5).float()

    return (votes > K / 2).float()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 130: VQ-Temporal NBS (Discrete Stochastic Resonance)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Generate dataset
    print("\n[Step 1] Generating dataset...")
    dataset = generate_meta_dataset(grid_sz=8, n_tasks_per_rule=100, n_demos=2)
    train_data = dataset[:int(len(dataset)*0.8)]
    test_data = dataset[int(len(dataset)*0.8):]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Train VQ-NCA
    print("\n[Step 2] Training VQ-NCA...")
    model = VQNCA(hidden_ch=16, n_codes=32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(60):
        model.train()
        random.shuffle(train_data)
        for item in train_data:
            inp = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred, vq_loss, _ = model(inp, n_steps=5)
            loss = F.binary_cross_entropy(pred, out) + vq_loss
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/60")

    # Evaluate: sweep Gumbel scale and K
    print("\n[Step 3] Gumbel NBS sweep...")

    # First: baseline (no noise)
    model.eval()
    base_px = 0; base_exact = 0; n = 0
    with torch.no_grad():
        for item in test_data:
            inp = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred, _, _ = model(inp, n_steps=5, gumbel_scale=0.0)
            pred_b = (pred > 0.5).float()
            base_px += (pred_b == out).float().mean().item()
            base_exact += (pred_b == out).all().item()
            n += 1
    print(f"  Baseline: pixel={base_px/n*100:.2f}%, exact={base_exact}/{n}")

    # Sweep
    results = {'baseline': {'pixel': base_px/n, 'exact': base_exact, 'total': n}}

    gumbel_scales = [0.1, 0.3, 0.5, 1.0, 2.0]
    K_values = [11, 21, 51]

    for gs in gumbel_scales:
        for K in K_values:
            total_px = 0; total_exact = 0; n = 0
            with torch.no_grad():
                for item in test_data:
                    inp = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    pred = vq_temporal_nbs(model, inp, K=K, n_steps=5,
                                          gumbel_scale=gs)
                    total_px += (pred == out).float().mean().item()
                    total_exact += (pred == out).all().item()
                    n += 1

            key = f"G={gs}_K={K}"
            px_acc = total_px / n
            results[key] = {'pixel': px_acc, 'exact': total_exact, 'total': n,
                           'gumbel': gs, 'K': K}
            gap_px = (px_acc - base_px/len(test_data)) * 100
            gap_exact = total_exact - base_exact
            print(f"    {key:15s}: pixel={px_acc*100:.2f}%, "
                  f"exact={total_exact}/{n}, "
                  f"gap_px={gap_px:+.2f}%, gap_exact={gap_exact:+d}")

    # Find best
    best_key = max(
        [(k, v) for k, v in results.items() if k != 'baseline'],
        key=lambda x: (x[1]['exact'], x[1]['pixel'])
    )
    print(f"\n  Best: {best_key[0]} → exact={best_key[1]['exact']}/{best_key[1]['total']}, "
          f"pixel={best_key[1]['pixel']*100:.2f}%")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase130_vq_nbs.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 130: VQ-Temporal NBS',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for K in K_values:
            gs_vals = []; px_vals = []; ex_vals = []
            for gs in gumbel_scales:
                key = f"G={gs}_K={K}"
                if key in results:
                    gs_vals.append(gs)
                    px_vals.append(results[key]['pixel'] * 100)
                    ex_vals.append(results[key]['exact'])
            axes[0].plot(gs_vals, px_vals, 'o-', label=f'K={K}', markersize=4)
            axes[1].plot(gs_vals, ex_vals, 's-', label=f'K={K}', markersize=4)

        axes[0].axhline(y=base_px/len(test_data)*100, color='red',
                       linestyle='--', label='Baseline')
        axes[0].set_xlabel('Gumbel Scale'); axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title('Discrete SR: Pixel Accuracy'); axes[0].legend()

        axes[1].axhline(y=base_exact, color='red', linestyle='--', label='Baseline')
        axes[1].set_xlabel('Gumbel Scale'); axes[1].set_ylabel('Exact Matches')
        axes[1].set_title('Discrete SR: Exact Match'); axes[1].legend()

        plt.suptitle('Phase 130: VQ-Temporal NBS (Gumbel Noise)', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase130_vq_nbs.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 130 complete!")


if __name__ == '__main__':
    main()
