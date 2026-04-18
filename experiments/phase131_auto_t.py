"""
Phase 131: Discrete Auto-T & Soft-Landing

VQ space enables perfect convergence detection:
  - Count cells whose VQ index changed between steps
  - If 0 → "crystallized" → stop immediately (Auto-T)
  - If oscillating → approaching collapse → slow down tau (Soft-Landing)

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

    def forward(self, z):
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
        dists = (z_flat ** 2).sum(1, keepdim=True) + \
                (self.codebook.weight ** 2).sum(1) - \
                2 * z_flat @ self.codebook.weight.t()
        indices = dists.argmin(dim=1)
        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        commit_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment * commit_loss
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss, indices.view(B, H, W)


class VQNCA_AutoT(nn.Module):
    """VQ-NCA with discrete energy monitoring for Auto-T and Soft-Landing."""
    def __init__(self, hidden_ch=16, n_codes=32, max_steps=20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_ch, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau_base = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1), nn.Sigmoid())
        self.vq = VectorQuantizer(n_codes, hidden_ch)
        self.decoder = nn.Conv2d(hidden_ch, 1, 1)
        self.max_steps = max_steps

    def forward(self, x, mode='fixed', n_steps=5, soft_landing=False,
                return_trajectory=False):
        """
        Modes:
          'fixed': fixed T steps (standard)
          'auto_t': stop when crystallized (0 index changes)
          'soft_landing': reduce tau when oscillation detected
        """
        state = self.stem(x)
        vq_total = 0
        prev_indices = None
        trajectory = []
        actual_steps = 0

        T = n_steps if mode == 'fixed' else self.max_steps

        # Track energy for soft-landing
        energy_history = []
        tau_scale = 1.0

        for t in range(T):
            delta = self.update(state)
            beta = self.tau_base(state) * tau_scale

            state = beta * state + (1 - beta) * delta
            state, vq_loss, indices = self.vq(state)
            vq_total += vq_loss
            actual_steps += 1

            # Discrete energy: count of changed VQ indices
            if prev_indices is not None:
                n_changed = (indices != prev_indices).sum().item()
                total_cells = indices.numel()
                change_rate = n_changed / total_cells
                energy_history.append(change_rate)

                if return_trajectory:
                    trajectory.append({
                        'step': t, 'n_changed': n_changed,
                        'change_rate': change_rate, 'tau_scale': tau_scale
                    })

                # Auto-T: crystallized (0 changes)
                if mode == 'auto_t' and n_changed == 0:
                    break

                # Soft-Landing: detect oscillation and cool down
                if soft_landing and len(energy_history) >= 3:
                    recent = energy_history[-3:]
                    # Oscillation: energy not monotonically decreasing
                    if recent[-1] > recent[-2] and recent[-2] < recent[-3]:
                        tau_scale = max(0.1, tau_scale * 0.8)
                    # Also cool if energy is too high
                    if change_rate > 0.5:
                        tau_scale = max(0.1, tau_scale * 0.9)

            prev_indices = indices

        output = torch.sigmoid(self.decoder(state))
        info = {
            'actual_steps': actual_steps,
            'final_energy': energy_history[-1] if energy_history else 1.0,
            'final_tau_scale': tau_scale,
            'vq_loss': vq_total / max(actual_steps, 1),
        }
        if return_trajectory:
            info['trajectory'] = trajectory

        return output, info


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 131: Discrete Auto-T & Soft-Landing")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Generating dataset...")
    dataset = generate_meta_dataset(grid_sz=8, n_tasks_per_rule=100, n_demos=2)
    train_data = dataset[:int(len(dataset)*0.8)]
    test_data = dataset[int(len(dataset)*0.8):]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Train VQ-NCA
    print("\n[Step 2] Training VQ-NCA (with variable T)...")
    model = VQNCA_AutoT(hidden_ch=16, n_codes=32, max_steps=20).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(60):
        model.train()
        random.shuffle(train_data)
        for item in train_data:
            inp = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            # Train with fixed T=5
            pred, info = model(inp, mode='fixed', n_steps=5)
            loss = F.binary_cross_entropy(pred, out) + info['vq_loss']
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/60")

    # Evaluate all modes
    print("\n[Step 3] Evaluating modes...")
    model.eval()

    modes = [
        ('Fixed T=5', 'fixed', 5, False),
        ('Fixed T=10', 'fixed', 10, False),
        ('Fixed T=20', 'fixed', 20, False),
        ('Auto-T (max=20)', 'auto_t', 20, False),
        ('Auto-T + SL', 'auto_t', 20, True),
        ('Fixed T=5 + SL', 'fixed', 5, True),
        ('Fixed T=10 + SL', 'fixed', 10, True),
        ('Fixed T=20 + SL', 'fixed', 20, True),
    ]

    all_results = {}

    for mode_name, mode, T, sl in modes:
        total_px = 0; total_exact = 0; total_steps = 0
        total_energy = 0; n = 0

        with torch.no_grad():
            for item in test_data:
                inp = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
                out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

                pred, info = model(inp, mode=mode, n_steps=T, soft_landing=sl)
                pred_b = (pred > 0.5).float()
                total_px += (pred_b == out).float().mean().item()
                total_exact += (pred_b == out).all().item()
                total_steps += info['actual_steps']
                total_energy += info['final_energy']
                n += 1

        px_acc = total_px / n
        exact_rate = total_exact / n
        avg_steps = total_steps / n
        avg_energy = total_energy / n

        all_results[mode_name] = {
            'pixel_acc': px_acc, 'exact_rate': exact_rate,
            'exact_count': total_exact, 'total': n,
            'avg_steps': avg_steps, 'avg_energy': avg_energy,
        }
        print(f"  {mode_name:22s}: pixel={px_acc*100:.2f}%, "
              f"exact={total_exact}/{n}, "
              f"steps={avg_steps:.1f}, energy={avg_energy:.4f}")

    # Collect trajectory for one example
    print("\n[Step 4] Collecting crystallization trajectory...")
    sample = test_data[0]
    inp = torch.tensor(sample['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _, info = model(inp, mode='auto_t', n_steps=20,
                       soft_landing=True, return_trajectory=True)
    traj = info.get('trajectory', [])
    if traj:
        print(f"  Trajectory ({len(traj)} steps):")
        for t in traj[:10]:
            print(f"    t={t['step']:2d}: changed={t['n_changed']:3d}, "
                  f"rate={t['change_rate']:.3f}, tau={t['tau_scale']:.3f}")
        if info['actual_steps'] < 20:
            print(f"  → Crystallized at step {info['actual_steps']}!")

    # Summary
    print(f"\n{'='*70}")
    print("  DISCRETE AUTO-T & SOFT-LANDING RESULTS")
    print(f"{'='*70}")
    print(f"  {'Mode':25s} {'Pixel':>8s} {'Exact':>10s} {'Steps':>7s} {'Energy':>8s}")
    for name, res in all_results.items():
        print(f"  {name:25s} {res['pixel_acc']*100:7.2f}% "
              f"{res['exact_count']:>3d}/{res['total']:<3d}  "
              f"{res['avg_steps']:6.1f} {res['avg_energy']:7.4f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase131_auto_t.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 131: Discrete Auto-T & Soft-Landing',
                   'timestamp': datetime.now().isoformat(),
                   'results': all_results,
                   'sample_trajectory': traj}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        names = list(all_results.keys())
        px = [all_results[n]['pixel_acc']*100 for n in names]
        ex = [all_results[n]['exact_count'] for n in names]
        steps = [all_results[n]['avg_steps'] for n in names]

        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
                  '#1abc9c', '#3498db', '#2980b9', '#9b59b6']

        axes[0].barh(range(len(names)), px, color=colors)
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=7)
        axes[0].set_xlabel('Pixel Accuracy (%)')
        axes[0].set_title('Pixel Accuracy by Mode')

        axes[1].barh(range(len(names)), ex, color=colors)
        axes[1].set_yticks(range(len(names)))
        axes[1].set_yticklabels(names, fontsize=7)
        axes[1].set_xlabel('Exact Matches')
        axes[1].set_title('Exact Match Count')

        axes[2].barh(range(len(names)), steps, color=colors)
        axes[2].set_yticks(range(len(names)))
        axes[2].set_yticklabels(names, fontsize=7)
        axes[2].set_xlabel('Avg Steps')
        axes[2].set_title('Computational Cost')

        # Trajectory subplot
        if traj:
            # Add as inset
            ax_inset = fig.add_axes([0.72, 0.55, 0.25, 0.35])
            ts = [t['step'] for t in traj]
            rates = [t['change_rate'] for t in traj]
            taus = [t['tau_scale'] for t in traj]
            ax_inset.plot(ts, rates, 'b-o', label='Change rate', markersize=3)
            ax_inset.plot(ts, taus, 'r--s', label='Tau scale', markersize=3)
            ax_inset.set_xlabel('Step', fontsize=7)
            ax_inset.set_title('Crystallization', fontsize=8)
            ax_inset.legend(fontsize=6)
            ax_inset.tick_params(labelsize=6)

        plt.suptitle('Phase 131: Discrete Auto-T & Soft-Landing', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase131_auto_t.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 131 complete!")


if __name__ == '__main__':
    main()
