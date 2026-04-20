"""
Phase 155: Turing Morphogenesis - Reaction-Diffusion Pattern Formation

Phase 152 showed NCA can dream (entropy drop), but it converged to
boring single-color states. Turing showed that interplay between
activation and inhibition creates beautiful patterns (stripes, spots).

Here we train NCA with NO target - only two competing objectives:
  1. MINIMIZE local entropy (sharpen each pixel to a definite color)
  2. MAXIMIZE spatial variance (prevent collapse to single color)

This tension should produce Turing-like morphogenetic patterns.

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
GRID_SIZE = 32
N_COLORS = 4  # Simpler color space for pattern emergence


class MorphoNCA(nn.Module):
    """NCA for unsupervised pattern formation."""
    def __init__(self, ch=32, n_colors=N_COLORS, steps=20):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(n_colors, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, n_colors, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)


def local_entropy(logits, kernel_size=5):
    """Compute local entropy in a sliding window."""
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    ent = -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    # Average entropy in local patches
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=logits.device) / (kernel_size**2)
    pad = kernel_size // 2
    local_ent = F.conv2d(F.pad(ent, (pad, pad, pad, pad), mode='circular'), kernel)
    return local_ent.mean()


def spatial_variance(logits):
    """Spatial variance of predictions - higher = more diverse patterns."""
    pred = logits.argmax(dim=1).float()  # (B, H, W)
    return pred.var(dim=(-2, -1)).mean()


def color_distribution_entropy(logits):
    """Entropy of global color distribution - maximize for diversity."""
    pred = logits.argmax(dim=1)  # (B, H, W)
    B = pred.shape[0]
    ent = 0
    for b in range(B):
        counts = torch.bincount(pred[b].ravel(), minlength=logits.shape[1]).float()
        probs = counts / counts.sum()
        ent -= (probs * (probs + 1e-8).log()).sum()
    return ent / B


def pattern_complexity(pred_grid):
    """Count color boundaries - more = more complex pattern."""
    if pred_grid.dim() == 2:
        pred_grid = pred_grid.unsqueeze(0)
    h_diff = (pred_grid[:, 1:, :] != pred_grid[:, :-1, :]).float().sum()
    v_diff = (pred_grid[:, :, 1:] != pred_grid[:, :, :-1]).float().sum()
    total = pred_grid.shape[0] * pred_grid.shape[1] * pred_grid.shape[2]
    return (h_diff + v_diff) / total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 155: Turing Morphogenesis")
    print(f"  Unsupervised pattern formation via competing objectives")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Colors: {N_COLORS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Test different balance ratios
    ratios = [
        (1.0, 0.1, "sharp_dominant"),   # Strong entropy min, weak variance max
        (1.0, 1.0, "balanced"),         # Equal
        (0.1, 1.0, "diverse_dominant"), # Weak entropy min, strong variance max
        (1.0, 0.5, "moderate"),         # Moderate balance
    ]

    results = {}
    final_grids = {}

    for ent_w, var_w, name in ratios:
        print(f"\n[Config: {name}] ent_weight={ent_w}, var_weight={var_w}")

        model = MorphoNCA(ch=32, n_colors=N_COLORS, steps=20).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

        loss_history = []
        ent_history = []
        var_history = []

        for epoch in range(200):
            model.train()
            # Random noise input each time
            noise = torch.randn(8, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3

            logits = model(noise, steps=20)

            # Competing objectives
            ent_loss = local_entropy(logits, kernel_size=5)        # Minimize
            var_loss = -spatial_variance(logits)                   # Maximize (negate)
            color_ent = -color_distribution_entropy(logits)        # Maximize color diversity

            loss = ent_w * ent_loss + var_w * (var_loss + 0.5 * color_ent)

            opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            loss_history.append(loss.item())
            ent_history.append(ent_loss.item())
            var_history.append(-var_loss.item())

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/200: loss={loss.item():.4f}, "
                      f"ent={ent_loss.item():.4f}, var={-var_loss.item():.4f}")

        # Final pattern analysis
        model.eval()
        with torch.no_grad():
            test_noise = torch.randn(4, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3
            final_logits = model(test_noise, steps=30)  # More steps for clearer pattern
            final_pred = final_logits.argmax(dim=1)  # (4, H, W)

            n_unique = len(torch.unique(final_pred[0]))
            complexity = pattern_complexity(final_pred).item()
            final_ent = local_entropy(final_logits).item()

        final_grids[name] = final_pred[0].cpu().numpy().tolist()

        results[name] = {
            'ent_weight': ent_w, 'var_weight': var_w,
            'final_entropy': final_ent,
            'final_variance': var_history[-1],
            'n_unique_colors': int(n_unique),
            'pattern_complexity': complexity,
            'loss_history': loss_history[::10],  # Subsample
            'ent_history': ent_history[::10],
            'var_history': var_history[::10],
        }
        print(f"  Final: {n_unique} colors, complexity={complexity:.4f}, ent={final_ent:.4f}")

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 155 Complete ({elapsed:.0f}s)")
    for name, r in results.items():
        print(f"  {name:>18}: colors={r['n_unique_colors']}, "
              f"complexity={r['pattern_complexity']:.4f}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase155_morphogenesis.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 155: Turing Morphogenesis',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'final_grids': final_grids,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        # Show final patterns for each config
        cmap = plt.cm.get_cmap('Set1', N_COLORS)
        for i, (_, _, name) in enumerate(ratios):
            grid = np.array(final_grids[name])
            axes[i].imshow(grid, cmap=cmap, vmin=0, vmax=N_COLORS-1, interpolation='nearest')
            r = results[name]
            axes[i].set_title(f'{name}\ncolors={r["n_unique_colors"]}, '
                            f'cmplx={r["pattern_complexity"]:.3f}',
                            fontweight='bold', fontsize=9)
            axes[i].axis('off')

        fig.suptitle('Phase 155: Turing Morphogenesis (Unsupervised Pattern Formation)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.98, wspace=0.15)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase155_morphogenesis.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
