"""
Phase 166: Differentiable Homeostasis - Life from Survival Rules Only

Phase 164 bugfix: argmax breaks gradient flow.
Fix: use continuous probabilities (sigmoid/softmax) for all loss terms.

No target image. Loss = "don't die, don't overgrow, stay complex"

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
N_COLORS = 4
TARGET_DENSITY_LOW = 0.10
TARGET_DENSITY_HIGH = 0.35


class LifeNCA(nn.Module):
    def __init__(self, n_colors=N_COLORS, ch=32, steps=30):
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


def differentiable_homeostasis_loss(logits):
    """
    Fully differentiable homeostasis loss using continuous probabilities.
    No argmax anywhere - all operations maintain gradient flow.
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W) - differentiable

    # "Active" = probability of NOT being background (color 0)
    # This is differentiable: 1 - P(background)
    active_prob = 1.0 - probs[:, 0:1, :, :]  # (B, 1, H, W)
    density = active_prob.mean(dim=(-2, -1, -3))  # (B,)

    # 1. Death penalty: density below threshold
    death_pen = F.relu(TARGET_DENSITY_LOW - density).mean() * 15

    # 2. Cancer penalty: density above threshold
    cancer_pen = F.relu(density - TARGET_DENSITY_HIGH).mean() * 15

    # 3. Spatial complexity: neighboring pixels should differ
    # Use probabilities directly for differentiable boundary detection
    h_diff = ((probs[:, :, 1:, :] - probs[:, :, :-1, :]) ** 2).sum(dim=1).mean()
    v_diff = ((probs[:, :, :, 1:] - probs[:, :, :, :-1]) ** 2).sum(dim=1).mean()
    complexity_reward = -(h_diff + v_diff) * 5  # Reward structure

    # 4. Color diversity: entropy of mean color distribution
    mean_probs = probs.mean(dim=(-2, -1))  # (B, C)
    color_entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=1).mean()
    diversity_reward = -color_entropy * 2  # Reward high entropy

    # 5. Per-pixel entropy: each pixel should commit to one color
    pixel_ent = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
    commitment_pen = pixel_ent * 0.5  # Encourage sharp decisions

    loss = death_pen + cancer_pen + complexity_reward + diversity_reward + commitment_pen
    return loss, {
        'density': density.mean().item(),
        'death_pen': death_pen.item(),
        'cancer_pen': cancer_pen.item(),
        'complexity': (h_diff + v_diff).item(),
        'color_entropy': color_entropy.item(),
    }


def analyze_life_form(probs_or_grid):
    """Analyze the emerged life form."""
    if probs_or_grid.dim() == 3:  # (C, H, W) probability
        grid = probs_or_grid.argmax(dim=0).cpu().numpy()
    else:
        grid = probs_or_grid.cpu().numpy()

    n_colors = len(np.unique(grid))
    density = (grid != 0).mean()

    # Connected components
    visited = np.zeros_like(grid, dtype=bool)
    n_blobs = 0; blob_sizes = []

    def flood(y, x, color):
        stack = [(y, x)]; size = 0
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= grid.shape[0] or cx < 0 or cx >= grid.shape[1]:
                continue
            if visited[cy, cx] or grid[cy, cx] != color:
                continue
            visited[cy, cx] = True; size += 1
            stack.extend([(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)])
        return size

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if not visited[y, x] and grid[y, x] != 0:
                size = flood(y, x, grid[y, x])
                if size > 0:
                    n_blobs += 1; blob_sizes.append(size)

    # Boundary density
    h_diff = (grid[1:, :] != grid[:-1, :]).mean()
    v_diff = (grid[:, 1:] != grid[:, :-1]).mean()

    return {
        'n_colors': int(n_colors), 'density': float(density),
        'n_blobs': n_blobs, 'boundary_density': float(h_diff + v_diff),
        'avg_blob': float(np.mean(blob_sizes)) if blob_sizes else 0,
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 166: Differentiable Homeostasis")
    print(f"  Life from survival rules only (bugfix: no argmax)")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Colors: {N_COLORS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    step_configs = [30, 50, 100]
    results = {}
    final_grids = {}

    for total_steps in step_configs:
        name = f"T={total_steps}"
        print(f"\n[Config: {name}]")

        model = LifeNCA(n_colors=N_COLORS, ch=32, steps=total_steps).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)

        density_hist = []; complexity_hist = []

        for epoch in range(400):
            model.train()
            noise = torch.randn(4, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3
            logits = model(noise, steps=total_steps)
            loss, info = differentiable_homeostasis_loss(logits)

            opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            density_hist.append(info['density'])
            complexity_hist.append(info['complexity'])

            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/400: loss={loss.item():.4f}, "
                      f"density={info['density']*100:.1f}%, "
                      f"complexity={info['complexity']:.3f}, "
                      f"colors_ent={info['color_entropy']:.3f}")

        # Analyze
        model.eval()
        with torch.no_grad():
            test_noise = torch.randn(8, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3
            final_logits = model(test_noise, steps=total_steps)
            final_probs = F.softmax(final_logits, dim=1)
            final_pred = final_logits.argmax(dim=1)

        analyses = [analyze_life_form(final_pred[b]) for b in range(min(8, final_pred.shape[0]))]
        avg_a = {k: np.mean([a[k] for a in analyses]) for k in analyses[0]}
        final_grids[name] = final_pred[0].cpu().numpy().tolist()

        alive = TARGET_DENSITY_LOW <= avg_a['density'] <= TARGET_DENSITY_HIGH
        results[name] = {
            'density_history': density_hist[::20],
            'complexity_history': complexity_hist[::20],
            'analysis': avg_a, 'alive': alive,
        }
        print(f"  Final: density={avg_a['density']*100:.1f}%, colors={avg_a['n_colors']:.1f}, "
              f"blobs={avg_a['n_blobs']:.0f}, {'ALIVE' if alive else 'DEAD/CANCER'}")

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    any_alive = any(r['alive'] for r in results.values())
    print(f"\n{'='*70}")
    print(f"Phase 166 Complete ({elapsed:.0f}s)")
    for name, r in results.items():
        a = r['analysis']
        print(f"  {name}: {'ALIVE' if r['alive'] else 'DEAD'}, "
              f"density={a['density']*100:.1f}%, blobs={a['n_blobs']:.0f}")
    print(f"  Autopoiesis achieved: {any_alive}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase166_homeostasis.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 166: Differentiable Homeostasis',
            'timestamp': datetime.now().isoformat(),
            'results': results, 'final_grids': final_grids,
            'autopoiesis': any_alive, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        cmap = plt.cm.get_cmap('Set1', N_COLORS)
        for i, ts in enumerate(step_configs):
            name = f"T={ts}"
            grid = np.array(final_grids[name])
            axes[i].imshow(grid, cmap=cmap, vmin=0, vmax=N_COLORS-1, interpolation='nearest')
            a = results[name]['analysis']
            axes[i].set_title(f'{name} ({"ALIVE" if results[name]["alive"] else "DEAD"})\n'
                            f'd={a["density"]*100:.0f}%, blobs={a["n_blobs"]:.0f}',
                            fontweight='bold', fontsize=9)
            axes[i].axis('off')
        fig.suptitle('Phase 166: Differentiable Homeostasis (Life Without Target)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.98, wspace=0.15)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase166_homeostasis.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
