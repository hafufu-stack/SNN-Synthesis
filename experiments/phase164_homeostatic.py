"""
Phase 164: Homeostatic Genesis - Open-Ended Life from Survival Rules Only

No target image. No teacher. Only survival constraints:
  1. Don't die (density > 5%)
  2. Don't overgrow (density < 40%)
  3. Stay interesting (maximize spatial structure)

What strange life forms will the NCA invent to stay alive?

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

    def evolve_sequence(self, x, total_steps, record_every=5):
        """Run NCA and record snapshots."""
        h = F.relu(self.proj_in(x))
        snapshots = []
        for t in range(total_steps):
            h = F.relu(h + self.rule(h))
            if t % record_every == 0 or t == total_steps - 1:
                pred = self.proj_out(h).argmax(dim=1)
                snapshots.append(pred.cpu())
        return self.proj_out(h), snapshots


def homeostasis_loss(logits, target_low=TARGET_DENSITY_LOW, target_high=TARGET_DENSITY_HIGH):
    """Survival loss: maintain density in healthy range + be interesting."""
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    pred = logits.argmax(dim=1)  # (B, H, W)

    # Active cells = non-background (not argmax=0)
    active = (pred != 0).float()
    density = active.mean(dim=(-2, -1))  # (B,)

    # 1. Death penalty: density too low
    death_penalty = F.relu(target_low - density).mean() * 10

    # 2. Cancer penalty: density too high
    cancer_penalty = F.relu(density - target_high).mean() * 10

    # 3. Homeostasis reward: density in sweet spot
    in_range = ((density >= target_low) & (density <= target_high)).float()
    homeostasis_reward = -in_range.mean() * 2  # Negative = reward

    # 4. Structural complexity: reward boundaries between different colors
    h_diff = (pred[:, 1:, :] != pred[:, :-1, :]).float().mean()
    v_diff = (pred[:, :, 1:] != pred[:, :, :-1]).float().mean()
    complexity_reward = -(h_diff + v_diff) * 3  # Reward structure

    # 5. Color diversity: use multiple colors
    color_ent = 0
    for b in range(pred.shape[0]):
        counts = torch.bincount(pred[b].ravel(), minlength=logits.shape[1]).float()
        p = counts / counts.sum()
        color_ent -= (p * (p + 1e-8).log()).sum()
    color_ent = -color_ent / pred.shape[0]  # Reward diversity

    # 6. Temporal variation: output should change across steps (anti-crystallization)
    # This is handled by running multiple steps with different noise

    loss = death_penalty + cancer_penalty + homeostasis_reward + complexity_reward + color_ent
    return loss, {
        'density': density.mean().item(),
        'death_pen': death_penalty.item(),
        'cancer_pen': cancer_penalty.item(),
        'complexity': (h_diff + v_diff).item(),
    }


def analyze_life_form(grid):
    """Analyze properties of the emerged life form."""
    if isinstance(grid, torch.Tensor):
        grid = grid.numpy()

    n_colors = len(np.unique(grid))
    density = (grid != 0).mean()

    # Connected components (simple flood fill)
    visited = np.zeros_like(grid, dtype=bool)
    n_blobs = 0
    blob_sizes = []

    def flood(y, x, color):
        stack = [(y, x)]
        size = 0
        while stack:
            cy, cx = stack.pop()
            if cy < 0 or cy >= grid.shape[0] or cx < 0 or cx >= grid.shape[1]:
                continue
            if visited[cy, cx] or grid[cy, cx] != color:
                continue
            visited[cy, cx] = True
            size += 1
            stack.extend([(cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)])
        return size

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if not visited[y, x] and grid[y, x] != 0:
                size = flood(y, x, grid[y, x])
                if size > 0:
                    n_blobs += 1
                    blob_sizes.append(size)

    # Symmetry check
    h_sym = np.mean(grid == np.fliplr(grid))
    v_sym = np.mean(grid == np.flipud(grid))

    return {
        'n_colors': int(n_colors),
        'density': float(density),
        'n_blobs': n_blobs,
        'avg_blob_size': float(np.mean(blob_sizes)) if blob_sizes else 0,
        'max_blob_size': max(blob_sizes) if blob_sizes else 0,
        'h_symmetry': float(h_sym),
        'v_symmetry': float(v_sym),
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 164: Homeostatic Genesis")
    print(f"  Open-ended life from survival rules only")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Colors: {N_COLORS}")
    print(f"  Target density: {TARGET_DENSITY_LOW*100:.0f}%-{TARGET_DENSITY_HIGH*100:.0f}%")
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
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)

        density_history = []
        complexity_history = []

        for epoch in range(300):
            model.train()
            # Random seed each time (open-ended)
            noise = torch.randn(4, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3
            logits = model(noise, steps=total_steps)
            loss, info = homeostasis_loss(logits)

            opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            density_history.append(info['density'])
            complexity_history.append(info['complexity'])

            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/300: loss={loss.item():.4f}, "
                      f"density={info['density']*100:.1f}%, "
                      f"complexity={info['complexity']:.3f}")

        # Analyze final life forms
        model.eval()
        with torch.no_grad():
            test_noise = torch.randn(8, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3
            final_logits, snapshots = model.evolve_sequence(test_noise, total_steps, record_every=10)
            final_pred = final_logits.argmax(dim=1)

        # Analyze each sample
        analyses = []
        for b in range(min(8, final_pred.shape[0])):
            a = analyze_life_form(final_pred[b].cpu())
            analyses.append(a)

        avg_analysis = {k: np.mean([a[k] for a in analyses]) for k in analyses[0]}
        final_grids[name] = final_pred[0].cpu().numpy().tolist()

        results[name] = {
            'density_history': density_history[::15],
            'complexity_history': complexity_history[::15],
            'final_analysis': avg_analysis,
            'alive': TARGET_DENSITY_LOW <= avg_analysis['density'] <= TARGET_DENSITY_HIGH,
        }
        print(f"  Final: density={avg_analysis['density']*100:.1f}%, "
              f"colors={avg_analysis['n_colors']:.1f}, "
              f"blobs={avg_analysis['n_blobs']:.0f}, "
              f"complexity={complexity_history[-1]:.3f}")

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 164 Complete ({elapsed:.0f}s)")
    for name, r in results.items():
        a = r['final_analysis']
        status = "ALIVE" if r['alive'] else "DEAD/CANCER"
        print(f"  {name}: {status}, density={a['density']*100:.1f}%, "
              f"blobs={a['n_blobs']:.0f}, colors={a['n_colors']:.1f}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase164_homeostatic.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 164: Homeostatic Genesis',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'final_grids': final_grids,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        cmap = plt.cm.get_cmap('Set1', N_COLORS)

        for i, ts in enumerate(step_configs):
            name = f"T={ts}"
            grid = np.array(final_grids[name])
            axes[i].imshow(grid, cmap=cmap, vmin=0, vmax=N_COLORS-1, interpolation='nearest')
            a = results[name]['final_analysis']
            status = "ALIVE" if results[name]['alive'] else "DEAD"
            axes[i].set_title(f'{name} ({status})\n'
                            f'density={a["density"]*100:.0f}%, blobs={a["n_blobs"]:.0f}',
                            fontweight='bold', fontsize=9)
            axes[i].axis('off')

        fig.suptitle('Phase 164: Homeostatic Genesis (Life from Survival Rules)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.98, wspace=0.15)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase164_homeostatic.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
