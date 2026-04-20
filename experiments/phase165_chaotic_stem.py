"""
Phase 165: Chaotic Stem-Cell Agent - ALife meets ARC

Combine P161 (chaos revival via noise) and P162 (stem cell
differentiation) into an ARC-solving prototype.

The NCA receives:
  - Input grid (the ARC problem)
  - Control channel (task embedding: mean color, density, etc.)
  - Stochastic noise injection at each step

Tests on simple ARC-like tasks:
  1. Fill (ctrl=0.2): fill all empty cells with dominant color
  2. Mirror (ctrl=0.5): horizontally mirror the pattern
  3. Invert (ctrl=0.8): swap foreground/background colors

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
GRID_SIZE = 8
N_COLORS = 4


# ================================================================
# ARC-like task generation
# ================================================================
def generate_fill_data(n, grid_size=GRID_SIZE):
    """Fill task: fill empty cells with the dominant non-zero color."""
    X, Y = [], []
    for _ in range(n):
        grid = torch.zeros(grid_size, grid_size, dtype=torch.long)
        color = random.randint(1, N_COLORS - 1)
        n_cells = random.randint(3, grid_size * grid_size // 3)
        positions = random.sample(range(grid_size * grid_size), n_cells)
        for p in positions:
            grid[p // grid_size, p % grid_size] = color
        target = torch.full((grid_size, grid_size), color, dtype=torch.long)
        X.append(grid)
        Y.append(target)
    return X, Y


def generate_mirror_data(n, grid_size=GRID_SIZE):
    """Mirror task: horizontally flip the pattern."""
    X, Y = [], []
    for _ in range(n):
        grid = torch.zeros(grid_size, grid_size, dtype=torch.long)
        n_cells = random.randint(3, grid_size * grid_size // 2)
        for _ in range(n_cells):
            y, x = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            grid[y, x] = random.randint(1, N_COLORS - 1)
        target = grid.fliplr()
        X.append(grid)
        Y.append(target)
    return X, Y


def generate_invert_data(n, grid_size=GRID_SIZE):
    """Invert task: swap 0<->dominant color."""
    X, Y = [], []
    for _ in range(n):
        grid = torch.zeros(grid_size, grid_size, dtype=torch.long)
        color = random.randint(1, N_COLORS - 1)
        for y in range(grid_size):
            for x in range(grid_size):
                if random.random() < 0.4:
                    grid[y, x] = color
        target = torch.where(grid == 0, torch.tensor(color), torch.tensor(0))
        X.append(grid)
        Y.append(target)
    return X, Y


def to_onehot(grids, n_colors=N_COLORS, grid_size=GRID_SIZE):
    """Convert list of grids to one-hot tensor."""
    oh = torch.zeros(len(grids), n_colors, grid_size, grid_size)
    for i, g in enumerate(grids):
        for c in range(n_colors):
            oh[i, c] = (g == c).float()
    return oh


# ================================================================
# Chaotic Stem Cell NCA
# ================================================================
class ChaoticStemNCA(nn.Module):
    """NCA with control channel + stochastic noise injection."""
    def __init__(self, n_colors=N_COLORS, ch=48, steps=15):
        super().__init__()
        self.steps = steps
        # +1 for control channel
        self.proj_in = nn.Conv2d(n_colors + 1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, n_colors, 1)

    def forward(self, x_with_ctrl, steps=None, noise_sigma=0.0):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x_with_ctrl))
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
            # Inject stochastic noise (P161's chaos revival)
            if noise_sigma > 0 and self.training:
                h = h + torch.randn_like(h) * noise_sigma
        return self.proj_out(h)


def prepare_data_with_ctrl(task_type, n, ctrl_val, grid_size=GRID_SIZE):
    """Generate data with control channel."""
    if task_type == 'fill':
        X_raw, Y_raw = generate_fill_data(n, grid_size)
    elif task_type == 'mirror':
        X_raw, Y_raw = generate_mirror_data(n, grid_size)
    elif task_type == 'invert':
        X_raw, Y_raw = generate_invert_data(n, grid_size)
    else:
        raise ValueError(f"Unknown task: {task_type}")

    X_oh = to_onehot(X_raw, N_COLORS, grid_size)
    ctrl = torch.full((len(X_raw), 1, grid_size, grid_size), ctrl_val)
    X = torch.cat([X_oh, ctrl], dim=1)  # (N, C+1, H, W)

    Y_labels = torch.stack(Y_raw)  # (N, H, W)
    return X, Y_labels


class MultiTaskDS(torch.utils.data.Dataset):
    def __init__(self, n_per_task=2000):
        tasks = [('fill', 0.2), ('mirror', 0.5), ('invert', 0.8)]
        all_X, all_Y = [], []
        for task, ctrl in tasks:
            X, Y = prepare_data_with_ctrl(task, n_per_task, ctrl)
            all_X.append(X)
            all_Y.append(Y)
        self.X = torch.cat(all_X)
        self.Y = torch.cat(all_Y)
        # Shuffle
        perm = torch.randperm(len(self.X))
        self.X = self.X[perm]
        self.Y = self.Y[perm]

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


def eval_per_task(model, noise_sigma=0.0):
    """Evaluate per-task accuracy."""
    model.eval()
    tasks = [('fill', 0.2), ('mirror', 0.5), ('invert', 0.8)]
    results = {}

    for task, ctrl in tasks:
        X, Y = prepare_data_with_ctrl(task, 200, ctrl)
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        with torch.no_grad():
            logits = model(X, noise_sigma=noise_sigma)
            pred = logits.argmax(dim=1)
            pa = (pred == Y).float().mean().item()
            em = all((pred[i] == Y[i]).all() for i in range(len(Y)))
        results[task] = {'pixel_acc': pa, 'exact_match': float(em)}

    return results


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 165: Chaotic Stem-Cell Agent")
    print(f"  ALife meets ARC: differentiation + chaos for rule learning")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Colors: {N_COLORS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    train_ds = MultiTaskDS(3000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    # Train with noise injection
    noise_levels = [0.0, 0.1, 0.3]
    results = {}

    for sigma in noise_levels:
        name = f"sigma={sigma}"
        print(f"\n[Config: {name}]")

        model = ChaoticStemNCA(n_colors=N_COLORS, ch=48, steps=15).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters())
        if sigma == 0:
            print(f"  Training ({n_p:,} params)...")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        for epoch in range(100):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x, noise_sigma=sigma)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if (epoch+1) % 50 == 0:
                task_r = eval_per_task(model, noise_sigma=0)
                parts = [f"{t}={r['pixel_acc']*100:.1f}%" for t, r in task_r.items()]
                print(f"  Epoch {epoch+1}/100: {', '.join(parts)}")

        # Eval with and without noise at inference
        task_r_clean = eval_per_task(model, noise_sigma=0)
        task_r_noisy = eval_per_task(model, noise_sigma=0.1)

        results[name] = {
            'clean_eval': task_r_clean,
            'noisy_eval': task_r_noisy,
        }

        avg_pa_clean = np.mean([r['pixel_acc'] for r in task_r_clean.values()])
        avg_pa_noisy = np.mean([r['pixel_acc'] for r in task_r_noisy.values()])
        print(f"  Clean eval: avg PA={avg_pa_clean*100:.1f}%")
        print(f"  Noisy eval: avg PA={avg_pa_noisy*100:.1f}%")

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Determine if stem cell differentiation works
    best_config = max(results.keys(),
                      key=lambda k: np.mean([r['pixel_acc'] for r in results[k]['clean_eval'].values()]))
    best_clean = results[best_config]['clean_eval']
    differentiation = all(r['pixel_acc'] > 0.7 for r in best_clean.values())

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 165 Complete ({elapsed:.0f}s)")
    print(f"  Best config: {best_config}")
    for task, r in best_clean.items():
        print(f"    {task}: PA={r['pixel_acc']*100:.2f}%")
    print(f"  Differentiation achieved: {differentiation}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase165_chaotic_stem.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 165: Chaotic Stem-Cell Agent',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'best_config': best_config,
            'differentiation': differentiation,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Panel 1: Per-task accuracy for best config
        tasks = list(best_clean.keys())
        pas = [best_clean[t]['pixel_acc']*100 for t in tasks]
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        bars = axes[0].bar(tasks, pas, color=colors, alpha=0.85, edgecolor='black')
        for bar, pa in zip(bars, pas):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{pa:.1f}%', ha='center', fontweight='bold', fontsize=9)
        axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title(f'Task Performance ({best_config})', fontweight='bold', fontsize=10)

        # Panel 2: Noise level comparison
        sigmas = list(results.keys())
        avg_pas = [np.mean([r['pixel_acc'] for r in results[s]['clean_eval'].values()])*100 for s in sigmas]
        axes[1].bar(sigmas, avg_pas, color=['#3498db', '#f39c12', '#e74c3c'],
                   alpha=0.85, edgecolor='black')
        axes[1].set_ylabel('Avg PA (%)')
        axes[1].set_title('Training Noise Impact', fontweight='bold', fontsize=10)

        # Panel 3: Clean vs noisy inference
        for s in sigmas:
            clean = np.mean([r['pixel_acc'] for r in results[s]['clean_eval'].values()])*100
            noisy = np.mean([r['pixel_acc'] for r in results[s]['noisy_eval'].values()])*100
            axes[2].scatter(clean, noisy, s=100, label=s, zorder=5)
        axes[2].plot([80, 100], [80, 100], 'k--', alpha=0.3)
        axes[2].set_xlabel('Clean Eval PA (%)')
        axes[2].set_ylabel('Noisy Eval PA (%)')
        axes[2].set_title('Robustness to Inference Noise', fontweight='bold', fontsize=10)
        axes[2].legend(fontsize=8)

        fig.suptitle('Phase 165: Chaotic Stem-Cell Agent (ALife x ARC)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase165_chaotic_stem.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
