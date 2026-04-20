"""
Phase 162: Stem Cell NCA - Programmable Matter via Environmental Control

Unify self-replication (P153) and glider movement (P159) into a
single NCA "DNA" (weights). A control channel acts as a chemical
signal that determines cell fate:
  - control=0.1 -> Glider (move right by 1 pixel)
  - control=0.5 -> Replication (tile the grid)
  - control=0.9 -> Oscillator (blink on/off)

Same weights, different behaviors = stem cell differentiation.

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
GRID_SIZE = 16


# ================================================================
# Data generation for 3 tasks
# ================================================================
def make_seed():
    """Create a 2x2 seed pattern."""
    p = torch.zeros(2, 2)
    p[0, 0] = 1; p[0, 1] = 1; p[1, 0] = 1
    return p


def generate_glider_target(seed, start_y, start_x, shift=1, grid_size=GRID_SIZE):
    """Target: seed moved right by shift pixels, original cleared."""
    ph, pw = seed.shape
    target = torch.zeros(grid_size, grid_size)
    ty, tx = start_y, start_x + shift
    if tx + pw <= grid_size:
        target[ty:ty+ph, tx:tx+pw] = seed
    return target


def generate_replication_target(seed, grid_size=GRID_SIZE):
    """Target: tile the seed across the entire grid."""
    ph, pw = seed.shape
    target = torch.zeros(grid_size, grid_size)
    for y in range(0, grid_size, ph):
        for x in range(0, grid_size, pw):
            h = min(ph, grid_size - y)
            w = min(pw, grid_size - x)
            target[y:y+h, x:x+w] = seed[:h, :w]
    return target


def generate_oscillator_target(seed, start_y, start_x, phase, grid_size=GRID_SIZE):
    """Target: seed appears/disappears based on phase (blink)."""
    ph, pw = seed.shape
    target = torch.zeros(grid_size, grid_size)
    if phase == 0:  # ON phase: seed visible
        target[start_y:start_y+ph, start_x:start_x+pw] = seed
    # phase == 1: OFF phase: empty grid
    return target


def generate_stem_cell_data(n_samples, grid_size=GRID_SIZE):
    """Generate multi-task data with control channel."""
    X, Y = [], []
    seed = make_seed()
    ph, pw = seed.shape

    per_task = n_samples // 3

    # Task 1: Glider (control=0.1)
    for _ in range(per_task):
        sy = random.randint(2, grid_size - ph - 2)
        sx = random.randint(2, grid_size - pw - 3)
        inp_grid = torch.zeros(grid_size, grid_size)
        inp_grid[sy:sy+ph, sx:sx+pw] = seed
        control = torch.full((grid_size, grid_size), 0.1)
        target = generate_glider_target(seed, sy, sx, shift=1, grid_size=grid_size)
        inp = torch.stack([inp_grid, control])  # (2, H, W)
        X.append(inp)
        Y.append(target.unsqueeze(0))

    # Task 2: Replication (control=0.5)
    for _ in range(per_task):
        inp_grid = torch.zeros(grid_size, grid_size)
        inp_grid[:ph, :pw] = seed
        control = torch.full((grid_size, grid_size), 0.5)
        target = generate_replication_target(seed, grid_size)
        inp = torch.stack([inp_grid, control])
        X.append(inp)
        Y.append(target.unsqueeze(0))

    # Task 3: Oscillator (control=0.9)
    for _ in range(n_samples - 2 * per_task):
        sy = random.randint(2, grid_size - ph - 2)
        sx = random.randint(2, grid_size - pw - 2)
        inp_grid = torch.zeros(grid_size, grid_size)
        inp_grid[sy:sy+ph, sx:sx+pw] = seed
        control = torch.full((grid_size, grid_size), 0.9)
        phase = random.choice([0, 1])
        target = generate_oscillator_target(seed, sy, sx, phase, grid_size)
        inp = torch.stack([inp_grid, control])
        X.append(inp)
        Y.append(target.unsqueeze(0))

    return torch.stack(X), torch.stack(Y)


class StemCellDS(torch.utils.data.Dataset):
    def __init__(self, n=6000):
        self.X, self.Y = generate_stem_cell_data(n)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Stem Cell NCA (2 input channels: grid + control)
# ================================================================
class StemCellNCA(nn.Module):
    def __init__(self, ch=48, steps=10):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(2, ch, 3, padding=1)  # 2 channels: grid + control
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)


def eval_per_task(model, ds, grid_size=GRID_SIZE):
    """Evaluate accuracy per task type."""
    model.eval()
    per_task = len(ds) // 3
    task_names = ['glider', 'replication', 'oscillator']
    results = {}

    with torch.no_grad():
        for t, name in enumerate(task_names):
            start = t * per_task
            end = start + per_task if t < 2 else len(ds)
            correct = total = exact = n = 0

            for i in range(start, end):
                x, y = ds[i]
                x = x.unsqueeze(0).to(DEVICE)
                y = y.unsqueeze(0).to(DEVICE)
                pred = (torch.sigmoid(model(x)) > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.numel()
                if (pred == y).all():
                    exact += 1
                n += 1

            results[name] = {
                'pixel_acc': correct / total,
                'exact_match': exact / n,
            }

    return results


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 162: Stem Cell NCA - Programmable Matter")
    print(f"  One DNA, three fates: Glider / Replication / Oscillator")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    train_ds = StemCellDS(6000)
    test_ds = StemCellDS(900)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    # Train
    model = StemCellNCA(ch=48, steps=15).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"\n[Training] StemCellNCA ({n_p:,} params)...")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
    for epoch in range(120):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch+1) % 30 == 0:
            task_results = eval_per_task(model, test_ds)
            parts = [f"{n}={r['pixel_acc']*100:.1f}%" for n, r in task_results.items()]
            print(f"  Epoch {epoch+1}/120: {', '.join(parts)}")

    # Final evaluation
    task_results = eval_per_task(model, test_ds)
    differentiation = all(r['pixel_acc'] > 0.7 for r in task_results.values())

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 162 Complete ({elapsed:.0f}s)")
    for name, r in task_results.items():
        print(f"  {name:>15}: PA={r['pixel_acc']*100:.2f}%, EM={r['exact_match']*100:.1f}%")
    print(f"  Differentiation achieved: {differentiation}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase162_stem_cell.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 162: Stem Cell NCA',
            'timestamp': datetime.now().isoformat(),
            'task_results': task_results,
            'differentiation': differentiation,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        names = list(task_results.keys())
        pas = [task_results[n]['pixel_acc']*100 for n in names]
        ems = [task_results[n]['exact_match']*100 for n in names]
        colors = ['#e74c3c', '#3498db', '#2ecc71']

        # Panel 1: PA per task
        bars = axes[0].bar(names, pas, color=colors, alpha=0.85, edgecolor='black')
        for bar, pa in zip(bars, pas):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{pa:.1f}%', ha='center', fontweight='bold', fontsize=9)
        axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title('Task Performance', fontweight='bold', fontsize=10)

        # Panel 2: EM per task
        bars = axes[1].bar(names, ems, color=colors, alpha=0.85, edgecolor='black')
        for bar, em in zip(bars, ems):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{em:.1f}%', ha='center', fontweight='bold', fontsize=9)
        axes[1].set_ylabel('Exact Match (%)')
        axes[1].set_title('Perfect Accuracy', fontweight='bold', fontsize=10)

        # Panel 3: Example outputs
        model.eval()
        with torch.no_grad():
            ctrl_vals = [0.1, 0.5, 0.9]
            seed = make_seed()
            combined = []
            for cv in ctrl_vals:
                inp_grid = torch.zeros(GRID_SIZE, GRID_SIZE)
                inp_grid[:2, :2] = seed
                control = torch.full((GRID_SIZE, GRID_SIZE), cv)
                x = torch.stack([inp_grid, control]).unsqueeze(0).to(DEVICE)
                pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()
                combined.append(pred)

            row = np.concatenate(combined, axis=1)
            axes[2].imshow(row, cmap='hot', vmin=0, vmax=1, aspect='auto')
            axes[2].set_title('ctrl=0.1 | ctrl=0.5 | ctrl=0.9', fontweight='bold', fontsize=10)
            axes[2].axis('off')

        fig.suptitle('Phase 162: Stem Cell NCA (One DNA, Three Fates)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.85, bottom=0.08, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase162_stem_cell.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return task_results


if __name__ == '__main__':
    main()
