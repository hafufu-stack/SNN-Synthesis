"""
Phase 159: Glider Synthesis - Self-Moving Digital Life

Conway's Game of Life is famous for gliders: patterns that move
through space while preserving their shape. Can we TRAIN an NCA
to produce gliders?

Task: Given a seed pattern at position (y, x), produce the SAME
pattern at position (y+dy, x+dx) after T steps, with the original
position cleared to empty.

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
# Glider data generation
# ================================================================
def make_glider_pair(pattern, start_y, start_x, dy, dx, grid_size=GRID_SIZE):
    """Create (input, target) where target is pattern shifted by (dy, dx)."""
    ph, pw = pattern.shape
    inp = torch.zeros(grid_size, grid_size)
    inp[start_y:start_y+ph, start_x:start_x+pw] = pattern

    target = torch.zeros(grid_size, grid_size)
    ty, tx = start_y + dy, start_x + dx
    if 0 <= ty and ty + ph <= grid_size and 0 <= tx and tx + pw <= grid_size:
        target[ty:ty+ph, tx:tx+pw] = pattern

    return inp, target


def generate_glider_data(n_samples, pattern_size=3, shift=2, grid_size=GRID_SIZE):
    """Generate glider training data with various patterns and shifts."""
    X, Y = [], []
    directions = [(shift, 0), (0, shift), (shift, shift), (-shift, 0), (0, -shift)]

    for _ in range(n_samples):
        # Random non-trivial pattern
        pattern = torch.rand(pattern_size, pattern_size).round()
        while pattern.sum() < 2:  # Need at least 2 cells
            pattern = torch.rand(pattern_size, pattern_size).round()

        dy, dx = random.choice(directions)

        # Random starting position (with room to move)
        margin = pattern_size + abs(shift) + 1
        start_y = random.randint(margin, grid_size - margin)
        start_x = random.randint(margin, grid_size - margin)

        inp, target = make_glider_pair(pattern, start_y, start_x, dy, dx, grid_size)
        X.append(inp)
        Y.append(target)

    return torch.stack(X).unsqueeze(1), torch.stack(Y).unsqueeze(1)


class GliderDS(torch.utils.data.Dataset):
    def __init__(self, n=5000, ps=3, shift=2):
        self.X, self.Y = generate_glider_data(n, ps, shift)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Glider NCA
# ================================================================
class GliderNCA(nn.Module):
    def __init__(self, ch=48, steps=10):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
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


class GliderClockNCA(nn.Module):
    """With external clock - might help the NCA know when to stop."""
    def __init__(self, ch=48, steps=10):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch + 1, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for t in range(steps):
            B, C, H, W = h.shape
            clock = torch.full((B, 1, H, W), t / steps, device=h.device)
            h_aug = torch.cat([h, clock], dim=1)
            h = F.relu(h + self.rule(h_aug))
        return self.proj_out(h)


# ================================================================
# Evaluation
# ================================================================
def eval_glider(model, loader):
    model.eval()
    correct = total = 0
    shape_preserved = 0
    position_correct = 0
    n_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = (torch.sigmoid(model(x)) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()

            for b in range(y.size(0)):
                # Check shape preservation
                pred_cells = pred[b, 0].nonzero(as_tuple=False)
                target_cells = y[b, 0].nonzero(as_tuple=False)
                input_cells = x[b, 0].nonzero(as_tuple=False)

                if len(pred_cells) > 0 and len(target_cells) > 0:
                    # Shape preserved = same number of cells
                    if len(pred_cells) == len(target_cells):
                        shape_preserved += 1

                    # Position correct = centroids match
                    pred_center = pred_cells.float().mean(0)
                    target_center = target_cells.float().mean(0)
                    if torch.abs(pred_center - target_center).max() < 1.5:
                        position_correct += 1

                n_samples += 1

    pa = correct / total
    sp = shape_preserved / max(1, n_samples)
    pc = position_correct / max(1, n_samples)
    return pa, sp, pc


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 159: Glider Synthesis")
    print(f"  Training NCA to create self-moving patterns")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    configs = [
        (3, 1, 10, "3x3_shift1"),
        (3, 2, 10, "3x3_shift2"),
        (3, 3, 15, "3x3_shift3"),
        (2, 1, 8, "2x2_shift1"),
    ]

    results = {}

    for ps, shift, steps, name in configs:
        print(f"\n{'='*50}")
        print(f"Config: {name} (pattern={ps}x{ps}, shift={shift}, steps={steps})")
        print(f"{'='*50}")

        train_ds = GliderDS(5000, ps, shift)
        test_ds = GliderDS(500, ps, shift)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

        # Baseline NCA
        model = GliderNCA(ch=48, steps=steps).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters())
        print(f"  [Baseline] Training ({n_p:,} params)...")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        for epoch in range(100):
            model.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = F.binary_cross_entropy_with_logits(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        pa, sp, pc = eval_glider(model, test_loader)
        print(f"  PA={pa*100:.2f}%, ShapePreserved={sp*100:.1f}%, PositionCorrect={pc*100:.1f}%")

        # Clock NCA
        model_clock = GliderClockNCA(ch=48, steps=steps).to(DEVICE)
        n_p2 = sum(p.numel() for p in model_clock.parameters())
        print(f"  [Clock] Training ({n_p2:,} params)...")

        opt = torch.optim.Adam(model_clock.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        for epoch in range(100):
            model_clock.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = F.binary_cross_entropy_with_logits(model_clock(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        pa2, sp2, pc2 = eval_glider(model_clock, test_loader)
        print(f"  PA={pa2*100:.2f}%, ShapePreserved={sp2*100:.1f}%, PositionCorrect={pc2*100:.1f}%")

        results[name] = {
            'baseline': {'pixel_acc': pa, 'shape_preserved': sp, 'position_correct': pc, 'params': n_p},
            'clock': {'pixel_acc': pa2, 'shape_preserved': sp2, 'position_correct': pc2, 'params': n_p2},
        }

        del model, model_clock; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Summary
    elapsed = time.time() - t0
    can_glide = any(r['baseline']['position_correct'] > 0.3 or r['clock']['position_correct'] > 0.3
                    for r in results.values())

    print(f"\n{'='*70}")
    print(f"Phase 159 Complete ({elapsed:.0f}s)")
    print(f"{'Config':>15} {'BL PA':>8} {'BL Pos':>8} {'CL PA':>8} {'CL Pos':>8}")
    print("-"*50)
    for name, r in results.items():
        print(f"{name:>15} {r['baseline']['pixel_acc']*100:>7.2f}% "
              f"{r['baseline']['position_correct']*100:>7.1f}% "
              f"{r['clock']['pixel_acc']*100:>7.2f}% "
              f"{r['clock']['position_correct']*100:>7.1f}%")
    print(f"\n  Glider synthesis achieved: {can_glide}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase159_glider.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 159: Glider Synthesis',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'can_glide': can_glide,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        names = list(results.keys())
        bl_pa = [results[n]['baseline']['pixel_acc']*100 for n in names]
        cl_pa = [results[n]['clock']['pixel_acc']*100 for n in names]
        x = np.arange(len(names))
        axes[0].bar(x - 0.15, bl_pa, 0.3, label='Baseline', color='#3498db', alpha=0.8)
        axes[0].bar(x + 0.15, cl_pa, 0.3, label='Clock', color='#2ecc71', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, fontsize=8, rotation=20)
        axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title('Glider Pixel Accuracy', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=8)

        bl_pc = [results[n]['baseline']['position_correct']*100 for n in names]
        cl_pc = [results[n]['clock']['position_correct']*100 for n in names]
        axes[1].bar(x - 0.15, bl_pc, 0.3, label='Baseline', color='#3498db', alpha=0.8)
        axes[1].bar(x + 0.15, cl_pc, 0.3, label='Clock', color='#2ecc71', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, fontsize=8, rotation=20)
        axes[1].set_ylabel('Position Correct (%)')
        axes[1].set_title('Movement Accuracy', fontweight='bold', fontsize=10)
        axes[1].legend(fontsize=8)

        bl_sp = [results[n]['baseline']['shape_preserved']*100 for n in names]
        cl_sp = [results[n]['clock']['shape_preserved']*100 for n in names]
        axes[2].bar(x - 0.15, bl_sp, 0.3, label='Baseline', color='#3498db', alpha=0.8)
        axes[2].bar(x + 0.15, cl_sp, 0.3, label='Clock', color='#2ecc71', alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names, fontsize=8, rotation=20)
        axes[2].set_ylabel('Shape Preserved (%)')
        axes[2].set_title('Shape Preservation', fontweight='bold', fontsize=10)
        axes[2].legend(fontsize=8)

        fig.suptitle('Phase 159: Glider Synthesis (Self-Moving Digital Life)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.15, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase159_glider.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
