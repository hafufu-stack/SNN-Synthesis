"""
Phase 153: Von Neumann Self-Replicator  -  NCA as Artificial Life

Phase 148 proved NCA is Turing complete. Von Neumann showed that
Turing-complete cellular automata can self-replicate.

Task: Place a seed pattern in the corner of a large grid. Train NCA
to replicate that pattern across the entire grid. If successful,
this proves NCA can implement the fundamental operation of life:
self-reproduction.

Tests: 1x1→2x2, 2x2→full tiling, 3x3→full tiling, L-shape→full tiling.

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
# Pattern generation (seed → full tiling)
# ================================================================
def create_tiling_pair(pattern, grid_size=GRID_SIZE):
    """Create input (seed in corner) and output (full tiling) pair."""
    ph, pw = pattern.shape
    # Input: pattern in top-left corner, rest is 0
    inp = torch.zeros(grid_size, grid_size)
    inp[:ph, :pw] = pattern

    # Output: tile pattern across entire grid
    out = torch.zeros(grid_size, grid_size)
    for y in range(0, grid_size, ph):
        for x in range(0, grid_size, pw):
            h = min(ph, grid_size - y)
            w = min(pw, grid_size - x)
            out[y:y+h, x:x+w] = pattern[:h, :w]

    return inp, out


def generate_replication_data(n_samples, pattern_type='2x2'):
    """Generate seed → replicated grid pairs."""
    X, Y = [], []
    for _ in range(n_samples):
        if pattern_type == '1x1':
            val = random.choice([0.3, 0.5, 0.7, 1.0])
            pattern = torch.tensor([[val]])
        elif pattern_type == '2x2':
            pattern = torch.rand(2, 2).round()  # Binary 2x2
            if pattern.sum() == 0:
                pattern[0, 0] = 1.0  # Avoid empty
        elif pattern_type == '3x3':
            pattern = torch.rand(3, 3).round()
            if pattern.sum() == 0:
                pattern[1, 1] = 1.0
        elif pattern_type == 'L_shape':
            pattern = torch.zeros(3, 3)
            pattern[0, 0] = 1.0
            pattern[1, 0] = 1.0
            pattern[2, 0] = 1.0
            pattern[2, 1] = 1.0
            # Randomly flip/rotate
            if random.random() > 0.5:
                pattern = pattern.flip(0)
            if random.random() > 0.5:
                pattern = pattern.flip(1)
        elif pattern_type == 'random_nxn':
            n = random.choice([2, 3, 4])
            pattern = torch.rand(n, n).round()
            if pattern.sum() == 0:
                pattern[0, 0] = 1.0
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        inp, out = create_tiling_pair(pattern, GRID_SIZE)
        X.append(inp)
        Y.append(out)

    return torch.stack(X).unsqueeze(1), torch.stack(Y).unsqueeze(1)


class ReplicatorDS(torch.utils.data.Dataset):
    def __init__(self, n=3000, pattern_type='2x2'):
        self.X, self.Y = generate_replication_data(n, pattern_type)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Self-Replicator NCA
# ================================================================
class ReplicatorNCA(nn.Module):
    """NCA that learns to tile a seed pattern across the grid."""
    def __init__(self, ch=48, steps=20):
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


class ReplicatorClockNCA(nn.Module):
    """Replicator with external clock for phase-aware replication."""
    def __init__(self, ch=48, steps=20):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        # +2: clock channel + "filled" fraction channel
        self.rule = nn.Sequential(
            nn.Conv2d(ch + 2, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for t in range(steps):
            B, C, H, W = h.shape
            clock = torch.full((B, 1, H, W), t / steps, device=h.device)
            # Fraction of non-zero cells (growth indicator)
            with torch.no_grad():
                filled = (torch.sigmoid(self.proj_out(h)) > 0.5).float().mean(dim=(2,3), keepdim=True)
                filled = filled.expand(B, 1, H, W)
            h_aug = torch.cat([h, clock, filled], dim=1)
            h = F.relu(h + self.rule(h_aug))
        return self.proj_out(h)


# ================================================================
# Training & evaluation
# ================================================================
def train_replicator(model, loader, epochs=80, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

def eval_replicator(model, loader):
    model.eval()
    correct = pixels = exact = total = 0
    repl_scores = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = (torch.sigmoid(model(x)) > 0.5).float()
            correct += (pred == y).sum().item()
            pixels += y.numel()
            for b in range(y.size(0)):
                match = (pred[b] == y[b]).all().item()
                exact += match
                # Replication score: fraction of correctly tiled cells
                repl_scores.append((pred[b] == y[b]).float().mean().item())
                total += 1
    return correct/pixels, exact/total, np.mean(repl_scores)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 153: Von Neumann Self-Replicator")
    print(f"  Task: Seed pattern → Full-grid tiling")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    pattern_types = ['1x1', '2x2', '3x3', 'L_shape', 'random_nxn']
    results = {}

    for ptype in pattern_types:
        print(f"\n{'='*50}")
        print(f"Pattern: {ptype}")
        print(f"{'='*50}")

        train_ds = ReplicatorDS(3000, ptype)
        test_ds = ReplicatorDS(500, ptype)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

        # Baseline NCA
        model = ReplicatorNCA(ch=48, steps=20).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters())
        print(f"  [Baseline] Training ({n_p:,} params)...")
        train_replicator(model, train_loader, epochs=80)
        pa, em, rs = eval_replicator(model, test_loader)
        print(f"  PA={pa*100:.2f}%, EM={em*100:.2f}%, ReplScore={rs*100:.2f}%")

        # Clock NCA
        model_clock = ReplicatorClockNCA(ch=48, steps=20).to(DEVICE)
        n_p2 = sum(p.numel() for p in model_clock.parameters())
        print(f"  [Clock] Training ({n_p2:,} params)...")
        train_replicator(model_clock, train_loader, epochs=80)
        pa2, em2, rs2 = eval_replicator(model_clock, test_loader)
        print(f"  PA={pa2*100:.2f}%, EM={em2*100:.2f}%, ReplScore={rs2*100:.2f}%")

        # Step generalization (train at 20 steps, test at 40)
        model.eval()
        model.steps = 40
        pa_40, em_40, rs_40 = eval_replicator(model, test_loader)
        model.steps = 20
        print(f"  [Baseline@40 steps] PA={pa_40*100:.2f}%, EM={em_40*100:.2f}%")

        results[ptype] = {
            'baseline': {'pixel_acc': pa, 'exact_match': em, 'repl_score': rs, 'params': n_p},
            'clock': {'pixel_acc': pa2, 'exact_match': em2, 'repl_score': rs2, 'params': n_p2},
            'baseline_40': {'pixel_acc': pa_40, 'exact_match': em_40, 'repl_score': rs_40},
        }

        del model, model_clock; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 153 Complete ({elapsed:.0f}s)")
    print(f"{'Pattern':>12} {'Baseline PA':>12} {'Clock PA':>10} {'BL EM':>8} {'CL EM':>8}")
    print("-"*54)
    for ptype, r in results.items():
        print(f"{ptype:>12} {r['baseline']['pixel_acc']*100:>11.2f}% "
              f"{r['clock']['pixel_acc']*100:>9.2f}% "
              f"{r['baseline']['exact_match']*100:>7.2f}% "
              f"{r['clock']['exact_match']*100:>7.2f}%")

    can_replicate = any(r['baseline']['exact_match'] > 0.5 or r['clock']['exact_match'] > 0.5
                       for r in results.values())
    print(f"\n  Self-replication achieved: {can_replicate}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase153_self_replicator.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 153: Von Neumann Self-Replicator',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'can_replicate': can_replicate,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        ptypes = list(results.keys())
        bl_pa = [results[p]['baseline']['pixel_acc']*100 for p in ptypes]
        cl_pa = [results[p]['clock']['pixel_acc']*100 for p in ptypes]
        x = np.arange(len(ptypes))
        axes[0].bar(x - 0.15, bl_pa, 0.3, label='Baseline', color='#3498db', alpha=0.8)
        axes[0].bar(x + 0.15, cl_pa, 0.3, label='Clock', color='#2ecc71', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ptypes, fontsize=8, rotation=20)
        axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title('Self-Replication Accuracy', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=8)

        bl_em = [results[p]['baseline']['exact_match']*100 for p in ptypes]
        cl_em = [results[p]['clock']['exact_match']*100 for p in ptypes]
        axes[1].bar(x - 0.15, bl_em, 0.3, label='Baseline', color='#3498db', alpha=0.8)
        axes[1].bar(x + 0.15, cl_em, 0.3, label='Clock', color='#2ecc71', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(ptypes, fontsize=8, rotation=20)
        axes[1].set_ylabel('Exact Match (%)')
        axes[1].set_title('Perfect Replication Rate', fontweight='bold', fontsize=10)
        axes[1].legend(fontsize=8)

        # Step generalization
        bl_20 = [results[p]['baseline']['pixel_acc']*100 for p in ptypes]
        bl_40 = [results[p]['baseline_40']['pixel_acc']*100 for p in ptypes]
        axes[2].bar(x - 0.15, bl_20, 0.3, label='20 steps', color='#3498db', alpha=0.8)
        axes[2].bar(x + 0.15, bl_40, 0.3, label='40 steps', color='#e74c3c', alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(ptypes, fontsize=8, rotation=20)
        axes[2].set_ylabel('Pixel Accuracy (%)')
        axes[2].set_title('Temporal Generalization', fontweight='bold', fontsize=10)
        axes[2].legend(fontsize=8)

        fig.suptitle('Phase 153: Von Neumann Self-Replicator (Seed → Full Tiling)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.15, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase153_self_replicator.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
