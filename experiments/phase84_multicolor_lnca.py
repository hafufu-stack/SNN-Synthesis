"""
Phase 84: Multi-Color L-NCA (10-color ARC Vision)

Extends L-NCA from binary (0/1) to 10-color ARC format.
Input: 10-channel one-hot per pixel.
Output: 10-class per-pixel classification.

Tasks (color-dependent ARC-like):
  A) Color Gravity: only red (color 1) falls, blue (color 2) stays
  B) Color Expand: each color expands into its own 4-neighbors

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

N_COLORS = 10
HIDDEN_CH = 16
NCA_STEPS = 10
GRID_SIZE = 8
EPOCHS = 60
LR = 1e-3
BATCH_SIZE = 32
N_TRAIN = 3000
N_TEST = 500


def to_onehot(grid, n_colors=N_COLORS):
    """Convert integer grid (H,W) to one-hot (n_colors, H, W)."""
    h, w = grid.shape
    oh = np.zeros((n_colors, h, w), dtype=np.float32)
    for c in range(n_colors):
        oh[c] = (grid == c).astype(np.float32)
    return oh


def gen_color_gravity(n, gs=8, seed=None):
    """Color 1 (red) falls to bottom. Color 2 (blue) stays. Color 0 = background."""
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        n_red = rng.randint(2, 5)
        n_blue = rng.randint(1, 4)
        for _ in range(n_red):
            r, c = rng.randint(0, gs, size=2)
            grid[r, c] = 1  # red
        for _ in range(n_blue):
            r, c = rng.randint(0, gs, size=2)
            if grid[r, c] == 0:
                grid[r, c] = 2  # blue

        result = np.zeros_like(grid)
        # Blue stays in place
        for r in range(gs):
            for c in range(gs):
                if grid[r, c] == 2:
                    result[r, c] = 2
        # Red falls (skip cells with blue)
        for c in range(gs):
            red_count = 0
            for r in range(gs):
                if grid[r, c] == 1:
                    red_count += 1
            # Stack red from bottom, skipping blue cells
            row = gs - 1
            placed = 0
            while placed < red_count and row >= 0:
                if result[row, c] == 0:
                    result[row, c] = 1
                    placed += 1
                row -= 1

        inputs.append(to_onehot(grid))
        targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def gen_color_expand(n, gs=8, seed=None):
    """Each colored pixel expands cross-wise. Different colors don't overlap (first wins)."""
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        n_seeds = rng.randint(2, 5)
        for _ in range(n_seeds):
            y, x = rng.randint(1, gs-1, size=2)
            color = rng.randint(1, 5)  # colors 1-4
            if grid[y, x] == 0:
                grid[y, x] = color

        result = grid.copy()
        for y in range(gs):
            for x in range(gs):
                if grid[y, x] > 0:
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < gs and 0 <= nx < gs and result[ny, nx] == 0:
                            result[ny, nx] = grid[y, x]

        inputs.append(to_onehot(grid))
        targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


class MultiColorLNCA(nn.Module):
    """L-NCA for 10-color ARC grids. Weight-shared, size-free."""
    def __init__(self, n_colors=10, hidden_ch=16):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.perceive = nn.Conv2d(n_colors + hidden_ch, hidden_ch * 2, 3, padding=1)
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch, 1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau_gate = nn.Conv2d(n_colors + hidden_ch * 2, hidden_ch, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hidden_ch, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hidden_ch, n_colors, 1)

    def step(self, x_input, state):
        combined = torch.cat([x_input, state], dim=1)
        perception = self.perceive(combined)
        delta = self.update(perception)
        tau_input = torch.cat([x_input, state, delta], dim=1)
        beta = torch.sigmoid(self.tau_gate(tau_input) + self.b_tau)
        beta = torch.clamp(beta, 0.01, 0.99)
        state = beta * state + (1 - beta) * delta
        return state

    def forward(self, x, n_steps=NCA_STEPS):
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hidden_ch, h, w, device=x.device)
        for _ in range(n_steps):
            state = self.step(x, state)
        return self.readout(state)  # (B, n_colors, H, W)


def train_and_eval(model, x_train, y_train, x_test, y_test):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n = x_train.size(0)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb = x_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)  # (B, 10, H, W)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/n:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(DEVICE))
        preds = logits.argmax(dim=1)
        target = y_test.to(DEVICE)
        pixel_acc = (preds == target).float().mean().item()
        exact = (preds.reshape(preds.size(0), -1) ==
                 target.reshape(target.size(0), -1)).all(dim=1).float().mean().item()
    return pixel_acc, exact


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase84_multicolor_lnca.json"), 'w',
              encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 84: Multi-Color L-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 84: Multi-Color L-NCA (10-color ARC Vision)")
    print("=" * 70)

    TASKS = {
        'color_gravity': gen_color_gravity,
        'color_expand': gen_color_expand,
    }
    results = {}

    for task_name, task_fn in TASKS.items():
        print(f"\n  --- Task: {task_name} ---")
        x_train, y_train = task_fn(N_TRAIN, GRID_SIZE, seed=SEED)
        x_test, y_test = task_fn(N_TEST, GRID_SIZE, seed=SEED+1)

        model = MultiColorLNCA().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")

        t0 = time.time()
        pixel_acc, exact = train_and_eval(model, x_train, y_train, x_test, y_test)
        elapsed = time.time() - t0

        results[task_name] = {
            'pixel_acc': pixel_acc, 'exact_match': exact,
            'n_params': n_params, 'time': elapsed
        }
        print(f"    Pixel: {pixel_acc*100:.1f}%, Exact: {exact*100:.1f}%, Time: {elapsed:.0f}s")

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _save(results)

    # Size-free test: train 8x8, test 12x12
    print(f"\n  --- Size-Free: Train 8x8, Test 12x12 ---")
    x_train_8, y_train_8 = gen_color_expand(N_TRAIN, 8, seed=SEED)
    x_test_12, y_test_12 = gen_color_expand(N_TEST, 12, seed=SEED+2)
    model = MultiColorLNCA().to(DEVICE)
    train_and_eval(model, x_train_8, y_train_8, x_train_8[:200], y_train_8[:200])
    model.eval()
    with torch.no_grad():
        logits = model(x_test_12.to(DEVICE))
        preds = logits.argmax(dim=1)
        target = y_test_12.to(DEVICE)
        px_12 = (preds == target).float().mean().item()
        ex_12 = (preds.reshape(-1, 12*12) == target.reshape(-1, 12*12)).all(dim=1).float().mean().item()
    results['size_free_12'] = {'pixel_acc': px_12, 'exact_match': ex_12}
    print(f"    12x12: pixel={px_12*100:.1f}%, exact={ex_12*100:.1f}%")
    _save(results)

    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Multi-Color L-NCA")
    print(f"{'='*70}")
    for k, r in results.items():
        print(f"  {k:20s}: pixel={r['pixel_acc']*100:.1f}% exact={r['exact_match']*100:.1f}%")

    _generate_figure(results)
    print("\nPhase 84 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tasks = [k for k in results if k != 'size_free_12']
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        x_pos = np.arange(len(tasks))
        pixel = [results[t]['pixel_acc']*100 for t in tasks]
        exact = [results[t]['exact_match']*100 for t in tasks]
        w = 0.35
        b1 = ax.bar(x_pos - w/2, pixel, w, label='Pixel Acc', color='#3B82F6')
        b2 = ax.bar(x_pos + w/2, exact, w, label='Exact Match', color='#EC4899')
        for b, v in zip(b1, pixel):
            ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=10)
        for b, v in zip(b2, exact):
            ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=10)
        ax.set_xticks(x_pos); ax.set_xticklabels(tasks)
        ax.set_ylabel('Accuracy (%)'); ax.legend(); ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 110)
        fig.suptitle('Phase 84: Multi-Color L-NCA\n10-color ARC-like Tasks',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase84_multicolor_lnca.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
