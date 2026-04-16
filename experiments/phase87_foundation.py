"""
Phase 87: Multi-Task Foundation L-NCA

Train L-NCA on multiple ARC-like tasks simultaneously to create
a "Foundation Backbone" that can adapt to any new task.

Tasks: gravity, expand, color_invert, move_right, fill_border
All 10-color, all size-free (3x3 local rules).

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
N_PER_TASK = 1000
N_TEST = 200


def to_onehot(grid, nc=N_COLORS):
    h, w = grid.shape
    oh = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): oh[c] = (grid == c).astype(np.float32)
    return oh


# === Task generators ===
def gen_gravity(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)):
            r, c = rng.randint(0, gs, size=2); grid[r, c] = 1
        for _ in range(rng.randint(1, 4)):
            r, c = rng.randint(0, gs, size=2)
            if grid[r, c] == 0: grid[r, c] = 2
        result = np.zeros_like(grid)
        for r in range(gs):
            for c in range(gs):
                if grid[r, c] == 2: result[r, c] = 2
        for c in range(gs):
            cnt = sum(1 for r in range(gs) if grid[r, c] == 1)
            row, placed = gs-1, 0
            while placed < cnt and row >= 0:
                if result[row, c] == 0: result[row, c] = 1; placed += 1
                row -= 1
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def gen_expand(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)):
            y, x = rng.randint(1, gs-1, size=2)
            c = rng.randint(1, 5)
            if grid[y, x] == 0: grid[y, x] = c
        result = grid.copy()
        for y in range(gs):
            for x in range(gs):
                if grid[y, x] > 0:
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < gs and 0 <= nx < gs and result[ny, nx] == 0:
                            result[ny, nx] = grid[y, x]
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def gen_color_invert(n, gs=8, seed=None):
    """Swap color 1 and color 2, keep everything else."""
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(3, 8)):
            r, c = rng.randint(0, gs, size=2)
            grid[r, c] = rng.randint(1, 4)
        result = grid.copy()
        result[grid == 1] = 2
        result[grid == 2] = 1
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def gen_move_right(n, gs=8, seed=None):
    """Move all non-zero pixels 1 step right (wrap around)."""
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 6)):
            r, c = rng.randint(0, gs, size=2)
            grid[r, c] = rng.randint(1, 5)
        result = np.zeros_like(grid)
        for r in range(gs):
            for c in range(gs):
                if grid[r, c] > 0:
                    result[r, (c+1) % gs] = grid[r, c]
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def gen_fill_border(n, gs=8, seed=None):
    """Fill the border of the grid with color 3, keep interior."""
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 6)):
            r, c = rng.randint(0, gs, size=2)
            grid[r, c] = rng.randint(1, 5)
        result = grid.copy()
        result[0, :] = 3; result[-1, :] = 3
        result[:, 0] = 3; result[:, -1] = 3
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


ALL_TASKS = {
    'gravity': gen_gravity,
    'expand': gen_expand,
    'color_invert': gen_color_invert,
    'move_right': gen_move_right,
    'fill_border': gen_fill_border,
}


class MultiColorLNCA(nn.Module):
    def __init__(self, nc=10, hc=16):
        super().__init__()
        self.hc = hc
        self.perceive = nn.Conv2d(nc + hc, hc * 2, 3, padding=1)
        self.update = nn.Sequential(
            nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(nc + hc * 2, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)

    def forward(self, x, n_steps=NCA_STEPS):
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        for _ in range(n_steps):
            combined = torch.cat([x, state], dim=1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x, state, delta], dim=1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)


def _save(results, fname="phase87_foundation.json"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, fname), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 87: Multi-Task Foundation L-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 87: Multi-Task Foundation L-NCA")
    print(f"  Tasks: {list(ALL_TASKS.keys())}")
    print("=" * 70)

    # Generate all tasks
    all_x, all_y = [], []
    for task_name, gen_fn in ALL_TASKS.items():
        x, y = gen_fn(N_PER_TASK, GRID_SIZE, seed=SEED)
        all_x.append(x); all_y.append(y)

    x_train = torch.cat(all_x)
    y_train = torch.cat(all_y)
    n = x_train.size(0)
    print(f"  Total samples: {n} ({N_PER_TASK} x {len(ALL_TASKS)} tasks)")

    # Train foundation model
    model = MultiColorLNCA().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = x_train[idx].to(DEVICE), y_train[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n:.4f}")

    # Evaluate per-task
    results = {'n_params': n_params}
    offset = 0
    for task_name, gen_fn in ALL_TASKS.items():
        x_test, y_test = gen_fn(N_TEST, GRID_SIZE, seed=SEED+1)
        model.eval()
        with torch.no_grad():
            preds = model(x_test.to(DEVICE)).argmax(1)
            target = y_test.to(DEVICE)
            pixel = (preds == target).float().mean().item()
            exact = (preds.reshape(N_TEST, -1) == target.reshape(N_TEST, -1)).all(1).float().mean().item()
        results[task_name] = {'pixel_acc': pixel, 'exact_match': exact}
        print(f"    {task_name:15s}: pixel={pixel*100:.1f}% exact={exact*100:.1f}%")

    # Size-free test
    x12, y12 = gen_expand(N_TEST, 12, seed=SEED+2)
    model.eval()
    with torch.no_grad():
        p12 = model(x12.to(DEVICE)).argmax(1)
        t12 = y12.to(DEVICE)
        px12 = (p12 == t12).float().mean().item()
        ex12 = (p12.reshape(N_TEST, -1) == t12.reshape(N_TEST, -1)).all(1).float().mean().item()
    results['size_free_expand_12'] = {'pixel_acc': px12, 'exact_match': ex12}
    print(f"    {'size_free(12x12)':15s}: pixel={px12*100:.1f}% exact={ex12*100:.1f}%")

    # Save model weights for Phase 88/89
    model_path = os.path.join(RESULTS_DIR, "foundation_lnca.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Backbone saved: {model_path}")

    _save(results)

    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Foundation L-NCA")
    print(f"{'='*70}")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k:20s}: pixel={v['pixel_acc']*100:.1f}% exact={v['exact_match']*100:.1f}%")

    _generate_figure(results)
    print("\nPhase 87 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tasks = [k for k, v in results.items() if isinstance(v, dict)]
        fig, ax = plt.subplots(figsize=(10, 5))
        pixel = [results[t]['pixel_acc']*100 for t in tasks]
        exact = [results[t]['exact_match']*100 for t in tasks]
        x_pos = np.arange(len(tasks))
        w = 0.35
        b1 = ax.bar(x_pos-w/2, pixel, w, label='Pixel', color='#3B82F6')
        b2 = ax.bar(x_pos+w/2, exact, w, label='Exact', color='#EC4899')
        for b, v in zip(b1, pixel):
            ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=8)
        for b, v in zip(b2, exact):
            ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=8)
        ax.set_xticks(x_pos); ax.set_xticklabels(tasks, rotation=15)
        ax.set_ylabel('Accuracy (%)'); ax.legend(); ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 115)
        fig.suptitle('Phase 87: Multi-Task Foundation L-NCA\n5 ARC-like tasks, single backbone',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase87_foundation.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
