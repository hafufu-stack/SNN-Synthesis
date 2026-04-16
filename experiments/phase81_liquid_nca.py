"""
Phase 81: Liquid Neural Cellular Automata (L-NCA)

Each pixel = 1 Liquid-LIF neuron. Only interacts with 3x3 neighbors.
Weight-shared across all cells -> size-free (works on any grid size).

Tasks (synthetic ARC-like):
  A) Fill: given a seed pixel, fill the connected region
  B) Gravity: pixels "fall" to the bottom of the grid
  C) Expand: a shape grows by 1 pixel in each direction

Compare: Standard NCA (no tau) vs L-NCA (dynamic tau)

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

HIDDEN_CH = 8
NCA_STEPS = 10
GRID_SIZE = 8
EPOCHS = 50
LR = 1e-3
N_TRAIN = 3000
N_TEST = 500
BATCH_SIZE = 32


# ==============================================================
# Synthetic ARC-like Tasks
# ==============================================================
def gen_gravity_task(n_samples, grid_size=8, seed=None):
    """Pixels 'fall' to the bottom row. Input: scattered pixels, Output: gravity-applied."""
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n_samples):
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        n_pixels = rng.randint(2, grid_size)
        cols = rng.choice(grid_size, size=n_pixels, replace=True)
        rows = rng.randint(0, grid_size, size=n_pixels)
        for r, c in zip(rows, cols):
            grid[r, c] = 1.0

        # Apply gravity: for each column, count active pixels and stack at bottom
        result = np.zeros_like(grid)
        for c in range(grid_size):
            count = int(grid[:, c].sum())
            for r in range(count):
                result[grid_size - 1 - r, c] = 1.0

        inputs.append(grid)
        targets.append(result)
    return (torch.tensor(np.array(inputs)).unsqueeze(1),
            torch.tensor(np.array(targets)).unsqueeze(1))


def gen_expand_task(n_samples, grid_size=8, seed=None):
    """A shape expands by 1 pixel in each direction (cross expansion)."""
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n_samples):
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        # Place 1-3 seed pixels
        n_seeds = rng.randint(1, 4)
        for _ in range(n_seeds):
            y, x = rng.randint(1, grid_size - 1, size=2)
            grid[y, x] = 1.0

        # Expand: each active pixel activates its 4-neighbors
        result = grid.copy()
        for y in range(grid_size):
            for x in range(grid_size):
                if grid[y, x] > 0:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < grid_size and 0 <= nx < grid_size:
                            result[ny, nx] = 1.0

        inputs.append(grid)
        targets.append(result)
    return (torch.tensor(np.array(inputs)).unsqueeze(1),
            torch.tensor(np.array(targets)).unsqueeze(1))


# ==============================================================
# L-NCA: Liquid Neural Cellular Automata
# ==============================================================
class LiquidNCA(nn.Module):
    """Neural Cellular Automata with Liquid-LIF dynamics.
    Each cell has per-pixel dynamic tau. Weight-shared 3x3 kernel."""
    def __init__(self, in_ch=1, hidden_ch=8, out_ch=1, use_liquid=True):
        super().__init__()
        self.use_liquid = use_liquid
        self.hidden_ch = hidden_ch

        # Perception: 3x3 conv to read neighborhood
        self.perceive = nn.Conv2d(in_ch + hidden_ch, hidden_ch * 2, 3, padding=1)
        # Update rule
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
        )
        # Tau gate (liquid dynamics)
        if use_liquid:
            self.tau_gate = nn.Conv2d(in_ch + hidden_ch * 2, hidden_ch, 3, padding=1)
            self.b_tau = nn.Parameter(torch.ones(1, hidden_ch, 1, 1) * 1.5)

        # Readout
        self.readout = nn.Conv2d(hidden_ch, out_ch, 1)

    def step(self, x_input, state):
        """One NCA step. x_input: (B,1,H,W), state: (B,hidden,H,W)"""
        combined = torch.cat([x_input, state], dim=1)
        perception = self.perceive(combined)
        delta = self.update(perception)

        if self.use_liquid:
            tau_input = torch.cat([x_input, state, delta], dim=1)
            beta = torch.sigmoid(self.tau_gate(tau_input) + self.b_tau)
            beta = torch.clamp(beta, 0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        else:
            state = 0.9 * state + 0.1 * delta

        return state

    def forward(self, x, n_steps=NCA_STEPS):
        """Run NCA for n_steps."""
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hidden_ch, h, w, device=x.device)

        for _ in range(n_steps):
            state = self.step(x, state)

        return torch.sigmoid(self.readout(state))


# ==============================================================
# Training
# ==============================================================
def train_and_eval(model, x_train, y_train, x_test, y_test, task_name):
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
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}: loss={total_loss/n:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(DEVICE))
        pred_binary = (pred > 0.5).float()
        target = y_test.to(DEVICE)
        # Per-pixel accuracy
        pixel_acc = (pred_binary == target).float().mean().item()
        # Per-sample exact match
        exact = (pred_binary.reshape(pred_binary.size(0), -1) ==
                 target.reshape(target.size(0), -1)).all(dim=1).float().mean().item()

    return pixel_acc, exact


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase81_liquid_nca.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 81: Liquid Neural Cellular Automata',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 81: Liquid Neural Cellular Automata (L-NCA)")
    print("  Size-free local computation for ARC-like tasks")
    print("=" * 70)

    TASKS = {
        'gravity': gen_gravity_task,
        'expand': gen_expand_task,
    }

    all_results = {}

    for task_name, task_fn in TASKS.items():
        print(f"\n  --- Task: {task_name} ---")
        x_train, y_train = task_fn(N_TRAIN, GRID_SIZE, seed=SEED)
        x_test, y_test = task_fn(N_TEST, GRID_SIZE, seed=SEED+1)

        for model_name, use_liquid in [('NCA', False), ('L-NCA', True)]:
            print(f"\n    === {model_name} ===")
            model = LiquidNCA(use_liquid=use_liquid).to(DEVICE)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"      Params: {n_params:,}")

            t0 = time.time()
            pixel_acc, exact_match = train_and_eval(
                model, x_train, y_train, x_test, y_test, task_name)
            elapsed = time.time() - t0

            key = f"{task_name}_{model_name}"
            all_results[key] = {
                'task': task_name, 'model': model_name,
                'pixel_acc': pixel_acc, 'exact_match': exact_match,
                'n_params': n_params, 'time': elapsed,
            }
            print(f"      Pixel acc: {pixel_acc*100:.1f}%, "
                  f"Exact match: {exact_match*100:.1f}%, Time: {elapsed:.0f}s")

            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        _save(all_results)

    # Test size-free property: train on 8x8, test on 12x12
    print(f"\n  --- Size-Free Test: Train 8x8, Test 12x12 ---")
    x_train_8, y_train_8 = gen_expand_task(N_TRAIN, 8, seed=SEED)
    x_test_12, y_test_12 = gen_expand_task(N_TEST, 12, seed=SEED+2)

    model = LiquidNCA(use_liquid=True).to(DEVICE)
    pixel_acc_8, exact_8 = train_and_eval(
        model, x_train_8, y_train_8, x_train_8[:N_TEST], y_train_8[:N_TEST], 'expand')

    model.eval()
    with torch.no_grad():
        pred_12 = model(x_test_12.to(DEVICE))
        pred_binary = (pred_12 > 0.5).float()
        target = y_test_12.to(DEVICE)
        pixel_acc_12 = (pred_binary == target).float().mean().item()
        exact_12 = (pred_binary.reshape(-1, 12*12) ==
                    target.reshape(-1, 12*12)).all(dim=1).float().mean().item()

    all_results['size_free'] = {
        'train_size': 8, 'test_size': 12,
        'pixel_acc_train': pixel_acc_8, 'exact_train': exact_8,
        'pixel_acc_transfer': pixel_acc_12, 'exact_transfer': exact_12,
    }
    print(f"    8x8 train: pixel={pixel_acc_8*100:.1f}%, exact={exact_8*100:.1f}%")
    print(f"    12x12 test: pixel={pixel_acc_12*100:.1f}%, exact={exact_12*100:.1f}%")
    _save(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: L-NCA")
    print(f"{'='*70}")
    for key, r in all_results.items():
        if key != 'size_free':
            print(f"  {key:25s}: pixel={r['pixel_acc']*100:.1f}% "
                  f"exact={r['exact_match']*100:.1f}% params={r['n_params']:,}")
    sf = all_results.get('size_free', {})
    if sf:
        print(f"  {'SIZE-FREE (8->12)':25s}: pixel={sf['pixel_acc_transfer']*100:.1f}% "
              f"exact={sf['exact_transfer']*100:.1f}%")

    _generate_figure(all_results)
    print("\nPhase 81 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        tasks = ['gravity', 'expand']
        for i, task in enumerate(tasks):
            ax = axes[i]
            models = ['NCA', 'L-NCA']
            pixel = [results[f'{task}_{m}']['pixel_acc']*100 for m in models]
            exact = [results[f'{task}_{m}']['exact_match']*100 for m in models]
            x_pos = np.arange(len(models))
            w = 0.35
            b1 = ax.bar(x_pos - w/2, pixel, w, label='Pixel Acc', color='#3B82F6')
            b2 = ax.bar(x_pos + w/2, exact, w, label='Exact Match', color='#EC4899')
            for b, v in zip(b1, pixel):
                ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=9)
            for b, v in zip(b2, exact):
                ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=9)
            ax.set_xticks(x_pos); ax.set_xticklabels(models)
            ax.set_ylabel('Accuracy (%)'); ax.set_title(f'Task: {task}', fontweight='bold')
            ax.legend(); ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 110)

        fig.suptitle('Phase 81: Liquid Neural Cellular Automata\nARC-like Local Rule Learning',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase81_liquid_nca.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
