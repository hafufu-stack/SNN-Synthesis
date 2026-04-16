"""
Phase 85: TTT Budget Optimization

How many Backprop steps can we fit in Kaggle's time budget?
Measures latency vs accuracy trade-off for Test-Time Training.

Protocol:
  1. Train L-NCA backbone on 'expand' (offline)
  2. For each step_count in [0, 10, 20, 50, 100, 200]:
     - Fine-tune on 2 demo pairs (like ARC train examples)
     - Measure adaptation latency (ms)
     - Measure test accuracy on novel samples

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy
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
BACKBONE_EPOCHS = 40
LR = 1e-3
TTT_LR = 0.01
BATCH_SIZE = 32
N_TRAIN = 2000
N_DEMO = 3   # ARC gives 2-3 demo pairs
N_TEST = 200
STEP_COUNTS = [0, 10, 20, 50, 100, 200]


def to_onehot(grid, nc=N_COLORS):
    h, w = grid.shape
    oh = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc):
        oh[c] = (grid == c).astype(np.float32)
    return oh


def gen_color_gravity(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)):
            r, c = rng.randint(0, gs, size=2)
            grid[r, c] = 1
        for _ in range(rng.randint(1, 4)):
            r, c = rng.randint(0, gs, size=2)
            if grid[r, c] == 0: grid[r, c] = 2
        result = np.zeros_like(grid)
        for r in range(gs):
            for c in range(gs):
                if grid[r, c] == 2: result[r, c] = 2
        for c in range(gs):
            red_count = sum(1 for r in range(gs) if grid[r, c] == 1)
            row, placed = gs - 1, 0
            while placed < red_count and row >= 0:
                if result[row, c] == 0:
                    result[row, c] = 1; placed += 1
                row -= 1
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def gen_color_expand(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)):
            y, x = rng.randint(1, gs-1, size=2)
            color = rng.randint(1, 5)
            if grid[y, x] == 0: grid[y, x] = color
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
            perception = self.perceive(combined)
            delta = self.update(perception)
            tau_input = torch.cat([x, state, delta], dim=1)
            beta = torch.sigmoid(self.tau_gate(tau_input) + self.b_tau)
            beta = torch.clamp(beta, 0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)


def train_backbone(model, x_train, y_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n = x_train.size(0)
    for epoch in range(BACKBONE_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = x_train[idx].to(DEVICE), y_train[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                p = model(x_train[:200].to(DEVICE)).argmax(1)
                acc = (p == y_train[:200].to(DEVICE)).float().mean()
            print(f"      Epoch {epoch+1}: pixel={acc*100:.1f}%")


def ttt_adapt(model, demo_x, demo_y, n_steps, lr=TTT_LR):
    """Test-Time Training: fine-tune on demo pairs."""
    if n_steps == 0:
        return 0.0
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad()
        logits = model(demo_x)
        loss = F.cross_entropy(logits, demo_y)
        loss.backward()
        optimizer.step()
    latency = (time.perf_counter() - t0) * 1000
    return latency


def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(x_test.to(DEVICE)).argmax(1)
        target = y_test.to(DEVICE)
        pixel = (preds == target).float().mean().item()
        exact = (preds.reshape(preds.size(0), -1) ==
                 target.reshape(target.size(0), -1)).all(dim=1).float().mean().item()
    return pixel, exact


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase85_ttt_budget.json"), 'w',
              encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 85: TTT Budget Optimization',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 85: TTT Budget Optimization")
    print("  How many Backprop steps fit in 450ms?")
    print("=" * 70)

    # Train backbone on 'expand'
    print("\n  Training backbone on 'color_expand'...")
    x_train, y_train = gen_color_expand(N_TRAIN, GRID_SIZE, seed=SEED)
    backbone = MultiColorLNCA().to(DEVICE)
    train_backbone(backbone, x_train, y_train)

    # Prepare novel task (gravity - not seen during backbone training)
    x_novel, y_novel = gen_color_gravity(N_TEST + N_DEMO + 10, GRID_SIZE, seed=SEED+10)
    demo_x = x_novel[:N_DEMO].to(DEVICE)
    demo_y = y_novel[:N_DEMO].to(DEVICE)
    test_x = x_novel[N_DEMO+10:]
    test_y = y_novel[N_DEMO+10:]

    results = {'step_counts': []}

    for n_steps in STEP_COUNTS:
        print(f"\n  --- TTT steps: {n_steps} ---")
        # Fresh copy of backbone
        model = copy.deepcopy(backbone).to(DEVICE)

        # Warm up for latency measurement
        if n_steps > 0:
            warmup_model = copy.deepcopy(backbone).to(DEVICE)
            ttt_adapt(warmup_model, demo_x, demo_y, 5)
            del warmup_model; gc.collect()

        # Actual TTT
        latency = ttt_adapt(model, demo_x, demo_y, n_steps)
        pixel, exact = evaluate(model, test_x, test_y)

        entry = {
            'n_steps': n_steps, 'latency_ms': latency,
            'pixel_acc': pixel, 'exact_match': exact
        }
        results['step_counts'].append(entry)
        print(f"    Latency: {latency:.1f}ms, Pixel: {pixel*100:.1f}%, Exact: {exact*100:.1f}%")

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _save(results)

    # Find optimal budget (best accuracy within 450ms)
    feasible = [e for e in results['step_counts'] if e['latency_ms'] <= 450]
    if feasible:
        best = max(feasible, key=lambda e: e['pixel_acc'])
        results['optimal'] = best
        print(f"\n  OPTIMAL: {best['n_steps']} steps, "
              f"{best['latency_ms']:.1f}ms, pixel={best['pixel_acc']*100:.1f}%")

    print(f"\n{'='*70}")
    print("GRAND SUMMARY: TTT Budget")
    print(f"{'='*70}")
    for e in results['step_counts']:
        flag = " <-- OPTIMAL" if e == results.get('optimal') else ""
        print(f"  steps={e['n_steps']:3d}: {e['latency_ms']:7.1f}ms  "
              f"pixel={e['pixel_acc']*100:.1f}%  exact={e['exact_match']*100:.1f}%{flag}")
    _save(results)

    _generate_figure(results)
    print("\nPhase 85 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        entries = results['step_counts']
        steps = [e['n_steps'] for e in entries]
        latencies = [e['latency_ms'] for e in entries]
        pixels = [e['pixel_acc']*100 for e in entries]

        ax1.plot(steps, pixels, 'o-', color='#EC4899', linewidth=2, markersize=8)
        ax1.set_xlabel('TTT Steps'); ax1.set_ylabel('Pixel Accuracy (%)')
        ax1.set_title('Accuracy vs TTT Steps', fontweight='bold')
        ax1.grid(alpha=0.3)

        ax2.plot(steps[1:], latencies[1:], 's-', color='#3B82F6', linewidth=2, markersize=8)
        ax2.axhline(y=450, color='red', linestyle='--', alpha=0.7, label='Budget (450ms)')
        ax2.set_xlabel('TTT Steps'); ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Latency vs TTT Steps', fontweight='bold')
        ax2.legend(); ax2.grid(alpha=0.3)

        fig.suptitle('Phase 85: Test-Time Training Budget\n'
                    'How much learning fits in 450ms?',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase85_ttt_budget.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
