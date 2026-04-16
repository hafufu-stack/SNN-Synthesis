"""
Phase 86: v17 L-NCA Agent Prototype (Kaggle ARC Dry Run)

Full ARC simulation:
  - Multiple levels with varying grid sizes (8x8, 10x10, 12x12)
  - Each level: 3 train pairs (demo) + 1 test pair
  - Agent: TTT (Backprop on demos) + Temporal NBS (K=11) inference
  - Time budget: 500ms per level

Measures: solve rate, average latency, timeout rate

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
TTT_STEPS = 50  # From Phase 85 optimization
TTT_LR = 0.01
K = 11
TAU_NOISE_SIGMAS = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
TIME_BUDGET_MS = 500
N_LEVELS = 20
BACKBONE_EPOCHS = 40
LR = 1e-3
BATCH_SIZE = 32
N_TRAIN = 2000


def to_onehot(grid, nc=N_COLORS):
    h, w = grid.shape
    oh = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): oh[c] = (grid == c).astype(np.float32)
    return oh


# ARC-like task generators with variable grid sizes
def gen_expand_level(gs, n_pairs, seed):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n_pairs):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(1, 4)):
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


def gen_gravity_level(gs, n_pairs, seed):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n_pairs):
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
        self.temporal_sigma = 0.0

    def forward(self, x, n_steps=NCA_STEPS):
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        for _ in range(n_steps):
            combined = torch.cat([x, state], dim=1)
            perception = self.perceive(combined)
            delta = self.update(perception)
            tau_input = torch.cat([x, state, delta], dim=1)
            tau_logit = self.tau_gate(tau_input) + self.b_tau
            if self.temporal_sigma > 0:
                tau_logit = tau_logit + torch.randn_like(tau_logit) * self.temporal_sigma
            beta = torch.sigmoid(tau_logit)
            beta = torch.clamp(beta, 0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)


def solve_level(backbone, train_x, train_y, test_x, test_y, time_budget_ms):
    """Full ARC agent: TTT + Temporal NBS."""
    t_start = time.perf_counter()

    # Phase 1: Test-Time Training (adapt to demos)
    model = copy.deepcopy(backbone).to(DEVICE)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=TTT_LR)
    demo_x = train_x.to(DEVICE)
    demo_y = train_y.to(DEVICE)

    for _ in range(TTT_STEPS):
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if elapsed_ms > time_budget_ms * 0.8:  # Save 20% for inference
            break
        optimizer.zero_grad()
        loss = F.cross_entropy(model(demo_x), demo_y)
        loss.backward()
        optimizer.step()

    ttt_ms = (time.perf_counter() - t_start) * 1000

    # Phase 2: Temporal NBS inference
    model.eval()
    gs = test_x.size(-1)
    vote_sum = torch.zeros(1, N_COLORS, gs, gs, device=DEVICE)

    with torch.no_grad():
        tx = test_x.to(DEVICE)
        for sigma in TAU_NOISE_SIGMAS[:K]:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            if elapsed_ms > time_budget_ms:
                break
            model.temporal_sigma = sigma
            vote_sum += F.softmax(model(tx), dim=1)

    model.temporal_sigma = 0.0
    pred = vote_sum.argmax(dim=1)
    target = test_y.to(DEVICE)

    total_ms = (time.perf_counter() - t_start) * 1000
    pixel_acc = (pred == target).float().mean().item()
    exact = (pred.reshape(1, -1) == target.reshape(1, -1)).all(dim=1).float().mean().item()
    timeout = total_ms > time_budget_ms

    del model; gc.collect()
    return {
        'pixel_acc': pixel_acc, 'exact_match': exact,
        'ttt_ms': ttt_ms, 'total_ms': total_ms, 'timeout': timeout
    }


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase86_agent_prototype.json"), 'w',
              encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 86: v17 L-NCA Agent Prototype',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 86: v17 L-NCA Agent Prototype")
    print(f"  Kaggle ARC Dry Run: {N_LEVELS} levels, {TIME_BUDGET_MS}ms budget")
    print(f"  TTT steps={TTT_STEPS}, NBS K={K}")
    print("=" * 70)

    # Train backbone offline on expand task (8x8)
    print("\n  Training backbone on 'expand' (8x8)...")
    x_train, y_train = gen_expand_level(8, N_TRAIN, seed=SEED)
    backbone = MultiColorLNCA().to(DEVICE)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=LR)
    for epoch in range(BACKBONE_EPOCHS):
        backbone.train()
        perm = torch.randperm(N_TRAIN)
        for i in range(0, N_TRAIN, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = x_train[idx].to(DEVICE), y_train[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(backbone(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
        if (epoch+1) % 10 == 0:
            backbone.eval()
            with torch.no_grad():
                p = backbone(x_train[:200].to(DEVICE)).argmax(1)
                acc = (p == y_train[:200].to(DEVICE)).float().mean()
            print(f"    Epoch {epoch+1}: pixel={acc*100:.1f}%")

    # Freeze backbone for copying
    for p in backbone.parameters():
        p.requires_grad = True  # TTT needs gradients

    # Generate levels
    rng = np.random.RandomState(SEED + 100)
    TASK_GENERATORS = [gen_expand_level, gen_gravity_level]
    grid_sizes = [8, 8, 10, 10, 12, 12, 8, 10, 12, 8,
                  10, 12, 8, 10, 12, 8, 10, 12, 8, 10]
    task_types = [rng.randint(0, 2) for _ in range(N_LEVELS)]

    results = {'levels': [], 'summary': {}}

    print(f"\n  Running {N_LEVELS} levels...")
    total_solved = 0
    total_timeouts = 0

    for level_idx in range(N_LEVELS):
        gs = grid_sizes[level_idx % len(grid_sizes)]
        gen_fn = TASK_GENERATORS[task_types[level_idx]]
        task_name = 'expand' if task_types[level_idx] == 0 else 'gravity'

        # Generate train + test for this level
        all_x, all_y = gen_fn(gs, 4, seed=SEED + 200 + level_idx)
        train_x, train_y = all_x[:3], all_y[:3]
        test_x, test_y = all_x[3:4], all_y[3:4]

        result = solve_level(backbone, train_x, train_y, test_x, test_y, TIME_BUDGET_MS)
        result['level'] = level_idx
        result['grid_size'] = gs
        result['task'] = task_name
        results['levels'].append(result)

        solved = result['exact_match'] > 0.5
        total_solved += int(solved)
        total_timeouts += int(result['timeout'])

        status = "SOLVED" if solved else ("TIMEOUT" if result['timeout'] else "WRONG")
        print(f"    Level {level_idx:2d} ({task_name:>7s} {gs}x{gs}): "
              f"{status:7s}  pixel={result['pixel_acc']*100:.0f}%  "
              f"total={result['total_ms']:.0f}ms")

    # Summary
    solve_rate = total_solved / N_LEVELS
    timeout_rate = total_timeouts / N_LEVELS
    avg_ms = np.mean([l['total_ms'] for l in results['levels']])

    results['summary'] = {
        'solve_rate': solve_rate, 'solved': total_solved,
        'total_levels': N_LEVELS, 'timeout_rate': timeout_rate,
        'avg_latency_ms': avg_ms
    }
    _save(results)

    print(f"\n{'='*70}")
    print("GRAND SUMMARY: v17 L-NCA Agent Prototype")
    print(f"{'='*70}")
    print(f"  Solve Rate: {total_solved}/{N_LEVELS} = {solve_rate*100:.0f}%")
    print(f"  Timeout Rate: {timeout_rate*100:.0f}%")
    print(f"  Avg Latency: {avg_ms:.0f}ms (budget: {TIME_BUDGET_MS}ms)")

    _generate_figure(results)
    print("\nPhase 86 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        levels = results['levels']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Solve rate by grid size
        sizes = sorted(set(l['grid_size'] for l in levels))
        for gs in sizes:
            subset = [l for l in levels if l['grid_size'] == gs]
            sr = sum(1 for l in subset if l['exact_match'] > 0.5) / len(subset)
            ax1.bar(f'{gs}x{gs}', sr * 100,
                   color='#EC4899' if sr > 0.5 else '#9CA3AF',
                   edgecolor='white', linewidth=1.5)
            ax1.text(ax1.patches[-1].get_x() + ax1.patches[-1].get_width()/2,
                    sr*100+2, f'{sr*100:.0f}%', ha='center', fontweight='bold')
        ax1.set_ylabel('Solve Rate (%)'); ax1.set_ylim(0, 110)
        ax1.set_title('Solve Rate by Grid Size', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Latency distribution
        latencies = [l['total_ms'] for l in levels]
        ax2.bar(range(len(latencies)), latencies,
               color=['#10B981' if l < TIME_BUDGET_MS else '#EF4444' for l in latencies],
               edgecolor='white', linewidth=0.5)
        ax2.axhline(y=TIME_BUDGET_MS, color='red', linestyle='--', alpha=0.7,
                   label=f'Budget ({TIME_BUDGET_MS}ms)')
        ax2.set_xlabel('Level'); ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Per-Level Latency', fontweight='bold')
        ax2.legend(); ax2.grid(axis='y', alpha=0.3)

        sr = results['summary']['solve_rate']
        fig.suptitle(f'Phase 86: v17 L-NCA Agent Prototype\n'
                    f'Solve Rate: {sr*100:.0f}% | Avg Latency: {results["summary"]["avg_latency_ms"]:.0f}ms',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase86_agent_prototype.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
