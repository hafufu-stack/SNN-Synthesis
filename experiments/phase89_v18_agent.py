"""
Phase 89: v18 Ultimate Liquid Agent

Multi-Task Foundation Backbone + Cellular Prompt Tuning + Temporal NBS.

Full ARC simulation with 30 levels, variable grid sizes, mixed tasks.
500ms budget per level.

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
CTX_CH = 4
NCA_STEPS = 10
TTT_STEPS = 15
TTT_LR = 0.1
K = 11
TAU_NOISE_SIGMAS = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
TIME_BUDGET_MS = 500
N_LEVELS = 30


def to_onehot(grid, nc=N_COLORS):
    h, w = grid.shape
    oh = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): oh[c] = (grid == c).astype(np.float32)
    return oh


# Task generators
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
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(3, 8)):
            r, c = rng.randint(0, gs, size=2); grid[r, c] = rng.randint(1, 4)
        result = grid.copy()
        result[grid == 1] = 2; result[grid == 2] = 1
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))

def gen_move_right(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 6)):
            r, c = rng.randint(0, gs, size=2); grid[r, c] = rng.randint(1, 5)
        result = np.zeros_like(grid)
        for r in range(gs):
            for c in range(gs):
                if grid[r, c] > 0: result[r, (c+1) % gs] = grid[r, c]
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))

def gen_fill_border(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 6)):
            r, c = rng.randint(0, gs, size=2); grid[r, c] = rng.randint(1, 5)
        result = grid.copy()
        result[0, :] = 3; result[-1, :] = 3; result[:, 0] = 3; result[:, -1] = 3
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


ALL_TASKS = [
    ('gravity', gen_gravity),
    ('expand', gen_expand),
    ('color_invert', gen_color_invert),
    ('move_right', gen_move_right),
    ('fill_border', gen_fill_border),
]


class PromptLNCA(nn.Module):
    def __init__(self, nc=10, hc=16, ctx_ch=4):
        super().__init__()
        self.hc = hc
        self.ctx_proj = nn.Conv2d(ctx_ch, nc, 1, bias=False)
        self.perceive = nn.Conv2d(nc + hc, hc * 2, 3, padding=1)
        self.update = nn.Sequential(
            nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(nc + hc * 2, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)
        self.task_context = nn.Parameter(torch.zeros(1, ctx_ch, 1, 1))
        self.temporal_sigma = 0.0

    def forward(self, x, n_steps=NCA_STEPS):
        b, c, h, w = x.shape
        ctx = self.ctx_proj(self.task_context.expand(b, -1, h, w))
        x_aug = x + ctx
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        for _ in range(n_steps):
            combined = torch.cat([x_aug, state], dim=1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x_aug, state, delta], dim=1)
            tau_logit = self.tau_gate(tau_in) + self.b_tau
            if self.temporal_sigma > 0:
                tau_logit = tau_logit + torch.randn_like(tau_logit) * self.temporal_sigma
            beta = torch.sigmoid(tau_logit).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)

    def freeze_backbone(self):
        for name, p in self.named_parameters():
            if name != 'task_context':
                p.requires_grad = False


def load_foundation(model):
    path = os.path.join(RESULTS_DIR, "foundation_lnca.pt")
    if os.path.exists(path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v; loaded += 1
        model.load_state_dict(model_state)
        return True
    return False


def solve_level(backbone_state, train_x, train_y, test_x, test_y, budget_ms):
    t_start = time.perf_counter()

    model = PromptLNCA().to(DEVICE)
    model.load_state_dict(backbone_state)
    model.freeze_backbone()
    model.task_context = nn.Parameter(torch.zeros(1, CTX_CH, 1, 1, device=DEVICE))

    # TTT: Prompt tuning on demos
    optimizer = torch.optim.Adam([model.task_context], lr=TTT_LR)
    dx, dy = train_x.to(DEVICE), train_y.to(DEVICE)
    for _ in range(TTT_STEPS):
        if (time.perf_counter() - t_start) * 1000 > budget_ms * 0.7:
            break
        optimizer.zero_grad()
        F.cross_entropy(model(dx), dy).backward()
        optimizer.step()
    ttt_ms = (time.perf_counter() - t_start) * 1000

    # NBS inference
    model.eval()
    gs = test_x.size(-1)
    vote_sum = torch.zeros(1, N_COLORS, gs, gs, device=DEVICE)
    with torch.no_grad():
        tx = test_x.to(DEVICE)
        for sigma in TAU_NOISE_SIGMAS[:K]:
            if (time.perf_counter() - t_start) * 1000 > budget_ms:
                break
            model.temporal_sigma = sigma
            vote_sum += F.softmax(model(tx), dim=1)
    model.temporal_sigma = 0.0

    pred = vote_sum.argmax(dim=1)
    target = test_y.to(DEVICE)
    total_ms = (time.perf_counter() - t_start) * 1000
    pixel = (pred == target).float().mean().item()
    exact = (pred.reshape(1, -1) == target.reshape(1, -1)).all(1).float().mean().item()

    del model; gc.collect()
    return {
        'pixel_acc': pixel, 'exact_match': exact,
        'ttt_ms': ttt_ms, 'total_ms': total_ms,
        'timeout': total_ms > budget_ms
    }


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase89_v18_agent.json"), 'w',
              encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 89: v18 Ultimate Liquid Agent',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 89: v18 Ultimate Liquid Agent")
    print(f"  {N_LEVELS} levels, {TIME_BUDGET_MS}ms budget")
    print(f"  TTT: {TTT_STEPS}-step Prompt Tuning + NBS K={K}")
    print("=" * 70)

    # Load foundation backbone
    backbone = PromptLNCA().to(DEVICE)
    if load_foundation(backbone):
        print("  Foundation backbone loaded!")
    else:
        print("  WARNING: No foundation backbone, using random init")

    backbone_state = backbone.state_dict()
    del backbone

    # Generate levels
    rng = np.random.RandomState(SEED + 300)
    grid_sizes = [8, 10, 12]
    results = {'levels': []}

    total_solved = 0
    task_solved = {name: [0, 0] for name, _ in ALL_TASKS}

    for level_idx in range(N_LEVELS):
        gs = grid_sizes[level_idx % len(grid_sizes)]
        task_idx = level_idx % len(ALL_TASKS)
        task_name, gen_fn = ALL_TASKS[task_idx]

        all_x, all_y = gen_fn(4, gs, seed=SEED + 400 + level_idx)
        train_x, train_y = all_x[:3], all_y[:3]
        test_x, test_y = all_x[3:4], all_y[3:4]

        result = solve_level(backbone_state, train_x, train_y,
                           test_x, test_y, TIME_BUDGET_MS)
        result['level'] = level_idx
        result['grid_size'] = gs
        result['task'] = task_name
        results['levels'].append(result)

        solved = result['exact_match'] > 0.5
        total_solved += int(solved)
        task_solved[task_name][0] += int(solved)
        task_solved[task_name][1] += 1

        status = "SOLVED" if solved else ("TIMEOUT" if result['timeout'] else "WRONG")
        print(f"  Lv{level_idx:2d} ({task_name:>13s} {gs}x{gs}): "
              f"{status:7s} pixel={result['pixel_acc']*100:.0f}% "
              f"{result['total_ms']:.0f}ms")

    solve_rate = total_solved / N_LEVELS
    avg_ms = np.mean([l['total_ms'] for l in results['levels']])
    timeouts = sum(1 for l in results['levels'] if l['timeout'])

    results['summary'] = {
        'solve_rate': solve_rate, 'solved': total_solved,
        'total_levels': N_LEVELS, 'timeout_rate': timeouts/N_LEVELS,
        'avg_latency_ms': avg_ms,
        'per_task': {k: {'solved': v[0], 'total': v[1],
                         'rate': v[0]/v[1] if v[1] > 0 else 0}
                    for k, v in task_solved.items()}
    }
    _save(results)

    print(f"\n{'='*70}")
    print("GRAND SUMMARY: v18 Ultimate Liquid Agent")
    print(f"{'='*70}")
    print(f"  Overall Solve Rate: {total_solved}/{N_LEVELS} = {solve_rate*100:.0f}%")
    print(f"  Timeout Rate: {timeouts/N_LEVELS*100:.0f}%")
    print(f"  Avg Latency: {avg_ms:.0f}ms (budget: {TIME_BUDGET_MS}ms)")
    print(f"\n  Per-task breakdown:")
    for task, stats in results['summary']['per_task'].items():
        print(f"    {task:15s}: {stats['solved']}/{stats['total']} "
              f"= {stats['rate']*100:.0f}%")

    _generate_figure(results)
    print("\nPhase 89 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        levels = results['levels']

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1) Per-task solve rate
        ax = axes[0]
        per_task = results['summary']['per_task']
        tasks = list(per_task.keys())
        rates = [per_task[t]['rate']*100 for t in tasks]
        colors = ['#EC4899' if r > 50 else '#9CA3AF' for r in rates]
        bars = ax.bar(range(len(tasks)), rates, color=colors, edgecolor='white')
        for b, v in zip(bars, rates):
            ax.text(b.get_x()+b.get_width()/2, v+2, f'{v:.0f}%',
                   ha='center', fontweight='bold', fontsize=9)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=20, fontsize=8)
        ax.set_ylabel('Solve Rate (%)'); ax.set_ylim(0, 115)
        ax.set_title('Per-Task Success', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 2) Per-grid-size solve rate
        ax = axes[1]
        sizes = sorted(set(l['grid_size'] for l in levels))
        for gs in sizes:
            sub = [l for l in levels if l['grid_size'] == gs]
            sr = sum(1 for l in sub if l['exact_match'] > 0.5) / len(sub) * 100
            ax.bar(f'{gs}x{gs}', sr, color='#3B82F6', edgecolor='white')
            ax.text(ax.patches[-1].get_x()+ax.patches[-1].get_width()/2,
                   sr+2, f'{sr:.0f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Solve Rate (%)'); ax.set_ylim(0, 115)
        ax.set_title('Per-Size Success', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 3) Latency
        ax = axes[2]
        lats = [l['total_ms'] for l in levels]
        ax.bar(range(len(lats)), lats,
              color=['#10B981' if l < TIME_BUDGET_MS else '#EF4444' for l in lats],
              edgecolor='white', linewidth=0.3)
        ax.axhline(y=TIME_BUDGET_MS, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Level'); ax.set_ylabel('Latency (ms)')
        ax.set_title('Per-Level Latency', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        sr = results['summary']['solve_rate']
        fig.suptitle(f'Phase 89: v18 Ultimate Liquid Agent\n'
                    f'Solve Rate: {sr*100:.0f}% | Avg: {results["summary"]["avg_latency_ms"]:.0f}ms',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase89_v18_agent.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
