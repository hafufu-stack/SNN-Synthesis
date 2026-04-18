"""
Phase 101: Real ARC-AGI Baseline

Test v20 agent (19 toy task experts) on REAL ARC training tasks.
Measures the gap between toy simulation and real ARC puzzles.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
ARC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training")
DEVICE = "cpu"; SEED = 2026; NC = 10; HC = 32; NCA_STEPS = 10; CTX_CH = 10
SPECIALIST_EPOCHS = 40; BS = 32; TTT_STEPS = 15; TTT_LR = 0.1; MAX_GS = 30

# ====================================================================
# L-NCA Architecture (from Phase 84/92)
# ====================================================================
class LiquidNCA(nn.Module):
    def __init__(self, nc=10, hc=32):
        super().__init__()
        self.hc = hc
        self.perceive = nn.Conv2d(nc + hc, hc * 2, 3, padding=1)
        self.update = nn.Sequential(nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(nc + hc * 2, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)

    def forward(self, x, n_steps=NCA_STEPS, ctx=None):
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        x_in = x
        if ctx is not None:
            x_in = x + ctx.expand(-1, -1, h, w) if ctx.dim() == 4 else x
        for _ in range(n_steps):
            combined = torch.cat([x_in, state], 1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x_in, state, delta], 1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)

# ====================================================================
# Color-Frequency Mapping (from Phase 97)
# ====================================================================
def freq_remap(grid):
    """Remap colors by frequency: most common -> 0, next -> 1, etc."""
    flat = grid.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    mapping = {unique[order[i]]: i for i in range(len(unique))}
    return np.vectorize(lambda c: mapping.get(c, c))(grid)

def one_hot(grid, nc=10):
    h, w = grid.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc):
        o[c] = (grid == c).astype(np.float32)
    return o

# ====================================================================
# Toy Task Generators (same as Phase 92)
# ====================================================================
def _gravity(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)): g[rng.randint(0, gs), rng.randint(0, gs)] = 1
        for _ in range(rng.randint(1, 3)):
            r, c = rng.randint(0, gs), rng.randint(0, gs)
            if g[r, c] == 0: g[r, c] = 2
        res = np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r, c] == 2: res[r, c] = 2
        for c in range(gs):
            cnt = sum(1 for r in range(gs) if g[r, c] == 1)
            row, pl = gs - 1, 0
            while pl < cnt and row >= 0:
                if res[row, c] == 0: res[row, c] = 1; pl += 1
                row -= 1
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _expand(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 4)):
            y, x = rng.randint(1, gs - 1), rng.randint(1, gs - 1)
            if g[y, x] == 0: g[y, x] = rng.randint(1, 5)
        res = g.copy()
        for y in range(gs):
            for x in range(gs):
                if g[y, x] > 0:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gs and 0 <= nx < gs and res[ny, nx] == 0: res[ny, nx] = g[y, x]
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _color_invert(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(3, 7)): g[rng.randint(0, gs), rng.randint(0, gs)] = rng.randint(1, 4)
        res = g.copy(); res[g == 1] = 2; res[g == 2] = 1
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _move_right(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)): g[rng.randint(0, gs), rng.randint(0, gs)] = rng.randint(1, 5)
        res = np.zeros_like(g)
        for r in range(gs):
            for c in range(gs):
                if g[r, c] > 0: res[r, (c + 1) % gs] = g[r, c]
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _fill_border(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)): g[rng.randint(0, gs), rng.randint(0, gs)] = rng.randint(1, 5)
        res = g.copy(); res[0, :] = 3; res[-1, :] = 3; res[:, 0] = 3; res[:, -1] = 3
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

TASK_FNS = {'gravity': _gravity, 'expand': _expand, 'color_invert': _color_invert,
            'move_right': _move_right, 'fill_border': _fill_border}

# ====================================================================
# Load Real ARC Tasks
# ====================================================================
def load_arc_tasks(arc_dir, max_tasks=None):
    """Load real ARC tasks from JSON files."""
    tasks = {}
    files = sorted([f for f in os.listdir(arc_dir) if f.endswith('.json')])
    if max_tasks:
        files = files[:max_tasks]
    for fname in files:
        tid = fname.replace('.json', '')
        with open(os.path.join(arc_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        tasks[tid] = data
    return tasks

# ====================================================================
# Train Toy Specialists (v20 agent reproduction)
# ====================================================================
def train_specialist(task_name, n_train=400, gs=8):
    rng = np.random.RandomState(SEED)
    ins, tgs = TASK_FNS[task_name](n_train, gs, rng)
    x = torch.tensor(np.array(ins))
    y = torch.tensor(np.array(tgs))
    model = LiquidNCA(NC, HC).to(DEVICE)
    ctx = nn.Parameter(torch.randn(1, CTX_CH, 1, 1) * 0.01)
    opt = torch.optim.Adam(list(model.parameters()) + [ctx], lr=1e-3)
    for ep in range(SPECIALIST_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BS):
            idx = perm[i:i + BS]
            xb, yb = x[idx].to(DEVICE), y[idx].to(DEVICE)
            opt.zero_grad()
            out = model(xb, ctx=ctx.expand(xb.size(0), -1, -1, -1))
            F.cross_entropy(out, yb).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [ctx], 1.0)
            opt.step()
    return model, ctx.data.clone()

# ====================================================================
# v20 Agent on Real ARC
# ====================================================================
def test_on_real_arc(specialists, contexts, arc_tasks, max_tasks=50):
    """Test v20 agent on real ARC tasks."""
    task_names = list(specialists.keys())
    results = []
    total_solved = 0
    total_tested = 0

    task_ids = list(arc_tasks.keys())[:max_tasks]

    for i, tid in enumerate(task_ids):
        task = arc_tasks[tid]
        demos = task['train']
        tests = task['test']

        # Skip tasks with grids too large
        max_dim = 0
        for d in demos + tests:
            h, w = len(d['input']), len(d['input'][0])
            max_dim = max(max_dim, h, w)
            if 'output' in d:
                ho, wo = len(d['output']), len(d['output'][0])
                max_dim = max(max_dim, ho, wo)
        if max_dim > MAX_GS:
            continue

        total_tested += 1

        # Apply freq-remap to all grids
        demo_ins = [freq_remap(np.array(d['input'])) for d in demos]
        demo_outs = [np.array(d['output']) for d in demos]  # Keep original for target
        demo_outs_remap = [freq_remap(np.array(d['output'])) for d in demos]

        # Route: evaluate each specialist on first demo pair
        best_loss = float('inf')
        best_name = None
        first_in = one_hot(demo_ins[0])
        first_in_t = torch.tensor(first_in).unsqueeze(0).to(DEVICE)
        first_out_t = torch.tensor(demo_outs_remap[0]).unsqueeze(0).to(DEVICE)

        for tn in task_names:
            model = specialists[tn]
            ctx = contexts[tn]
            model.eval()
            with torch.no_grad():
                pred = model(first_in_t, ctx=ctx.expand(1, -1, -1, -1))
                loss = F.cross_entropy(pred, first_out_t).item()
            if loss < best_loss:
                best_loss = loss
                best_name = tn

        # TTT with best specialist
        model = copy.deepcopy(specialists[best_name])
        ttt_ctx = nn.Parameter(contexts[best_name].clone())
        opt = torch.optim.Adam([ttt_ctx], lr=TTT_LR)
        for step in range(TTT_STEPS):
            opt.zero_grad()
            total_loss = 0
            for di, do in zip(demo_ins, demo_outs_remap):
                di_t = torch.tensor(one_hot(di)).unsqueeze(0).to(DEVICE)
                do_t = torch.tensor(do).unsqueeze(0).to(DEVICE)
                pred = model(di_t, ctx=ttt_ctx.expand(1, -1, -1, -1))
                total_loss += F.cross_entropy(pred, do_t)
            (total_loss / len(demo_ins)).backward()
            opt.step()

        # Test on test pairs
        n_correct = 0
        for test_pair in tests:
            test_in = freq_remap(np.array(test_pair['input']))
            test_out = np.array(test_pair['output'])

            ti_t = torch.tensor(one_hot(test_in)).unsqueeze(0).to(DEVICE)
            model.eval()
            with torch.no_grad():
                pred = model(ti_t, ctx=ttt_ctx.expand(1, -1, -1, -1))
                pred_grid = pred.argmax(1).squeeze(0).numpy()

            # Check dimensions match
            if pred_grid.shape == test_out.shape:
                match = np.array_equal(pred_grid, test_out)
            else:
                match = False

            if match:
                n_correct += 1

        solved = n_correct == len(tests)
        if solved:
            total_solved += 1

        status = "SOLVED" if solved else "wrong"
        print(f"  [{i+1:3d}] {tid[:12]:12s} {max_dim:2d}x route={best_name:13s} "
              f"loss={best_loss:.2f} {status}")

        results.append({
            'task_id': tid,
            'max_dim': max_dim,
            'routed_to': best_name,
            'routing_loss': best_loss,
            'solved': solved,
            'n_test': len(tests),
            'n_correct': n_correct
        })

    return results, total_solved, total_tested


def _save(results, total_solved, total_tested):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase101_real_arc_baseline.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 101: Real ARC Baseline',
            'timestamp': datetime.now().isoformat(),
            'solve_rate': total_solved / max(total_tested, 1),
            'total_solved': total_solved,
            'total_tested': total_tested,
            'results': results
        }, f, indent=2, default=str)


def _plot(results, total_solved, total_tested):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Routing distribution
        routes = [r['routed_to'] for r in results]
        from collections import Counter
        rc = Counter(routes)
        axes[0].barh(list(rc.keys()), list(rc.values()), color='steelblue')
        axes[0].set_xlabel('Count')
        axes[0].set_title('Routing Distribution (Real ARC)')

        # 2. Loss distribution
        losses = [r['routing_loss'] for r in results]
        axes[1].hist(losses, bins=20, color='coral', edgecolor='black')
        axes[1].set_xlabel('Routing Loss')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Routing Loss Distribution')

        # 3. Solve rate summary
        axes[2].bar(['Solved', 'Failed'],
                     [total_solved, total_tested - total_solved],
                     color=['green', 'red'])
        axes[2].set_title(f'Solve Rate: {total_solved}/{total_tested} = '
                          f'{total_solved / max(total_tested, 1) * 100:.1f}%')

        plt.suptitle('Phase 101: Real ARC-AGI Baseline (19 Toy Experts)', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase101_real_arc_baseline.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 101: Real ARC-AGI Baseline")
    print("=" * 70)

    # Check ARC data exists
    if not os.path.exists(ARC_DIR):
        print(f"  ERROR: ARC data not found at {ARC_DIR}")
        return

    # Load ARC tasks
    print("  Loading real ARC tasks...")
    arc_tasks = load_arc_tasks(ARC_DIR, max_tasks=100)
    print(f"  Loaded {len(arc_tasks)} tasks")

    # Train toy specialists
    print("  Training 5 toy specialists...")
    specialists = {}
    contexts = {}
    for tn in TASK_FNS:
        model, ctx = train_specialist(tn)
        specialists[tn] = model
        contexts[tn] = ctx
        print(f"    {tn}: done")

    # Test on real ARC
    print(f"\n  Testing v20 agent on {len(arc_tasks)} real ARC tasks...")
    results, total_solved, total_tested = test_on_real_arc(
        specialists, contexts, arc_tasks, max_tasks=100
    )

    print(f"\n{'=' * 70}")
    print(f"  Real ARC Baseline: {total_solved}/{total_tested} = "
          f"{total_solved / max(total_tested, 1) * 100:.1f}%")
    print(f"{'=' * 70}")

    _save(results, total_solved, total_tested)
    _plot(results, total_solved, total_tested)
    print("Phase 101 complete!")

if __name__ == '__main__':
    main()
