"""
Phase 106: Auto-T Scaling by Task Difficulty

Measure Auto-T stopping step across difficulty levels.
Hypothesis: harder tasks need more NCA steps.
"Cognitive load as a physical quantity"

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
DEVICE = "cpu"; SEED = 2026; NC = 10; HC = 32; CTX_CH = 10
SPECIALIST_EPOCHS = 60; BS = 32; MAX_T = 50; THETA = 0.01; N_TEST = 100

class LiquidNCA(nn.Module):
    def __init__(self, nc=10, hc=32):
        super().__init__()
        self.hc = hc
        self.perceive = nn.Conv2d(nc + hc, hc * 2, 3, padding=1)
        self.update = nn.Sequential(nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(nc + hc * 2, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)
    def forward_auto_t(self, x, max_steps=50, theta=0.01, ctx=None):
        """Forward with Auto-T: stop when state converges."""
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        x_in = x if ctx is None else x + ctx.expand(-1, -1, h, w)
        steps_used = 0
        for t in range(max_steps):
            combined = torch.cat([x_in, state], 1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x_in, state, delta], 1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            new_state = beta * state + (1 - beta) * delta
            steps_used = t + 1
            mse = F.mse_loss(new_state, state).item()
            state = new_state
            if mse < theta:
                break
        return self.readout(state), steps_used

def one_hot(grid, nc=10):
    h, w = grid.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): o[c] = (grid == c).astype(np.float32)
    return o

# Task generators with configurable difficulty
def _gravity(n, gs, rng, n_objects=3):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(n_objects): g[rng.randint(0, gs), rng.randint(0, gs)] = rng.randint(1, 5)
        res = np.zeros_like(g)
        for c in range(gs):
            vals = [g[r, c] for r in range(gs) if g[r, c] > 0]
            for idx, v in enumerate(vals): res[gs-1-idx, c] = v
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _expand(n, gs, rng, n_objects=3):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(n_objects):
            y, x = rng.randint(1, gs-1), rng.randint(1, gs-1)
            if g[y, x] == 0: g[y, x] = rng.randint(1, 5)
        res = g.copy()
        for y in range(gs):
            for x in range(gs):
                if g[y, x] > 0:
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0<=ny<gs and 0<=nx<gs and res[ny,nx]==0: res[ny,nx] = g[y,x]
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _color_invert(n, gs, rng, n_objects=3):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(n_objects): g[rng.randint(0, gs), rng.randint(0, gs)] = rng.randint(1, 4)
        res = g.copy(); res[g == 1] = 2; res[g == 2] = 1
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

TASK_FNS = {'gravity': _gravity, 'expand': _expand, 'color_invert': _color_invert}
DIFFICULTIES = {
    'easy': {'gs': 6, 'n_obj': 2},
    'medium': {'gs': 8, 'n_obj': 4},
    'hard': {'gs': 12, 'n_obj': 6},
    'extreme': {'gs': 16, 'n_obj': 10}
}

def train_specialist_with_attractor(task_name, gs, n_obj, n_train=400):
    rng = np.random.RandomState(SEED)
    ins, tgs = TASK_FNS[task_name](n_train, gs, rng, n_obj)
    x = torch.tensor(np.array(ins)); y = torch.tensor(np.array(tgs))
    model = LiquidNCA(NC, HC).to(DEVICE)
    ctx = nn.Parameter(torch.randn(1, CTX_CH, 1, 1) * 0.01)
    opt = torch.optim.Adam(list(model.parameters()) + [ctx], lr=1e-3)
    for ep in range(SPECIALIST_EPOCHS):
        model.train(); perm = torch.randperm(n_train)
        # Random T for attractor regularization
        T = random.randint(5, 20)
        for i in range(0, n_train, BS):
            idx = perm[i:i+BS]; opt.zero_grad()
            xb, yb = x[idx], y[idx]
            # Forward with T steps
            b, c, h, w = xb.shape
            state = torch.zeros(b, model.hc, h, w)
            x_in = xb + ctx.expand(b, -1, h, w)
            prev_state = state
            for _ in range(T):
                combined = torch.cat([x_in, state], 1)
                delta = model.update(model.perceive(combined))
                tau_in = torch.cat([x_in, state, delta], 1)
                beta = torch.sigmoid(model.tau_gate(tau_in) + model.b_tau).clamp(0.01, 0.99)
                prev_state = state
                state = beta * state + (1 - beta) * delta
            out = model.readout(state)
            loss = F.cross_entropy(out, yb) + 0.1 * F.mse_loss(state, prev_state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [ctx], 1.0); opt.step()
    return model, ctx.data.clone()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 106: Auto-T Scaling by Task Difficulty")
    print("=" * 70)

    results = []

    for task_name in TASK_FNS:
        print(f"\n  Task: {task_name}")
        for diff_name, diff_params in DIFFICULTIES.items():
            gs = diff_params['gs']
            n_obj = diff_params['n_obj']

            print(f"    {diff_name} (gs={gs}, n_obj={n_obj}): ", end="")

            # Train with attractor regularization
            model, ctx = train_specialist_with_attractor(task_name, gs, n_obj)

            # Test with Auto-T
            rng = np.random.RandomState(42)
            ins, tgs = TASK_FNS[task_name](N_TEST, gs, rng, n_obj)

            steps_list = []; em_count = 0
            model.eval()
            with torch.no_grad():
                for i in range(N_TEST):
                    x_t = torch.tensor(ins[i]).unsqueeze(0)
                    pred, steps = model.forward_auto_t(x_t, MAX_T, THETA,
                                    ctx=ctx.expand(1,-1,-1,-1))
                    pred_grid = pred.argmax(1).squeeze(0).numpy()
                    if np.array_equal(pred_grid, tgs[i]):
                        em_count += 1
                    steps_list.append(steps)

            avg_steps = np.mean(steps_list)
            em = em_count / N_TEST * 100
            print(f"avg_T={avg_steps:.1f}, EM={em:.0f}%")

            results.append({
                'task': task_name,
                'difficulty': diff_name,
                'grid_size': gs,
                'n_objects': n_obj,
                'avg_auto_t': avg_steps,
                'std_auto_t': np.std(steps_list),
                'exact_match': em,
                'steps_distribution': np.histogram(steps_list, bins=10)[0].tolist()
            })

            del model; gc.collect()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Task Difficulty → Auto-T Scaling:")
    for r in results:
        print(f"    {r['task']:14s} {r['difficulty']:8s} gs={r['grid_size']:2d} "
              f"avg_T={r['avg_auto_t']:5.1f}  EM={r['exact_match']:.0f}%")
    print(f"{'=' * 70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase106_auto_t_scaling.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 106: Auto-T Scaling by Task Difficulty',
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Auto-T vs difficulty
        for tn in TASK_FNS:
            task_results = [r for r in results if r['task'] == tn]
            diffs = [r['difficulty'] for r in task_results]
            ts = [r['avg_auto_t'] for r in task_results]
            axes[0].plot(diffs, ts, 'o-', label=tn, linewidth=2, markersize=8)
        axes[0].set_xlabel('Difficulty'); axes[0].set_ylabel('Average Auto-T Steps')
        axes[0].set_title('Cognitive Load: Harder Tasks → More Steps')
        axes[0].legend()

        # 2. EM vs Auto-T
        for tn in TASK_FNS:
            task_results = [r for r in results if r['task'] == tn]
            ts = [r['avg_auto_t'] for r in task_results]
            ems = [r['exact_match'] for r in task_results]
            axes[1].scatter(ts, ems, s=100, label=tn)
        axes[1].set_xlabel('Average Auto-T Steps'); axes[1].set_ylabel('Exact Match (%)')
        axes[1].set_title('Auto-T Steps vs Accuracy')
        axes[1].legend()

        plt.suptitle('Phase 106: Auto-T as Cognitive Load Indicator', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase106_auto_t_scaling.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 106 complete!")

if __name__ == '__main__':
    main()
