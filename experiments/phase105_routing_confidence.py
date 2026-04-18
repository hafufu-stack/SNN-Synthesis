"""
Phase 105: Routing Confidence Calibration

Quantify routing uncertainty via loss gap (top-1 vs top-2).
Auto-fallback to compositional routing on low confidence.
Analyze v20's 12% failure cases.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cpu"; SEED = 2026; NC = 10; HC = 32; NCA_STEPS = 10; CTX_CH = 10
SPECIALIST_EPOCHS = 40; BS = 32; GS = 8; N_TEST = 200

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
        x_in = x if ctx is None else x + ctx.expand(-1, -1, h, w)
        for _ in range(n_steps):
            combined = torch.cat([x_in, state], 1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x_in, state, delta], 1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)

def one_hot(grid, nc=10):
    h, w = grid.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): o[c] = (grid == c).astype(np.float32)
    return o

# Toy task generators
def _gravity(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)): g[rng.randint(0, gs), rng.randint(0, gs)] = 1
        res = np.zeros_like(g)
        for c in range(gs):
            vals = [g[r, c] for r in range(gs) if g[r, c] > 0]
            for idx, v in enumerate(vals): res[gs-1-idx, c] = v
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _expand(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 4)):
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
                if g[r, c] > 0: res[r, (c+1)%gs] = g[r, c]
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _fill_border(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)): g[rng.randint(0, gs), rng.randint(0, gs)] = rng.randint(1, 5)
        res = g.copy(); res[0,:]=3; res[-1,:]=3; res[:,0]=3; res[:,-1]=3
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

TASK_FNS = {'gravity': _gravity, 'expand': _expand, 'color_invert': _color_invert,
            'move_right': _move_right, 'fill_border': _fill_border}

def train_specialist(task_name, n_train=400):
    rng = np.random.RandomState(SEED)
    ins, tgs = TASK_FNS[task_name](n_train, GS, rng)
    x = torch.tensor(np.array(ins)); y = torch.tensor(np.array(tgs))
    model = LiquidNCA(NC, HC).to(DEVICE)
    ctx = nn.Parameter(torch.randn(1, CTX_CH, 1, 1) * 0.01)
    opt = torch.optim.Adam(list(model.parameters()) + [ctx], lr=1e-3)
    for ep in range(SPECIALIST_EPOCHS):
        model.train(); perm = torch.randperm(n_train)
        for i in range(0, n_train, BS):
            idx = perm[i:i+BS]; opt.zero_grad()
            out = model(x[idx], ctx=ctx.expand(len(idx),-1,-1,-1))
            F.cross_entropy(out, y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [ctx], 1.0); opt.step()
    return model, ctx.data.clone()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 105: Routing Confidence Calibration")
    print("=" * 70)

    # Train specialists
    print("  Training 5 specialists...")
    specialists = {}; contexts = {}
    for tn in TASK_FNS:
        model, ctx = train_specialist(tn)
        specialists[tn] = model; contexts[tn] = ctx
        print(f"    {tn}: done")

    task_names = list(TASK_FNS.keys())
    rng = np.random.RandomState(42)
    results = []

    # Generate test data for each task (including some cross-task to test confidence)
    for gt_task in task_names:
        ins, tgs = TASK_FNS[gt_task](N_TEST, GS, rng)

        for i in range(N_TEST):
            x_t = torch.tensor(ins[i]).unsqueeze(0)
            y_t = torch.tensor(tgs[i]).unsqueeze(0)

            # Evaluate all specialists
            losses = {}
            for tn in task_names:
                specialists[tn].eval()
                with torch.no_grad():
                    pred = specialists[tn](x_t, ctx=contexts[tn].expand(1,-1,-1,-1))
                    loss = F.cross_entropy(pred, y_t).item()
                losses[tn] = loss

            # Rank by loss
            ranked = sorted(losses.items(), key=lambda x: x[1])
            top1_name, top1_loss = ranked[0]
            top2_name, top2_loss = ranked[1]
            gap = top2_loss - top1_loss

            # Check correctness
            specialists[top1_name].eval()
            with torch.no_grad():
                pred = specialists[top1_name](x_t, ctx=contexts[top1_name].expand(1,-1,-1,-1))
                pred_grid = pred.argmax(1).squeeze(0).numpy()
            correct = np.array_equal(pred_grid, tgs[i])

            results.append({
                'gt_task': gt_task,
                'routed_to': top1_name,
                'correct_route': (top1_name == gt_task),
                'correct_output': correct,
                'top1_loss': top1_loss,
                'top2_loss': top2_loss,
                'confidence_gap': gap
            })

    # Analysis
    correct_routes = [r for r in results if r['correct_route']]
    wrong_routes = [r for r in results if not r['correct_route']]
    correct_gaps = [r['confidence_gap'] for r in correct_routes]
    wrong_gaps = [r['confidence_gap'] for r in wrong_routes]

    route_acc = len(correct_routes) / len(results) * 100
    avg_gap_correct = np.mean(correct_gaps) if correct_gaps else 0
    avg_gap_wrong = np.mean(wrong_gaps) if wrong_gaps else 0

    # Find confidence threshold for compositional fallback
    solved_direct = sum(1 for r in results if r['correct_output'])
    solve_rate = solved_direct / len(results) * 100

    print(f"\n{'=' * 70}")
    print(f"  Routing Accuracy: {route_acc:.1f}%")
    print(f"  Direct Solve Rate: {solve_rate:.1f}%")
    print(f"  Avg Confidence Gap (correct): {avg_gap_correct:.3f}")
    print(f"  Avg Confidence Gap (wrong):   {avg_gap_wrong:.3f}")
    print(f"  Wrong routes: {len(wrong_routes)}/{len(results)}")

    # Threshold analysis
    thresholds = np.arange(0, 3, 0.1)
    fallback_results = []
    for thresh in thresholds:
        n_confident = sum(1 for r in results if r['confidence_gap'] >= thresh)
        n_correct_confident = sum(1 for r in results if r['confidence_gap'] >= thresh and r['correct_route'])
        precision = n_correct_confident / max(n_confident, 1)
        fallback_results.append({'threshold': thresh, 'n_confident': n_confident,
                                  'precision': precision, 'coverage': n_confident / len(results)})

    print(f"{'=' * 70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase105_routing_confidence.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 105: Routing Confidence Calibration',
            'timestamp': datetime.now().isoformat(),
            'routing_accuracy': route_acc,
            'solve_rate': solve_rate,
            'avg_gap_correct': avg_gap_correct,
            'avg_gap_wrong': avg_gap_wrong,
            'fallback_analysis': fallback_results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Gap distribution
        if correct_gaps: axes[0].hist(correct_gaps, bins=30, alpha=0.7, label=f'Correct ({len(correct_gaps)})', color='green')
        if wrong_gaps: axes[0].hist(wrong_gaps, bins=30, alpha=0.7, label=f'Wrong ({len(wrong_gaps)})', color='red')
        axes[0].legend(); axes[0].set_xlabel('Confidence Gap'); axes[0].set_title('Gap Distribution')

        # 2. Precision-Coverage tradeoff
        precs = [r['precision'] for r in fallback_results]
        covs = [r['coverage'] for r in fallback_results]
        axes[1].plot(covs, precs, 'b-o', markersize=3)
        axes[1].set_xlabel('Coverage'); axes[1].set_ylabel('Precision'); axes[1].set_title('Precision-Coverage Tradeoff')

        # 3. Per-task routing accuracy
        per_task = {}
        for r in results:
            gt = r['gt_task']
            if gt not in per_task: per_task[gt] = {'correct': 0, 'total': 0}
            per_task[gt]['total'] += 1
            if r['correct_route']: per_task[gt]['correct'] += 1
        names = list(per_task.keys())
        accs = [per_task[n]['correct']/per_task[n]['total']*100 for n in names]
        axes[2].barh(names, accs, color='steelblue')
        axes[2].set_xlabel('Routing Accuracy (%)'); axes[2].set_title('Per-Task Routing Accuracy')

        plt.suptitle('Phase 105: Routing Confidence Calibration', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase105_routing_confidence.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 105 complete!")

if __name__ == '__main__':
    main()
