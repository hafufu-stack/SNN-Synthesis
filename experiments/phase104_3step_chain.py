"""
Phase 104: 3-Step Compositional Chain

Extend Phase 94's 2-step expert chaining to 3-step chains: C(B(A(x))).
Beam search over K^3 space with pruning to discover triple-composite rules.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cpu"; SEED = 2026; NC = 10; HC = 32; NCA_STEPS = 10; CTX_CH = 10
SPECIALIST_EPOCHS = 40; BS = 32; GS = 8

# L-NCA
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

def one_hot(grid, nc=10):
    h, w = grid.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): o[c] = (grid == c).astype(np.float32)
    return o

# ====================================================================
# Toy Task Generators
# ====================================================================
def _gravity(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)): g[rng.randint(0, gs), rng.randint(0, gs)] = 1
        res = np.zeros_like(g)
        for c in range(gs):
            cnt = sum(1 for r in range(gs) if g[r, c] == 1)
            row, pl = gs - 1, 0
            while pl < cnt and row >= 0: res[row, c] = 1; pl += 1; row -= 1
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
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
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
                if g[r, c] > 0: res[r, (c+1)%gs] = g[r, c]
        ins.append(one_hot(g)); tgs.append(res)
    return ins, tgs

def _fill_border(n, gs, rng):
    ins, tgs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)): g[rng.randint(0, gs), rng.randint(0, gs)] = rng.randint(1, 5)
        res = g.copy(); res[0,:] = 3; res[-1,:] = 3; res[:,0] = 3; res[:,-1] = 3
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
            idx = perm[i:i+BS]
            xb, yb = x[idx], y[idx]
            opt.zero_grad()
            out = model(xb, ctx=ctx.expand(xb.size(0),-1,-1,-1))
            F.cross_entropy(out, yb).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [ctx], 1.0)
            opt.step()
    return model, ctx.data.clone()

def apply_expert(model, ctx, input_oh):
    """Apply expert and return one-hot result."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(input_oh).unsqueeze(0)
        pred = model(x, ctx=ctx.expand(1, -1, -1, -1))
        pred_class = pred.argmax(1).squeeze(0).numpy()
    return one_hot(pred_class)


def generate_composite_ground_truth(task_chain, n, gs, rng):
    """Generate ground truth for a composite task = chain of transforms."""
    # Start with random input
    inputs, outputs = [], []
    for _ in range(n):
        g = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)):
            g[rng.randint(0, gs), rng.randint(0, gs)] = rng.randint(1, 4)

        # Apply chain of transforms
        current = g.copy()
        for task_name in task_chain:
            i_oh, t = TASK_FNS[task_name](1, gs, np.random.RandomState(hash(tuple(current.flatten())) % 2**31))
            # Actually apply the transform properly
            if task_name == 'color_invert':
                res = current.copy()
                res[current == 1] = 2; res[current == 2] = 1
                current = res
            elif task_name == 'move_right':
                res = np.zeros_like(current)
                for r in range(gs):
                    for c in range(gs):
                        if current[r, c] > 0: res[r, (c+1)%gs] = current[r, c]
                current = res
            elif task_name == 'gravity':
                res = np.zeros_like(current)
                for c in range(gs):
                    cnt = sum(1 for r in range(gs) if current[r, c] > 0)
                    vals = [current[r, c] for r in range(gs) if current[r, c] > 0]
                    for idx, v in enumerate(vals):
                        res[gs-1-idx, c] = v
                current = res
            elif task_name == 'fill_border':
                current[0,:] = 3; current[-1,:] = 3; current[:,0] = 3; current[:,-1] = 3
            elif task_name == 'expand':
                res = current.copy()
                for y in range(gs):
                    for x in range(gs):
                        if current[y, x] > 0:
                            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                                ny, nx = y+dy, x+dx
                                if 0 <= ny < gs and 0 <= nx < gs and res[ny, nx] == 0:
                                    res[ny, nx] = current[y, x]
                current = res

        inputs.append(one_hot(g))
        outputs.append(current)
    return inputs, outputs


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 104: 3-Step Compositional Chain")
    print("=" * 70)

    # Train specialists
    print("  Training 5 specialists...")
    specialists = {}; contexts = {}
    for tn in TASK_FNS:
        model, ctx = train_specialist(tn)
        specialists[tn] = model; contexts[tn] = ctx
        print(f"    {tn}: done")

    task_names = list(TASK_FNS.keys())
    rng = np.random.RandomState(SEED)

    # Define 3-step composite tasks
    chains_3 = [
        ('color_invert', 'gravity', 'move_right'),
        ('move_right', 'color_invert', 'gravity'),
        ('gravity', 'move_right', 'fill_border'),
        ('expand', 'color_invert', 'move_right'),
    ]

    # Also include 2-step for comparison
    chains_2 = [
        ('color_invert', 'gravity'),
        ('gravity', 'move_right'),
        ('expand', 'color_invert'),
    ]

    results = []
    N_TEST = 50

    # Test 2-step chains
    print(f"\n  === 2-Step Chains ===")
    for chain in chains_2:
        test_ins, test_outs = generate_composite_ground_truth(chain, N_TEST, GS, rng)

        # Exhaustive search over all K^2 pairs
        best_em = 0; best_chain_found = None
        for a, b in itertools.product(task_names, repeat=2):
            n_correct = 0
            for i in range(N_TEST):
                step1 = apply_expert(specialists[a], contexts[a], test_ins[i])
                step2 = apply_expert(specialists[b], contexts[b], step1)
                pred = np.argmax(step2, axis=0)
                if np.array_equal(pred, test_outs[i]):
                    n_correct += 1
            em = n_correct / N_TEST
            if em > best_em:
                best_em = em; best_chain_found = (a, b)

        match = best_chain_found == chain if best_chain_found else False
        print(f"    {' -> '.join(chain):40s} EM={best_em*100:.0f}%  "
              f"Found={best_chain_found}  {'OK' if match else 'WRONG'}")
        results.append({
            'chain': chain, 'depth': 2, 'exact_match': best_em,
            'found_chain': best_chain_found, 'correct_route': match
        })

    # Test 3-step chains with beam-pruned search
    print(f"\n  === 3-Step Chains ===")
    for chain in chains_3:
        test_ins, test_outs = generate_composite_ground_truth(chain, N_TEST, GS, rng)

        # Beam search: evaluate all K^3 = 125 combinations (feasible for K=5)
        best_em = 0; best_chain_found = None
        for a, b, c in itertools.product(task_names, repeat=3):
            n_correct = 0
            for i in range(N_TEST):
                step1 = apply_expert(specialists[a], contexts[a], test_ins[i])
                step2 = apply_expert(specialists[b], contexts[b], step1)
                step3 = apply_expert(specialists[c], contexts[c], step2)
                pred = np.argmax(step3, axis=0)
                if np.array_equal(pred, test_outs[i]):
                    n_correct += 1
            em = n_correct / N_TEST
            if em > best_em:
                best_em = em; best_chain_found = (a, b, c)

        match = best_chain_found == chain if best_chain_found else False
        print(f"    {' -> '.join(chain):40s} EM={best_em*100:.0f}%  "
              f"Found={best_chain_found}  {'OK' if match else 'WRONG'}")
        results.append({
            'chain': chain, 'depth': 3, 'exact_match': best_em,
            'found_chain': best_chain_found, 'correct_route': match
        })

    # Summary
    print(f"\n{'=' * 70}")
    r2 = [r for r in results if r['depth'] == 2]
    r3 = [r for r in results if r['depth'] == 3]
    avg2 = np.mean([r['exact_match'] for r in r2]) if r2 else 0
    avg3 = np.mean([r['exact_match'] for r in r3]) if r3 else 0
    print(f"  2-Step: avg EM={avg2*100:.1f}%  correct routes={sum(r['correct_route'] for r in r2)}/{len(r2)}")
    print(f"  3-Step: avg EM={avg3*100:.1f}%  correct routes={sum(r['correct_route'] for r in r3)}/{len(r3)}")
    print(f"{'=' * 70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase104_3step_chain.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 104: 3-Step Compositional Chain',
            'timestamp': datetime.now().isoformat(),
            'avg_em_2step': avg2, 'avg_em_3step': avg3,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        labels = [' -> '.join(r['chain']) for r in results]
        ems = [r['exact_match'] * 100 for r in results]
        colors = ['#2ecc71' if r['correct_route'] else '#e74c3c' for r in results]
        bars = ax.barh(labels, ems, color=colors, edgecolor='black')
        ax.set_xlabel('Exact Match (%)')
        ax.set_title('Phase 104: 3-Step Compositional Chain\n(Green=correct route, Red=wrong route)')
        ax.axvline(x=100, color='gray', linestyle='--', alpha=0.3)
        for bar, em in zip(bars, ems):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{em:.0f}%', va='center')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase104_3step_chain.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 104 complete!")

if __name__ == '__main__':
    main()
