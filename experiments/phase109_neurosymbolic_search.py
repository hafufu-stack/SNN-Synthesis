"""
Phase 109: Neuro-Symbolic Expert Search

Beam search over compositions of L-NCA experts + geometric ops.
Instead of training one model to learn complex ARC rules, discover
the correct expert chain that transforms demo inputs to outputs.

DSL primitives:
- Trained L-NCA experts (from Phase 102 library)
- Geometric ops: rot90, rot180, rot270, flipud, fliplr, transpose

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
ARC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training")
EXPERT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experts")
DEVICE = "cpu"; SEED = 2026; NC = 10; HC = 32; NCA_STEPS = 10
MAX_GS = 30; MAX_CHAIN_DEPTH = 3; BEAM_K = 5

# ====================================================================
# L-NCA (for loading experts)
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
        x_in = x if ctx is None else x + ctx.expand(-1, -1, h, w)
        for _ in range(n_steps):
            combined = torch.cat([x_in, state], 1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x_in, state, delta], 1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)

# ====================================================================
# Geometric Operations (pure numpy, no learning)
# ====================================================================
def op_identity(grid): return grid
def op_rot90(grid): return np.rot90(grid, 1)
def op_rot180(grid): return np.rot90(grid, 2)
def op_rot270(grid): return np.rot90(grid, 3)
def op_flipud(grid): return np.flipud(grid).copy()
def op_fliplr(grid): return np.fliplr(grid).copy()
def op_transpose(grid): return grid.T.copy()

GEO_OPS = {
    'identity': op_identity, 'rot90': op_rot90, 'rot180': op_rot180,
    'rot270': op_rot270, 'flipud': op_flipud, 'fliplr': op_fliplr,
    'transpose': op_transpose
}

# ====================================================================
# Helpers
# ====================================================================
def one_hot(grid, nc=10):
    h, w = grid.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): o[c] = (grid == c).astype(np.float32)
    return o

def freq_remap(grid):
    flat = grid.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    mapping = {unique[order[i]]: i for i in range(len(unique))}
    return np.vectorize(lambda c: mapping.get(c, c))(grid)

def pad_to(grid, max_h, max_w):
    h, w = grid.shape
    out = np.zeros((max_h, max_w), dtype=grid.dtype)
    out[:h, :w] = grid
    return out

# ====================================================================
# Expert Application
# ====================================================================
def apply_expert(model, ctx, grid):
    """Apply L-NCA expert to a grid. Returns predicted grid."""
    h, w = grid.shape
    x = torch.tensor(one_hot(grid)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        if ctx is not None:
            pred = model(x, ctx=ctx.expand(1, -1, -1, -1))
        else:
            pred = model(x)
        return pred.argmax(1).squeeze(0).numpy()

def apply_chain(chain, grid, experts):
    """Apply a chain of operations/experts to a grid."""
    current = grid.copy()
    for op_name in chain:
        if op_name in GEO_OPS:
            current = GEO_OPS[op_name](current)
        elif op_name in experts:
            e = experts[op_name]
            # Expert only works on same-size, pad if needed
            h, w = current.shape
            if h > MAX_GS or w > MAX_GS:
                return None
            current = apply_expert(e['model'], e['ctx'], current)
        else:
            return None
    return current

# ====================================================================
# Beam Search over Expert Chains
# ====================================================================
def beam_search(demos, experts, max_depth=MAX_CHAIN_DEPTH, beam_k=BEAM_K):
    """Search for the best chain of ops/experts that transforms demo inputs to outputs."""
    
    # All available primitives
    primitives = list(GEO_OPS.keys())
    # Add top-K experts by name (don't use all 262, just sample)
    expert_names = list(experts.keys())[:20]  # Use first 20 experts
    all_ops = primitives + expert_names
    
    demo_ins = [freq_remap(np.array(d['input'])) for d in demos]
    demo_outs = [np.array(d['output']) for d in demos]
    
    # Check same-size constraint
    for di, do in zip(demo_ins, demo_outs):
        if di.shape != do.shape:
            return None, 0.0  # Can't handle size-changing yet
    
    best_chain = None
    best_score = 0.0
    
    # Depth 1: try each single primitive
    candidates = []
    for op in all_ops:
        score = 0
        for di, do in zip(demo_ins, demo_outs):
            result = apply_chain([op], di, experts)
            if result is not None and result.shape == do.shape:
                if np.array_equal(result, do):
                    score += 1
        score /= len(demos)
        if score > best_score:
            best_score = score; best_chain = [op]
        if score > 0:
            candidates.append(([op], score))
    
    # Sort and keep top-K
    candidates.sort(key=lambda x: -x[1])
    candidates = candidates[:beam_k]
    
    if best_score >= 1.0:
        return best_chain, best_score
    
    # Depth 2: extend top candidates
    new_candidates = []
    for chain, prev_score in candidates:
        for op in all_ops:
            new_chain = chain + [op]
            score = 0
            for di, do in zip(demo_ins, demo_outs):
                result = apply_chain(new_chain, di, experts)
                if result is not None and result.shape == do.shape:
                    if np.array_equal(result, do):
                        score += 1
            score /= len(demos)
            if score > best_score:
                best_score = score; best_chain = new_chain
            if score > 0:
                new_candidates.append((new_chain, score))
    
    if best_score >= 1.0:
        return best_chain, best_score
    
    # Depth 3: extend again
    new_candidates.sort(key=lambda x: -x[1])
    new_candidates = new_candidates[:beam_k]
    
    for chain, prev_score in new_candidates:
        for op in all_ops:
            new_chain = chain + [op]
            score = 0
            for di, do in zip(demo_ins, demo_outs):
                result = apply_chain(new_chain, di, experts)
                if result is not None and result.shape == do.shape:
                    if np.array_equal(result, do):
                        score += 1
            score /= len(demos)
            if score > best_score:
                best_score = score; best_chain = new_chain
    
    return best_chain, best_score


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 109: Neuro-Symbolic Expert Search")
    print("=" * 70)

    # Load expert library
    print("  Loading expert library...")
    experts = {}
    if os.path.exists(EXPERT_DIR):
        for f in sorted(os.listdir(EXPERT_DIR))[:20]:  # Top 20 experts
            if not f.endswith('.pt'): continue
            tid = f.replace('.pt', '')
            data = torch.load(os.path.join(EXPERT_DIR, f), map_location=DEVICE, weights_only=False)
            model = LiquidNCA(NC, HC)
            model.load_state_dict(data['model'])
            model.eval()
            experts[tid] = {'model': model, 'ctx': data['ctx']}
    print(f"  Loaded {len(experts)} experts + {len(GEO_OPS)} geometric ops")

    # Load ARC tasks (same-size only for now)
    files = sorted([f for f in os.listdir(ARC_DIR) if f.endswith('.json')])
    tasks = {}
    for fname in files:
        tid = fname.replace('.json', '')
        with open(os.path.join(ARC_DIR, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Same-size and within limits
        valid = True
        for pair in data['train'] + data['test']:
            inp, out = np.array(pair['input']), np.array(pair['output'])
            if inp.shape != out.shape or max(inp.shape) > MAX_GS:
                valid = False; break
        if valid:
            tasks[tid] = data
        if len(tasks) >= 30:
            break
    print(f"  Loaded {len(tasks)} same-size ARC tasks")

    # Search
    results = []
    n_solved = 0
    n_geo_solved = 0

    for i, (tid, task) in enumerate(tasks.items()):
        t0 = time.time()
        chain, demo_score = beam_search(task['train'], experts)
        elapsed = time.time() - t0

        # Test on held-out test pairs
        test_score = 0
        if chain:
            tests = task['test']
            for test_pair in tests:
                ti = freq_remap(np.array(test_pair['input']))
                to = np.array(test_pair['output'])
                result = apply_chain(chain, ti, experts)
                if result is not None and result.shape == to.shape and np.array_equal(result, to):
                    test_score += 1
            test_score /= max(len(tests), 1)
        
        solved = test_score >= 1.0
        if solved: n_solved += 1
        is_geo = chain and all(op in GEO_OPS for op in chain) if chain else False
        if solved and is_geo: n_geo_solved += 1

        chain_str = ' → '.join(chain) if chain else 'NONE'
        status = "SOLVED" if solved else f"demo={demo_score*100:.0f}%"
        print(f"  [{i+1:2d}] {tid[:12]:12s}  {status:12s}  chain=[{chain_str}]  ({elapsed:.1f}s)")

        results.append({
            'task_id': tid, 'chain': chain, 'demo_score': demo_score,
            'test_score': test_score, 'solved': solved, 'is_geo': is_geo,
            'time_s': elapsed
        })

    sr = n_solved / max(len(tasks), 1) * 100
    print(f"\n{'=' * 70}")
    print(f"  Neuro-Symbolic Search: {n_solved}/{len(tasks)} = {sr:.1f}% solved")
    print(f"  Geometric-only solves: {n_geo_solved}")
    print(f"  Expert-chain solves: {n_solved - n_geo_solved}")
    print(f"{'=' * 70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase109_neurosymbolic_search.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 109: Neuro-Symbolic Expert Search',
            'timestamp': datetime.now().isoformat(),
            'solve_rate': sr, 'n_solved': n_solved, 'n_tested': len(tasks),
            'n_geo_solved': n_geo_solved,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        solved_r = [r for r in results if r['solved']]
        unsolved_r = [r for r in results if not r['solved']]
        
        axes[0].bar(['Solved', 'Unsolved'], [len(solved_r), len(unsolved_r)],
                    color=['#2ecc71', '#e74c3c'], edgecolor='black')
        axes[0].set_title(f'Solve Rate: {sr:.1f}%')

        # Chain depth distribution
        depths = [len(r['chain']) for r in solved_r if r['chain']]
        if depths:
            axes[1].hist(depths, bins=range(1, max(depths)+2), color='steelblue', edgecolor='black')
            axes[1].set_xlabel('Chain Depth'); axes[1].set_ylabel('Count')
            axes[1].set_title('Solution Chain Depth Distribution')

        plt.suptitle('Phase 109: Neuro-Symbolic Expert Search', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase109_neurosymbolic_search.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 109 complete!")

if __name__ == '__main__':
    main()
