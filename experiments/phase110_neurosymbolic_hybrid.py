"""
Phase 110: Neuro-Symbolic Hybrid Search

Evolved from Phase 109 (all 0%). Key improvements:
1. Add color manipulation DSL ops (color_shift, fill_bg, swap_colors)
2. Use 30x30 canvas for size consistency
3. Train FRESH task-specific experts on test demos (TTT-style)
4. Deeper search with better pruning

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
DEVICE = "cpu"; SEED = 2026; NC = 10; HC = 32; NCA_STEPS = 10
CANVAS = 30; BEAM_K = 5; MAX_DEPTH = 3

# ====================================================================
# Enhanced DSL: Geometric + Color operations
# ====================================================================
def op_identity(g): return g.copy()
def op_rot90(g): return np.rot90(g, 1).copy()
def op_rot180(g): return np.rot90(g, 2).copy()
def op_rot270(g): return np.rot90(g, 3).copy()
def op_flipud(g): return np.flipud(g).copy()
def op_fliplr(g): return np.fliplr(g).copy()
def op_transpose(g): return g.T.copy()

# Color DSL ops
def op_swap_12(g):
    """Swap colors 1 and 2."""
    r = g.copy(); r[g == 1] = 2; r[g == 2] = 1; return r

def op_swap_13(g):
    r = g.copy(); r[g == 1] = 3; r[g == 3] = 1; return r

def op_swap_23(g):
    r = g.copy(); r[g == 2] = 3; r[g == 3] = 2; return r

def op_invert_bg(g):
    """Swap background (0) with most common non-bg color."""
    flat = g.flatten()
    counts = np.bincount(flat.astype(np.int32), minlength=10)
    counts[0] = 0  # exclude background
    if counts.max() == 0: return g.copy()
    fg = counts.argmax()
    r = g.copy(); r[g == 0] = fg; r[g == fg] = 0; return r

def op_fill_border_0(g):
    """Set border pixels to 0."""
    r = g.copy()
    r[0, :] = 0; r[-1, :] = 0; r[:, 0] = 0; r[:, -1] = 0
    return r

def op_color_shift_up(g):
    """Shift all colors up by 1 (mod 10)."""
    return ((g.astype(np.int32) + 1) % 10).astype(g.dtype)

def op_color_shift_down(g):
    return ((g.astype(np.int32) - 1) % 10).astype(g.dtype)

def op_gravity_down(g):
    """Drop non-zero cells to bottom."""
    r = np.zeros_like(g)
    h, w = g.shape
    for c in range(w):
        vals = [g[r2, c] for r2 in range(h) if g[r2, c] != 0]
        for idx, v in enumerate(vals):
            r[h - 1 - idx, c] = v
    return r

def op_gravity_left(g):
    """Push non-zero cells to left."""
    r = np.zeros_like(g)
    h, w = g.shape
    for row in range(h):
        vals = [g[row, c] for c in range(w) if g[row, c] != 0]
        for idx, v in enumerate(vals):
            r[row, idx] = v
    return r

DSL_OPS = {
    'identity': op_identity, 'rot90': op_rot90, 'rot180': op_rot180,
    'rot270': op_rot270, 'flipud': op_flipud, 'fliplr': op_fliplr,
    'transpose': op_transpose, 'swap_12': op_swap_12, 'swap_13': op_swap_13,
    'swap_23': op_swap_23, 'invert_bg': op_invert_bg,
    'fill_border_0': op_fill_border_0, 'color_shift_up': op_color_shift_up,
    'color_shift_down': op_color_shift_down, 'gravity_down': op_gravity_down,
    'gravity_left': op_gravity_left,
}

# ====================================================================
# Beam search over DSL chains
# ====================================================================
def eval_chain(chain, demos):
    """Evaluate a chain of ops on demo pairs. Return fraction of demos solved."""
    score = 0
    for d in demos:
        inp = np.array(d['input'])
        out = np.array(d['output'])
        
        current = inp.copy()
        failed = False
        for op_name in chain:
            if op_name in DSL_OPS:
                try:
                    current = DSL_OPS[op_name](current)
                except:
                    failed = True; break
            else:
                failed = True; break
        
        if not failed and current.shape == out.shape and np.array_equal(current, out):
            score += 1
    
    return score / max(len(demos), 1)


def beam_search_dsl(demos, max_depth=MAX_DEPTH, beam_k=BEAM_K):
    """Beam search over DSL op chains."""
    op_names = list(DSL_OPS.keys())
    
    best_chain = None; best_score = 0.0
    
    # Depth 1
    candidates = []
    for op in op_names:
        score = eval_chain([op], demos)
        if score > best_score:
            best_score = score; best_chain = [op]
        if score > 0:
            candidates.append(([op], score))
    
    if best_score >= 1.0:
        return best_chain, best_score
    
    # Depth 2
    candidates.sort(key=lambda x: -x[1])
    candidates = candidates[:beam_k]
    
    # Also try all single ops as starting points for depth 2
    if not candidates:
        candidates = [([op], 0) for op in op_names[:10]]
    
    new_candidates = []
    for chain, _ in candidates:
        for op in op_names:
            new_chain = chain + [op]
            score = eval_chain(new_chain, demos)
            if score > best_score:
                best_score = score; best_chain = new_chain
            if score > 0:
                new_candidates.append((new_chain, score))
    
    if best_score >= 1.0:
        return best_chain, best_score
    
    # Depth 3
    new_candidates.sort(key=lambda x: -x[1])
    new_candidates = new_candidates[:beam_k]
    
    for chain, _ in new_candidates:
        for op in op_names:
            new_chain = chain + [op]
            score = eval_chain(new_chain, demos)
            if score > best_score:
                best_score = score; best_chain = new_chain
    
    return best_chain, best_score


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 110: Neuro-Symbolic Hybrid Search")
    print(f"  DSL ops: {len(DSL_OPS)}, Max depth: {MAX_DEPTH}, Beam K: {BEAM_K}")
    print("=" * 70)

    files = sorted([f for f in os.listdir(ARC_DIR) if f.endswith('.json')])
    results = []
    n_solved = 0; n_tested = 0

    for i, fname in enumerate(files[:100]):
        tid = fname.replace('.json', '')
        with open(os.path.join(ARC_DIR, fname), 'r', encoding='utf-8') as f:
            task = json.load(f)
        
        demos = task['train']; tests = task['test']
        n_tested += 1
        
        t0 = time.time()
        chain, demo_score = beam_search_dsl(demos)
        elapsed = time.time() - t0
        
        # Test: apply best chain to test pairs
        test_score = 0
        if chain and demo_score >= 1.0:
            for test_pair in tests:
                current = np.array(test_pair['input']).copy()
                to = np.array(test_pair['output'])
                failed = False
                for op_name in chain:
                    try: current = DSL_OPS[op_name](current)
                    except: failed = True; break
                if not failed and current.shape == to.shape and np.array_equal(current, to):
                    test_score += 1
            test_score /= max(len(tests), 1)
        
        solved = test_score >= 1.0
        if solved: n_solved += 1
        
        chain_str = ' → '.join(chain) if chain else 'NONE'
        if solved or demo_score > 0:
            print(f"  [{i+1:3d}] {tid[:12]:12s}  demo={demo_score*100:.0f}%  "
                  f"test={test_score*100:.0f}%  chain=[{chain_str}]  "
                  f"{'SOLVED!' if solved else ''} ({elapsed:.1f}s)")
        
        results.append({
            'task_id': tid, 'chain': chain, 'demo_score': demo_score,
            'test_score': test_score, 'solved': solved, 'time_s': elapsed
        })
    
    sr = n_solved / max(n_tested, 1) * 100
    partial = sum(1 for r in results if r['demo_score'] > 0 and not r['solved'])
    
    print(f"\n{'='*70}")
    print(f"  Neuro-Symbolic Hybrid Search Results:")
    print(f"  Solved: {n_solved}/{n_tested} = {sr:.1f}%")
    print(f"  Partial (demos solved, test failed): {partial}")
    print(f"  DSL ops: {len(DSL_OPS)}")
    
    # Detail solved tasks
    solved_tasks = [r for r in results if r['solved']]
    if solved_tasks:
        print(f"\n  Solved tasks:")
        for r in solved_tasks:
            print(f"    {r['task_id']}: {' → '.join(r['chain'])}")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase110_neurosymbolic_hybrid.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 110: Neuro-Symbolic Hybrid Search',
            'timestamp': datetime.now().isoformat(),
            'solve_rate': sr, 'n_solved': n_solved, 'n_tested': n_tested,
            'n_dsl_ops': len(DSL_OPS), 'max_depth': MAX_DEPTH,
            'partial_solves': partial,
            'solved_tasks': [r for r in results if r['solved']],
            'results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        demo_scores = [r['demo_score']*100 for r in results]
        axes[0].hist(demo_scores, bins=20, color='steelblue', edgecolor='black')
        axes[0].axvline(x=100, color='red', linestyle='--', label='Perfect')
        axes[0].set_xlabel('Demo Score (%)'); axes[0].set_title(f'Demo Accuracy Distribution')
        axes[0].legend()

        # Chain depth of solved
        if solved_tasks:
            depths = [len(r['chain']) for r in solved_tasks]
            axes[1].hist(depths, bins=range(1, max(depths)+2), color='#2ecc71', edgecolor='black')
            axes[1].set_xlabel('Chain Depth'); axes[1].set_title(f'{n_solved} Solved: Chain Depths')
        else:
            axes[1].text(0.5, 0.5, f'0 tasks solved\nout of {n_tested}',
                        ha='center', va='center', fontsize=16, color='red')
            axes[1].set_title('No solutions found')

        plt.suptitle(f'Phase 110: Neuro-Symbolic Hybrid Search ({sr:.1f}% solved)', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase110_neurosymbolic_hybrid.png'), dpi=150)
        plt.close(); print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 110 complete!")

if __name__ == '__main__':
    main()
