"""
Phase 234: Neuro-Symbolic Compiler

NCA gives ~60% PA (high-quality "sketch"), but misses 1-2 pixels.
DSL search alone is exponentially expensive.
Solution: Use NCA output as the HEURISTIC for DSL program search.

"Neural intuition guides symbolic precision."

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import load_arc_training, prepare_arc_meta_dataset
from phase199_gated import GatedHybridNCA


# ==============================================================
# Minimal DSL for ARC-like transforms
# ==============================================================
def dsl_identity(grid):
    return grid.copy()

def dsl_flip_h(grid):
    return grid[:, ::-1].copy()

def dsl_flip_v(grid):
    return grid[::-1, :].copy()

def dsl_rotate90(grid):
    return np.rot90(grid).copy()

def dsl_rotate180(grid):
    return np.rot90(grid, 2).copy()

def dsl_rotate270(grid):
    return np.rot90(grid, 3).copy()

def dsl_transpose(grid):
    if grid.shape[0] == grid.shape[1]:
        return grid.T.copy()
    return grid.copy()

def make_color_swap(c1, c2):
    def fn(grid):
        out = grid.copy()
        out[grid == c1] = c2
        out[grid == c2] = c1
        return out
    fn.__name__ = f'swap_{c1}_{c2}'
    return fn

def make_fill_color(src, dst):
    def fn(grid):
        out = grid.copy()
        out[grid == src] = dst
        return out
    fn.__name__ = f'fill_{src}to{dst}'
    return fn

def dsl_border_fill(grid):
    out = grid.copy()
    bg = int(np.bincount(grid.ravel().astype(int), minlength=11).argmax())
    for c in range(10):
        if c != bg:
            out[0, :] = c; out[-1, :] = c
            out[:, 0] = c; out[:, -1] = c
            return out
    return out

# Build DSL program library
def build_dsl_library(input_grid):
    """Build all single-step DSL programs."""
    programs = [
        ('identity', dsl_identity),
        ('flip_h', dsl_flip_h),
        ('flip_v', dsl_flip_v),
        ('rotate90', dsl_rotate90),
        ('rotate180', dsl_rotate180),
        ('rotate270', dsl_rotate270),
        ('transpose', dsl_transpose),
    ]
    # Color swaps for colors present in input
    colors = list(set(input_grid.ravel().astype(int)))
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            programs.append((f'swap_{colors[i]}_{colors[j]}',
                           make_color_swap(colors[i], colors[j])))
    # Color fills
    for c1 in colors[:5]:
        for c2 in range(10):
            if c1 != c2:
                programs.append((f'fill_{c1}to{c2}',
                               make_fill_color(c1, c2)))
    return programs


def search_dsl(input_grid, nca_output, gt_output, max_depth=2):
    """Search for DSL programs that match NCA output (and GT)."""
    programs = build_dsl_library(input_grid)
    h, w = gt_output.shape

    best_nca_score = -1
    best_gt_score = -1
    best_nca_prog = 'none'
    best_gt_prog = 'none'
    best_nca_em = 0
    best_gt_em = 0

    # Single-step search
    for name, fn in programs:
        try:
            result = fn(input_grid)
            if result.shape != gt_output.shape:
                continue
            # Score against NCA output
            nca_match = (result == nca_output).mean()
            gt_match = (result == gt_output).mean()
            gt_em = float((result == gt_output).all())

            if nca_match > best_nca_score:
                best_nca_score = nca_match
                best_nca_prog = name
                best_nca_em = float((result == gt_output).all())

            if gt_match > best_gt_score:
                best_gt_score = gt_match
                best_gt_prog = name
                best_gt_em = gt_em
        except Exception:
            continue

    # Two-step search (compose top-5 programs)
    if max_depth >= 2:
        # Pre-compute single-step results
        single_results = []
        for name, fn in programs:
            try:
                result = fn(input_grid)
                if result.shape == gt_output.shape:
                    nca_match = (result == nca_output).mean()
                    single_results.append((name, fn, result, nca_match))
            except Exception:
                continue
        # Sort by NCA match (use NCA as heuristic!)
        single_results.sort(key=lambda x: x[3], reverse=True)
        top_k = min(10, len(single_results))

        for i in range(top_k):
            n1, fn1, r1, _ = single_results[i]
            for j in range(top_k):
                n2, fn2, r2, _ = single_results[j]
                if i == j:
                    continue
                try:
                    composed = fn2(r1)
                    if composed.shape != gt_output.shape:
                        continue
                    nca_match = (composed == nca_output).mean()
                    gt_match = (composed == gt_output).mean()

                    if nca_match > best_nca_score:
                        best_nca_score = nca_match
                        best_nca_prog = f'{n1}+{n2}'
                        best_nca_em = float((composed == gt_output).all())
                    if gt_match > best_gt_score:
                        best_gt_score = gt_match
                        best_gt_prog = f'{n1}+{n2}'
                        best_gt_em = float((composed == gt_output).all())
                except Exception:
                    continue

    return {
        'nca_guided': {'prog': best_nca_prog, 'pa': float(best_nca_score),
                       'em': best_nca_em},
        'oracle': {'prog': best_gt_prog, 'pa': float(best_gt_score),
                   'em': best_gt_em},
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 234: Neuro-Symbolic Compiler")
    print(f"  NCA sketch + DSL precision = perfect answer")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train NCA
    print(f"\n[Training GatedHybridNCA]")
    torch.manual_seed(SEED)
    model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        model.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    # Evaluate: NCA alone, DSL alone (random), DSL guided by NCA
    nca_pa, nca_em = 0, 0
    dsl_nca_pa, dsl_nca_em = 0, 0
    dsl_oracle_pa, dsl_oracle_em = 0, 0
    nca_only_pa_list = []
    dsl_nca_pa_list = []

    print(f"\n[Evaluating Neuro-Symbolic Compiler]")
    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            # NCA prediction
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            nca_pred = logits[0, :, :oh, :ow].argmax(dim=0).cpu().numpy()
            gt_np = gt[:oh, :ow].cpu().numpy()
            input_np = ti[0, :11, :oh, :ow].argmax(dim=0).cpu().numpy()

            pa_val = (nca_pred == gt_np).mean()
            em_val = float((nca_pred == gt_np).all())
            nca_pa += pa_val; nca_em += em_val
            nca_only_pa_list.append(pa_val)

            # DSL search
            dsl_result = search_dsl(input_np, nca_pred, gt_np, max_depth=2)

            dsl_nca_pa += dsl_result['nca_guided']['pa']
            dsl_nca_em += dsl_result['nca_guided']['em']
            dsl_nca_pa_list.append(dsl_result['nca_guided']['pa'])
            dsl_oracle_pa += dsl_result['oracle']['pa']
            dsl_oracle_em += dsl_result['oracle']['em']

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)} "
                      f"NCA={pa_val*100:.0f}% "
                      f"DSL(NCA)={dsl_result['nca_guided']['pa']*100:.0f}% "
                      f"prog={dsl_result['nca_guided']['prog']}")

    n_test = len(test)
    nca_pa /= n_test; nca_em /= n_test
    dsl_nca_pa /= n_test; dsl_nca_em /= n_test
    dsl_oracle_pa /= n_test; dsl_oracle_em /= n_test

    # Count how many tasks DSL improved over NCA
    improved = sum(1 for a, b in zip(nca_only_pa_list, dsl_nca_pa_list) if b > a)
    degraded = sum(1 for a, b in zip(nca_only_pa_list, dsl_nca_pa_list) if b < a)

    print(f"\n{'='*70}")
    print(f"  NEURO-SYMBOLIC COMPILER:")
    print(f"  NCA only      : PA={nca_pa*100:.1f}%, EM={nca_em*100:.1f}%")
    print(f"  DSL(NCA-guided): PA={dsl_nca_pa*100:.1f}%, EM={dsl_nca_em*100:.1f}%")
    print(f"  DSL(oracle)   : PA={dsl_oracle_pa*100:.1f}%, EM={dsl_oracle_em*100:.1f}%")
    print(f"  Improved: {improved}/{n_test}, Degraded: {degraded}/{n_test}")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    results = {
        'nca_only': {'pa': nca_pa, 'em': nca_em},
        'dsl_nca_guided': {'pa': dsl_nca_pa, 'em': dsl_nca_em},
        'dsl_oracle': {'pa': dsl_oracle_pa, 'em': dsl_oracle_em},
        'improved': improved, 'degraded': degraded
    }

    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase234_neurosymbolic.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = ['NCA only', 'DSL(NCA)', 'DSL(oracle)']
        pa_vals = [nca_pa*100, dsl_nca_pa*100, dsl_oracle_pa*100]
        em_vals = [nca_em*100, dsl_nca_em*100, dsl_oracle_em*100]
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        x = np.arange(3); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('%'); ax.set_title('Phase 234: Neuro-Symbolic Compiler', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase234_neurosymbolic.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
