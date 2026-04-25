"""
Phase 236: Object-Centric DSL

Upgrade from pixel-level DSL to OBJECT-level DSL.
P234 showed DSL oracle PA=75.2% with pixel ops.
Object ops should push oracle PA/EM much higher.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime
from scipy import ndimage

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import load_arc_training, prepare_arc_meta_dataset
from phase199_gated import GatedHybridNCA


# ==============================================================
# Object Extraction
# ==============================================================
def extract_objects(grid):
    """Extract connected components as objects."""
    objects = []
    h, w = grid.shape
    bg = int(np.bincount(grid.ravel().astype(int), minlength=11).argmax())
    for color in range(11):
        if color == bg:
            continue
        mask = (grid == color)
        if not mask.any():
            continue
        labeled, n_components = ndimage.label(mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            ys, xs = np.where(comp_mask)
            if len(ys) == 0:
                continue
            obj = {
                'color': color,
                'pixels': list(zip(ys.tolist(), xs.tolist())),
                'y_min': int(ys.min()), 'y_max': int(ys.max()),
                'x_min': int(xs.min()), 'x_max': int(xs.max()),
                'size': len(ys),
                'mask': comp_mask,
            }
            objects.append(obj)
    return objects, bg


# ==============================================================
# Object-Centric DSL Operations
# ==============================================================
def dsl_move(grid, obj, dy, dx, bg=0):
    out = grid.copy()
    # Clear old position
    for y, x in obj['pixels']:
        out[y, x] = bg
    h, w = grid.shape
    # Place at new position
    for y, x in obj['pixels']:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            out[ny, nx] = obj['color']
    return out


def dsl_recolor(grid, obj, new_color):
    out = grid.copy()
    for y, x in obj['pixels']:
        out[y, x] = new_color
    return out


def dsl_mirror_h(grid, obj, bg=0):
    out = grid.copy()
    cx = (obj['x_min'] + obj['x_max']) / 2.0
    for y, x in obj['pixels']:
        out[y, x] = bg
    h, w = grid.shape
    for y, x in obj['pixels']:
        nx = int(round(2 * cx - x))
        if 0 <= nx < w:
            out[y, nx] = obj['color']
    return out


def dsl_mirror_v(grid, obj, bg=0):
    out = grid.copy()
    cy = (obj['y_min'] + obj['y_max']) / 2.0
    for y, x in obj['pixels']:
        out[y, x] = bg
    h, w = grid.shape
    for y, x in obj['pixels']:
        ny = int(round(2 * cy - y))
        if 0 <= ny < h:
            out[ny, x] = obj['color']
    return out


def dsl_copy_to(grid, obj, dy, dx):
    out = grid.copy()
    h, w = grid.shape
    for y, x in obj['pixels']:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            out[ny, nx] = obj['color']
    return out


def dsl_fill_bbox(grid, obj, color):
    out = grid.copy()
    out[obj['y_min']:obj['y_max']+1, obj['x_min']:obj['x_max']+1] = color
    return out


def dsl_delete(grid, obj, bg=0):
    out = grid.copy()
    for y, x in obj['pixels']:
        out[y, x] = bg
    return out


def dsl_global_flip_h(grid):
    return grid[:, ::-1].copy()

def dsl_global_flip_v(grid):
    return grid[::-1, :].copy()

def dsl_global_rot90(grid):
    return np.rot90(grid).copy()

def dsl_global_rot180(grid):
    return np.rot90(grid, 2).copy()

def dsl_global_color_swap(grid, c1, c2):
    out = grid.copy()
    out[grid == c1] = c2
    out[grid == c2] = c1
    return out

def dsl_global_fill(grid, src, dst):
    out = grid.copy()
    out[grid == src] = dst
    return out


# ==============================================================
# Oracle Search (knows ground truth, measures DSL expressiveness)
# ==============================================================
def oracle_search(input_grid, gt_grid, max_time=2.0):
    """Search for best DSL program matching ground truth."""
    t0 = time.time()
    h, w = gt_grid.shape
    if input_grid.shape != gt_grid.shape:
        return {'prog': 'shape_mismatch', 'pa': 0.0, 'em': 0}

    best_pa = (input_grid == gt_grid).mean()
    best_prog = 'identity'
    best_em = float((input_grid == gt_grid).all())
    best_grid = input_grid.copy()

    objects, bg = extract_objects(input_grid)

    def try_candidate(cand, prog_name):
        nonlocal best_pa, best_prog, best_em, best_grid
        if cand.shape != gt_grid.shape:
            return
        pa = (cand == gt_grid).mean()
        em = float((cand == gt_grid).all())
        if pa > best_pa or (pa == best_pa and em > best_em):
            best_pa = pa
            best_prog = prog_name
            best_em = em
            best_grid = cand

    # Global transforms
    try_candidate(dsl_global_flip_h(input_grid), 'flip_h')
    try_candidate(dsl_global_flip_v(input_grid), 'flip_v')
    if h == w:
        try_candidate(dsl_global_rot90(input_grid), 'rot90')
        try_candidate(dsl_global_rot180(input_grid), 'rot180')

    # Color swaps/fills
    colors = list(set(input_grid.ravel().astype(int)))
    for c1 in colors[:6]:
        for c2 in range(10):
            if c1 != c2:
                try_candidate(dsl_global_color_swap(input_grid, c1, c2), f'swap_{c1}_{c2}')
                try_candidate(dsl_global_fill(input_grid, c1, c2), f'fill_{c1}to{c2}')
            if time.time() - t0 > max_time:
                break

    # Object-level operations
    for oi, obj in enumerate(objects[:8]):
        if time.time() - t0 > max_time:
            break
        # Move in all directions
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dy == 0 and dx == 0:
                    continue
                cand = dsl_move(input_grid, obj, dy, dx, bg)
                try_candidate(cand, f'move_o{oi}({dy},{dx})')
                if time.time() - t0 > max_time:
                    break
            if time.time() - t0 > max_time:
                break

        # Recolor
        for c in range(10):
            if c != obj['color']:
                try_candidate(dsl_recolor(input_grid, obj, c), f'recolor_o{oi}_to{c}')

        # Mirror
        try_candidate(dsl_mirror_h(input_grid, obj, bg), f'mirror_h_o{oi}')
        try_candidate(dsl_mirror_v(input_grid, obj, bg), f'mirror_v_o{oi}')

        # Copy
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dy == 0 and dx == 0:
                    continue
                try_candidate(dsl_copy_to(input_grid, obj, dy, dx), f'copy_o{oi}({dy},{dx})')
                if time.time() - t0 > max_time:
                    break
            if time.time() - t0 > max_time:
                break

        # Delete
        try_candidate(dsl_delete(input_grid, obj, bg), f'del_o{oi}')

        # Fill bbox
        for c in range(10):
            try_candidate(dsl_fill_bbox(input_grid, obj, c), f'fill_bbox_o{oi}_c{c}')

    # Two-step: best single + object op
    if best_pa < 1.0 and time.time() - t0 < max_time * 0.7:
        for oi, obj in enumerate(objects[:5]):
            if time.time() - t0 > max_time:
                break
            objs2, bg2 = extract_objects(best_grid)
            for oj, obj2 in enumerate(objs2[:5]):
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        if dy == 0 and dx == 0:
                            continue
                        cand = dsl_move(best_grid, obj2, dy, dx, bg2)
                        try_candidate(cand, f'{best_prog}+move_o{oj}({dy},{dx})')
                        if time.time() - t0 > max_time:
                            break
                    if time.time() - t0 > max_time:
                        break
                if time.time() - t0 > max_time:
                    break

    return {'prog': best_prog, 'pa': float(best_pa), 'em': int(best_em)}


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 236: Object-Centric DSL")
    print(f"  Upgrade DSL expressiveness with object-level operations")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    test = all_tasks[200:250]

    # Oracle search with object DSL
    total_pa, total_em = 0, 0
    prog_types = {}

    print(f"\n[Oracle Object-DSL Search on {len(test)} tasks]")
    for tidx, item in enumerate(test):
        inp = item['test_input'][:11].argmax(dim=0).numpy()
        gt = item['test_output'][:11].argmax(dim=0).numpy()
        oh, ow = item['out_h'], item['out_w']
        inp_crop = inp[:oh, :ow]
        gt_crop = gt[:oh, :ow]

        result = oracle_search(inp_crop, gt_crop, max_time=3.0)
        total_pa += result['pa']
        total_em += result['em']

        prog_base = result['prog'].split('_')[0] if '_' in result['prog'] else result['prog']
        prog_types[prog_base] = prog_types.get(prog_base, 0) + 1

        if (tidx + 1) % 10 == 0:
            print(f"    Task {tidx+1}/{len(test)}: "
                  f"prog={result['prog'][:30]}, PA={result['pa']*100:.0f}%, EM={result['em']}")

    n_test = len(test)
    avg_pa = total_pa / n_test
    avg_em = total_em / n_test

    print(f"\n{'='*70}")
    print(f"  OBJECT-CENTRIC DSL (Oracle):")
    print(f"  PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%")
    print(f"  (Compare P234 pixel DSL: PA=75.2%, EM=4.0%)")
    print(f"  Program distribution: {dict(sorted(prog_types.items(), key=lambda x:-x[1]))}")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    results = {'oracle_pa': avg_pa, 'oracle_em': avg_em,
               'prog_types': prog_types, 'n_test': n_test}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase236_object_dsl.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ['P234 Pixel DSL', 'P236 Object DSL']
        pa_vals = [75.2, avg_pa*100]; em_vals = [4.0, avg_em*100]
        colors = ['#95a5a6', '#2ecc71']; x = np.arange(2); w = 0.35
        axes[0].bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
        axes[0].set_ylabel('%'); axes[0].set_title('DSL Oracle Comparison', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')
        top_progs = sorted(prog_types.items(), key=lambda x:-x[1])[:8]
        axes[1].barh([p[0] for p in top_progs], [p[1] for p in top_progs], color='#3498db', alpha=0.7)
        axes[1].set_xlabel('Count'); axes[1].set_title('Top Programs Found', fontweight='bold')
        fig.suptitle('Phase 236: Object-Centric DSL', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.85, wspace=0.35)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase236_object_dsl.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
