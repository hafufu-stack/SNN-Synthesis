"""
Phase 239: DSL-Driven Data Engine

P228's 100K synthetic failed because data quality was too low.
Solution: Use P236's Object DSL as a DATA GENERATOR.
Generate tasks that embody real ARC object physics.

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
from phase224_synthetic import to_one_hot


# ==============================================================
# DSL-Driven Task Generator (Object-level)
# ==============================================================
def place_random_object(grid, rng, color=None):
    """Place a random rectangular or L-shaped object on grid."""
    h, w = grid.shape
    if color is None:
        color = rng.randint(1, 10)
    shape_type = rng.choice(['rect', 'line_h', 'line_v', 'l_shape', 'dot'])
    if shape_type == 'rect':
        oh = rng.randint(2, min(5, h-1))
        ow = rng.randint(2, min(5, w-1))
        oy = rng.randint(0, h - oh)
        ox = rng.randint(0, w - ow)
        grid[oy:oy+oh, ox:ox+ow] = color
    elif shape_type == 'line_h':
        length = rng.randint(2, min(6, w))
        oy = rng.randint(0, h-1)
        ox = rng.randint(0, w - length)
        grid[oy, ox:ox+length] = color
    elif shape_type == 'line_v':
        length = rng.randint(2, min(6, h))
        oy = rng.randint(0, h - length)
        ox = rng.randint(0, w-1)
        grid[oy:oy+length, ox] = color
    elif shape_type == 'l_shape':
        size = rng.randint(2, min(4, min(h, w)-1))
        oy = rng.randint(0, h - size)
        ox = rng.randint(0, w - size)
        grid[oy:oy+size, ox] = color
        grid[oy+size-1, ox:ox+size] = color
    else:  # dot
        oy = rng.randint(0, h-1)
        ox = rng.randint(0, w-1)
        grid[oy, ox] = color
    return grid


def gen_dsl_task(rng):
    """Generate a task using Object DSL operations."""
    h = rng.randint(5, 15)
    w = rng.randint(5, 15)
    grid = np.zeros((h, w), dtype=np.int64)

    # Place 1-4 objects
    n_obj = rng.randint(1, 4)
    colors_used = []
    for _ in range(n_obj):
        c = rng.randint(1, 10)
        colors_used.append(c)
        place_random_object(grid, rng, c)

    inp = grid.copy()

    # Apply 1-2 DSL operations
    n_ops = rng.choice([1, 1, 1, 2])
    out = grid.copy()

    for _ in range(n_ops):
        op = rng.choice([
            'move', 'recolor', 'flip_h', 'flip_v', 'copy',
            'color_swap', 'fill_color', 'delete', 'border',
            'rot180', 'mirror_obj'
        ])
        try:
            if op == 'move':
                # Move a colored region
                c = rng.choice(colors_used) if colors_used else 1
                mask = (out == c)
                if mask.any():
                    dy, dx = rng.randint(-3, 4), rng.randint(-3, 4)
                    new = np.zeros_like(out)
                    new[out != c] = out[out != c]
                    ys, xs = np.where(mask)
                    for y, x in zip(ys, xs):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w:
                            new[ny, nx] = c
                    out = new

            elif op == 'recolor':
                c_old = rng.choice(colors_used) if colors_used else 1
                c_new = rng.randint(1, 10)
                out[out == c_old] = c_new

            elif op == 'flip_h':
                out = out[:, ::-1].copy()

            elif op == 'flip_v':
                out = out[::-1, :].copy()

            elif op == 'rot180':
                out = np.rot90(out, 2).copy()

            elif op == 'copy':
                c = rng.choice(colors_used) if colors_used else 1
                mask = (out == c)
                if mask.any():
                    dy, dx = rng.randint(-4, 5), rng.randint(-4, 5)
                    ys, xs = np.where(mask)
                    for y, x in zip(ys, xs):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w:
                            out[ny, nx] = c

            elif op == 'color_swap':
                if len(colors_used) >= 2:
                    c1, c2 = rng.choice(colors_used, 2, replace=False)
                    tmp = out.copy()
                    out[tmp == c1] = c2
                    out[tmp == c2] = c1

            elif op == 'fill_color':
                bg = 0
                c_new = rng.randint(1, 10)
                out[out == bg] = c_new

            elif op == 'delete':
                c = rng.choice(colors_used) if colors_used else 1
                out[out == c] = 0

            elif op == 'border':
                bc = rng.randint(1, 10)
                out[0, :] = bc; out[-1, :] = bc
                out[:, 0] = bc; out[:, -1] = bc

            elif op == 'mirror_obj':
                c = rng.choice(colors_used) if colors_used else 1
                mask = (out == c)
                if mask.any():
                    ys, xs = np.where(mask)
                    cx = (xs.min() + xs.max()) / 2.0
                    for y, x in zip(ys, xs):
                        nx = int(round(2*cx - x))
                        if 0 <= nx < w:
                            out[y, nx] = c
        except Exception:
            pass

    return inp, out


def generate_dsl_batch(rng, batch_size=32, max_h=15, max_w=15):
    items = []
    for _ in range(batch_size):
        try:
            inp, out = gen_dsl_task(rng)
        except Exception:
            inp = np.zeros((5, 5), dtype=np.int64)
            out = inp[:, ::-1].copy()
        ih, iw = inp.shape
        oh, ow = out.shape
        inp_pad = np.zeros((max_h, max_w), dtype=np.int64)
        out_pad = np.zeros((max_h, max_w), dtype=np.int64)
        inp_pad[:ih, :iw] = inp
        out_pad[:oh, :ow] = out
        inp_oh = to_one_hot(inp_pad)
        out_oh = to_one_hot(out_pad)
        items.append((inp_oh, out_oh, oh, ow))
    return items


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 239: DSL-Driven Data Engine")
    print(f"  High-quality synthetic data from Object DSL")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Baseline
    print(f"\n[Baseline: real ARC only]")
    torch.manual_seed(SEED)
    m_base = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    opt = torch.optim.Adam(m_base.parameters(), lr=1e-3)
    for epoch in range(100):
        m_base.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = m_base.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = m_base(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    m_base.eval()
    bpa, bem = 0, 0
    with torch.no_grad():
        for item in test:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = m_base.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = m_base(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits[0, :, :oh, :ow].argmax(dim=0)
            bpa += (pred == gt[:oh, :ow]).float().mean().item()
            bem += float((pred == gt[:oh, :ow]).all().item())
    bpa /= len(test); bem /= len(test)
    print(f"  Baseline: PA={bpa*100:.1f}%, EM={bem*100:.1f}%")
    del m_base; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # DSL Co-Training 1:5 (best ratio from P229)
    configs = [('DSL_CT_10K', 10000), ('DSL_CT_50K', 50000)]
    results = {'baseline': {'pa': bpa, 'em': bem}}

    for label, n_syn in configs:
        print(f"\n[{label}: Co-Training 1:5 with {n_syn} DSL tasks]")
        torch.manual_seed(SEED)
        model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng_syn = np.random.RandomState(SEED + n_syn)

        syn_per_real = 5
        n_epochs = 100
        for epoch in range(n_epochs):
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
                # Interleave DSL synthetic
                sbatch = generate_dsl_batch(rng_syn, batch_size=syn_per_real)
                for inp_oh, out_oh, soh, sow in sbatch:
                    inp_t = inp_oh.unsqueeze(0).to(DEVICE)
                    out_gt = out_oh[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
                    semb = model.encode_task([out_oh.to(DEVICE)])
                    sout = model(inp_t, semb)
                    slogits = sout[0] if isinstance(sout, tuple) else sout
                    sloss = F.cross_entropy(slogits[:, :, :soh, :sow], out_gt[:, :soh, :sow])
                    opt.zero_grad(); sloss.backward(); opt.step()

        model.eval()
        tpa, tem = 0, 0
        with torch.no_grad():
            for item in test:
                do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.encode_task(do_t)
                ti = item['test_input'].unsqueeze(0).to(DEVICE)
                gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                oh, ow = item['out_h'], item['out_w']
                out = model(ti, emb)
                logits = out[0] if isinstance(out, tuple) else out
                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                tpa += (pred == gt[:oh, :ow]).float().mean().item()
                tem += float((pred == gt[:oh, :ow]).all().item())
        tpa /= len(test); tem /= len(test)
        results[label] = {'pa': tpa, 'em': tem}
        print(f"  {label}: PA={tpa*100:.1f}%, EM={tem*100:.1f}%")
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DSL-DRIVEN DATA ENGINE:")
    for k, r in results.items():
        print(f"  {k:15s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"  (Compare P229 simple CT 1:5: PA=61.5%)")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase239_dsl_engine.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(results.keys())
        pa_vals = [r['pa']*100 for r in results.values()]
        em_vals = [r['em']*100 for r in results.values()]
        colors = ['#95a5a6', '#3498db', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors[:len(labels)], alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors[:len(labels)], alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 239: DSL-Driven Data Engine', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase239_dsl_engine.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
