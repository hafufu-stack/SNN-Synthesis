"""
Phase 227: Synthetic Complexity - Composite Multi-Rule Tasks

P224 showed 10K synthetic tasks boost PA to 60.2%, but EM dropped
because tasks were too simple (single rule). This phase generates
COMPOSITE tasks (2-3 rules chained) that match ARC's complexity.

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
from phase224_synthetic import to_one_hot, GENERATORS


# --- Complex composite generators ---
def gen_composite_2(rng, h=None, w=None):
    """Chain 2 random transformations."""
    g1 = rng.choice(GENERATORS)
    g2 = rng.choice(GENERATORS)
    try:
        inp, mid = g1(rng)
        # Apply g2 to mid (same size)
        h2, w2 = mid.shape
        _, out = g2(rng, h=h2, w=w2)
        # But we want the INPUT of g2 to be mid, so reconstruct
        # Actually just apply the transform logic to mid directly
        out = _apply_transform(g2, rng, mid)
        return inp, out
    except Exception:
        return gen_simple_fallback(rng)

def _apply_transform(gen_func, rng, grid):
    """Apply a generator's transform to an existing grid."""
    h, w = grid.shape
    n_c = max(2, int(grid.max()) + 1)

    if gen_func.__name__ == 'gen_flip_h':
        return grid[:, ::-1].copy()
    elif gen_func.__name__ == 'gen_flip_v':
        return grid[::-1, :].copy()
    elif gen_func.__name__ == 'gen_rotate90' and h == w:
        return np.rot90(grid).copy()
    elif gen_func.__name__ == 'gen_color_swap':
        colors = list(set(grid.ravel()))
        if len(colors) >= 2:
            c1, c2 = rng.choice(colors, 2, replace=False)
            out = grid.copy()
            out[grid == c1] = c2
            out[grid == c2] = c1
            return out
    elif gen_func.__name__ == 'gen_border':
        bc = rng.randint(0, 10)
        out = grid.copy()
        out[0, :] = bc; out[-1, :] = bc
        out[:, 0] = bc; out[:, -1] = bc
        return out
    elif gen_func.__name__ == 'gen_denoise':
        out = grid.copy()
        n_noise = max(1, int(h * w * 0.05))
        for _ in range(n_noise):
            ny, nx = rng.randint(0, h), rng.randint(0, w)
            out[ny, nx] = rng.randint(0, 10)
        return out
    elif gen_func.__name__ == 'gen_fill_color':
        bg = int(np.bincount(grid.ravel().astype(int), minlength=11).argmax())
        fill = rng.randint(0, 10)
        out = grid.copy()
        out[grid == bg] = fill
        return out
    # Default: just return grid
    return grid.copy()


def gen_multi_object(rng):
    """Generate grid with multiple distinct objects + transformation."""
    h, w = rng.randint(6, 15), rng.randint(6, 15)
    grid = np.zeros((h, w), dtype=np.int64)
    n_objects = rng.randint(2, 4)
    for _ in range(n_objects):
        oh = rng.randint(2, min(4, h-1))
        ow = rng.randint(2, min(4, w-1))
        oy = rng.randint(0, h - oh)
        ox = rng.randint(0, w - ow)
        color = rng.randint(1, 10)
        grid[oy:oy+oh, ox:ox+ow] = color
    # Transform: color swap on objects
    out = grid.copy()
    colors = list(set(grid.ravel()) - {0})
    if len(colors) >= 2:
        c1, c2 = rng.choice(colors, 2, replace=False)
        out[grid == c1] = c2
        out[grid == c2] = c1
    return grid, out


def gen_pattern_fill(rng):
    """Create a pattern and fill a region with it."""
    h, w = rng.randint(6, 12), rng.randint(6, 12)
    ph, pw = rng.randint(2, 4), rng.randint(2, 4)
    pattern = rng.randint(0, 5, (ph, pw))
    grid = np.zeros((h, w), dtype=np.int64)
    # Place pattern once
    grid[:ph, :pw] = pattern
    # Output: tile the pattern
    out = np.zeros((h, w), dtype=np.int64)
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            eh = min(ph, h - y)
            ew = min(pw, w - x)
            out[y:y+eh, x:x+ew] = pattern[:eh, :ew]
    return grid, out


def gen_simple_fallback(rng):
    h, w = rng.randint(3, 12), rng.randint(3, 12)
    grid = rng.randint(0, 5, (h, w))
    return grid, grid[:, ::-1].copy()


COMPLEX_GENERATORS = GENERATORS + [gen_composite_2, gen_multi_object, gen_pattern_fill]


def generate_complex_batch(rng, batch_size=32, max_h=15, max_w=15):
    items = []
    for _ in range(batch_size):
        gen = rng.choice(COMPLEX_GENERATORS)
        try:
            inp, out = gen(rng)
        except Exception:
            inp, out = gen_simple_fallback(rng)
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
    print("Phase 227: Synthetic Complexity (Composite Tasks)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Baseline
    print(f"\n[Baseline: scratch on real ARC]")
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

    # Complex synthetic 10K pre-train + FT
    configs = {'simple_10K': False, 'complex_10K': True}
    results = {'baseline': {'pa': bpa, 'em': bem}}

    for label, use_complex in configs.items():
        print(f"\n[{label} pre-train + FT]")
        torch.manual_seed(SEED)
        model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng_syn = np.random.RandomState(SEED + hash(label) % 10000)
        model.train()
        n_batches = 10000 // 32
        for bi in range(n_batches):
            if use_complex:
                batch = generate_complex_batch(rng_syn, batch_size=32)
            else:
                from phase224_synthetic import generate_synthetic_batch
                batch = generate_synthetic_batch(rng_syn, batch_size=32)
            for inp_oh, out_oh, oh, ow in batch:
                inp_t = inp_oh.unsqueeze(0).to(DEVICE)
                out_gt = out_oh[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
                emb = model.encode_task([out_oh.to(DEVICE)])
                out_pred = model(inp_t, emb)
                logits = out_pred[0] if isinstance(out_pred, tuple) else out_pred
                loss = F.cross_entropy(logits[:, :, :oh, :ow], out_gt[:, :oh, :ow])
                opt.zero_grad(); loss.backward(); opt.step()
            if (bi + 1) % (n_batches // 3) == 0:
                print(f"    Pre-train batch {bi+1}/{n_batches}")
        # Fine-tune
        opt2 = torch.optim.Adam(model.parameters(), lr=5e-4)
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
                opt2.zero_grad(); loss.backward(); opt2.step()
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
    print(f"  SYNTHETIC COMPLEXITY:")
    for k, r in results.items():
        print(f"  {k:15s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase227_complexity.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(results.keys()); pa_vals = [r['pa']*100 for r in results.values()]
        em_vals = [r['em']*100 for r in results.values()]
        colors = ['#95a5a6', '#3498db', '#e74c3c']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors[:len(labels)], alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors[:len(labels)], alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 227: Synthetic Complexity', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase227_complexity.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
