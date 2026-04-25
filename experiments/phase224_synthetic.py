"""
Phase 224: Synthetic Massive Pre-training

Generate thousands of synthetic ARC-like tasks programmatically,
pre-train GatedHybridNCA, then fine-tune on real ARC data.

"Data quantity as the fundamental driver of intelligence."

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
# Synthetic Task Generator
# ==============================================================
def to_one_hot(grid, n_colors=11):
    """Convert (H, W) int grid to (n_colors, H, W) one-hot."""
    t = torch.from_numpy(grid.astype(np.int64))
    oh = F.one_hot(t, n_colors).permute(2, 0, 1).float()
    return oh


def gen_color_swap(rng, h=None, w=None):
    """Swap two colors in a random grid."""
    h = h or rng.randint(3, 15)
    w = w or rng.randint(3, 15)
    n_colors_used = rng.randint(2, 6)
    grid = rng.randint(0, n_colors_used, (h, w))
    c1, c2 = rng.choice(n_colors_used, 2, replace=False)
    out = grid.copy()
    out[grid == c1] = c2
    out[grid == c2] = c1
    return grid, out


def gen_fill_color(rng, h=None, w=None):
    """Fill background with a specific color."""
    h = h or rng.randint(3, 12)
    w = w or rng.randint(3, 12)
    n_c = rng.randint(2, 5)
    grid = rng.randint(0, n_c, (h, w))
    bg = int(np.bincount(grid.ravel()).argmax())
    fill = rng.randint(0, 10)
    out = grid.copy()
    out[grid == bg] = fill
    return grid, out


def gen_flip_h(rng, h=None, w=None):
    """Horizontal flip."""
    h = h or rng.randint(3, 12)
    w = w or rng.randint(3, 12)
    n_c = rng.randint(2, 6)
    grid = rng.randint(0, n_c, (h, w))
    return grid, grid[:, ::-1].copy()


def gen_flip_v(rng, h=None, w=None):
    """Vertical flip."""
    h = h or rng.randint(3, 12)
    w = w or rng.randint(3, 12)
    n_c = rng.randint(2, 6)
    grid = rng.randint(0, n_c, (h, w))
    return grid, grid[::-1, :].copy()


def gen_rotate90(rng, h=None, w=None):
    """Rotate 90 degrees (only for square grids)."""
    s = rng.randint(3, 12)
    n_c = rng.randint(2, 6)
    grid = rng.randint(0, n_c, (s, s))
    return grid, np.rot90(grid).copy()


def gen_denoise(rng, h=None, w=None):
    """Add noise to grid, task is to remove it."""
    h = h or rng.randint(4, 12)
    w = w or rng.randint(4, 12)
    n_c = rng.randint(2, 4)
    clean = rng.randint(0, n_c, (h, w))
    noisy = clean.copy()
    n_noise = max(1, int(h * w * 0.1))
    for _ in range(n_noise):
        ny, nx = rng.randint(0, h), rng.randint(0, w)
        noisy[ny, nx] = rng.randint(0, 10)
    return noisy, clean


def gen_translate(rng, h=None, w=None):
    """Translate pattern by (dy, dx)."""
    h = h or rng.randint(5, 12)
    w = w or rng.randint(5, 12)
    n_c = rng.randint(2, 5)
    grid = rng.randint(0, n_c, (h, w))
    dy = rng.randint(-2, 3)
    dx = rng.randint(-2, 3)
    out = np.zeros_like(grid)
    for y in range(h):
        for x in range(w):
            sy, sx = y - dy, x - dx
            if 0 <= sy < h and 0 <= sx < w:
                out[y, x] = grid[sy, sx]
    return grid, out


def gen_border(rng, h=None, w=None):
    """Add a border of a specific color."""
    h = h or rng.randint(4, 10)
    w = w or rng.randint(4, 10)
    n_c = rng.randint(2, 5)
    grid = rng.randint(0, n_c, (h, w))
    bc = rng.randint(0, 10)
    out = grid.copy()
    out[0, :] = bc; out[-1, :] = bc
    out[:, 0] = bc; out[:, -1] = bc
    return grid, out


GENERATORS = [gen_color_swap, gen_fill_color, gen_flip_h, gen_flip_v,
              gen_rotate90, gen_denoise, gen_translate, gen_border]


def generate_synthetic_batch(rng, batch_size=32, max_h=15, max_w=15):
    """Generate a batch of synthetic tasks."""
    items = []
    for _ in range(batch_size):
        gen = rng.choice(GENERATORS)
        try:
            inp, out = gen(rng)
        except Exception:
            inp, out = gen_flip_h(rng)

        # Pad to max_h x max_w
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
    print("Phase 224: Synthetic Massive Pre-training")
    print(f"  Pre-train on synthetic data, fine-tune on real ARC")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    rng_syn = np.random.RandomState(SEED)

    # --- Baseline: train from scratch on real data ---
    print(f"\n[Baseline: Train from scratch on real ARC]")
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
    base_pa, base_em = 0, 0
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
            base_pa += (pred == gt[:oh, :ow]).float().mean().item()
            base_em += float((pred == gt[:oh, :ow]).all().item())
    base_pa /= len(test); base_em /= len(test)
    print(f"  Baseline: PA={base_pa*100:.1f}%, EM={base_em*100:.1f}%")
    del m_base; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # --- Pre-train on synthetic, then fine-tune ---
    syn_configs = [1000, 5000, 10000]
    results = {'baseline': {'pa': base_pa, 'em': base_em}}

    for n_syn in syn_configs:
        print(f"\n[Pre-train on {n_syn} synthetic tasks, then fine-tune]")
        torch.manual_seed(SEED)
        model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Phase 1: Pre-train on synthetic data
        rng_syn = np.random.RandomState(SEED + n_syn)
        n_batches = n_syn // 32
        model.train()
        for batch_idx in range(n_batches):
            batch = generate_synthetic_batch(rng_syn, batch_size=32)
            for inp_oh, out_oh, oh, ow in batch:
                inp_t = inp_oh.unsqueeze(0).to(DEVICE)
                out_gt = out_oh[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
                # Use output as "demo" for task embedding (self-supervised)
                emb = model.encode_task([out_oh.to(DEVICE)])
                out_pred = model(inp_t, emb)
                logits = out_pred[0] if isinstance(out_pred, tuple) else out_pred
                loss = F.cross_entropy(logits[:, :, :oh, :ow], out_gt[:, :oh, :ow])
                opt.zero_grad(); loss.backward(); opt.step()
            if (batch_idx + 1) % (n_batches // 3) == 0:
                print(f"    Pre-train batch {batch_idx+1}/{n_batches}")

        # Phase 2: Fine-tune on real ARC
        opt2 = torch.optim.Adam(model.parameters(), lr=5e-4)  # Lower LR
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

        # Evaluate
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
        results[f'syn_{n_syn}'] = {'pa': tpa, 'em': tem}
        print(f"  Syn({n_syn})+FT: PA={tpa*100:.1f}%, EM={tem*100:.1f}%")
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  SYNTHETIC MASSIVE PRE-TRAINING:")
    for k, r in results.items():
        print(f"  {k:15s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase224_synthetic.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(results.keys())
        pa_vals = [results[k]['pa']*100 for k in labels]
        em_vals = [results[k]['em']*100 for k in labels]
        colors = ['#95a5a6', '#3498db', '#2980b9', '#1a5276']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors[:len(labels)], alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors[:len(labels)], alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 224: Synthetic Pre-training', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase224_synthetic.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
