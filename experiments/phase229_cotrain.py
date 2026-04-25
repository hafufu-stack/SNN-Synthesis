"""
Phase 229: Synthetic-Real Co-Training

Instead of Pre-train -> Fine-tune (2 stages, risk of forgetting),
mix synthetic and real ARC data in the SAME training loop.

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
from phase227_complexity import generate_complex_batch


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 229: Synthetic-Real Co-Training")
    print(f"  Mix real ARC + synthetic in same batch (no forgetting)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Ratios: how many synthetic per 1 real
    ratios = [('1:0 (baseline)', 0), ('1:3', 3), ('1:5', 5), ('1:10', 10)]
    results = {}

    for label, syn_per_real in ratios:
        print(f"\n[Co-Training ratio: {label}]")
        torch.manual_seed(SEED)
        model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rng_syn = np.random.RandomState(SEED + syn_per_real * 100)

        for epoch in range(100):
            model.train(); random.shuffle(train)
            # Real ARC items
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

                # Interleave synthetic items
                if syn_per_real > 0:
                    syn_batch = generate_complex_batch(rng_syn, batch_size=syn_per_real)
                    for inp_oh, out_oh, soh, sow in syn_batch:
                        inp_t = inp_oh.unsqueeze(0).to(DEVICE)
                        out_gt = out_oh[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
                        semb = model.encode_task([out_oh.to(DEVICE)])
                        sout = model(inp_t, semb)
                        slogits = sout[0] if isinstance(sout, tuple) else sout
                        sloss = F.cross_entropy(slogits[:, :, :soh, :sow], out_gt[:, :soh, :sow])
                        opt.zero_grad(); sloss.backward(); opt.step()

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
        results[label] = {'pa': tpa, 'em': tem}
        print(f"  {label}: PA={tpa*100:.1f}%, EM={tem*100:.1f}%")
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  CO-TRAINING RESULTS:")
    for k, r in results.items():
        print(f"  {k:20s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase229_cotrain.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(results.keys()); pa_vals = [r['pa']*100 for r in results.values()]
        em_vals = [r['em']*100 for r in results.values()]
        colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 229: Synthetic-Real Co-Training', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase229_cotrain.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
