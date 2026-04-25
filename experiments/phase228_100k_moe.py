"""
Phase 228: The 100K MoE Awakening

P224: 10K synthetic -> PA 60.2% (best ever)
P225: MoE(32,k2) 1.3M params overfits on 400 tasks

Hypothesis: Feed the hungry MoE beast 50K-100K synthetic tasks.
Massive data + massive model = breakthrough?

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
from phase225_moe import MoEGatedHybridNCA
from phase227_complexity import generate_complex_batch


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 228: The 100K MoE Awakening")
    print(f"  Massive data + Massive model")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    data_sizes = [50000, 100000]
    results = {}

    for n_syn in data_sizes:
        label = f"MoE_syn{n_syn//1000}K"
        print(f"\n[{label}: Pre-train MoE(32,k2) on {n_syn} tasks + FT]")

        torch.manual_seed(SEED)
        model = MoEGatedHybridNCA(11, 64, 32, 10, n_experts=32, top_k=2).to(DEVICE)
        print(f"  Params: {model.count_params():,}")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Pre-train on synthetic
        rng_syn = np.random.RandomState(SEED + n_syn)
        n_batches = n_syn // 32
        model.train()
        for bi in range(n_batches):
            batch = generate_complex_batch(rng_syn, batch_size=32)
            for inp_oh, out_oh, oh, ow in batch:
                inp_t = inp_oh.unsqueeze(0).to(DEVICE)
                out_gt = out_oh[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
                emb = model.encode_task([out_oh.to(DEVICE)])
                out_pred = model(inp_t, emb)
                logits = out_pred[0] if isinstance(out_pred, tuple) else out_pred
                loss = F.cross_entropy(logits[:, :, :oh, :ow], out_gt[:, :oh, :ow])
                opt.zero_grad(); loss.backward(); opt.step()
            if (bi + 1) % (n_batches // 5) == 0:
                print(f"    Pre-train: {bi+1}/{n_batches} batches")

        # Fine-tune on real ARC
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
        results[label] = {'pa': tpa, 'em': tem, 'params': model.count_params(), 'n_syn': n_syn}
        print(f"  {label}: PA={tpa*100:.1f}%, EM={tem*100:.1f}%")
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  100K MoE AWAKENING:")
    for k, r in results.items():
        print(f"  {k:20s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}% ({r['params']:,} params)")
    print(f"  (Compare: P225 MoE(32) no pretrain: PA=59.0%, EM=4.0%)")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase228_100k_moe.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # Include P225 baseline for comparison
        labels = ['MoE(no PT)'] + list(results.keys())
        pa_vals = [59.0] + [r['pa']*100 for r in results.values()]
        em_vals = [4.0] + [r['em']*100 for r in results.values()]
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 228: 100K MoE Awakening', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase228_100k_moe.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
