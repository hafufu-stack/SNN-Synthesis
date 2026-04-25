"""
Phase 238: v27 "The Neuro-Symbolic Goose" Design Validation

Simulate the Neuro-Symbolic pipeline:
1. NCA predicts target grid (intuition)
2. Object DSL searches for matching program (logic)
3. If found, use DSL output (pixel-perfect); else fallback to NCA

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
from phase237_guided import nca_guided_search


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 238: v27 Neuro-Symbolic Goose Validation")
    print(f"  NCA intuition + Object DSL logic = best of both")
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

    # Full pipeline evaluation
    nca_pa, nca_em = 0, 0
    hybrid_pa, hybrid_em = 0, 0
    dsl_used = 0
    nca_fallback = 0

    # Confidence threshold: if DSL matches NCA output > threshold, trust DSL
    CONFIDENCE = 0.85

    print(f"\n[Neuro-Symbolic Pipeline (threshold={CONFIDENCE})]")
    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            # Step 1: NCA prediction
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            nca_pred = logits[0, :, :oh, :ow].argmax(dim=0).cpu().numpy()
            gt_np = gt[:oh, :ow].cpu().numpy()
            inp_np = ti[0, :11, :oh, :ow].argmax(dim=0).cpu().numpy()

            pa_nca = (nca_pred == gt_np).mean()
            em_nca = float((nca_pred == gt_np).all())
            nca_pa += pa_nca; nca_em += em_nca

            # Step 2: DSL search guided by NCA
            result = nca_guided_search(inp_np, nca_pred, gt_np, max_time=2.0)

            # Step 3: Decision - use DSL or NCA fallback?
            if result['nca_pa'] >= CONFIDENCE:
                # DSL found a program that closely matches NCA output
                # Use the DSL output (it's pixel-perfect for that program)
                hybrid_pa += result['gt_pa']
                hybrid_em += result['gt_em']
                dsl_used += 1
            else:
                # NCA is our best guess
                hybrid_pa += pa_nca
                hybrid_em += em_nca
                nca_fallback += 1

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)}: "
                      f"NCA={pa_nca*100:.0f}% DSL_match={result['nca_pa']*100:.0f}% "
                      f"{'DSL' if result['nca_pa'] >= CONFIDENCE else 'NCA'}")

    n = len(test)
    nca_pa /= n; nca_em /= n
    hybrid_pa /= n; hybrid_em /= n

    print(f"\n{'='*70}")
    print(f"  NEURO-SYMBOLIC PIPELINE (v27 design):")
    print(f"  NCA alone    : PA={nca_pa*100:.1f}%, EM={nca_em*100:.1f}%")
    print(f"  Hybrid (NS)  : PA={hybrid_pa*100:.1f}%, EM={hybrid_em*100:.1f}%")
    print(f"  DSL used: {dsl_used}/{n}, NCA fallback: {nca_fallback}/{n}")
    delta = (hybrid_pa - nca_pa) * 100
    print(f"  Delta PA = {delta:+.1f}pp")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    results = {
        'nca_only': {'pa': nca_pa, 'em': nca_em},
        'hybrid': {'pa': hybrid_pa, 'em': hybrid_em},
        'dsl_used': dsl_used, 'nca_fallback': nca_fallback,
        'delta_pa': delta, 'confidence_threshold': CONFIDENCE,
    }
    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase238_neurosymbolic_v27.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ['NCA only', 'Hybrid (v27)']
        pa_vals = [nca_pa*100, hybrid_pa*100]
        em_vals = [nca_em*100, hybrid_em*100]
        colors = ['#95a5a6', '#2ecc71']; x = np.arange(2); w = 0.35
        axes[0].bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
        axes[0].set_ylabel('%'); axes[0].set_title('PA/EM Comparison', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')
        axes[1].pie([dsl_used, nca_fallback], labels=['DSL used', 'NCA fallback'],
                   colors=['#2ecc71', '#95a5a6'], autopct='%1.0f%%', startangle=90)
        axes[1].set_title('Decision Distribution', fontweight='bold')
        fig.suptitle('Phase 238: v27 Neuro-Symbolic Goose', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.85, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase238_v27.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
