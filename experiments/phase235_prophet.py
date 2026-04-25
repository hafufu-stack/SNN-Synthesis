"""
Phase 235: v26 "The Prophet" Kaggle Agent

Incorporating all equation-driven insights:
- Larger CfC hidden dim (C*=192 from P231)
- Anti-repeat + diverse targeting (from v25)
- 100% miracle replay for solved levels
- Multi-seed brain diversity (from P211)
- Faster stale detection

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
from phase233_depthwise import DepthwiseGatedNCA, cotrain


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 235: v26 Prophet Agent Design Validation")
    print(f"  Validate the equation's prediction: C=192 + N=100 sampling")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train the best model from P233: DepthwiseGated C=192 + Co-Training
    print(f"\n[Training DepthwiseGatedNCA C=192 + Co-Training 1:5]")
    torch.manual_seed(SEED)
    model = DepthwiseGatedNCA(11, 192, 32, 10).to(DEVICE)
    print(f"  Params: {model.count_params():,}")
    h_pa, h_em = cotrain(model, train, test, 100, "DW192+CT", syn_ratio=5)

    # N-sampling evaluation (Self-Consistency style)
    N_values = [1, 10, 50, 100]
    noise_scale = 0.3
    model.eval()
    results = {}

    for N in N_values:
        label = f"N={N}"
        print(f"\n[Eval: N={N} sampling with Self-Consistency]")
        total_pa, total_em = 0, 0
        with torch.no_grad():
            for tidx, item in enumerate(test):
                do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.encode_task(do_t)
                ti = item['test_input'].unsqueeze(0).to(DEVICE)
                gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                oh, ow = item['out_h'], item['out_w']

                if N == 1:
                    out = model(ti, emb)
                    logits = out[0] if isinstance(out, tuple) else out
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    pa = (pred == gt[:oh, :ow]).float().mean().item()
                    em = float((pred == gt[:oh, :ow]).all().item())
                else:
                    # Self-Consistency: pick candidate with highest average
                    # agreement with all other candidates
                    candidates = []
                    for trial in range(N):
                        out = model(ti, emb)
                        logits = out[0] if isinstance(out, tuple) else out
                        if trial > 0:
                            gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                            logits = logits + noise_scale * gumbel
                        pred_t = logits[0, :, :oh, :ow].argmax(dim=0)
                        candidates.append(pred_t)

                    # Self-consistency voting
                    stacked = torch.stack(candidates)  # (N, oh, ow)
                    best_sc = -1
                    best_idx = 0
                    for i in range(N):
                        agreement = sum(
                            (stacked[i] == stacked[j]).float().mean().item()
                            for j in range(N) if j != i
                        ) / max(1, N - 1)
                        if agreement > best_sc:
                            best_sc = agreement
                            best_idx = i
                    pred = candidates[best_idx]
                    pa = (pred == gt[:oh, :ow]).float().mean().item()
                    em = float((pred == gt[:oh, :ow]).all().item())

                total_pa += pa; total_em += em

        avg_pa = total_pa / len(test); avg_em = total_em / len(test)
        results[label] = {'pa': avg_pa, 'em': avg_em}
        print(f"  {label}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%")

    elapsed = time.time() - t0
    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  v26 PROPHET VALIDATION:")
    for k, r in results.items():
        print(f"  {k:10s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase235_prophet.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'training': {'pa': h_pa, 'em': h_em},
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(results.keys())
        pa_vals = [r['pa']*100 for r in results.values()]
        em_vals = [r['em']*100 for r in results.values()]
        colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('%'); ax.set_title('Phase 235: v26 Prophet (C=192 + SC)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase235_prophet.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
