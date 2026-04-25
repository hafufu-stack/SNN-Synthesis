"""
Phase 210: The Verification Gap - Meta-Cognition and Self-Evaluation

Given N=100 stochastic trials, compare selection strategies:
  1. Oracle: Pick the trial with highest actual PA (upper bound)
  2. AE Critic: Pick the trial with lowest AE reconstruction error
  3. Margin: Pick the trial with highest confidence margin

Measures the "Verification Gap" = Oracle - Best_Selector.

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
from phase206_causal_volume import ParametricNCA
from phase205_critic import ArcAutoEncoder, train_autoencoder


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 210: The Verification Gap")
    print(f"  Oracle vs AE Critic vs Margin selection from N trials")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train model (K=1, C=64, T=1)
    print(f"\n[Training K=1, C=64, T=1 Model]")
    torch.manual_seed(SEED)
    model = ParametricNCA(11, 64, 1, 1, 32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        model.train()
        random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            logits = model(ti, emb)
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  Params: {model.count_params():,}")

    # Train AE Critic
    print(f"\n[Training AE Critic]")
    output_grids = []
    for item in train:
        output_grids.append(item['test_output'][:11])
        for do in item['demo_outputs']:
            output_grids.append(do)
    ae = ArcAutoEncoder(11, 32).to(DEVICE)
    train_autoencoder(ae, output_grids, n_epochs=200, lr=1e-3)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # Evaluate with N=100 trials
    N = 100
    noise_scale = 0.3
    print(f"\n[Generating {N} trials per task]")

    model.eval()
    oracle_pa, oracle_em = 0, 0
    ae_pa, ae_em = 0, 0
    margin_pa, margin_em = 0, 0
    greedy_pa, greedy_em = 0, 0

    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            candidates = []  # (pred, pa, em, ae_err, margin)
            for trial in range(N):
                logits = model(ti, emb)
                if trial > 0:
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                    logits = logits + noise_scale * gumbel

                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                pa = (pred == gt[:oh, :ow]).float().mean().item()
                em = (pred == gt[:oh, :ow]).all().item()

                # AE score
                soft = F.softmax(logits[0:1, :, :oh, :ow], dim=1)
                ae_err = ae.reconstruction_error(soft).item()

                # Margin score
                probs = F.softmax(logits[0, :, :oh, :ow], dim=0)
                top2 = probs.topk(2, dim=0).values
                margin = (top2[0] - top2[1]).mean().item()

                candidates.append((pred, pa, em, ae_err, margin))

            # Greedy (trial 0)
            greedy_pa += candidates[0][1]
            greedy_em += candidates[0][2]

            # Oracle: best actual PA
            best = max(candidates, key=lambda c: c[1])
            oracle_pa += best[1]
            oracle_em += best[2]

            # AE Critic: lowest reconstruction error
            ae_best = min(candidates, key=lambda c: c[3])
            ae_pa += ae_best[1]
            ae_em += ae_best[2]

            # Margin: highest margin
            margin_best = max(candidates, key=lambda c: c[4])
            margin_pa += margin_best[1]
            margin_em += margin_best[2]

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    results = {
        'greedy': {'pa': greedy_pa/n_test, 'em': greedy_em/n_test},
        'oracle': {'pa': oracle_pa/n_test, 'em': oracle_em/n_test},
        'ae_critic': {'pa': ae_pa/n_test, 'em': ae_em/n_test},
        'margin': {'pa': margin_pa/n_test, 'em': margin_em/n_test},
    }

    v_gap_ae = (oracle_pa - ae_pa) / n_test * 100
    v_gap_margin = (oracle_pa - margin_pa) / n_test * 100

    print(f"\n{'='*70}")
    print(f"  THE VERIFICATION GAP (N={N}):")
    print(f"  Greedy:     PA={results['greedy']['pa']*100:.1f}%, EM={results['greedy']['em']*100:.1f}%")
    print(f"  Oracle:     PA={results['oracle']['pa']*100:.1f}%, EM={results['oracle']['em']*100:.1f}%")
    print(f"  AE Critic:  PA={results['ae_critic']['pa']*100:.1f}%, EM={results['ae_critic']['em']*100:.1f}%")
    print(f"  Margin:     PA={results['margin']['pa']*100:.1f}%, EM={results['margin']['em']*100:.1f}%")
    print(f"  Verification Gap (AE):     {v_gap_ae:.1f}pp")
    print(f"  Verification Gap (Margin): {v_gap_margin:.1f}pp")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model, ae; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase210_verification.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'N': N, 'results': results,
            'verification_gap_ae_pp': v_gap_ae,
            'verification_gap_margin_pp': v_gap_margin,
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = ['Greedy', f'Oracle\n(N={N})', 'AE Critic', 'Margin']
        pa_vals = [results[k]['pa']*100 for k in ['greedy','oracle','ae_critic','margin']]
        em_vals = [results[k]['em']*100 for k in ['greedy','oracle','ae_critic','margin']]
        colors = ['#95a5a6', '#2ecc71', '#e67e22', '#3498db']
        x = np.arange(4); w = 0.35
        axes[0].bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
        axes[0].set_ylabel('%'); axes[0].set_title('Selection Strategy Comparison', fontweight='bold')
        axes[0].legend()

        # Verification gap visualization
        gaps = [0, 0, v_gap_ae, v_gap_margin]
        axes[1].bar(labels, gaps, color=colors, alpha=0.85)
        axes[1].set_ylabel('Gap from Oracle (pp)')
        axes[1].set_title('Verification Gap', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Phase 210: The Verification Gap (N={N})', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase210_verification.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
