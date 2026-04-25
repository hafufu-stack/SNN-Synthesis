"""
Phase 211: The Diversity Equation - Model Diversity vs Trial Count

Fixed total compute budget (M*N = 100).
Compare: 1 model x 100 trials vs 5 models x 20 trials vs 10 models x 10 trials.

Each model trained with different random seed for genetic diversity.
Selection via Margin (best from P210).

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


def train_model_with_seed(train_tasks, seed, n_epochs=80):
    """Train K=1, C=64, T=1 model with specific seed."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = ParametricNCA(11, 64, 1, 1, 32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        model.train()
        random.shuffle(train_tasks)
        for item in train_tasks[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            logits = model(ti, emb)
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate_ensemble(models, test_tasks, N_per_model, noise_scale=0.3):
    """Run M models x N trials each, select best by margin."""
    M = len(models)
    total_trials = M * N_per_model

    oracle_pa, oracle_em = 0, 0
    margin_pa, margin_em = 0, 0

    for model in models:
        model.eval()

    with torch.no_grad():
        for item in test_tasks:
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            ti = item['test_input'].unsqueeze(0).to(DEVICE)

            candidates = []
            for model in models:
                do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.encode_task(do_t)

                for trial in range(N_per_model):
                    logits = model(ti, emb)
                    if trial > 0:
                        gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                        logits = logits + noise_scale * gumbel

                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    pa = (pred == gt[:oh, :ow]).float().mean().item()
                    em = (pred == gt[:oh, :ow]).all().item()

                    probs = F.softmax(logits[0, :, :oh, :ow], dim=0)
                    top2 = probs.topk(2, dim=0).values
                    margin = (top2[0] - top2[1]).mean().item()

                    candidates.append((pa, em, margin))

            # Oracle
            best = max(candidates, key=lambda c: c[0])
            oracle_pa += best[0]
            oracle_em += best[1]

            # Margin selection
            margin_best = max(candidates, key=lambda c: c[2])
            margin_pa += margin_best[0]
            margin_em += margin_best[1]

    n_test = len(test_tasks)
    return {
        'oracle_pa': oracle_pa / n_test,
        'oracle_em': oracle_em / n_test,
        'margin_pa': margin_pa / n_test,
        'margin_em': margin_em / n_test,
        'total_trials': total_trials,
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 211: The Diversity Equation")
    print(f"  M models x N trials = 100 total, varying M")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    TOTAL_BUDGET = 100
    configs = [
        (1, 100),   # 1 model, 100 trials
        (2, 50),    # 2 models, 50 trials
        (5, 20),    # 5 models, 20 trials
        (10, 10),   # 10 models, 10 trials
    ]

    results = []
    all_models = {}

    # Pre-train models with different seeds
    max_models = max(c[0] for c in configs)
    print(f"\n[Pre-training {max_models} models with different seeds]")
    for i in range(max_models):
        seed = SEED + i * 100
        print(f"  Training model {i+1}/{max_models} (seed={seed})...")
        model = train_model_with_seed(train, seed, n_epochs=80)
        all_models[i] = model
        print(f"    Params: {model.count_params():,}")

    # Evaluate each configuration
    for M, N in configs:
        print(f"\n[Config: M={M} models x N={N} trials = {M*N} total]")
        models = [all_models[i] for i in range(M)]
        result = evaluate_ensemble(models, test, N)
        result['M'] = M
        result['N'] = N
        results.append(result)
        print(f"  Oracle: PA={result['oracle_pa']*100:.1f}%, EM={result['oracle_em']*100:.1f}%")
        print(f"  Margin: PA={result['margin_pa']*100:.1f}%, EM={result['margin_em']*100:.1f}%")

    # Clean up
    for m in all_models.values():
        del m
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 211 Complete ({elapsed:.0f}s)")
    print(f"  THE DIVERSITY EQUATION (Total Budget = {TOTAL_BUDGET}):")
    for r in results:
        print(f"    M={r['M']:2d} x N={r['N']:3d}: "
              f"Oracle PA={r['oracle_pa']*100:.1f}%, Margin PA={r['margin_pa']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase211_diversity.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'total_budget': TOTAL_BUDGET, 'results': results,
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        Ms = [r['M'] for r in results]
        oracle_pas = [r['oracle_pa']*100 for r in results]
        margin_pas = [r['margin_pa']*100 for r in results]
        oracle_ems = [r['oracle_em']*100 for r in results]

        axes[0].plot(Ms, oracle_pas, 'o-', color='#2ecc71', lw=2, ms=10, label='Oracle PA')
        axes[0].plot(Ms, margin_pas, 's-', color='#3498db', lw=2, ms=10, label='Margin PA')
        axes[0].set_xlabel('M (model diversity)'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title(f'PA vs Diversity\n(Budget={TOTAL_BUDGET})', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(Ms, oracle_ems, 'D-', color='#e74c3c', lw=2, ms=10)
        axes[1].set_xlabel('M (model diversity)'); axes[1].set_ylabel('Oracle EM (%)')
        axes[1].set_title('EM Discovery vs Diversity', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        labels = [f'M={r["M"]}\nN={r["N"]}' for r in results]
        x = np.arange(len(results)); w = 0.35
        axes[2].bar(x-w/2, oracle_pas, w, color='#2ecc71', alpha=0.85, label='Oracle')
        axes[2].bar(x+w/2, margin_pas, w, color='#3498db', alpha=0.85, label='Margin')
        axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
        axes[2].set_ylabel('PA (%)'); axes[2].set_title('Oracle vs Margin', fontweight='bold')
        axes[2].legend()

        fig.suptitle('Phase 211: The Diversity Equation', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.82, bottom=0.12, left=0.06, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase211_diversity.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
