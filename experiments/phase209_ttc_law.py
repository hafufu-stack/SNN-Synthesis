"""
Phase 209: Test-Time Compute Law - Scaling of Trials and Intelligence

Measure Pass@N: probability that at least one of N stochastic trials
achieves exact match (or best PA), as N grows exponentially.

Uses P206's optimal config: K=1, C=64, T=1 (the "pure intuition" NCA).
Stochastic diversity via Gumbel noise on logits at inference.

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


def sample_with_noise(model, x, emb, oh, ow, noise_scale=0.3):
    """Run model with Gumbel noise for stochastic diversity."""
    logits = model(x, emb)
    # Add Gumbel noise for diverse sampling
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    noisy_logits = logits + noise_scale * gumbel
    pred = noisy_logits[0, :, :oh, :ow].argmax(dim=0)
    margin = F.softmax(noisy_logits[0, :, :oh, :ow], dim=0)
    top2 = margin.topk(2, dim=0).values
    avg_margin = (top2[0] - top2[1]).mean().item()
    return pred, avg_margin, noisy_logits[0, :, :oh, :ow]


def train_model(train_tasks, C=64, K=1, T=1, n_epochs=100):
    """Train the optimal intuition model."""
    torch.manual_seed(SEED)
    model = ParametricNCA(11, C, K, T, 32).to(DEVICE)
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


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 209: Test-Time Compute Law")
    print(f"  Pass@N scaling with stochastic trials")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train optimal intuition model (K=1, C=64, T=1)
    print(f"\n[Training K=1, C=64, T=1 Model]")
    model = train_model(train, C=64, K=1, T=1, n_epochs=100)
    params = model.count_params()
    print(f"  Params: {params:,}")

    # Test Pass@N for various N
    N_vals = [1, 5, 10, 50, 100, 500]
    results = []

    model.eval()
    for N in N_vals:
        print(f"\n  [N={N} trials]")
        pass_at_n_em = 0  # tasks where EM found in N trials
        pass_at_n_pa = 0  # best PA across N trials (averaged over tasks)
        greedy_pa = 0
        total_em_found = 0

        with torch.no_grad():
            for item in test:
                do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.encode_task(do_t)
                ti = item['test_input'].unsqueeze(0).to(DEVICE)
                gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                oh, ow = item['out_h'], item['out_w']

                best_pa = 0
                em_found = False

                for trial in range(N):
                    if trial == 0:
                        # Greedy (no noise)
                        logits = model(ti, emb)
                        pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    else:
                        pred, _, _ = sample_with_noise(model, ti, emb, oh, ow, noise_scale=0.3)

                    pa = (pred == gt[:oh, :ow]).float().mean().item()
                    em = (pred == gt[:oh, :ow]).all().item()

                    if pa > best_pa:
                        best_pa = pa
                    if em:
                        em_found = True

                    if trial == 0:
                        greedy_pa += pa

                pass_at_n_pa += best_pa
                if em_found:
                    pass_at_n_em += 1

        n_test = len(test)
        avg_best_pa = pass_at_n_pa / n_test
        avg_greedy_pa = greedy_pa / n_test
        pass_rate = pass_at_n_em / n_test
        print(f"    Greedy PA: {avg_greedy_pa*100:.1f}%")
        print(f"    Best@{N} PA: {avg_best_pa*100:.1f}%")
        print(f"    Pass@{N} EM: {pass_rate*100:.1f}%")

        results.append({
            'N': N, 'greedy_pa': avg_greedy_pa, 'best_pa': avg_best_pa,
            'pass_em': pass_rate, 'em_count': pass_at_n_em
        })

    # Fit power law: log(best_pa) = gamma * log(N) + const
    valid = [r for r in results if r['N'] > 0]
    log_N = np.array([np.log(r['N']) for r in valid])
    log_PA = np.array([np.log(r['best_pa']) for r in valid])
    A = np.column_stack([log_N, np.ones(len(valid))])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_PA, rcond=None)
    gamma, const = coeffs
    r2 = 1 - np.sum((log_PA - A @ coeffs)**2) / np.sum((log_PA - log_PA.mean())**2)

    print(f"\n{'='*70}")
    print(f"  TEST-TIME COMPUTE LAW:")
    print(f"  Best_PA ~ N^{gamma:.4f}")
    print(f"  gamma = {gamma:.4f} (trial scaling exponent)")
    print(f"  R^2   = {r2:.4f}")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase209_ttc_law.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'equation': {'gamma': float(gamma), 'const': float(const), 'r2': float(r2)},
            'model': {'K': 1, 'C': 64, 'T': 1, 'params': params},
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        Ns = [r['N'] for r in results]
        best_pas = [r['best_pa']*100 for r in results]
        pass_ems = [r['pass_em']*100 for r in results]

        axes[0].semilogx(Ns, best_pas, 'o-', color='#3498db', lw=2, ms=8, label='Best@N PA')
        axes[0].axhline(y=results[0]['greedy_pa']*100, color='#95a5a6', ls='--', label='Greedy')
        axes[0].set_xlabel('N (trials)'); axes[0].set_ylabel('Best PA (%)')
        axes[0].set_title(f'PA Scaling: γ={gamma:.3f}', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].semilogx(Ns, pass_ems, 's-', color='#e74c3c', lw=2, ms=8)
        axes[1].set_xlabel('N (trials)'); axes[1].set_ylabel('Pass@N EM (%)')
        axes[1].set_title('EM Discovery Rate', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Log-log fit
        axes[2].plot(log_N, log_PA, 'o', color='#3498db', ms=8)
        fit_x = np.linspace(min(log_N), max(log_N), 50)
        axes[2].plot(fit_x, gamma*fit_x + const, '--', color='#e74c3c', lw=2,
                    label=f'γ={gamma:.3f}, R²={r2:.3f}')
        axes[2].set_xlabel('log(N)'); axes[2].set_ylabel('log(Best PA)')
        axes[2].set_title('Log-Log Power Law Fit', fontweight='bold')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 209: Test-Time Compute Law', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.06, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase209_ttc_law.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'gamma': float(gamma), 'r2': float(r2)}


if __name__ == '__main__':
    main()
