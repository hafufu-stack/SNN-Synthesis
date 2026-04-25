"""
Phase 231: The Inverse Intelligence Algebra

Given the scaling law PA = f(C, N, D, P), INVERT it to solve:
"What C* and N* maximize PA under time budget constraint?"

Two equations, two unknowns:
  Eq1: PA(C, N) = target
  Eq2: Time(C, N) = C^2 * N * k <= budget

Solve with scipy.optimize.

Author: Hiroto Funasaki
"""
import os, json, time, gc, sys
import numpy as np
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 231: The Inverse Intelligence Algebra")
    print("  Solve: What C*, N* maximize PA under time constraint?")
    print("=" * 70)

    # Load scaling law from P230
    fp = os.path.join(RESULTS_DIR, "phase230_scaling_law.json")
    if not os.path.exists(fp):
        print("  ERROR: P230 results not found. Run Phase 230 first!")
        return {'error': 'no scaling law'}
    with open(fp, 'r', encoding='utf-8') as f:
        p230 = json.load(f)

    coefs = p230['coefficients']
    print(f"\n  Loaded scaling law (R2={coefs['R2']:.4f}):")
    print(f"    PA = {coefs['const']:.4f} + {coefs['alpha_C']:.4f}*log(C) "
          f"+ {coefs['beta_N']:.4f}*log(N) + {coefs['gamma_D']:.4f}*log(D) "
          f"+ {coefs['delta_P']:.4f}*log(P)")

    from scipy.optimize import minimize, minimize_scalar

    # Fixed parameters (from our environment)
    D_fixed = 200     # Real ARC tasks
    # P ≈ a * C^2 (empirically: C=16->6507, C=32->24K, C=64->167K)
    # Fit: P ≈ 40 * C^2 (rough estimate)
    P_factor = 40.0

    # Time model: time_per_trial ∝ C^2 * T_step (T_step ~= constant)
    # On our GPU: C=64 -> ~12ms/trial. So time_per_trial ≈ C^2 / (64^2) * 12ms
    ms_per_trial_base = 12.0  # ms at C=64
    C_base = 64.0

    def time_ms(C, N):
        return (C / C_base) ** 2 * ms_per_trial_base * N

    def pa_predict(C, N):
        P = P_factor * C ** 2
        return (coefs['const']
                + coefs['alpha_C'] * np.log(max(1, C))
                + coefs['beta_N'] * np.log(max(1, N))
                + coefs['gamma_D'] * np.log(max(1, D_fixed))
                + coefs['delta_P'] * np.log(max(1, P)))

    # Kaggle budget: 12 hours = 43,200,000 ms
    # But per-task budget: 400 tasks in 12h = 108,000 ms/task
    budgets_ms = {
        'per_task_108s': 108000,
        'per_task_30s': 30000,
        'per_task_10s': 10000,
    }

    results = {}

    for budget_name, budget_ms in budgets_ms.items():
        print(f"\n  --- Budget: {budget_name} ({budget_ms}ms) ---")

        best_pa = -1
        best_C = 16
        best_N = 1

        # Grid search over C and N
        for C in [16, 24, 32, 48, 64, 96, 128, 192, 256]:
            max_N = int(budget_ms / time_ms(C, 1))
            if max_N < 1:
                continue
            # Try N values
            for N in [1, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10000]:
                if N > max_N:
                    break
                total_time = time_ms(C, N)
                if total_time > budget_ms:
                    break
                pa = pa_predict(C, N)
                if pa > best_pa:
                    best_pa = pa
                    best_C = C
                    best_N = N

        P_star = P_factor * best_C ** 2
        time_star = time_ms(best_C, best_N)

        print(f"    Optimal C* = {best_C}")
        print(f"    Optimal N* = {best_N}")
        print(f"    Predicted PA = {best_pa*100:.1f}%")
        print(f"    Estimated P* = {int(P_star):,} params")
        print(f"    Time used = {time_star:.0f}ms ({time_star/1000:.1f}s)")
        print(f"    Synapses per cell = C* = {best_C}")

        results[budget_name] = {
            'C_star': best_C, 'N_star': best_N,
            'predicted_pa': best_pa, 'P_star': int(P_star),
            'time_ms': time_star, 'budget_ms': budget_ms
        }

    # Also solve the inverse: "How much time to achieve target PA?"
    print(f"\n  --- Inverse: Time needed for target PA ---")
    targets = [0.60, 0.65, 0.70, 0.75, 0.80]
    inverse_results = {}
    C_fixed = 64

    for target_pa in targets:
        # Solve: PA(64, N) = target for N
        # PA = const + alpha*log(64) + beta*log(N) + ...
        # beta*log(N) = target - const - alpha*log(64) - gamma*log(D) - delta*log(P)
        P_fixed = P_factor * C_fixed ** 2
        rhs = (target_pa - coefs['const']
               - coefs['alpha_C'] * np.log(C_fixed)
               - coefs['gamma_D'] * np.log(D_fixed)
               - coefs['delta_P'] * np.log(P_fixed))

        if coefs['beta_N'] != 0:
            log_N = rhs / coefs['beta_N']
            N_needed = np.exp(log_N)
            time_needed = time_ms(C_fixed, N_needed)
        else:
            N_needed = float('inf')
            time_needed = float('inf')

        feasible = N_needed < 1e12
        print(f"    PA={target_pa*100:.0f}%: N={N_needed:.0f} trials, "
              f"Time={time_needed/1000:.0f}s {'OK' if feasible and time_needed < 108000 else 'X'}")

        inverse_results[f'PA_{int(target_pa*100)}'] = {
            'target_pa': target_pa, 'N_needed': float(N_needed),
            'time_ms': float(time_needed), 'feasible_108s': bool(feasible and time_needed < 108000)
        }

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  INVERSE INTELLIGENCE ALGEBRA COMPLETE")
    print(f"  Kaggle-optimal design: C*={results.get('per_task_108s',{}).get('C_star','?')}, "
          f"N*={results.get('per_task_108s',{}).get('N_star','?')}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase231_inverse.json"), 'w', encoding='utf-8') as f:
        json.dump({'optimal_configs': results, 'inverse_time': inverse_results,
                   'scaling_law': coefs, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: C vs N Pareto front (under 108s budget)
        Cs = np.arange(16, 257, 8)
        budget = 108000
        pa_map = np.zeros((len(Cs), 50))
        N_vals = np.logspace(0, 4, 50).astype(int)
        for i, C in enumerate(Cs):
            for j, N in enumerate(N_vals):
                if time_ms(C, N) <= budget:
                    pa_map[i, j] = pa_predict(C, N) * 100
                else:
                    pa_map[i, j] = np.nan
        axes[0].imshow(pa_map, aspect='auto', origin='lower',
                      extent=[0, 4, 16, 256], cmap='viridis')
        axes[0].set_xlabel('log10(N)'); axes[0].set_ylabel('C (channels)')
        axes[0].set_title('PA(%) under 108s budget', fontweight='bold')
        cbar = fig.colorbar(axes[0].images[0], ax=axes[0])
        cbar.set_label('PA (%)')

        # Plot 2: Inverse - time needed
        tpas = [int(k.split('_')[1]) for k in inverse_results]
        times = [inverse_results[k]['time_ms']/1000 for k in inverse_results]
        feasible = [inverse_results[k]['feasible_108s'] for k in inverse_results]
        colors_inv = ['#2ecc71' if f else '#e74c3c' for f in feasible]
        axes[1].bar(range(len(tpas)), times, color=colors_inv, alpha=0.8)
        axes[1].axhline(108, color='r', linestyle='--', label='108s budget')
        axes[1].set_xticks(range(len(tpas)))
        axes[1].set_xticklabels([f'{t}%' for t in tpas])
        axes[1].set_ylabel('Time needed (s)')
        axes[1].set_title('Inverse: Time to achieve PA', fontweight='bold')
        axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_yscale('log')

        fig.suptitle('Phase 231: Inverse Intelligence Algebra', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.85, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase231_inverse.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")

    gc.collect()
    return {'optimal': results, 'inverse': inverse_results}

if __name__ == '__main__':
    main()
