"""
Phase 230: The Grand Unified Scaling Law

Collect data from all previous experiments and fit a single equation:
PA = Const + alpha*log(C) + beta*log(N) + gamma*log(D) + delta*log(P)

C: channels (synapses), N: trial count, D: data size, P: params

This is the "E=mc^2" of neural intelligence.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def load_result(name):
    fp = os.path.join(RESULTS_DIR, name)
    if os.path.exists(fp):
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 230: The Grand Unified Scaling Law")
    print("  PA = f(C, N, D, P) - the equation of intelligence")
    print("=" * 70)

    # Collect data points: (C, N, D, P, PA)
    # C=channels, N=trial_count, D=data_count, P=params, PA=pixel_accuracy
    data_points = []

    # From P206: causal volume (vary K, C, T)
    p206 = load_result("phase206_causal_volume.json")
    if p206:
        for r in p206['results']:
            C = r['C']; P = r['params']; pa = r['pa']
            data_points.append({'C': C, 'N': 1, 'D': 200, 'P': P, 'PA': pa, 'src': 'P206'})

    # From P209: TTC law (vary N)
    p209 = load_result("phase209_ttc_law.json")
    if p209:
        model_info = p209.get('model', {})
        C_base = model_info.get('C', 64)
        P_base = model_info.get('params', 167000)
        for r in p209['results']:
            N = r['N']; pa = r['best_pa']
            data_points.append({'C': C_base, 'N': N, 'D': 200, 'P': P_base, 'PA': pa, 'src': 'P209'})

    # From P224: synthetic data scaling (vary D)
    p224 = load_result("phase224_synthetic.json")
    if p224:
        for k, r in p224['results'].items():
            if k == 'baseline':
                D = 200
            else:
                D = int(k.split('_')[1]) + 200  # syn + real
            data_points.append({'C': 64, 'N': 1, 'D': D, 'P': 166679, 'PA': r['pa'], 'src': 'P224'})

    # From P225: MoE (vary P)
    p225 = load_result("phase225_moe.json")
    if p225:
        data_points.append({'C': 64, 'N': 1, 'D': 200, 'P': 166679,
                           'PA': p225['baseline']['pa'], 'src': 'P225_base'})
        for k, r in p225['moe'].items():
            data_points.append({'C': 64, 'N': 1, 'D': 200, 'P': r['params'],
                               'PA': r['pa'], 'src': f'P225_{k}'})

    # From P211: diversity (vary N)
    p211 = load_result("phase211_diversity.json")
    if p211:
        for r in p211['results']:
            N = r['total_trials']
            data_points.append({'C': 64, 'N': N, 'D': 200, 'P': 166679,
                               'PA': r['oracle_pa'], 'src': 'P211'})

    print(f"\n  Collected {len(data_points)} data points")

    if len(data_points) < 5:
        print("  Not enough data for regression!")
        return {'error': 'insufficient data'}

    # Print all data points
    print(f"\n  {'C':>5s} {'N':>5s} {'D':>6s} {'P':>8s} {'PA':>6s} {'src'}")
    for d in data_points:
        print(f"  {d['C']:5d} {d['N']:5d} {d['D']:6d} {d['P']:8d} {d['PA']:6.3f} {d['src']}")

    # Fit: PA = const + a*log(C) + b*log(N) + c*log(D) + d*log(P)
    from sklearn.linear_model import LinearRegression

    X = []
    y = []
    for d in data_points:
        X.append([
            np.log(max(1, d['C'])),
            np.log(max(1, d['N'])),
            np.log(max(1, d['D'])),
            np.log(max(1, d['P'])),
        ])
        y.append(d['PA'])

    X = np.array(X)
    y = np.array(y)

    reg = LinearRegression()
    reg.fit(X, y)
    r2 = reg.score(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred

    coefs = {
        'alpha_C': reg.coef_[0],
        'beta_N': reg.coef_[1],
        'gamma_D': reg.coef_[2],
        'delta_P': reg.coef_[3],
        'const': reg.intercept_,
        'R2': r2
    }

    print(f"\n{'='*70}")
    print(f"  GRAND UNIFIED SCALING LAW:")
    print(f"  PA = {coefs['const']:.4f}")
    print(f"       + {coefs['alpha_C']:.4f} * log(C)  [channels/synapses]")
    print(f"       + {coefs['beta_N']:.4f} * log(N)  [trial count]")
    print(f"       + {coefs['gamma_D']:.4f} * log(D)  [data quantity]")
    print(f"       + {coefs['delta_P']:.4f} * log(P)  [total params]")
    print(f"  R2 = {r2:.4f}")
    print(f"  RMSE = {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"{'='*70}")

    # Predictions
    print(f"\n  Predictions from the equation:")
    scenarios = [
        ('Current best', 64, 1, 200, 166679),
        ('10K data', 64, 1, 10200, 166679),
        ('100K data', 64, 1, 100200, 166679),
        ('MoE+100K', 64, 1, 100200, 1313527),
        ('100 trials', 64, 100, 200, 166679),
        ('C=128', 128, 1, 200, 600000),
        ('Dream: C=128, 100K, N=100', 128, 100, 100200, 600000),
    ]
    for name, C, N, D, P in scenarios:
        pred_pa = (coefs['const']
                   + coefs['alpha_C'] * np.log(C)
                   + coefs['beta_N'] * np.log(N)
                   + coefs['gamma_D'] * np.log(D)
                   + coefs['delta_P'] * np.log(P))
        print(f"    {name:35s}: PA = {pred_pa*100:.1f}%")

    elapsed = time.time() - t0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase230_scaling_law.json"), 'w', encoding='utf-8') as f:
        json.dump({'coefficients': coefs, 'data_points': data_points,
                   'n_points': len(data_points), 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Predicted vs Actual
        axes[0].scatter(y*100, y_pred*100, c='#3498db', alpha=0.7, s=40)
        axes[0].plot([45, 70], [45, 70], 'k--', alpha=0.3)
        axes[0].set_xlabel('Actual PA (%)'); axes[0].set_ylabel('Predicted PA (%)')
        axes[0].set_title(f'Fit Quality (R2={r2:.3f})', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Coefficient magnitudes
        coef_names = ['a(C)', 'b(N)', 'g(D)', 'd(P)']
        coef_vals = [coefs['alpha_C'], coefs['beta_N'], coefs['gamma_D'], coefs['delta_P']]
        colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in coef_vals]
        axes[1].barh(coef_names, coef_vals, color=colors, alpha=0.8)
        axes[1].axvline(0, color='k', linewidth=0.5)
        axes[1].set_xlabel('Coefficient'); axes[1].set_title('Scaling Exponents', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        # Plot 3: Residuals
        axes[2].scatter(y_pred*100, residuals*100, c='#9b59b6', alpha=0.6, s=30)
        axes[2].axhline(0, color='k', linewidth=0.5)
        axes[2].set_xlabel('Predicted PA (%)'); axes[2].set_ylabel('Residual (pp)')
        axes[2].set_title('Residuals', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 230: Grand Unified Scaling Law of Intelligence', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.85, wspace=0.35)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase230_scaling_law.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")

    gc.collect()
    return coefs

if __name__ == '__main__':
    main()
