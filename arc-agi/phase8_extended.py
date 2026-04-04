"""
Phase 8 Extended: Fine-grained σ sweep (N=30) + N=100 validation
Runs sequentially: part1 → part2 → plot → done
Uses arc_micro_brain.py functions directly.
"""
import sys, os, json, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # arc-agi/
REPO_DIR = os.path.dirname(SCRIPT_DIR)  # snn-synthesis/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "environment_files", "ls20", "9607627b"))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

# Import everything from arc_micro_brain
from arc_micro_brain import (
    MicroBrainSmall, MicroBrainLarge,
    train, build_oracle_cache, run_cnn, sweep,
    ACT_IDX, IDX_ACT
)
import arc_agi

print(f"[{time.strftime('%H:%M:%S')}] Phase 8 Extended starting...")
print(f"Results: {RESULTS_DIR}")
print(flush=True)

# ============================================================
# Setup
# ============================================================
arc = arc_agi.Arcade()
cache, meta, ncol, max_clr = build_oracle_cache()
print(f"[{time.strftime('%H:%M:%S')}] Oracle built: {max_clr} levels")

def train_fresh(model_cls):
    model = model_cls()
    acc = train(model)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  Model: {nparams:,} params, acc={acc:.4f}")
    return model

def run_l2_n(model, sigma, n_episodes):
    """Evaluate L2 only, N episodes"""
    c = 0
    for _ in range(n_episodes):
        r = run_cnn(model, arc, 1, sigma, cache, meta, ncol)  # target=1 = L2 (0-indexed)
        if r['cleared']:
            c += 1
    return c, n_episodes

# ============================================================
# PART 1: Fine-grained σ sweep (N=30, L2 only)
# ============================================================
print(f"\n{'='*60}")
print(f"  PART 1: Fine-grained σ sweep (σ=0.16-0.24, N=30)")
print(f"{'='*60}", flush=True)

fine_sigmas = [0.16, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23, 0.24]
N_FINE = 30
fine_results = {}

for mname, mcls in [("small_63K", MicroBrainSmall), ("large_244K", MicroBrainLarge)]:
    print(f"\n  Training {mname}...", flush=True)
    model = train_fresh(mcls)
    fine_results[mname] = {}
    for sigma in fine_sigmas:
        t0 = time.time()
        c, n = run_l2_n(model, sigma, N_FINE)
        rate = c / n * 100
        fine_results[mname][f"{sigma:.2f}"] = {"c": c, "n": n, "rate": round(rate, 1)}
        print(f"    σ={sigma:.2f}: L2={rate:5.1f}% ({c}/{n}) [{time.time()-t0:.0f}s]", flush=True)

fine_path = os.path.join(RESULTS_DIR, "fine_sigma_sweep.json")
with open(fine_path, "w") as f:
    json.dump(fine_results, f, indent=2)
print(f"\n  Saved: {fine_path}", flush=True)

# ============================================================
# PART 2: N=100 validation (key σ values, L2 only)
# ============================================================
print(f"\n{'='*60}")
print(f"  PART 2: N=100 validation (σ=0, 0.10, 0.15, 0.20, 0.25, 0.30)")
print(f"{'='*60}", flush=True)

val_sigmas = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
N_VAL = 100
val_results = {}

for mname, mcls in [("small_63K", MicroBrainSmall), ("large_244K", MicroBrainLarge)]:
    print(f"\n  Training {mname}...", flush=True)
    model = train_fresh(mcls)
    val_results[mname] = {}
    for sigma in val_sigmas:
        t0 = time.time()
        c, n = run_l2_n(model, sigma, N_VAL)
        rate = c / n * 100
        val_results[mname][f"{sigma:.2f}"] = {"c": c, "n": n, "rate": round(rate, 1)}
        print(f"    σ={sigma:.2f}: L2={rate:5.1f}% ({c}/{n}) [{time.time()-t0:.0f}s]", flush=True)

# Add statistics
from scipy import stats as sp_stats

for mname in val_results:
    bl = val_results[mname]["0.00"]
    for sigma_str, entry in val_results[mname].items():
        if sigma_str == "0.00":
            entry["fisher_p"] = None
            continue
        table = [[entry["c"], entry["n"] - entry["c"]],
                 [bl["c"], bl["n"] - bl["c"]]]
        _, p = sp_stats.fisher_exact(table)
        entry["fisher_p"] = round(p, 6)
        # Wilson CI
        k, nn = entry["c"], entry["n"]
        z = 1.96
        p_hat = k / nn
        denom = 1 + z**2 / nn
        center = (p_hat + z**2 / (2*nn)) / denom
        margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*nn)) / nn) / denom
        entry["ci_lower"] = round(max(0, center - margin) * 100, 1)
        entry["ci_upper"] = round(min(1, center + margin) * 100, 1)

val_path = os.path.join(RESULTS_DIR, "n100_validation.json")
with open(val_path, "w") as f:
    json.dump(val_results, f, indent=2)
print(f"\n  Saved: {val_path}", flush=True)

# ============================================================
# PART 3: Combined plot
# ============================================================
print(f"\n{'='*60}")
print(f"  PART 3: Generating plots")
print(f"{'='*60}", flush=True)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load original N=30 data
with open(os.path.join(RESULTS_DIR, "arc_noise_final.json")) as f:
    orig_data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for col, (mname, color, label) in enumerate([
    ("small_63K", '#2196F3', 'Small CNN (63K)'),
    ("large_244K", '#FF5722', 'Large CNN (244K)')
]):
    ax = axes[col]
    
    # Combine original + fine sigma data
    all_pts = {}
    for sigma_str, levels in orig_data[mname].items():
        if "L2" in levels:
            all_pts[float(sigma_str)] = levels["L2"]["rate"]
    for sigma_str, entry in fine_results[mname].items():
        all_pts[float(sigma_str)] = entry["rate"]
    
    ss = sorted(all_pts.keys())
    rr = [all_pts[s] for s in ss]
    
    ax.plot(ss, rr, '-o', color=color, lw=2, ms=5, label='N=30 (all σ)', alpha=0.7)
    ax.fill_between(ss, rr, alpha=0.15, color=color)
    
    # N=100 overlay with error bars
    if mname in val_results:
        vs = sorted([float(s) for s in val_results[mname].keys()])
        vr = [val_results[mname][f"{s:.2f}"]["rate"] for s in vs]
        vlo = [val_results[mname][f"{s:.2f}"].get("ci_lower", 0) for s in vs]
        vhi = [val_results[mname][f"{s:.2f}"].get("ci_upper", 0) for s in vs]
        
        ax.errorbar(vs, vr,
                     yerr=[np.array(vr) - np.array(vlo),
                           np.array(vhi) - np.array(vr)],
                     fmt='s', color='black', ms=8, capsize=4, lw=2,
                     label='N=100 (95% CI)', zorder=5)
    
    # Peak annotation
    pi = np.argmax(rr)
    if rr[pi] > 0:
        ax.annotate(f'Peak: {rr[pi]:.0f}%\nσ={ss[pi]:.2f}',
            xy=(ss[pi], rr[pi]),
            xytext=(ss[pi]+0.12, rr[pi]+5),
            fontsize=10, fontweight='bold', color=color,
            arrowprops=dict(arrowstyle='->', color=color))
    
    ax.set_xlabel('Noise σ', fontsize=12)
    ax.set_ylabel('L2 Clear Rate (%)', fontsize=12)
    ax.set_title(f'{label}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-3, max(max(rr)+15, 40))
    ax.set_xlim(-0.02, 0.52)

fig.suptitle('ARC-AGI Stochastic Resonance: Fine σ + N=100 Validation',
             fontsize=14, fontweight='bold')
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "extended_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
# Also save to figures/
import shutil
fig_path = os.path.join(FIGURES_DIR, "arc_extended_analysis.png")
shutil.copy(plot_path, fig_path)
print(f"  Plot: {plot_path}")
print(f"  Copy: {fig_path}", flush=True)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  FINAL SUMMARY")
print(f"{'='*60}")
print(f"\n  Fine-grained sweep (N={N_FINE}):")
for mname in fine_results:
    print(f"    {mname}:")
    for s in sorted(fine_results[mname].keys(), key=float):
        e = fine_results[mname][s]
        print(f"      σ={s}: {e['rate']:5.1f}% ({e['c']}/{e['n']})")

print(f"\n  N={N_VAL} validation:")
for mname in val_results:
    print(f"    {mname}:")
    for s in sorted(val_results[mname].keys(), key=float):
        e = val_results[mname][s]
        p_str = f"p={e['fisher_p']:.4f}" if e.get('fisher_p') else "(baseline)"
        ci = f"[{e.get('ci_lower',0):.0f}%,{e.get('ci_upper',0):.0f}%]" if 'ci_lower' in e else ""
        print(f"      σ={s}: {e['rate']:5.1f}% ({e['c']}/{e['n']}) {ci} {p_str}")

print(f"\n[{time.strftime('%H:%M:%S')}] All done!")
