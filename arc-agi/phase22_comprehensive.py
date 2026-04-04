"""
Phase 22: Comprehensive Cross-Game Analysis & Publication Figures
==================================================================
Generate paper-quality figures summarizing all ARC-AGI experiments.
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\results"
FIGURES_DIR = r"c:\Users\kyjan\研究\snn-synthesis\figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Load all data
# ============================================================

# Phase 16c: Random baseline across 7 games
p16 = json.load(open(os.path.join(RESULTS_DIR, "phase16c_multigame_parallel.json")))

# Phase 18: K scaling law
p18 = json.load(open(os.path.join(RESULTS_DIR, "phase18_extended_k.json")))

# Phase 14: CNN+Noise Trajectory Ensemble (LS20)
p14_path = os.path.join(RESULTS_DIR, "phase14_trajectory_ensemble.json")
p14 = json.load(open(p14_path)) if os.path.exists(p14_path) else None

# Phase 20: SNN-ExIt on LS20
p20 = json.load(open(os.path.join(RESULTS_DIR, "phase20_exit_ls20.json")))

# Phase 21: SNN-ExIt on TR87
p21 = json.load(open(os.path.join(RESULTS_DIR, "phase21_exit_tr87.json")))

plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans'})

# ============================================================
# Figure 1: The Money Shot — ExIt Self-Improvement on LS20
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1a: ExIt iteration curve
ax = axes[0]
iters = [r['iteration'] for r in p20]
cnn_k1 = [r['configs'].get(f"CNN(i{r['iteration']}) K=1", {}).get('rate', 0) for r in p20]
cnn_k1_noise = [r['configs'].get(f"CNN(i{r['iteration']}) K=1 \u03c3=0.2", {}).get('rate', 0) for r in p20]
cnn_k5 = [r['configs'].get(f"CNN(i{r['iteration']}) K=5 \u03c3=0.1", {}).get('rate', 0) for r in p20]
cnn_k11 = [r['configs'].get(f"CNN(i{r['iteration']}) K=11 \u03c3=0.1", {}).get('rate', 0) for r in p20]

ax.plot(iters, cnn_k1, 'o-', color='#9C27B0', linewidth=2, markersize=8, label='CNN K=1')
ax.plot(iters, cnn_k1_noise, 'd-', color='#E91E63', linewidth=2, markersize=8, label='CNN K=1 + σ=0.2')
ax.plot(iters, cnn_k5, 's-', color='#FF9800', linewidth=2, markersize=8, label='CNN K=5 + σ=0.1')
ax.plot(iters, cnn_k11, '^-', color='#4CAF50', linewidth=2.5, markersize=10, label='CNN K=11 + σ=0.1')
ax.axhline(y=1, color='#999', linestyle='--', linewidth=1, label='Random K=11 (1%)')
ax.axhline(y=78, color='#2196F3', linestyle=':', linewidth=2, alpha=0.5, label='Oracle CNN K=11 (78%)')

ax.fill_between(iters, 0, [1]*5, color='#999', alpha=0.05)
ax.set_xlabel('Self-Distillation Iteration', fontsize=12)
ax.set_ylabel('Level Clear Rate (%)', fontsize=12)
ax.set_title('(a) SNN-ExIt: Oracle-Free Self-Evolution\non LS20', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='center left')
ax.grid(True, alpha=0.2)
ax.set_ylim(-5, 105)
ax.set_xticks(iters)

# 1b: K scaling law across games
ax2 = axes[1]
colors = {'m0r0': '#E91E63', 'tr87': '#2196F3', 'ls20': '#4CAF50'}
labels = {'m0r0': 'M0R0 (easy)', 'tr87': 'TR87 (medium)', 'ls20': 'LS20 (hard)'}

for gid in ['m0r0', 'tr87', 'ls20']:
    if gid not in p18:
        continue
    ks = [int(k.split('=')[1]) for k in p18[gid]]
    rates = [p18[gid][f"K={k}"]["clear_rate"] for k in ks]
    ax2.plot(ks, rates, 'o-', color=colors[gid], linewidth=2, markersize=7, label=labels[gid])

ax2.set_xlabel('K (parallel trajectories)', fontsize=12)
ax2.set_ylabel('Level Clear Rate (%)', fontsize=12)
ax2.set_title('(b) Trajectory Ensemble Scaling Law\nRandom Policy Across Games', fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.set_xticks([1, 3, 5, 11, 21, 51, 101])
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)

# 1c: Cross-game summary bar chart
ax3 = axes[2]
games = ['LS20', 'TR87', 'M0R0']
random_k1 = [0, 0, 8]
random_k11 = [1, 3, 56.5]
exit_k11 = [99, 3, 61]  # From ExIt results

x = np.arange(len(games))
w = 0.25
ax3.bar(x - w, random_k1, w, label='Random K=1', color='#BBDEFB', edgecolor='#1565C0')
ax3.bar(x, random_k11, w, label='Random K=11', color='#90CAF9', edgecolor='#1565C0')
ax3.bar(x + w, exit_k11, w, label='ExIt K=11+σ', color='#E91E63', edgecolor='#880E4F')

# Add value labels
for i, v in enumerate(exit_k11):
    ax3.text(i + w, v + 2, f'{v}%', ha='center', fontsize=10, fontweight='bold')

ax3.set_xticks(x)
ax3.set_xticklabels(games, fontsize=12)
ax3.set_ylabel('Level Clear Rate (%)', fontsize=12)
ax3.set_title('(c) Oracle-Free ExIt Performance\nAcross ARC-AGI-3 Games', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2, axis='y')
ax3.set_ylim(0, 115)

plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, "fig_exit_comprehensive.png")
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"Figure 1: {fig_path}")
plt.close()

# ============================================================
# Figure 2: ExIt Learning Dynamics
# ============================================================
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))

# 2a: Miracle trajectory accumulation
games_exit = [('LS20', p20, '#4CAF50'), ('TR87', p21, '#2196F3')]
for gname, data, color in games_exit:
    iters = [r['iteration'] for r in data]
    totals = [r['miracles_total'] for r in data]
    ax_a.plot(iters, totals, 'o-', color=color, linewidth=2, markersize=8, label=gname)

ax_a.set_xlabel('ExIt Iteration', fontsize=12)
ax_a.set_ylabel('Cumulative Miracle Trajectories', fontsize=12)
ax_a.set_title('Training Data Accumulation', fontsize=13, fontweight='bold')
ax_a.legend(fontsize=11)
ax_a.grid(True, alpha=0.2)

# 2b: CNN K=1 improvement (policy quality without ensemble)
for gname, data, color in games_exit:
    iters = [r['iteration'] for r in data]
    k1_rates = [r['configs'].get(f"CNN(i{r['iteration']}) K=1", {}).get('rate', 0) for r in data]
    ax_b.plot(iters, k1_rates, 'o-', color=color, linewidth=2, markersize=8, label=gname)

ax_b.set_xlabel('ExIt Iteration', fontsize=12)
ax_b.set_ylabel('CNN K=1 Clear Rate (%)', fontsize=12)
ax_b.set_title('Policy Quality (Without Ensemble)', fontsize=13, fontweight='bold')
ax_b.legend(fontsize=11)
ax_b.grid(True, alpha=0.2)
ax_b.set_ylim(-5, max(40, max(cnn_k1) + 5))

plt.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, "fig_exit_dynamics.png")
plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
print(f"Figure 2: {fig2_path}")
plt.close()

# ============================================================
# Summary Table for Paper
# ============================================================
print(f"\n{'='*70}")
print(f"  COMPREHENSIVE RESULTS SUMMARY (SNN-Synthesis v3)")
print(f"{'='*70}")
print(f"\n  1. TRAJECTORY ENSEMBLE SCALING (Random Policy)")
print(f"  {'Game':8s} {'K=1':>8s} {'K=11':>8s} {'K=21':>8s} {'K=51':>8s} {'K=101':>8s}")
for gid in ['m0r0', 'tr87', 'ls20']:
    row = [f"{p18[gid].get(f'K={k}',{}).get('clear_rate',0):.1f}%" for k in [1,11,21,51,101]]
    print(f"  {gid:8s} {'  '.join(row)}")

print(f"\n  2. SNN-ExIt SELF-EVOLUTION (Oracle-Free)")
print(f"  {'Game':8s} {'Random K=11':>12s} {'ExIt K=1':>10s} {'ExIt K=11+σ':>12s} {'Miracles':>10s}")
for gname, data in [('LS20', p20), ('TR87', p21)]:
    last = data[-1]
    rk11 = last['configs'].get('Random K=11', {}).get('rate', 0)
    ck1 = last['configs'].get(f"CNN(i{last['iteration']}) K=1", {}).get('rate', 0)
    ck11 = last['configs'].get(f"CNN(i{last['iteration']}) K=11 σ=0.1", {}).get('rate', 0)
    print(f"  {gname:8s} {rk11:>10.1f}% {ck1:>8.1f}% {ck11:>10.1f}% {last['miracles_total']:>10d}")

print(f"\n  3. KEY FINDINGS")
print(f"  • Trajectory Ensemble is game-invariant (works across all 7 ARC-AGI-3 games)")
print(f"  • SNN-ExIt on LS20: 0% → 99% in 5 iterations WITHOUT human Oracle")
print(f"  • ExIt CNN (99%) SURPASSES Oracle-trained CNN (78%)")
print(f"  • TR87 ExIt limited by sparse state (dim=7) — identifies ExIt prerequisites")
print(f"  • M0R0: Random K=101 = 100% (trivial game, no CNN needed)")

print(f"\n[Done] All figures saved to {FIGURES_DIR}/")
