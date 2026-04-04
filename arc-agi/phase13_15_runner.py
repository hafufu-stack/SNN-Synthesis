"""
Phase 13-15 Combined Runner
============================
Phase 13: σ-K Phase Diagram (2D heatmap of σ × K) [Deep Think #1]
Phase 14: Trajectory Ensemble for L3/L4 breakthrough [Deep Think #2]
Phase 15: N=1000 definitive resonance curve [Gemini's idea]

All CPU-only, all results saved to arc-agi/results/
"""
import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "environment_files", "ls20", "9607627b"))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

from arc_micro_brain import (
    MicroBrainLarge, MicroBrainSmall,
    train, build_oracle_cache, run_cnn, p2g, IDX_ACT
)
import arc_agi
from arcengine import GameAction
from scipy import stats as sp_stats

print(f"[{time.strftime('%H:%M:%S')}] Phase 13-15 Runner")
print(f"Results: {RESULTS_DIR}", flush=True)

arc_inst = arc_agi.Arcade()
cache, meta, ncol, max_clr = build_oracle_cache()
print(f"[{time.strftime('%H:%M:%S')}] Oracle: {max_clr} levels", flush=True)

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0, 0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    return round(max(0, center - margin)*100, 1), round(min(1, center + margin)*100, 1)

# ============================================================
# Shared: Ensemble evaluation
# ============================================================
def run_l2_ensemble(model, sigma, K, arc_inst, cache, meta, ncol, max_steps=300):
    """K noisy forward passes per step, majority vote."""
    env = arc_inst.make("ls20")
    game = env._game
    obs = env.step(GameAction.RESET)
    for a in cache.get(1, []):
        obs = env.step(a)
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False, 'steps': 0}
    expected_lc = 1
    m = meta[1]
    sg, pxo, pyo = m['sg'], m['pxo'], m['pyo']
    gs0, gci0, gri0 = m['gs'], m['gci'], m['gri']
    model.eval()
    for step in range(max_steps):
        pc = np.zeros((1,12,12), dtype=np.float32)
        px, py = game.gudziatsk.x, game.gudziatsk.y
        x, y = p2g(px, py, pxo, pyo)
        if 0<=x<12 and 0<=y<12: pc[0,y,x] = 1.0
        full = np.concatenate([sg, pc], axis=0)
        sv = np.array([game.fwckfzsyc/5.0, game.hiaauhahz/max(1,ncol-1),
            game.cklxociuu/3.0, gs0/5.0, gci0/max(1,ncol-1), gri0/3.0,
            max(0,1-step/200), step/200], dtype=np.float32)
        grid_t = torch.from_numpy(full).unsqueeze(0)
        state_t = torch.from_numpy(sv).unsqueeze(0)
        with torch.no_grad():
            votes = np.zeros(4, dtype=int)
            for _ in range(K):
                logits = model(grid_t, state_t, noise_sigma=sigma)
                votes[logits.argmax(1).item()] += 1
            ai = int(np.argmax(votes))
        obs = env.step(IDX_ACT[ai])
        if obs.levels_completed > expected_lc or obs.state.value == 'WIN':
            return {'cleared': True, 'steps': step+1}
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False, 'steps': step+1}
    return {'cleared': False, 'steps': max_steps}

# ############################################################
# PHASE 13: σ-K Phase Diagram
# ############################################################
print(f"\n{'='*70}")
print(f"  PHASE 13: σ-K Phase Diagram (Large CNN, N=100)")
print(f"{'='*70}", flush=True)

SIGMAS_13 = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
KS_13 = [1, 3, 5, 7, 9, 11]
N_13 = 100

model_large = MicroBrainLarge()
acc = train(model_large)
print(f"  Large CNN trained: acc={acc:.4f}", flush=True)

p13 = {}
for sigma in SIGMAS_13:
    p13[f"{sigma:.2f}"] = {}
    for K in KS_13:
        t0 = time.time()
        c = sum(1 for _ in range(N_13) if run_l2_ensemble(
            model_large, sigma, K, arc_inst, cache, meta, ncol)['cleared'])
        rate = c / N_13 * 100
        ci = wilson_ci(c, N_13)
        p13[f"{sigma:.2f}"][f"K={K}"] = {"c": c, "n": N_13, "rate": round(rate,1), "ci": list(ci)}
        sig = "★" if rate > 0 else " "
        print(f"  {sig} σ={sigma:.2f} K={K:2d}: {rate:5.1f}% ({c}/{N_13}) [{time.time()-t0:.0f}s]", flush=True)

p13_path = os.path.join(RESULTS_DIR, "phase13_sigma_k_heatmap.json")
with open(p13_path, "w") as f:
    json.dump(p13, f, indent=2)
print(f"  Saved: {p13_path}", flush=True)

# Generate heatmap
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

heatmap = np.zeros((len(KS_13), len(SIGMAS_13)))
for i, K in enumerate(KS_13):
    for j, sigma in enumerate(SIGMAS_13):
        heatmap[i, j] = p13[f"{sigma:.2f}"][f"K={K}"]["rate"]

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=0,
               vmax=max(20, heatmap.max()+2), origin='lower')
ax.set_xticks(range(len(SIGMAS_13)))
ax.set_xticklabels([f"{s:.2f}" for s in SIGMAS_13], fontsize=11)
ax.set_yticks(range(len(KS_13)))
ax.set_yticklabels([str(k) for k in KS_13], fontsize=11)
ax.set_xlabel('Noise σ', fontsize=13, fontweight='bold')
ax.set_ylabel('Ensemble Size K', fontsize=13, fontweight='bold')

# Annotate cells
for i in range(len(KS_13)):
    for j in range(len(SIGMAS_13)):
        val = heatmap[i, j]
        color = 'white' if val > 10 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

# Mark the peak
peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
ax.add_patch(plt.Rectangle((peak_idx[1]-0.5, peak_idx[0]-0.5), 1, 1,
    fill=False, edgecolor='gold', linewidth=3))
ax.set_title(f'Phase 13: σ-K Phase Diagram (Large CNN 244K, N={N_13})\n'
             f'Peak: {heatmap.max():.0f}% at σ={SIGMAS_13[peak_idx[1]]:.2f}, K={KS_13[peak_idx[0]]}',
             fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='L2 Clear Rate (%)')
plt.tight_layout()
p13_plot = os.path.join(RESULTS_DIR, "phase13_heatmap.png")
plt.savefig(p13_plot, dpi=150, bbox_inches='tight')
shutil.copy(p13_plot, os.path.join(FIGURES_DIR, "arc_sigma_k_heatmap.png"))
print(f"  Heatmap: {p13_plot}", flush=True)
plt.close('all')

gc.collect()

# ############################################################
# PHASE 14: Trajectory Ensemble (Noisy Beam Search for L3/L4)
# ############################################################
print(f"\n{'='*70}")
print(f"  PHASE 14: Trajectory Ensemble (L2/L3/L4, Large CNN)")
print(f"  K parallel noisy trajectories, pick best one")
print(f"{'='*70}", flush=True)

def run_trajectory_ensemble(model, target_level, sigma, K_traj, arc_inst, cache, meta, ncol, max_steps=300):
    """Run K independent noisy trajectories, return the best outcome.
    'Best' = highest levels_completed, then fewest steps."""
    best = {'cleared': False, 'steps': max_steps, 'levels_extra': 0}
    
    for k in range(K_traj):
        env = arc_inst.make("ls20")
        game = env._game
        obs = env.step(GameAction.RESET)
        
        # Oracle skip to target level
        for a in cache.get(target_level, []):
            obs = env.step(a)
            if obs.state.value == 'GAME_OVER':
                break
        
        expected_lc = target_level
        m = meta.get(target_level)
        if m is None:
            continue
        sg, pxo, pyo = m['sg'], m['pxo'], m['pyo']
        gs0, gci0, gri0 = m['gs'], m['gci'], m['gri']
        model.eval()
        
        cleared = False
        for step in range(max_steps):
            pc = np.zeros((1,12,12), dtype=np.float32)
            px, py = game.gudziatsk.x, game.gudziatsk.y
            x, y = p2g(px, py, pxo, pyo)
            if 0<=x<12 and 0<=y<12: pc[0,y,x] = 1.0
            full = np.concatenate([sg, pc], axis=0)
            sv = np.array([game.fwckfzsyc/5.0, game.hiaauhahz/max(1,ncol-1),
                game.cklxociuu/3.0, gs0/5.0, gci0/max(1,ncol-1), gri0/3.0,
                max(0,1-step/200), step/200], dtype=np.float32)
            
            with torch.no_grad():
                ai = model(torch.from_numpy(full).unsqueeze(0),
                           torch.from_numpy(sv).unsqueeze(0),
                           noise_sigma=sigma).argmax(1).item()
            obs = env.step(IDX_ACT[ai])
            
            if obs.levels_completed > expected_lc or obs.state.value == 'WIN':
                cleared = True
                if not best['cleared'] or step+1 < best['steps']:
                    best = {'cleared': True, 'steps': step+1, 'levels_extra': 0}
                break
            if obs.state.value == 'GAME_OVER':
                break
        
        if not cleared and not best['cleared']:
            best = {'cleared': False, 'steps': max_steps, 'levels_extra': 0}
    
    return best

N_14 = 100
TRAJ_KS = [1, 3, 5, 7, 11]
TRAJ_SIGMAS = [0.20, 0.25, 0.30, 0.40]
LEVELS = [1, 2, 3]  # L2, L3, L4 (0-indexed: target=1=L2, target=2=L3, target=3=L4)
LEVEL_NAMES = {1: "L2", 2: "L3", 3: "L4"}

p14 = {}
for target in LEVELS:
    lname = LEVEL_NAMES[target]
    p14[lname] = {}
    print(f"\n  --- {lname} ---", flush=True)
    
    for sigma in TRAJ_SIGMAS:
        p14[lname][f"sigma_{sigma:.2f}"] = {}
        for K in TRAJ_KS:
            t0 = time.time()
            c = sum(1 for _ in range(N_14) if run_trajectory_ensemble(
                model_large, target, sigma, K, arc_inst, cache, meta, ncol)['cleared'])
            rate = c / N_14 * 100
            p14[lname][f"sigma_{sigma:.2f}"][f"K={K}"] = {
                "c": c, "n": N_14, "rate": round(rate, 1)
            }
            sig = "★" if rate > 0 else " "
            print(f"  {sig} {lname} σ={sigma:.2f} K_traj={K:2d}: {rate:5.1f}% ({c}/{N_14}) [{time.time()-t0:.0f}s]",
                  flush=True)

p14_path = os.path.join(RESULTS_DIR, "phase14_trajectory_ensemble.json")
with open(p14_path, "w") as f:
    json.dump(p14, f, indent=2)
print(f"  Saved: {p14_path}", flush=True)

# Phase 14 plot
fig, axes = plt.subplots(1, len(LEVELS), figsize=(6*len(LEVELS), 5))
if len(LEVELS) == 1: axes = [axes]
colors_14 = plt.cm.viridis(np.linspace(0.2, 0.9, len(TRAJ_SIGMAS)))

for col, target in enumerate(LEVELS):
    ax = axes[col]
    lname = LEVEL_NAMES[target]
    for si, sigma in enumerate(TRAJ_SIGMAS):
        skey = f"sigma_{sigma:.2f}"
        if skey not in p14[lname]: continue
        ks = [int(k.split('=')[1]) for k in p14[lname][skey]]
        rates = [p14[lname][skey][f"K={k}"]["rate"] for k in ks]
        ax.plot(ks, rates, '-o', color=colors_14[si], lw=2, ms=7, label=f'σ={sigma}')
    ax.set_xlabel('Trajectory K', fontsize=11)
    ax.set_ylabel('Clear Rate (%)', fontsize=11)
    ax.set_title(f'{lname}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(TRAJ_KS)

fig.suptitle('Phase 14: Trajectory Ensemble (Large CNN 244K, N=100)', fontsize=14, fontweight='bold')
plt.tight_layout()
p14_plot = os.path.join(RESULTS_DIR, "phase14_trajectory.png")
plt.savefig(p14_plot, dpi=150, bbox_inches='tight')
shutil.copy(p14_plot, os.path.join(FIGURES_DIR, "arc_trajectory_ensemble.png"))
print(f"  Plot: {p14_plot}", flush=True)
plt.close('all')

gc.collect()

# ############################################################
# PHASE 15: N=1000 Definitive Resonance Curve (Large CNN)
# ############################################################
print(f"\n{'='*70}")
print(f"  PHASE 15: N=1000 Definitive Resonance Curve")
print(f"{'='*70}", flush=True)

# Retrain fresh model for clean measurement
model_large2 = MicroBrainLarge()
acc = train(model_large2)
print(f"  Fresh Large CNN: acc={acc:.4f}", flush=True)

SIGMAS_15 = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.00]
N_15 = 1000
p15 = {}

for sigma in SIGMAS_15:
    t0 = time.time()
    c = 0
    for ep in range(N_15):
        r = run_cnn(model_large2, arc_inst, 1, sigma, cache, meta, ncol)
        if r['cleared']:
            c += 1
    rate = c / N_15 * 100
    ci = wilson_ci(c, N_15)
    
    # Fisher vs baseline
    if sigma == 0.0:
        p_val = None
        bl_c = c
    else:
        table = [[c, N_15-c], [bl_c, N_15-bl_c]]
        _, p_val = sp_stats.fisher_exact(table)
    
    p15[f"{sigma:.2f}"] = {
        "c": c, "n": N_15, "rate": round(rate, 1),
        "ci_lower": ci[0], "ci_upper": ci[1],
        "fisher_p": round(p_val, 8) if p_val is not None else None,
    }
    sig_str = ""
    if p_val is not None:
        if p_val < 0.001: sig_str = "***"
        elif p_val < 0.01: sig_str = "**"
        elif p_val < 0.05: sig_str = "*"
    
    print(f"  σ={sigma:.2f}: {rate:6.2f}% ({c}/{N_15}) CI=[{ci[0]:.1f}%,{ci[1]:.1f}%]"
          f" p={p_val if p_val else 'baseline':} {sig_str} [{time.time()-t0:.0f}s]", flush=True)

p15_path = os.path.join(RESULTS_DIR, "phase15_n1000_curve.json")
with open(p15_path, "w") as f:
    json.dump(p15, f, indent=2)
print(f"  Saved: {p15_path}", flush=True)

# N=1000 beautiful curve plot
fig, ax = plt.subplots(figsize=(12, 6))
sigmas = [float(s) for s in p15.keys()]
rates = [p15[s]["rate"] for s in p15]
ci_lo = [p15[s]["ci_lower"] for s in p15]
ci_hi = [p15[s]["ci_upper"] for s in p15]

ax.fill_between(sigmas, ci_lo, ci_hi, alpha=0.2, color='#FF5722')
ax.plot(sigmas, rates, '-o', color='#FF5722', lw=3, ms=10, label='Large CNN (244K)')
ax.errorbar(sigmas, rates,
            yerr=[np.array(rates)-np.array(ci_lo), np.array(ci_hi)-np.array(rates)],
            fmt='o', color='#FF5722', capsize=5, lw=2, ms=0)

# Peak annotation
peak_idx = np.argmax(rates)
ax.annotate(f'Peak: {rates[peak_idx]:.1f}%\nσ={sigmas[peak_idx]:.2f}\nN={N_15}',
    xy=(sigmas[peak_idx], rates[peak_idx]),
    xytext=(sigmas[peak_idx]+0.12, rates[peak_idx]+2),
    fontsize=12, fontweight='bold', color='#FF5722',
    arrowprops=dict(arrowstyle='->', color='#FF5722', lw=2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#FF5722'))

# Mark significance
for i, s in enumerate(sigmas):
    p_val = p15[f"{s:.2f}"].get("fisher_p")
    if p_val is not None and p_val < 0.05:
        ax.plot(s, rates[i], '*', color='gold', ms=20, zorder=10)

ax.set_xlabel('Noise σ', fontsize=14, fontweight='bold')
ax.set_ylabel('L2 Clear Rate (%)', fontsize=14, fontweight='bold')
ax.set_title(f'Phase 15: The Definitive Stochastic Resonance Curve\n'
             f'Large CNN (244K params), N={N_15}, 95% Wilson CI',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.05)
ax.set_ylim(-1, max(rates)+8)
plt.tight_layout()

p15_plot = os.path.join(RESULTS_DIR, "phase15_definitive_curve.png")
plt.savefig(p15_plot, dpi=200, bbox_inches='tight')
shutil.copy(p15_plot, os.path.join(FIGURES_DIR, "arc_definitive_resonance_n1000.png"))
print(f"  Plot: {p15_plot}", flush=True)

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"  FINAL SUMMARY")
print(f"{'='*70}")

print(f"\n  Phase 13 (σ-K Heatmap):")
peak_rate, peak_s, peak_k = 0, "", ""
for s in p13:
    for k in p13[s]:
        if p13[s][k]["rate"] > peak_rate:
            peak_rate = p13[s][k]["rate"]
            peak_s, peak_k = s, k
print(f"    Peak: {peak_rate}% at σ={peak_s}, {peak_k}")

print(f"\n  Phase 14 (Trajectory Ensemble):")
for lname in p14:
    best_rate, best_cfg = 0, ""
    for skey in p14[lname]:
        for kkey in p14[lname][skey]:
            r = p14[lname][skey][kkey]["rate"]
            if r > best_rate:
                best_rate = r
                best_cfg = f"{skey} {kkey}"
    print(f"    {lname}: best={best_rate}% ({best_cfg})")

print(f"\n  Phase 15 (N=1000 Curve):")
peak_idx = np.argmax(rates)
print(f"    Peak: {rates[peak_idx]:.1f}% at σ={sigmas[peak_idx]:.2f}")
print(f"    CI at peak: [{ci_lo[peak_idx]:.1f}%, {ci_hi[peak_idx]:.1f}%]")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 13-15 complete!", flush=True)
