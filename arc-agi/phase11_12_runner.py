"""
Phase 11: Bandit-Adaptive Noise + Synapse Scaling
Phase 12: Noise Ensemble Voting (Opus's addition)
==================================================
Runs both experiments sequentially.

Phase 11: Multi-Armed Bandit selects (σ, synapse_scale) at each step
  based on dense reward (distance-to-goal delta).
  "When the AI detects progress, adjust noise and synapse strength."

Phase 12: Instead of 1 noisy forward pass, do K passes and majority vote.
  Tests whether sampling multiple noisy decisions amplifies resonance.
"""
import sys, os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # arc-agi/
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "environment_files", "ls20", "9607627b"))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

from arc_micro_brain import (
    MicroBrainLarge, MicroBrainSmall,
    train, build_oracle_cache, p2g, IDX_ACT
)
import arc_agi
from arcengine import GameAction

print(f"[{time.strftime('%H:%M:%S')}] Phase 11+12 Runner Starting")
print(f"Results: {RESULTS_DIR}", flush=True)

arc_inst = arc_agi.Arcade()
cache, meta, ncol, max_clr = build_oracle_cache()
print(f"[{time.strftime('%H:%M:%S')}] Oracle: {max_clr} levels", flush=True)

# ============================================================
# Shared: get goal positions for distance computation
# ============================================================
def get_goal_positions(meta_entry):
    """Extract goal cell grid positions from level metadata"""
    sg = meta_entry['sg']
    # Channel 1 = goals
    goal_ys, goal_xs = np.where(sg[1] > 0)
    return list(zip(goal_xs.tolist(), goal_ys.tolist()))

def manhattan_to_nearest_goal(px, py, pxo, pyo, goals):
    """Manhattan distance from player to nearest goal in grid coords"""
    gx, gy = p2g(px, py, pxo, pyo)
    if not goals:
        return 999
    return min(abs(gx - gox) + abs(gy - goy) for gox, goy in goals)

# ============================================================
# PHASE 11: Multi-Armed Bandit Adaptive Noise + Synapse Scaling
# ============================================================

class UCBBandit:
    """Upper Confidence Bound bandit for (σ, synapse_scale) selection"""
    def __init__(self, arms):
        self.arms = arms  # list of (sigma, synapse_scale) tuples
        self.n_arms = len(arms)
        self.counts = np.zeros(self.n_arms)
        self.rewards = np.zeros(self.n_arms)
        self.total = 0
    
    def select(self):
        self.total += 1
        # Force explore each arm at least once
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        # UCB1
        avg = self.rewards / self.counts
        exploration = np.sqrt(2 * np.log(self.total) / self.counts)
        ucb = avg + exploration
        return int(np.argmax(ucb))
    
    def update(self, arm_idx, reward):
        self.counts[arm_idx] += 1
        self.rewards[arm_idx] += reward

# Bandit arms: (sigma, synapse_scale)
# synapse_scale multiplies fc1 output = controls signal gain
BANDIT_ARMS = [
    (0.10, 0.8),   # low noise, weak synapses
    (0.15, 1.0),   # moderate noise, normal synapses
    (0.20, 1.0),   # optimal static noise, normal
    (0.25, 1.0),   # high noise, normal
    (0.20, 1.2),   # optimal noise, strong synapses
    (0.25, 1.2),   # high noise, strong synapses
    (0.15, 0.8),   # moderate noise, weak synapses
    (0.30, 1.0),   # very high noise, normal
]

def run_l2_bandit(model, arc_inst, cache, meta, ncol, max_steps=300):
    """L2 with UCB bandit selecting (σ, synapse_scale) per step"""
    env = arc_inst.make("ls20")
    game = env._game
    obs = env.step(GameAction.RESET)
    
    for a in cache.get(1, []):
        obs = env.step(a)
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False, 'steps': 0, 'arm_counts': [0]*len(BANDIT_ARMS)}
    
    expected_lc = 1
    m = meta[1]
    sg, pxo, pyo = m['sg'], m['pxo'], m['pyo']
    gs0, gci0, gri0 = m['gs'], m['gci'], m['gri']
    goals = get_goal_positions(m)
    
    model.eval()
    bandit = UCBBandit(BANDIT_ARMS)
    prev_dist = manhattan_to_nearest_goal(game.gudziatsk.x, game.gudziatsk.y, pxo, pyo, goals)
    
    for step in range(max_steps):
        # Bandit selects arm
        arm = bandit.select()
        sigma, syn_scale = BANDIT_ARMS[arm]
        
        # Build observation
        pc = np.zeros((1,12,12), dtype=np.float32)
        px, py = game.gudziatsk.x, game.gudziatsk.y
        x, y = p2g(px, py, pxo, pyo)
        if 0<=x<12 and 0<=y<12: pc[0,y,x] = 1.0
        full = np.concatenate([sg, pc], axis=0)
        sv = np.array([game.fwckfzsyc/5.0, game.hiaauhahz/max(1,ncol-1),
            game.cklxociuu/3.0, gs0/5.0, gci0/max(1,ncol-1), gri0/3.0,
            max(0,1-step/200), step/200], dtype=np.float32)
        
        with torch.no_grad():
            grid_t = torch.from_numpy(full).unsqueeze(0)
            state_t = torch.from_numpy(sv).unsqueeze(0)
            
            # Manual forward with synapse scaling
            feat = model.conv(grid_t)
            feat = model.gap(feat).squeeze(-1).squeeze(-1)
            combined = torch.cat([feat, state_t], dim=1)
            h = model.head(combined) * syn_scale  # synapse scaling!
            if sigma > 0:
                h = h + torch.randn_like(h) * sigma
            logits = model.output(h)
            ai = logits.argmax(1).item()
        
        obs = env.step(IDX_ACT[ai])
        
        # Dense reward: distance improvement
        new_dist = manhattan_to_nearest_goal(game.gudziatsk.x, game.gudziatsk.y, pxo, pyo, goals)
        reward = (prev_dist - new_dist) / 12.0  # normalize by grid size
        
        # Bonus for level clear
        if obs.levels_completed > expected_lc or obs.state.value == 'WIN':
            bandit.update(arm, reward + 1.0)
            return {'cleared': True, 'steps': step+1, 
                    'arm_counts': bandit.counts.tolist()}
        
        if obs.state.value == 'GAME_OVER':
            bandit.update(arm, -1.0)
            return {'cleared': False, 'steps': step+1,
                    'arm_counts': bandit.counts.tolist()}
        
        bandit.update(arm, reward)
        prev_dist = new_dist
    
    return {'cleared': False, 'steps': max_steps, 
            'arm_counts': bandit.counts.tolist()}

def run_l2_static(model, sigma, arc_inst, cache, meta, ncol, max_steps=300):
    """Static baseline for comparison"""
    env = arc_inst.make("ls20")
    game = env._game
    obs = env.step(GameAction.RESET)
    for a in cache.get(1, []):
        obs = env.step(a)
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False}
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
        with torch.no_grad():
            ai = model(torch.from_numpy(full).unsqueeze(0),
                       torch.from_numpy(sv).unsqueeze(0),
                       noise_sigma=sigma).argmax(1).item()
        obs = env.step(IDX_ACT[ai])
        if obs.levels_completed > expected_lc or obs.state.value == 'WIN':
            return {'cleared': True}
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False}
    return {'cleared': False}

# ============================================================
# PHASE 12: Noise Ensemble Voting (Opus's idea)
# ============================================================

def run_l2_ensemble(model, sigma, K, arc_inst, cache, meta, ncol, max_steps=300):
    """Do K noisy forward passes per step, majority vote on action.
    Hypothesis: sampling amplifies stochastic resonance."""
    env = arc_inst.make("ls20")
    game = env._game
    obs = env.step(GameAction.RESET)
    for a in cache.get(1, []):
        obs = env.step(a)
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False}
    
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
            # K noisy forward passes
            votes = np.zeros(4, dtype=int)
            for _ in range(K):
                logits = model(grid_t, state_t, noise_sigma=sigma)
                votes[logits.argmax(1).item()] += 1
            ai = int(np.argmax(votes))  # majority vote
        
        obs = env.step(IDX_ACT[ai])
        if obs.levels_completed > expected_lc or obs.state.value == 'WIN':
            return {'cleared': True}
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False}
    return {'cleared': False}


# ############################################################
# RUN EXPERIMENTS
# ############################################################

from scipy import stats as sp_stats

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0, 0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
    return round(max(0, center - margin)*100, 1), round(min(1, center + margin)*100, 1)

N = 100

# ============================================================
# PHASE 11
# ============================================================
print(f"\n{'='*60}")
print(f"  PHASE 11: Bandit-Adaptive Noise + Synapse Scaling (N={N})")
print(f"{'='*60}", flush=True)

p11_results = {}

for model_name, model_cls in [("large_244K", MicroBrainLarge), ("small_63K", MicroBrainSmall)]:
    print(f"\n  {model_name}:", flush=True)
    model = model_cls()
    acc = train(model)
    print(f"  Trained: acc={acc:.4f}", flush=True)
    
    p11_results[model_name] = {}
    
    # Static baselines
    for sigma in [0.20, 0.25]:
        t0 = time.time()
        c = sum(1 for _ in range(N) if run_l2_static(model, sigma, arc_inst, cache, meta, ncol)['cleared'])
        rate = c / N * 100
        ci_lo, ci_hi = wilson_ci(c, N)
        p11_results[model_name][f"static_{sigma:.2f}"] = {"c": c, "n": N, "rate": round(rate,1), "ci": [ci_lo, ci_hi]}
        print(f"    static σ={sigma}: {rate:5.1f}% ({c}/{N}) [{time.time()-t0:.0f}s]", flush=True)
    
    # Bandit
    t0 = time.time()
    clears = 0
    total_arm_counts = np.zeros(len(BANDIT_ARMS))
    for ep in range(N):
        r = run_l2_bandit(model, arc_inst, cache, meta, ncol)
        if r['cleared']:
            clears += 1
        total_arm_counts += np.array(r['arm_counts'])
    
    rate = clears / N * 100
    ci_lo, ci_hi = wilson_ci(clears, N)
    
    # Fisher vs best static baseline
    best_static_c = max(p11_results[model_name][k]["c"] for k in p11_results[model_name])
    table = [[clears, N - clears], [best_static_c, N - best_static_c]]
    _, fisher_p = sp_stats.fisher_exact(table)
    
    # Arm preference
    arm_pct = total_arm_counts / total_arm_counts.sum() * 100
    preferred_arm = int(np.argmax(total_arm_counts))
    
    p11_results[model_name]["bandit"] = {
        "c": clears, "n": N, "rate": round(rate, 1), "ci": [ci_lo, ci_hi],
        "fisher_p_vs_best_static": round(fisher_p, 6),
        "arm_preference": {f"arm{i}_s{s:.2f}_syn{sc:.1f}": round(arm_pct[i], 1)
                          for i, (s, sc) in enumerate(BANDIT_ARMS)},
        "preferred_arm": f"σ={BANDIT_ARMS[preferred_arm][0]}, scale={BANDIT_ARMS[preferred_arm][1]}",
    }
    
    print(f"    ★ bandit:    {rate:5.1f}% ({clears}/{N}) p={fisher_p:.4f} [{time.time()-t0:.0f}s]")
    print(f"      Preferred arm: σ={BANDIT_ARMS[preferred_arm][0]}, syn={BANDIT_ARMS[preferred_arm][1]}")
    print(f"      Arm distribution: {' '.join(f'{p:.0f}%' for p in arm_pct)}", flush=True)

p11_path = os.path.join(RESULTS_DIR, "phase11_bandit.json")
with open(p11_path, "w") as f:
    json.dump(p11_results, f, indent=2)
print(f"\n  Saved: {p11_path}", flush=True)

gc.collect()

# ============================================================
# PHASE 12: Noise Ensemble Voting
# ============================================================
print(f"\n{'='*60}")
print(f"  PHASE 12: Noise Ensemble Voting (N={N})")
print(f"  Hypothesis: K noisy votes > 1 noisy decision")
print(f"{'='*60}", flush=True)

ENSEMBLE_K = [1, 3, 5, 7, 11]  # number of votes
ENSEMBLE_SIGMAS = [0.20, 0.25, 0.30]  # test wider noise with voting
p12_results = {}

for model_name, model_cls in [("large_244K", MicroBrainLarge), ("small_63K", MicroBrainSmall)]:
    print(f"\n  {model_name}:", flush=True)
    model = model_cls()
    acc = train(model)
    print(f"  Trained: acc={acc:.4f}", flush=True)
    
    p12_results[model_name] = {}
    
    for sigma in ENSEMBLE_SIGMAS:
        p12_results[model_name][f"sigma_{sigma:.2f}"] = {}
        print(f"\n    σ={sigma}:", flush=True)
        
        for K in ENSEMBLE_K:
            t0 = time.time()
            c = sum(1 for _ in range(N) if run_l2_ensemble(model, sigma, K, arc_inst, cache, meta, ncol)['cleared'])
            rate = c / N * 100
            ci_lo, ci_hi = wilson_ci(c, N)
            elapsed = time.time() - t0
            
            # Fisher vs K=1
            if K == 1:
                p12_results[model_name][f"sigma_{sigma:.2f}"][f"K={K}"] = {
                    "c": c, "n": N, "rate": round(rate, 1), "ci": [ci_lo, ci_hi],
                    "fisher_p": None
                }
            else:
                bl_c = p12_results[model_name][f"sigma_{sigma:.2f}"]["K=1"]["c"]
                table = [[c, N-c], [bl_c, N-bl_c]]
                _, p = sp_stats.fisher_exact(table)
                p12_results[model_name][f"sigma_{sigma:.2f}"][f"K={K}"] = {
                    "c": c, "n": N, "rate": round(rate, 1), "ci": [ci_lo, ci_hi],
                    "fisher_p": round(p, 6)
                }
            
            sig = "★" if rate > 0 else " "
            print(f"      {sig} K={K:2d}: {rate:5.1f}% ({c}/{N}) [{elapsed:.0f}s]", flush=True)

p12_path = os.path.join(RESULTS_DIR, "phase12_ensemble.json")
with open(p12_path, "w") as f:
    json.dump(p12_results, f, indent=2)
print(f"\n  Saved: {p12_path}", flush=True)

# ============================================================
# Combined Plot
# ============================================================
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

# Phase 12 ensemble plot
fig, axes = plt.subplots(1, len(ENSEMBLE_SIGMAS), figsize=(6*len(ENSEMBLE_SIGMAS), 5))
if len(ENSEMBLE_SIGMAS) == 1: axes = [axes]

for col, sigma in enumerate(ENSEMBLE_SIGMAS):
    ax = axes[col]
    skey = f"sigma_{sigma:.2f}"
    
    for mname, color, marker in [("large_244K", '#FF5722', 'o'), ("small_63K", '#2196F3', 's')]:
        if mname not in p12_results or skey not in p12_results[mname]: continue
        ks = [int(k.split('=')[1]) for k in p12_results[mname][skey]]
        rates = [p12_results[mname][skey][f"K={k}"]["rate"] for k in ks]
        ci_lo = [p12_results[mname][skey][f"K={k}"]["ci"][0] for k in ks]
        ci_hi = [p12_results[mname][skey][f"K={k}"]["ci"][1] for k in ks]
        
        ax.errorbar(ks, rates,
                    yerr=[np.array(rates)-np.array(ci_lo), np.array(ci_hi)-np.array(rates)],
                    fmt=f'-{marker}', color=color, lw=2, ms=8, capsize=4, label=mname)
    
    ax.set_xlabel('Ensemble Size K', fontsize=11)
    ax.set_ylabel('L2 Clear Rate (%)', fontsize=11)
    ax.set_title(f'σ = {sigma}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ENSEMBLE_K)

fig.suptitle('Phase 12: Noise Ensemble Voting (N=100)', fontsize=14, fontweight='bold')
plt.tight_layout()
p12_plot = os.path.join(RESULTS_DIR, "phase12_ensemble.png")
plt.savefig(p12_plot, dpi=150, bbox_inches='tight')
shutil.copy(p12_plot, os.path.join(FIGURES_DIR, "arc_ensemble_voting.png"))

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  FINAL SUMMARY")
print(f"{'='*60}")

print(f"\n  Phase 11 (Bandit):")
for mname in p11_results:
    print(f"    {mname}:")
    for k, v in p11_results[mname].items():
        print(f"      {k}: {v['rate']}% ({v['c']}/{v['n']}) CI={v['ci']}")

print(f"\n  Phase 12 (Ensemble):")
for mname in p12_results:
    print(f"    {mname}:")
    for skey in p12_results[mname]:
        rates = {k: v['rate'] for k, v in p12_results[mname][skey].items()}
        best_k = max(rates, key=rates.get)
        print(f"      {skey}: best={best_k} ({rates[best_k]}%)")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 11+12 complete!", flush=True)
