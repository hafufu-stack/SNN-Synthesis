"""
Phase 16b: Multi-Game Noisy Beam Search (OFFLINE)
==================================================
Tests trajectory ensemble on multiple ARC-AGI-3 games
WITHOUT internet connection.

Games tested:
  - ls20 (keyboard, 4-direction grid navigation)
  - ft09 (no tag, unknown mechanics)
  - g50t (keyboard)
  - wa30 (keyboard)
  - tr87 (keyboard)
  - m0r0 (keyboard_click)
  - sb26 (keyboard_click)
"""
import sys, os, json, time, random, gc
import numpy as np

# OFFLINE MODE
os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'environment_files')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

import arc_agi
from arcengine import GameAction

ALL_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2,
               GameAction.ACTION3, GameAction.ACTION4]

print(f"[{time.strftime('%H:%M:%S')}] Phase 16b: Multi-Game Offline Exploration")
print(f"  Mode: OFFLINE (no internet required)")
print(f"  Results: {RESULTS_DIR}", flush=True)

arcade = arc_agi.Arcade()

# All downloaded games
GAMES = ['ls20', 'ft09', 'g50t', 'wa30', 'tr87', 'm0r0', 'sb26']

# ============================================================
# Step 1: Quick exploration of each game
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 1: Quick exploration of all {len(GAMES)} games")
print(f"{'='*60}", flush=True)

game_profiles = {}

for gid in GAMES:
    print(f"\n  --- {gid} ---", flush=True)
    t0 = time.time()
    try:
        env = arcade.make(gid)
        obs = env.step(GameAction.RESET)
        game = env._game
        
        # State attributes
        state_nums = {}
        for attr in dir(game):
            if not attr.startswith('_'):
                try:
                    val = getattr(game, attr)
                    if isinstance(val, (int, float)):
                        state_nums[attr] = val
                except:
                    pass
        
        # Random play: 50 steps
        max_lc = 0
        for step in range(50):
            action = random.choice(ALL_ACTIONS)
            try:
                obs = env.step(action)
                if obs.levels_completed > max_lc:
                    max_lc = obs.levels_completed
                if obs.state.value in ('GAME_OVER', 'WIN'):
                    break
            except:
                break
        
        profile = {
            "class": type(game).__name__,
            "n_state_attrs": len(state_nums),
            "random_50_max_lc": max_lc,
            "final_state": obs.state.value,
            "time_s": round(time.time() - t0, 1)
        }
        game_profiles[gid] = profile
        print(f"    Class: {profile['class']}, state_attrs: {profile['n_state_attrs']}, "
              f"random_50_lc: {max_lc}, state: {profile['final_state']}")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        game_profiles[gid] = {"error": str(e)}

# ============================================================
# Step 2: Random Trajectory Ensemble on ALL games
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 2: Random Trajectory Ensemble (all games)")
print(f"{'='*60}", flush=True)

def run_random_trajectory_offline(game_id, arc_inst, max_steps=300):
    """Single random trajectory, returns levels_completed."""
    env = arc_inst.make(game_id)
    obs = env.step(GameAction.RESET)
    max_lc = 0
    
    for step in range(max_steps):
        action = random.choice(ALL_ACTIONS)
        try:
            obs = env.step(action)
            if obs.levels_completed > max_lc:
                max_lc = obs.levels_completed
            if obs.state.value in ('GAME_OVER', 'WIN'):
                break
        except:
            break
    
    return max_lc

def run_trajectory_ensemble_offline(game_id, arc_inst, K, max_steps=300):
    """K independent random trajectories, return best levels_completed."""
    best = 0
    for _ in range(K):
        lc = run_random_trajectory_offline(game_id, arc_inst, max_steps)
        if lc > best:
            best = lc
    return best

N = 100
KS = [1, 3, 5, 11, 21]

all_results = {}

for gid in GAMES:
    if game_profiles.get(gid, {}).get("error"):
        print(f"\n  Skipping {gid} (error in Step 1)")
        continue
    
    print(f"\n  --- {gid} (N={N}) ---", flush=True)
    game_results = {}
    
    for K in KS:
        t0 = time.time()
        level_counts = {}
        total_lc = 0
        
        for ep in range(N):
            lc = run_trajectory_ensemble_offline(gid, arcade, K, max_steps=300)
            total_lc += lc
            level_counts[lc] = level_counts.get(lc, 0) + 1
        
        avg = total_lc / N
        elapsed = time.time() - t0
        
        game_results[f"K={K}"] = {
            "avg_levels": round(avg, 3),
            "level_dist": {str(k): v for k, v in sorted(level_counts.items())},
            "time_s": round(elapsed, 1)
        }
        
        # Format distribution
        dist_str = " ".join(f"L{k}:{v}" for k, v in sorted(level_counts.items()))
        any_clear = sum(v for k, v in level_counts.items() if k > 0)
        star = "★" if any_clear > 0 else " "
        print(f"    {star} K={K:3d}: avg={avg:.3f}  any_clear={any_clear:3d}/100  {dist_str}  [{elapsed:.0f}s]",
              flush=True)
    
    all_results[gid] = game_results

# Save all results
out_path = os.path.join(RESULTS_DIR, "phase16b_multigame_offline.json")
with open(out_path, "w") as f:
    json.dump({"profiles": game_profiles, "ensemble": all_results}, f, indent=2)
print(f"\n  Saved: {out_path}", flush=True)

# ============================================================
# Step 3: Visualization
# ============================================================
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, gid in enumerate(GAMES):
    ax = axes[idx]
    if gid not in all_results:
        ax.set_title(f'{gid} (ERROR)', fontsize=11)
        continue
    
    ks = [int(k.split('=')[1]) for k in all_results[gid]]
    avgs = [all_results[gid][f"K={k}"]["avg_levels"] for k in ks]
    any_clears = []
    for k in ks:
        dist = all_results[gid][f"K={k}"]["level_dist"]
        ac = sum(v for key, v in dist.items() if int(key) > 0)
        any_clears.append(ac)
    
    ax.bar(range(len(ks)), any_clears, color='#4CAF50', alpha=0.7)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('K (trajectories)')
    ax.set_ylabel('Episodes with any clear (/100)')
    ax.set_title(f'{gid.upper()} (random policy)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

# Hide last subplot if odd number
if len(GAMES) < len(axes):
    for i in range(len(GAMES), len(axes)):
        axes[i].set_visible(False)

fig.suptitle('Phase 16b: Random Trajectory Ensemble Across ARC-AGI-3 Games\n'
             '(Offline, N=100, no learned policy)', fontsize=14, fontweight='bold')
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "phase16b_multigame.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"  Plot: {plot_path}", flush=True)
plt.close('all')

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")

for gid in GAMES:
    if gid not in all_results:
        print(f"  {gid}: SKIPPED")
        continue
    k1 = all_results[gid].get("K=1", {})
    k21 = all_results[gid].get("K=21", {})
    
    k1_clear = sum(int(v) for k, v in k1.get("level_dist", {}).items() if int(k) > 0)
    k21_clear = sum(int(v) for k, v in k21.get("level_dist", {}).items() if int(k) > 0)
    
    boost = f"{k21_clear/max(1,k1_clear):.1f}x" if k1_clear > 0 else "inf" if k21_clear > 0 else "0"
    print(f"  {gid:6s}: K=1 clear={k1_clear:3d}/100, K=21 clear={k21_clear:3d}/100, boost={boost}")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 16b complete!", flush=True)
