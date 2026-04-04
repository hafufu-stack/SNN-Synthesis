"""
Phase 16: Multi-Game Exploration & Noisy Beam Search Extension
===============================================================
Step 1: Discover all available ARC-AGI-3 games
Step 2: Explore ft09 (already downloaded) 
Step 3: Build a game-agnostic random-policy baseline
Step 4: Test Noisy Beam Search with random+noise approach
"""
import sys, os, json, time, random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

import arc_agi
from arcengine import GameAction

ALL_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, 
               GameAction.ACTION3, GameAction.ACTION4]

print(f"[{time.strftime('%H:%M:%S')}] Phase 16: Multi-Game Exploration")

# ============================================================
# Step 1: List all available games
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 1: Discovering all ARC-AGI-3 games")
print(f"{'='*60}", flush=True)

arc = arc_agi.Arcade()

# Try to get game list from API
try:
    envs = arc._client.get_environments()
    print(f"  Found {len(envs)} environments:")
    for e in envs:
        eid = e.get('id', e.get('environment_id', str(e)))
        print(f"    - {eid}")
except Exception as ex:
    print(f"  Could not list: {ex}")
    envs = []

# Test known games
GAMES_TO_TRY = ["ls20", "ft09"]
game_info = {}

for gid in GAMES_TO_TRY:
    print(f"\n  --- Exploring {gid} ---", flush=True)
    try:
        env = arc.make(gid)
        game = env._game
        obs = env.step(GameAction.RESET)
        
        # Discover game structure
        info = {
            "game_id": gid,
            "game_class": type(game).__name__,
            "game_attrs": [a for a in dir(game) if not a.startswith('_')],
        }
        
        # Observation structure
        info["obs_type"] = type(obs).__name__
        info["obs_attrs"] = {k: type(v).__name__ for k, v in obs.__dict__.items() 
                            if not k.startswith('_')}
        
        # Try taking actions and observe state changes
        level_clears = 0
        total_steps = 0
        action_results = {a.name: {"count": 0, "state_changes": 0} for a in ALL_ACTIONS}
        
        for step in range(200):
            action = random.choice(ALL_ACTIONS)
            try:
                prev_lc = obs.levels_completed
                obs = env.step(action)
                total_steps += 1
                action_results[action.name]["count"] += 1
                if obs.levels_completed > prev_lc:
                    level_clears += 1
                    action_results[action.name]["state_changes"] += 1
                if obs.state.value == 'GAME_OVER':
                    break
                if obs.state.value == 'WIN':
                    level_clears = obs.levels_completed
                    break
            except Exception as e:
                info.setdefault("errors", []).append(str(e))
                break
        
        info["random_200_steps"] = {
            "steps_taken": total_steps,
            "levels_cleared": level_clears,
            "final_state": obs.state.value,
            "final_levels_completed": obs.levels_completed,
        }
        
        # Grid/pixel info
        if hasattr(obs, 'pixels'):
            info["pixel_shape"] = list(obs.pixels.shape) if hasattr(obs.pixels, 'shape') else "unknown"
        
        game_info[gid] = info
        print(f"    Class: {info['game_class']}")
        print(f"    Random 200 steps: {level_clears} levels cleared, state={obs.state.value}")
        print(f"    Obs attrs: {list(info['obs_attrs'].keys())}")
        
        # Check game internal state
        game_state_attrs = []
        for attr in dir(game):
            if not attr.startswith('_'):
                try:
                    val = getattr(game, attr)
                    if isinstance(val, (int, float, str, bool)):
                        game_state_attrs.append(f"{attr}={val}")
                except:
                    pass
        print(f"    Game state: {game_state_attrs[:15]}")
        
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        game_info[gid] = {"error": str(e)}

# ============================================================
# Step 2: Game-agnostic Noisy Beam Search (Random Policy + Noise)
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 2: Game-Agnostic Trajectory Ensemble")  
print(f"  (Random policy with weighted action sampling)")
print(f"{'='*60}", flush=True)

def run_random_trajectory(game_id, arc_inst, max_steps=300, action_weights=None):
    """Run a single random trajectory. Returns levels cleared."""
    env = arc_inst.make(game_id)
    obs = env.step(GameAction.RESET)
    
    for step in range(max_steps):
        if action_weights is not None:
            action = random.choices(ALL_ACTIONS, weights=action_weights, k=1)[0]
        else:
            action = random.choice(ALL_ACTIONS)
        obs = env.step(action)
        if obs.state.value in ('GAME_OVER', 'WIN'):
            break
    
    return {
        'levels_completed': obs.levels_completed,
        'steps': step + 1,
        'state': obs.state.value
    }

def run_trajectory_ensemble_generic(game_id, arc_inst, K, max_steps=300, action_weights=None):
    """Run K independent random trajectories, return best result."""
    best_levels = 0
    best_steps = max_steps
    
    for k in range(K):
        r = run_random_trajectory(game_id, arc_inst, max_steps, action_weights)
        if r['levels_completed'] > best_levels or \
           (r['levels_completed'] == best_levels and r['steps'] < best_steps):
            best_levels = r['levels_completed']
            best_steps = r['steps']
    
    return {'levels_completed': best_levels, 'steps': best_steps}

# Test trajectory ensemble on available games
N_EPISODES = 100
KS = [1, 3, 5, 11, 21, 51]

for gid in GAMES_TO_TRY:
    if game_info.get(gid, {}).get("error"):
        continue
    
    print(f"\n  --- {gid}: Random Trajectory Ensemble (N={N_EPISODES}) ---", flush=True)
    results_game = {}
    
    for K in KS:
        t0 = time.time()
        total_levels = 0
        level_dist = {}
        
        for ep in range(N_EPISODES):
            r = run_trajectory_ensemble_generic(gid, arc, K, max_steps=300)
            lc = r['levels_completed']
            total_levels += lc
            level_dist[lc] = level_dist.get(lc, 0) + 1
        
        avg_levels = total_levels / N_EPISODES
        elapsed = time.time() - t0
        
        results_game[f"K={K}"] = {
            "avg_levels": round(avg_levels, 2),
            "level_distribution": level_dist,
            "time_s": round(elapsed, 1)
        }
        
        dist_str = " ".join(f"L{k}:{v}" for k, v in sorted(level_dist.items()))
        print(f"    K={K:3d}: avg={avg_levels:.2f} levels  {dist_str}  [{elapsed:.0f}s]", flush=True)
    
    # Save
    out_path = os.path.join(RESULTS_DIR, f"phase16_{gid}_random_ensemble.json")
    with open(out_path, "w") as f:
        json.dump(results_game, f, indent=2)
    print(f"    Saved: {out_path}", flush=True)

# ============================================================
# Step 3: For LS20, compare CNN+Noise vs Random Trajectory
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 3: LS20 CNN+Noise vs Random Trajectory Comparison")
print(f"{'='*60}", flush=True)

# LS20 CNN results are already in Phase 14
# Just need to load and compare
p14_path = os.path.join(RESULTS_DIR, "phase14_trajectory_ensemble.json")
if os.path.exists(p14_path):
    p14 = json.load(open(p14_path))
    print(f"  LS20 CNN Trajectory Ensemble (Phase 14):")
    for lname in ['L2', 'L3', 'L4']:
        if lname in p14:
            best_rate, best_cfg = 0, ""
            for skey in p14[lname]:
                for kkey in p14[lname][skey]:
                    r = p14[lname][skey][kkey]["rate"]
                    if r > best_rate:
                        best_rate = r
                        best_cfg = f"{skey} {kkey}"
            print(f"    {lname}: best={best_rate}% ({best_cfg})")

# Load random baseline for comparison
rand_path = os.path.join(RESULTS_DIR, "phase16_ls20_random_ensemble.json")
if os.path.exists(rand_path):
    rand_data = json.load(open(rand_path))
    print(f"\n  LS20 Random Trajectory Ensemble (Phase 16):")
    for K in KS:
        kkey = f"K={K}"
        if kkey in rand_data:
            print(f"    {kkey}: avg={rand_data[kkey]['avg_levels']:.2f} levels "
                  f"dist={rand_data[kkey]['level_distribution']}")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 16 complete!", flush=True)
