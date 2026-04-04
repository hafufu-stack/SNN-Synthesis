"""
Phase 16c: Multi-Game Noisy Beam Search (OFFLINE + PARALLEL)
=============================================================
Uses multiprocessing to utilize all CPU cores.
12 cores / 24 threads on Ryzen AI 9 HX 375.
"""
import sys, os, json, time, random, gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# OFFLINE MODE - must be set before importing arc_agi
os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'environment_files')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

N_WORKERS = min(cpu_count() - 4, 20)  # Leave 4 threads for GPU runner + OS

print(f"[{time.strftime('%H:%M:%S')}] Phase 16c: Multi-Game PARALLEL Offline")
print(f"  Workers: {N_WORKERS} (of {cpu_count()} cores)")
print(f"  Results: {RESULTS_DIR}", flush=True)

# ============================================================
# Worker function (runs in separate process)
# ============================================================
def worker_trajectory_ensemble(args):
    """Worker: run K trajectories for one episode, return best levels_completed."""
    game_id, K, max_steps, seed = args
    
    # Must set env vars in each subprocess
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi
    from arcengine import GameAction
    
    ALL_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2,
                   GameAction.ACTION3, GameAction.ACTION4]
    
    rng = random.Random(seed)
    arcade = arc_agi.Arcade()
    
    best_lc = 0
    for k in range(K):
        try:
            env = arcade.make(game_id)
            obs = env.step(GameAction.RESET)
            max_lc = 0
            
            for step in range(max_steps):
                action = rng.choice(ALL_ACTIONS)
                try:
                    obs = env.step(action)
                    if obs.levels_completed > max_lc:
                        max_lc = obs.levels_completed
                    if obs.state.value in ('GAME_OVER', 'WIN'):
                        break
                except:
                    break
            
            if max_lc > best_lc:
                best_lc = max_lc
        except Exception:
            pass
    
    return best_lc

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import arc_agi
    from arcengine import GameAction
    
    GAMES = ['ls20', 'ft09', 'g50t', 'wa30', 'tr87', 'm0r0', 'sb26']
    N = 100
    KS = [1, 3, 5, 11, 21]
    MAX_STEPS = 300
    
    all_results = {}
    
    for gid in GAMES:
        print(f"\n{'='*60}")
        print(f"  {gid.upper()} (N={N}, parallel={N_WORKERS} workers)")
        print(f"{'='*60}", flush=True)
        
        game_results = {}
        
        for K in KS:
            t0 = time.time()
            
            # Create task list: N episodes, each with unique seed
            base_seed = hash(f"{gid}_{K}") & 0xFFFFFFFF
            tasks = [(gid, K, MAX_STEPS, base_seed + ep) for ep in range(N)]
            
            # Run in parallel
            level_counts = {}
            total_lc = 0
            
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = [executor.submit(worker_trajectory_ensemble, t) for t in tasks]
                for future in as_completed(futures):
                    try:
                        lc = future.result(timeout=120)
                        total_lc += lc
                        level_counts[lc] = level_counts.get(lc, 0) + 1
                    except Exception as e:
                        level_counts[0] = level_counts.get(0, 0) + 1
            
            avg = total_lc / N
            elapsed = time.time() - t0
            
            game_results[f"K={K}"] = {
                "avg_levels": round(avg, 3),
                "level_dist": {str(k): v for k, v in sorted(level_counts.items())},
                "time_s": round(elapsed, 1)
            }
            
            dist_str = " ".join(f"L{k}:{v}" for k, v in sorted(level_counts.items()))
            any_clear = sum(v for k, v in level_counts.items() if k > 0)
            star = "★" if any_clear > 0 else " "
            print(f"  {star} K={K:3d}: avg={avg:.3f}  clear={any_clear:3d}/100  "
                  f"{dist_str}  [{elapsed:.0f}s]", flush=True)
        
        all_results[gid] = game_results
        
        # Save incrementally (in case of crash)
        out_path = os.path.join(RESULTS_DIR, "phase16c_multigame_parallel.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    # ============================================================
    # Visualization
    # ============================================================
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import shutil
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, gid in enumerate(GAMES):
        ax = axes[idx]
        if gid not in all_results:
            ax.set_title(f'{gid} (skipped)', fontsize=11)
            continue
        
        ks = [int(k.split('=')[1]) for k in all_results[gid]]
        any_clears = []
        for k in ks:
            dist = all_results[gid][f"K={k}"]["level_dist"]
            ac = sum(v for key, v in dist.items() if int(key) > 0)
            any_clears.append(ac)
        
        colors = ['#4CAF50' if c > 0 else '#ccc' for c in any_clears]
        ax.bar(range(len(ks)), any_clears, color=colors, alpha=0.8)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel('K (trajectories)')
        ax.set_ylabel('Any level cleared (/100)')
        ax.set_title(f'{gid.upper()}', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
    
    for i in range(len(GAMES), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Phase 16c: Random Trajectory Ensemble × 7 ARC-AGI-3 Games\n'
                 f'(Offline, N={N}, {N_WORKERS} parallel workers)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "phase16c_multigame.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    shutil.copy(plot_path, os.path.join(FIGURES_DIR, "arc_multigame_ensemble.png"))
    print(f"\n  Plot: {plot_path}", flush=True)
    plt.close('all')
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for gid in GAMES:
        if gid not in all_results:
            continue
        k1 = all_results[gid].get("K=1", {})
        k21 = all_results[gid].get("K=21", {})
        k1_c = sum(int(v) for k, v in k1.get("level_dist", {}).items() if int(k) > 0)
        k21_c = sum(int(v) for k, v in k21.get("level_dist", {}).items() if int(k) > 0)
        boost = f"{k21_c/max(1,k1_c):.1f}x" if k1_c > 0 else ("inf" if k21_c > 0 else "n/a")
        print(f"  {gid:6s}: K=1={k1_c:3d}/100  K=21={k21_c:3d}/100  boost={boost}")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 16c complete!", flush=True)
