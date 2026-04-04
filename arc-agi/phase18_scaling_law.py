"""
Phase 18: Extended K Scaling Law + Cross-Game Comparison
=========================================================
1. Push K to 51, 101, 201 on m0r0 and tr87
2. Verify scaling saturation
3. Generate comparative figures
"""
import sys, os, json, time, random, gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'environment_files')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

N_WORKERS = min(cpu_count() - 4, 20)

print(f"[{time.strftime('%H:%M:%S')}] Phase 18: Extended K Scaling + Comparison")
print(f"  Workers: {N_WORKERS}", flush=True)

# ============================================================
# Worker
# ============================================================
def worker_trajectory(args):
    """Run K random trajectories, return best levels_completed."""
    game_id, K, max_steps, seed = args
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi, random
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    
    rng = random.Random(seed)
    arcade = arc_agi.Arcade()
    best = 0
    
    for k in range(K):
        try:
            env = arcade.make(game_id)
            obs = env.step(GameAction.RESET)
            lc = 0
            for step in range(max_steps):
                obs = env.step(rng.choice(ALL_A))
                if obs.levels_completed > lc:
                    lc = obs.levels_completed
                if obs.state.value in ('GAME_OVER', 'WIN'):
                    break
            if lc > best:
                best = lc
        except:
            pass
    return best

# ============================================================
# Step 1: Extended K sweep on m0r0 and tr87
# ============================================================
if __name__ == '__main__':
    GAMES = ['m0r0', 'tr87', 'ls20']
    N = 200  # More episodes for better statistics
    KS_EXTENDED = [1, 3, 5, 11, 21, 51, 101]
    
    all_results = {}
    
    for gid in GAMES:
        print(f"\n{'='*60}")
        print(f"  {gid.upper()} Extended K Sweep (N={N})")
        print(f"{'='*60}", flush=True)
        
        game_results = {}
        
        for K in KS_EXTENDED:
            t0 = time.time()
            base_seed = hash(f"{gid}_ext_{K}") & 0xFFFFFFFF
            tasks = [(gid, K, 300, base_seed + ep) for ep in range(N)]
            
            level_counts = {}
            total_lc = 0
            max_lc_seen = 0
            
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                for lc in executor.map(worker_trajectory, tasks):
                    total_lc += lc
                    level_counts[lc] = level_counts.get(lc, 0) + 1
                    if lc > max_lc_seen:
                        max_lc_seen = lc
            
            avg = total_lc / N
            any_clear = sum(v for k, v in level_counts.items() if k > 0)
            elapsed = time.time() - t0
            
            # Wilson CI
            p_hat = any_clear / N
            z = 1.96
            denom = 1 + z**2/N
            center = (p_hat + z**2/(2*N)) / denom
            half_w = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*N)) / N) / denom
            ci_lo = max(0, center - half_w) * 100
            ci_hi = min(1, center + half_w) * 100
            
            game_results[f"K={K}"] = {
                "clear_rate": round(any_clear/N*100, 1),
                "clear_count": any_clear,
                "n": N,
                "avg_levels": round(avg, 3),
                "max_levels": max_lc_seen,
                "level_dist": {str(k): v for k, v in sorted(level_counts.items())},
                "ci_lower": round(ci_lo, 1),
                "ci_upper": round(ci_hi, 1),
                "time_s": round(elapsed, 1)
            }
            
            dist_str = " ".join(f"L{k}:{v}" for k, v in sorted(level_counts.items()) if k > 0)
            star = "★" if any_clear > 0 else " "
            print(f"  {star} K={K:4d}: {any_clear:3d}/{N} = {any_clear/N*100:5.1f}%  "
                  f"CI=[{ci_lo:.1f},{ci_hi:.1f}]  max_L={max_lc_seen}  "
                  f"{dist_str}  [{elapsed:.0f}s]", flush=True)
        
        all_results[gid] = game_results
        
        # Save incrementally
        out_path = os.path.join(RESULTS_DIR, "phase18_extended_k.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
    
    # ============================================================
    # Step 2: Generate comparison figures
    # ============================================================
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Figure 1: K scaling curves for all games
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'m0r0': '#E91E63', 'tr87': '#2196F3', 'ls20': '#4CAF50'}
    
    for gid in GAMES:
        if gid not in all_results:
            continue
        ks = [int(k.split('=')[1]) for k in all_results[gid]]
        rates = [all_results[gid][f"K={k}"]["clear_rate"] for k in ks]
        ci_lo = [all_results[gid][f"K={k}"]["ci_lower"] for k in ks]
        ci_hi = [all_results[gid][f"K={k}"]["ci_upper"] for k in ks]
        
        ax.plot(ks, rates, 'o-', color=colors[gid], linewidth=2, 
                markersize=8, label=f'{gid.upper()} (random)')
        ax.fill_between(ks, ci_lo, ci_hi, color=colors[gid], alpha=0.15)
    
    # Add LS20 CNN+Noise data from Phase 14
    p14_path = os.path.join(RESULTS_DIR, "phase14_trajectory_ensemble.json")
    if os.path.exists(p14_path):
        p14 = json.load(open(p14_path))
        if "L2" in p14:
            # Best sigma (0.25) for L2
            cnn_ks = [1, 3, 5, 7, 11]
            cnn_rates = []
            for k in cnn_ks:
                if f"K={k}" in p14["L2"].get("sigma_0.25", {}):
                    cnn_rates.append(p14["L2"]["sigma_0.25"][f"K={k}"]["rate"])
            if cnn_rates:
                ax.plot(cnn_ks, cnn_rates, 's--', color='#FF9800', linewidth=2,
                        markersize=8, label='LS20 CNN+Noise σ=0.25 (L2)')
    
    ax.set_xlabel('K (number of parallel trajectories)', fontsize=12)
    ax.set_ylabel('Clear Rate (%)', fontsize=12)
    ax.set_title('Trajectory Ensemble Scaling Law Across ARC-AGI-3 Games\n'
                 'Random Policy vs CNN+Noise Policy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 5, 11, 21, 51, 101])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plot1 = os.path.join(RESULTS_DIR, "phase18_scaling_law.png")
    plt.savefig(plot1, dpi=150, bbox_inches='tight')
    print(f"\n  Plot 1: {plot1}", flush=True)
    plt.close()
    
    # Figure 2: m0r0 detailed level distribution
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    m0 = all_results.get('m0r0', {})
    ks = [int(k.split('=')[1]) for k in m0]
    
    # Stack bar for levels cleared
    max_level = max(int(k) for res in m0.values() for k in res['level_dist'].keys())
    
    bottom = np.zeros(len(ks))
    cmap = plt.cm.viridis
    for lev in range(max_level + 1):
        vals = [m0[f"K={k}"]["level_dist"].get(str(lev), 0) for k in ks]
        color = '#ddd' if lev == 0 else cmap(lev / max(1, max_level))
        ax2.bar(range(len(ks)), vals, bottom=bottom, color=color, 
                label=f'L{lev}' if lev > 0 else 'No clear', alpha=0.8)
        bottom += np.array(vals)
    
    ax2.set_xticks(range(len(ks)))
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel('K (trajectories)')
    ax2.set_ylabel(f'Episodes (N={N})')
    ax2.set_title('M0R0: Level Distribution by K\n(Random Policy Trajectory Ensemble)',
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plot2 = os.path.join(RESULTS_DIR, "phase18_m0r0_levels.png")
    plt.savefig(plot2, dpi=150, bbox_inches='tight')
    print(f"  Plot 2: {plot2}", flush=True)
    plt.close()
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 18 SUMMARY: K Scaling Law")
    print(f"{'='*60}")
    for gid in GAMES:
        if gid not in all_results:
            continue
        k1 = all_results[gid].get("K=1", {})
        k101 = all_results[gid].get("K=101", {})
        r1 = k1.get("clear_rate", 0)
        r101 = k101.get("clear_rate", 0)
        boost = f"{r101/max(0.1,r1):.1f}x" if r1 > 0 else ("∞" if r101 > 0 else "n/a")
        print(f"  {gid:6s}: K=1={r1:5.1f}%  K=101={r101:5.1f}%  boost={boost}")
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 18 complete!", flush=True)
