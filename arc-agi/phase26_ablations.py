"""
Phase 26: Three Final Ablation Studies
========================================
26a: LS20 Miracle Rate Reduction (Gemini proposal)
    - Artificially halve LS20's miracles → does ExIt still work?
    - Tests "miracle rate ≥ 20% is NECESSARY"

26b: Test-Time Compute Scaling Law (Deep Think proposal)
    - Plot K × avg_steps vs clear rate from existing data
    - No new experiments needed, just analysis

26c: M0R0 ExIt Convergence Speed
    - K=10 vs K=100 bootstrap → how fast does ExIt converge on easy game?
"""
import sys, os, json, time, random, gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'environment_files')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_WORKERS = min(cpu_count() - 4, 16)

import torch
import torch.nn as nn
import torch.optim as optim

import logging
logging.disable(logging.CRITICAL)


# ============================================================
# Shared: state extraction + model + worker
# ============================================================
def extract_game_state(game):
    state = []
    for attr in sorted(dir(game)):
        if attr.startswith('_') or attr in ('level_index', 'win_score'):
            continue
        try:
            val = getattr(game, attr)
            if callable(val): continue
            if isinstance(val, (int, float, bool)):
                state.append(float(val))
            elif isinstance(val, (list, tuple)):
                for v in val[:4]:
                    if isinstance(v, (int, float, bool)):
                        state.append(float(v))
        except:
            pass
    return np.array(state, dtype=np.float32)


class StateBrain(nn.Module):
    def __init__(self, state_dim, n_actions=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x, noise_sigma=0.0):
        h = self.net[0](x); h = self.net[1](h); h = self.net[2](h)
        if noise_sigma > 0:
            h = h + torch.randn_like(h) * noise_sigma
        h = self.net[3](h); h = self.net[4](h); h = self.net[5](h)
        return h


def best_of_k_worker(args):
    game_id, K, max_steps, seed, model_path, noise_sigma = args
    
    import os, random, torch, torch.nn as nn, numpy as np, logging
    logging.disable(logging.CRITICAL)
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    rng = random.Random(seed)
    
    model = None
    x_mean = x_std = None
    if model_path and os.path.exists(model_path):
        try:
            data = torch.load(model_path, weights_only=True)
            sd = data['state_dim']
            x_mean = data.get('x_mean'); x_std = data.get('x_std')
            class SB(nn.Module):
                def __init__(self, sd, na=4, h=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(sd, h), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(h, na))
                def forward(self, x, ns=0.0):
                    h = self.net[0](x); h = self.net[1](h); h = self.net[2](h)
                    if ns > 0: h = h + torch.randn_like(h) * ns
                    h = self.net[3](h); h = self.net[4](h); h = self.net[5](h)
                    return h
            model = SB(sd=sd); model.load_state_dict(data['model']); model.eval()
        except: model = None
    
    best = None
    for k in range(K):
        try:
            arcade = arc_agi.Arcade()
            env = arcade.make(game_id)
            obs = env.step(GameAction.RESET)
            game = env._game
        except: continue
        
        states, actions, max_lc = [], [], 0
        for step in range(max_steps):
            try:
                s = extract_game_state(game)
                states.append(s.tolist())
            except:
                states.append([0.0] * 28)
            
            if model is not None:
                try:
                    x = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0)
                    if x_mean is not None: x = (x - x_mean) / x_std
                    with torch.no_grad():
                        logits = model(x, ns=noise_sigma)
                        probs = torch.softmax(logits, dim=1)
                        action_idx = torch.multinomial(probs, 1).item()
                    action = ALL_A[action_idx]
                except: action = rng.choice(ALL_A)
            else:
                action = rng.choice(ALL_A)
            
            actions.append(ALL_A.index(action))
            try:
                obs = env.step(action)
                if obs.levels_completed > max_lc:
                    max_lc = obs.levels_completed
                if obs.state.value in ('GAME_OVER', 'WIN'): break
            except: break
        
        result = {'states': states, 'actions': actions,
                  'levels_cleared': max_lc, 'n_steps': len(actions)}
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


def run_exit(game_id, miracles, state_dim, n_iters, collect_K, n_collect, n_eval, tag):
    """Run ExIt loop and return results."""
    cumulative = list(miracles)
    model_path = None
    results = []
    
    for iteration in range(n_iters):
        print(f"      ExIt iter {iteration+1}/{n_iters}: {len(cumulative)} miracles", flush=True)
        
        all_s, all_a = [], []
        for m in cumulative:
            for s, a in zip(m['states'], m['actions']):
                all_s.append(s); all_a.append(a)
        
        X = torch.tensor(all_s, dtype=torch.float32)
        Y = torch.tensor(all_a, dtype=torch.long)
        x_mean = X.mean(0, keepdim=True)
        x_std = X.std(0, keepdim=True).clamp(min=1e-6)
        X_n = (X - x_mean) / x_std
        
        model = StateBrain(state_dim=state_dim, n_actions=4, hidden=256)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
        
        model.train()
        best_acc = 0
        for ep in range(500):
            perm = torch.randperm(len(X_n))
            batch = perm[:min(256, len(X_n))]
            loss = crit(model(X_n[batch], 0.0), Y[batch])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (ep+1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    acc = (model(X_n, 0.0).argmax(1) == Y).float().mean().item()
                model.train()
                best_acc = max(best_acc, acc)
        
        mp = os.path.join(SCRIPT_DIR, "data", f"{tag}_iter{iteration+1}.pt")
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        torch.save({'model': model.state_dict(), 'state_dim': state_dim,
                    'x_mean': x_mean, 'x_std': x_std}, mp)
        model_path = mp
        
        # Collect more
        K_c = max(10, collect_K // (2 ** iteration))
        tasks = [(game_id, K_c, 200, (iteration+1)*10000000+ep, model_path, 0.1)
                 for ep in range(n_collect)]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            new = list(ex.map(best_of_k_worker, tasks))
        new_m = [r for r in new if r and r['levels_cleared'] > 0]
        cumulative.extend(new_m)
        
        # Eval
        iter_res = {"iteration": iteration+1, "miracles": len(cumulative),
                    "train_acc": best_acc, "configs": {}}
        
        for name, mp_e, K_e, sig in [("Random K=11", None, 11, 0.0),
                                      (f"CNN K=1", model_path, 1, 0.0),
                                      (f"CNN K=11 σ=0.1", model_path, 11, 0.1)]:
            tasks = [(game_id, K_e, 200, 99000000+iteration*10000+ep, mp_e, sig)
                     for ep in range(n_eval)]
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                ev = list(ex.map(best_of_k_worker, tasks))
            clears = sum(1 for r in ev if r and r['levels_cleared'] > 0)
            rate = clears / n_eval * 100
            iter_res["configs"][name] = {"clears": clears, "rate": rate}
            bar = "█" * int(rate / 2)
            print(f"        {name:30s}: {clears:3d}/{n_eval} = {rate:5.1f}%  {bar}", flush=True)
        
        results.append(iter_res)
        gc.collect()
    
    return results


if __name__ == '__main__':
    import arc_agi
    from arcengine import GameAction
    
    all_output = {}
    t_total = time.time()
    
    print(f"[{time.strftime('%H:%M:%S')}] Phase 26: Three Final Ablation Studies")
    print(f"  Workers: {N_WORKERS}")
    print(f"{'='*70}", flush=True)
    
    # ==============================================================
    # 26a: LS20 Miracle Rate Reduction
    # ==============================================================
    print(f"\n{'='*70}")
    print(f"  26a: LS20 Miracle Rate Reduction")
    print(f"  Keep only 50% of miracles → test if ExIt still works")
    print(f"{'='*70}", flush=True)
    
    t0 = time.time()
    
    # Collect LS20 miracles with K=100 (normal rate ~21%)
    N_BOOT = 400
    print(f"  Collecting {N_BOOT} LS20 episodes at K=100...", flush=True)
    tasks = [("ls20", 100, 200, 26100000+ep, None, 0.0) for ep in range(N_BOOT)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        boot = list(ex.map(best_of_k_worker, tasks))
    
    all_miracles = [r for r in boot if r and r['levels_cleared'] > 0]
    full_rate = len(all_miracles) / N_BOOT * 100
    print(f"  Full miracles: {len(all_miracles)}/{N_BOOT} ({full_rate:.1f}%)")
    
    # Determine state dim
    state_dim_ls20 = len(all_miracles[0]['states'][0])
    print(f"  State dim: {state_dim_ls20}")
    
    # Condition A: Full miracles (control)
    print(f"\n  --- Condition A: Full miracles ({len(all_miracles)}) ---", flush=True)
    res_full = run_exit("ls20", all_miracles, state_dim_ls20,
                        n_iters=3, collect_K=50, n_collect=100, n_eval=100,
                        tag="26a_full")
    
    # Condition B: 50% miracles
    half = all_miracles[:len(all_miracles)//2]
    half_rate = len(half) / N_BOOT * 100
    print(f"\n  --- Condition B: Half miracles ({len(half)}, effective rate {half_rate:.1f}%) ---", flush=True)
    res_half = run_exit("ls20", half, state_dim_ls20,
                        n_iters=3, collect_K=50, n_collect=100, n_eval=100,
                        tag="26a_half")
    
    # Condition C: 25% miracles (below threshold)
    quarter = all_miracles[:len(all_miracles)//4]
    quarter_rate = len(quarter) / N_BOOT * 100
    print(f"\n  --- Condition C: Quarter miracles ({len(quarter)}, effective rate {quarter_rate:.1f}%) ---", flush=True)
    res_quarter = run_exit("ls20", quarter, state_dim_ls20,
                           n_iters=3, collect_K=50, n_collect=100, n_eval=100,
                           tag="26a_quarter")
    
    all_output["26a"] = {
        "full_miracles": len(all_miracles), "full_rate": full_rate,
        "half_miracles": len(half), "half_rate": half_rate,
        "quarter_miracles": len(quarter), "quarter_rate": quarter_rate,
        "results_full": res_full, "results_half": res_half,
        "results_quarter": res_quarter,
        "elapsed_s": time.time() - t0,
    }
    print(f"  26a done in {time.time()-t0:.0f}s", flush=True)
    
    # ==============================================================
    # 26b: Test-Time Compute Scaling Law
    # ==============================================================
    print(f"\n{'='*70}")
    print(f"  26b: Test-Time Compute Scaling Law")
    print(f"  Plot from existing Phase 18 + 14 data")
    print(f"{'='*70}", flush=True)
    
    t0 = time.time()
    
    # Load existing data
    scaling_data = {}
    
    # Phase 18 data (random policy, multiple games)
    p18_path = os.path.join(RESULTS_DIR, "phase18_random_multigame.json")
    if os.path.exists(p18_path):
        with open(p18_path) as f:
            p18 = json.load(f)
        scaling_data["phase18"] = p18
        print(f"  Loaded Phase 18 data")
    
    # Phase 14 data (Oracle CNN + trajectory ensemble)
    p14_path = os.path.join(RESULTS_DIR, "phase14_trajectory_ensemble.json")
    if os.path.exists(p14_path):
        with open(p14_path) as f:
            p14 = json.load(f)
        scaling_data["phase14"] = p14
        print(f"  Loaded Phase 14 data")
    
    # Phase 16c data
    p16_path = os.path.join(RESULTS_DIR, "phase16c_multigame_ensemble.json")
    if os.path.exists(p16_path):
        with open(p16_path) as f:
            p16 = json.load(f)
        scaling_data["phase16c"] = p16
        print(f"  Loaded Phase 16c data")
    
    # Phase 24 data (activation energy)
    p24_path = os.path.join(RESULTS_DIR, "phase24_activation_energy.json")
    if os.path.exists(p24_path):
        with open(p24_path) as f:
            p24 = json.load(f)
        scaling_data["phase24"] = p24
        print(f"  Loaded Phase 24 data")
    
    all_output["26b"] = {"data_sources": list(scaling_data.keys()),
                         "elapsed_s": time.time() - t0}
    print(f"  26b data collected in {time.time()-t0:.0f}s", flush=True)
    
    # ==============================================================
    # 26c: M0R0 ExIt Convergence Speed
    # ==============================================================
    print(f"\n{'='*70}")
    print(f"  26c: M0R0 ExIt Convergence Speed")
    print(f"  K=10 vs K=100 bootstrap → convergence comparison")
    print(f"{'='*70}", flush=True)
    
    t0 = time.time()
    
    # Determine M0R0 state dim
    arcade = arc_agi.Arcade()
    env = arcade.make("m0r0")
    obs = env.step(GameAction.RESET)
    game = env._game
    state_dim_m0r0 = len(extract_game_state(game))
    print(f"  M0R0 state dim: {state_dim_m0r0}")
    
    for K_boot, label in [(10, "K10"), (100, "K100")]:
        print(f"\n  --- M0R0 ExIt with K={K_boot} bootstrap ---", flush=True)
        N_BOOT_M = 200
        tasks = [("m0r0", K_boot, 200, 26300000+K_boot*1000+ep, None, 0.0)
                 for ep in range(N_BOOT_M)]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            boot_m = list(ex.map(best_of_k_worker, tasks))
        
        miracles_m = [r for r in boot_m if r and r['levels_cleared'] > 0]
        rate_m = len(miracles_m) / N_BOOT_M * 100
        print(f"  Bootstrap miracles: {len(miracles_m)}/{N_BOOT_M} ({rate_m:.1f}%)")
        
        if len(miracles_m) >= 5:
            res_m = run_exit("m0r0", miracles_m, state_dim_m0r0,
                            n_iters=3, collect_K=K_boot, n_collect=100, n_eval=100,
                            tag=f"26c_m0r0_{label}")
        else:
            res_m = [{"iteration": 0, "note": "insufficient miracles"}]
        
        all_output[f"26c_{label}"] = {
            "bootstrap_K": K_boot, "miracles": len(miracles_m),
            "miracle_rate": rate_m, "results": res_m,
        }
    
    all_output["26c_elapsed_s"] = time.time() - t0
    print(f"  26c done in {time.time()-t0:.0f}s", flush=True)
    
    # ==============================================================
    # Save all results
    # ==============================================================
    out_path = os.path.join(RESULTS_DIR, "phase26_ablations.json")
    with open(out_path, "w") as f:
        json.dump(all_output, f, indent=2)
    print(f"\n  All results: {out_path}")
    
    # ==============================================================
    # Comprehensive visualization
    # ==============================================================
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 26: Final Ablation Studies", fontsize=15, fontweight='bold')
    
    # 26a: LS20 miracle reduction
    ax = axes[0, 0]
    for label, res, color, marker in [
        ("Full", res_full, "#4CAF50", "s"),
        ("Half", res_half, "#FF9800", "o"),
        ("Quarter", res_quarter, "#E91E63", "^")]:
        iters = [r['iteration'] for r in res]
        rates = [r['configs'].get("CNN K=11 σ=0.1", {}).get('rate', 0) for r in res]
        n_mir = res[0]['miracles'] if res else 0
        ax.plot(iters, rates, f'{marker}-', color=color, linewidth=2, markersize=8,
                label=f'{label} ({n_mir} miracles)')
    ax.set_xlabel('ExIt Iteration', fontsize=11)
    ax.set_ylabel('Clear Rate (%)', fontsize=11)
    ax.set_title('26a: LS20 Miracle Rate Reduction\n(CNN K=11+σ)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # 26b: Test-Time Compute Scaling
    ax = axes[0, 1]
    # Plot from Phase 24 data (TR87 miracle rate vs K)
    if "phase24" in scaling_data:
        p24d = scaling_data["phase24"]["miracle_rate_vs_K"]
        ks = sorted(p24d.keys(), key=lambda x: int(x))
        rates_24 = [p24d[k]['rate'] for k in ks]
        k_vals = [int(k) for k in ks]
        # Compute = K * avg_steps (assume ~100 steps)
        compute = [k * 100 for k in k_vals]
        ax.semilogx(compute, rates_24, 'o-', color='#E91E63', linewidth=2, markersize=8,
                    label='TR87 (random)')
    
    # LS20 from Phase 18 if available
    if "phase16c" in scaling_data:
        p16d = scaling_data["phase16c"]
        if "results" in p16d:
            for game_data in p16d["results"]:
                if isinstance(game_data, dict) and "game" in game_data:
                    gname = game_data["game"]
                    if "k_results" in game_data:
                        k_res = game_data["k_results"]
                        ks_g = sorted(k_res.keys(), key=lambda x: int(x))
                        compute_g = [int(k) * 100 for k in ks_g]
                        rates_g = [k_res[k] for k in ks_g]
                        ax.semilogx(compute_g, rates_g, 'o--', linewidth=1.5, markersize=6,
                                    label=gname, alpha=0.7)
    
    ax.set_xlabel('Test-Time Compute (K × steps)', fontsize=11)
    ax.set_ylabel('Clear Rate (%)', fontsize=11)
    ax.set_title('26b: Test-Time Compute Scaling Law', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # 26c: M0R0 convergence
    ax = axes[1, 0]
    for label, key, color in [("K=10", "26c_K10", "#2196F3"), ("K=100", "26c_K100", "#4CAF50")]:
        if key in all_output and "results" in all_output[key]:
            res_c = all_output[key]["results"]
            if res_c and "configs" in res_c[0]:
                iters_c = [r['iteration'] for r in res_c]
                rates_c = [r['configs'].get("CNN K=11 σ=0.1", {}).get('rate', 0) for r in res_c]
                n_mir = all_output[key].get("miracles", 0)
                ax.plot(iters_c, rates_c, 's-', color=color, linewidth=2, markersize=8,
                        label=f'{label} boot ({n_mir} miracles)')
    ax.set_xlabel('ExIt Iteration', fontsize=11)
    ax.set_ylabel('Clear Rate (%)', fontsize=11)
    ax.set_title('26c: M0R0 ExIt Convergence\nK=10 vs K=100 Bootstrap', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # 26 summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "Phase 26 Summary\n" + "="*40 + "\n\n"
    
    # 26a summary
    if res_full and res_quarter:
        full_final = res_full[-1]['configs'].get("CNN K=11 σ=0.1", {}).get('rate', 0)
        half_final = res_half[-1]['configs'].get("CNN K=11 σ=0.1", {}).get('rate', 0)
        qtr_final = res_quarter[-1]['configs'].get("CNN K=11 σ=0.1", {}).get('rate', 0)
        summary_text += f"26a: LS20 Miracle Reduction\n"
        summary_text += f"  Full:    {full_final:.0f}%\n"
        summary_text += f"  Half:    {half_final:.0f}%\n"
        summary_text += f"  Quarter: {qtr_final:.0f}%\n\n"
    
    # 26c summary
    for label, key in [("K=10", "26c_K10"), ("K=100", "26c_K100")]:
        if key in all_output and "results" in all_output[key]:
            res_c = all_output[key]["results"]
            if res_c and "configs" in res_c[-1]:
                rate_c = res_c[-1]['configs'].get("CNN K=11 σ=0.1", {}).get('rate', 0)
                summary_text += f"26c M0R0 {label}: {rate_c:.0f}%\n"
    
    summary_text += f"\nTotal time: {time.time()-t_total:.0f}s"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(RESULTS_DIR, "phase26_ablations.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Plot: {plot_path}")
    
    print(f"\n{'='*70}")
    print(f"  PHASE 26 COMPLETE: Three Final Ablation Studies")
    print(f"  Total elapsed: {time.time()-t_total:.0f}s")
    print(f"{'='*70}")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 26 complete!", flush=True)
