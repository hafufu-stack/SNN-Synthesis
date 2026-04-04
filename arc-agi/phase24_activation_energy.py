"""
Phase 24: Activation Energy Experiment
=======================================
Hypothesis: TR87's ExIt failure (Phase 21) was due to insufficient initial
miracle rate (13% at K=100). Can brute-force K (K=500, 1000, 2000) overcome
the "20% activation energy threshold"?

If yes: "Any game can be self-evolved with sufficient initial compute."
If no: "TR87 is structurally too hard—the 20% threshold is game-intrinsic."
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

N_WORKERS = min(cpu_count() - 4, 20)

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# State extraction (same as Phase 21)
# ============================================================
def extract_game_state(game):
    """Extract numeric state features from game object."""
    state = []
    for attr in sorted(dir(game)):
        if attr.startswith('_') or attr in ('level_index', 'win_score'):
            continue
        try:
            val = getattr(game, attr)
            if callable(val):
                continue
            if isinstance(val, (int, float, bool)):
                state.append(float(val))
            elif isinstance(val, (list, tuple)):
                for v in val[:4]:
                    if isinstance(v, (int, float, bool)):
                        state.append(float(v))
        except:
            pass
    return np.array(state, dtype=np.float32)


# ============================================================
# Worker: Single random trajectory (no CNN needed for bootstrap)
# ============================================================
def random_trajectory(args):
    """Single random trajectory on TR87."""
    game_id, max_steps, seed = args
    
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi, random
    import numpy as np
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    
    rng = random.Random(seed)
    
    try:
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        obs = env.step(GameAction.RESET)
        game = env._game
    except:
        return None
    
    states = []
    actions = []
    max_lc = 0
    
    for step in range(max_steps):
        try:
            s = extract_game_state(game)
            states.append(s.tolist())
        except:
            states.append([0.0] * 7)
        
        action = rng.choice(ALL_A)
        actions.append(ALL_A.index(action))
        
        try:
            obs = env.step(action)
            if obs.levels_completed > max_lc:
                max_lc = obs.levels_completed
            if obs.state.value in ('GAME_OVER', 'WIN'):
                break
        except:
            break
    
    return {
        'states': states,
        'actions': actions,
        'levels_cleared': max_lc,
        'n_steps': len(actions),
    }


def best_of_k_random(args):
    """Run K random trajectories, return the best."""
    game_id, K, max_steps, seed = args
    best = None
    for k in range(K):
        result = random_trajectory((game_id, max_steps, seed * 100000 + k))
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


# ============================================================
# StateBrain MLP (same as Phase 21)
# ============================================================
class StateBrain(nn.Module):
    def __init__(self, state_dim=7, n_actions=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x, noise_sigma=0.0):
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        if noise_sigma > 0:
            h = h + torch.randn_like(h) * noise_sigma
        h = self.net[3](h)
        h = self.net[4](h)
        h = self.net[5](h)
        return h


def collect_with_model(args):
    """Trajectory using trained model + noise."""
    game_id, K, max_steps, seed, model_path, noise_sigma = args
    
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi, random, torch, torch.nn as nn
    import numpy as np
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    
    rng = random.Random(seed)
    
    # Load model
    model = None
    if model_path and os.path.exists(model_path):
        try:
            class SB(nn.Module):
                def __init__(self, sd=7, na=4, h=256):
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
            data = torch.load(model_path, weights_only=True)
            sd = data['state_dim']
            model = SB(sd=sd)
            model.load_state_dict(data['model'])
            model.eval()
        except:
            model = None
    
    best = None
    for k in range(K):
        try:
            arcade = arc_agi.Arcade()
            env = arcade.make(game_id)
            obs = env.step(GameAction.RESET)
            game = env._game
        except:
            continue
        
        states, actions, max_lc = [], [], 0
        for step in range(max_steps):
            try:
                s = extract_game_state(game)
                states.append(s.tolist())
            except:
                states.append([0.0] * 7)
            
            if model is not None:
                try:
                    x = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(x, ns=noise_sigma)
                        probs = torch.softmax(logits, dim=1)
                        action_idx = torch.multinomial(probs, 1).item()
                    action = ALL_A[action_idx]
                except:
                    action = rng.choice(ALL_A)
            else:
                action = rng.choice(ALL_A)
            
            actions.append(ALL_A.index(action))
            try:
                obs = env.step(action)
                if obs.levels_completed > max_lc:
                    max_lc = obs.levels_completed
                if obs.state.value in ('GAME_OVER', 'WIN'):
                    break
            except:
                break
        
        result = {'states': states, 'actions': actions, 
                  'levels_cleared': max_lc, 'n_steps': len(actions)}
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    
    return best


if __name__ == '__main__':
    GAME_ID = "tr87"
    MAX_STEPS = 200
    
    print(f"[{time.strftime('%H:%M:%S')}] Phase 24: Activation Energy Experiment")
    print(f"  Game: {GAME_ID}")
    print(f"  Hypothesis: K>=1000 overcomes 20% miracle rate threshold")
    print(f"  Workers: {N_WORKERS}")
    print(f"{'='*60}", flush=True)
    
    # ============================================================
    # Step 1: Measure miracle rate at various K values
    # ============================================================
    print(f"\n  [Step 1] Measuring miracle rate vs K (random policy)...", flush=True)
    
    K_VALUES = [1, 10, 50, 100, 200, 500, 1000]
    N_EPISODES = 200  # per K value
    
    miracle_rates = {}
    for K in K_VALUES:
        t0 = time.time()
        # For large K, use fewer episodes to stay feasible
        n_ep = min(N_EPISODES, max(50, 200 // max(1, K // 100)))
        
        tasks = [(GAME_ID, K, MAX_STEPS, K * 1000000 + ep) for ep in range(n_ep)]
        
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(best_of_k_random, tasks))
        
        clears = sum(1 for r in results if r and r['levels_cleared'] > 0)
        rate = clears / n_ep * 100
        miracle_rates[K] = {'clears': clears, 'n': n_ep, 'rate': rate}
        
        bar = "█" * int(rate / 2)
        elapsed = time.time() - t0
        print(f"    K={K:5d}: {clears:3d}/{n_ep} = {rate:5.1f}%  {bar}  ({elapsed:.0f}s)", flush=True)
    
    # ============================================================
    # Step 2: If miracle rate > 20% at K>=500, run ExIt
    # ============================================================
    best_k = max(K_VALUES, key=lambda k: miracle_rates[k]['rate'])
    best_rate = miracle_rates[best_k]['rate']
    
    print(f"\n  Best miracle rate: K={best_k} → {best_rate:.1f}%")
    print(f"  Threshold: 20%")
    
    exit_results = []
    
    if best_rate >= 15:  # Try ExIt even at 15% to find exact boundary
        BOOTSTRAP_K = best_k
        print(f"\n  [Step 2] Running ExIt with K={BOOTSTRAP_K} bootstrap ({best_rate:.1f}% miracle rate)...", flush=True)
        
        # Collect initial miracles with massive K
        N_BOOTSTRAP = 300
        print(f"    Collecting {N_BOOTSTRAP} episodes at K={BOOTSTRAP_K}...", flush=True)
        
        tasks = [(GAME_ID, BOOTSTRAP_K, MAX_STEPS, 9000000 + ep) for ep in range(N_BOOTSTRAP)]
        
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            bootstrap_results = list(executor.map(best_of_k_random, tasks))
        
        miracles = [r for r in bootstrap_results if r and r['levels_cleared'] > 0]
        print(f"    Bootstrap miracles: {len(miracles)}/{N_BOOTSTRAP} ({len(miracles)/N_BOOTSTRAP*100:.1f}%)")
        
        if len(miracles) >= 10:
            # Determine state dim
            state_dim = len(miracles[0]['states'][0]) if miracles[0]['states'] else 7
            print(f"    State dim: {state_dim}")
            
            # ============================================================
            # ExIt Loop
            # ============================================================
            cumulative_miracles = list(miracles)
            model_path = None
            N_EXIT_ITERS = 4
            
            for iteration in range(N_EXIT_ITERS):
                print(f"\n    --- ExIt Iteration {iteration+1}/{N_EXIT_ITERS} ---")
                print(f"    Cumulative miracles: {len(cumulative_miracles)}")
                
                # Train CNN
                all_states, all_actions = [], []
                for m in cumulative_miracles:
                    for s, a in zip(m['states'], m['actions']):
                        all_states.append(s)
                        all_actions.append(a)
                
                X = torch.tensor(all_states, dtype=torch.float32)
                Y = torch.tensor(all_actions, dtype=torch.long)
                
                # Normalize
                x_mean = X.mean(0, keepdim=True)
                x_std = X.std(0, keepdim=True).clamp(min=1e-6)
                X = (X - x_mean) / x_std
                
                print(f"    Training on {len(X)} samples...")
                
                model = StateBrain(state_dim=state_dim, n_actions=4, hidden=256)
                optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
                
                model.train()
                best_acc = 0
                for epoch in range(500):
                    perm = torch.randperm(len(X))
                    batch = perm[:min(256, len(X))]
                    loss = criterion(model(X[batch], 0.0), Y[batch])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    if (epoch + 1) % 100 == 0:
                        model.eval()
                        with torch.no_grad():
                            acc = (model(X, 0.0).argmax(1) == Y).float().mean().item()
                        model.train()
                        best_acc = max(best_acc, acc)
                
                print(f"    Train acc: {best_acc:.3f}")
                
                # Save model
                mp = os.path.join(SCRIPT_DIR, "data", f"activation_tr87_iter{iteration+1}.pt")
                os.makedirs(os.path.dirname(mp), exist_ok=True)
                torch.save({'model': model.state_dict(), 'state_dim': state_dim,
                           'x_mean': x_mean, 'x_std': x_std}, mp)
                model_path = mp
                
                # Collect more miracles with trained CNN
                K_COLLECT = max(50, BOOTSTRAP_K // (2 ** iteration))
                N_COLLECT = 200
                print(f"    Collecting with CNN + K={K_COLLECT}...")
                
                tasks = [(GAME_ID, K_COLLECT, MAX_STEPS, 
                         (iteration+1) * 10000000 + ep, model_path, 0.1)
                        for ep in range(N_COLLECT)]
                
                with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                    new_results = list(executor.map(collect_with_model, tasks))
                
                new_miracles = [r for r in new_results if r and r['levels_cleared'] > 0]
                cumulative_miracles.extend(new_miracles)
                print(f"    New miracles: {len(new_miracles)}/{N_COLLECT} ({len(new_miracles)/N_COLLECT*100:.1f}%)")
                
                # Evaluate
                N_EVAL = 100
                eval_configs = [
                    ("Random K=11", None, 11, 0.0),
                    (f"CNN(i{iteration+1}) K=1", model_path, 1, 0.0),
                    (f"CNN(i{iteration+1}) K=11 σ=0.1", model_path, 11, 0.1),
                ]
                
                iter_result = {"iteration": iteration+1,
                              "bootstrap_K": BOOTSTRAP_K,
                              "miracles_total": len(cumulative_miracles),
                              "train_acc": best_acc, "configs": {}}
                
                for name, mp_eval, K_eval, sigma in eval_configs:
                    tasks = [(GAME_ID, K_eval, MAX_STEPS, 
                             88000000 + iteration*10000 + ep, mp_eval, sigma)
                            for ep in range(N_EVAL)]
                    
                    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                        eval_results = list(executor.map(collect_with_model, tasks))
                    
                    clears = sum(1 for r in eval_results if r and r['levels_cleared'] > 0)
                    rate = clears / N_EVAL * 100
                    iter_result["configs"][name] = {"clears": clears, "rate": rate}
                    
                    bar = "█" * int(rate / 2)
                    print(f"      {name:35s}: {clears:3d}/{N_EVAL} = {rate:5.1f}%  {bar}")
                
                exit_results.append(iter_result)
                gc.collect()
    else:
        print(f"\n  ⚠ Miracle rate never exceeds 15% even at K={best_k}.")
        print(f"  → TR87 is structurally too hard for random bootstrapping.")
    
    # ============================================================
    # Save and visualize
    # ============================================================
    output = {
        "miracle_rate_vs_K": {str(k): v for k, v in miracle_rates.items()},
        "exit_results": exit_results,
        "hypothesis": "K>=1000 overcomes 20% miracle rate threshold for TR87 ExIt",
    }
    
    out_path = os.path.join(RESULTS_DIR, "phase24_activation_energy.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results: {out_path}")
    
    # Visualization
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Miracle rate vs K
    ax1 = axes[0]
    ks = sorted(miracle_rates.keys())
    rates = [miracle_rates[k]['rate'] for k in ks]
    ax1.semilogx(ks, rates, 'o-', color='#E91E63', linewidth=2, markersize=8)
    ax1.axhline(y=20, color='#4CAF50', linestyle='--', linewidth=2, label='20% threshold')
    ax1.axhline(y=25, color='#2196F3', linestyle=':', linewidth=1, label='LS20 rate (K=100)')
    ax1.fill_between([min(ks), max(ks)], 20, 100, alpha=0.1, color='#4CAF50')
    ax1.set_xlabel('K (parallel trajectories)', fontsize=12)
    ax1.set_ylabel('Miracle Rate (%)', fontsize=12)
    ax1.set_title('TR87: Can Compute Overcome the\n20% Activation Energy Threshold?', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(-5, max(max(rates) * 1.2, 30))
    
    # Right: ExIt results (if available)
    ax2 = axes[1]
    if exit_results:
        iters = [r['iteration'] for r in exit_results]
        cnn_k11 = [r['configs'].get(f"CNN(i{r['iteration']}) K=11 σ=0.1", {}).get('rate', 0) for r in exit_results]
        random_k11 = [r['configs'].get("Random K=11", {}).get('rate', 0) for r in exit_results]
        
        ax2.plot(iters, cnn_k11, 's-', color='#FF9800', linewidth=2, label='CNN K=11+σ')
        if random_k11:
            ax2.axhline(y=random_k11[0], color='#999', linestyle='--', label=f'Random K=11 ({random_k11[0]:.0f}%)')
        ax2.axhline(y=3, color='#E91E63', linestyle=':', label='Phase 21 (K=100 bootstrap)')
    
    ax2.set_xlabel('ExIt Iteration', fontsize=12)
    ax2.set_ylabel('Clear Rate (%)', fontsize=12)
    ax2.set_title(f'Activation Energy ExIt\n(Bootstrap K={best_k})', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-5, 105)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "phase24_activation_energy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot: {plot_path}")
    plt.close('all')
    
    print(f"\n{'='*60}")
    print(f"  PHASE 24 COMPLETE: Activation Energy Experiment")
    print(f"  Miracle rate vs K:")
    for k in ks:
        print(f"    K={k:5d}: {miracle_rates[k]['rate']:.1f}%")
    if exit_results:
        last = exit_results[-1]
        print(f"  ExIt with K={best_k} bootstrap:")
        for name, data in last['configs'].items():
            print(f"    {name:35s}: {data['rate']:.1f}%")
    print(f"{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 24 complete!", flush=True)
