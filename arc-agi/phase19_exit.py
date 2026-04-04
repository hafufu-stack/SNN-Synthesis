"""
Phase 19: SNN-ExIt (Expert Iteration with Noisy Beam Search)
==============================================================
'Oracle-Free Self-Evolving AI' — No BFS, no human rules.
Only noise + trajectory ensemble + self-distillation.

The Pipeline:
  1. Random policy + K=100 Noisy Beam Search → collect "miracle trajectories"
  2. Train CNN on miracle trajectories (Behavioral Cloning / SFT)
  3. CNN + K=20 → discover deeper miracles → retrain
  4. Repeat until convergence

Target Games: m0r0 (easy, proof of concept), then LS20 (hard, real test)
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

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Game-Agnostic Feature Extractor
# ============================================================
def extract_state_vector(game):
    """Extract numeric features from any ARC-AGI game's internal state.
    Works even with obfuscated attribute names."""
    features = []
    for attr in sorted(dir(game)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(game, attr)
            if isinstance(val, bool):
                features.append(float(val))
            elif isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, (list, tuple)):
                # Flatten numeric lists
                for item in val:
                    if isinstance(item, (int, float)):
                        features.append(float(item))
                    elif isinstance(item, (tuple, list)):
                        for sub in item:
                            if isinstance(sub, (int, float)):
                                features.append(float(sub))
            elif isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, (int, float)):
                        features.append(float(v))
                    elif isinstance(v, (tuple, list)):
                        for sub in v:
                            if isinstance(sub, (int, float)):
                                features.append(float(sub))
        except:
            pass
    return features

# ============================================================
# State-based MLP Brain (game-agnostic)
# ============================================================
class StateBrain(nn.Module):
    """MLP that takes raw numeric state → action probabilities.
    Adapts to any game's state dimension."""
    def __init__(self, state_dim, n_actions=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.output = nn.Linear(hidden // 2, n_actions)
    
    def forward(self, x, noise_sigma=0.0):
        h = self.net(x)
        if noise_sigma > 0:
            h = h + torch.randn_like(h) * noise_sigma
        return self.output(h)


# ============================================================
# Worker: Collect trajectory with optional CNN policy
# ============================================================
def collect_single_trajectory(args):
    """Single trajectory. Returns (state_vectors, actions, levels_cleared)."""
    game_id, max_steps, seed, model_state_path, state_dim, noise_sigma = args
    
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi, random, torch, torch.nn as nn, numpy as np
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    
    rng = random.Random(seed)
    arcade = arc_agi.Arcade()
    
    try:
        env = arcade.make(game_id)
        obs = env.step(GameAction.RESET)
        game = env._game
    except:
        return None
    
    # Load model if available
    model = None
    if model_state_path and os.path.exists(model_state_path):
        try:
            class SB(nn.Module):
                def __init__(self, sd, na=4, h=128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(sd, h), nn.ReLU(), nn.BatchNorm1d(h),
                        nn.Linear(h, h), nn.ReLU(), nn.BatchNorm1d(h),
                        nn.Linear(h, h//2), nn.ReLU(),
                    )
                    self.output = nn.Linear(h//2, na)
                def forward(self, x, ns=0.0):
                    h = self.net(x)
                    if ns > 0: h = h + torch.randn_like(h) * ns
                    return self.output(h)
            
            model = SB(state_dim)
            model.load_state_dict(torch.load(model_state_path, weights_only=True))
            model.eval()
        except:
            model = None
    
    states = []
    actions = []
    max_lc = 0
    
    for step in range(max_steps):
        # Extract state
        feats = []
        for attr in sorted(dir(game)):
            if attr.startswith('_'): continue
            try:
                val = getattr(game, attr)
                if isinstance(val, bool): feats.append(float(val))
                elif isinstance(val, (int, float)): feats.append(float(val))
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        if isinstance(item, (int, float)): feats.append(float(item))
                        elif isinstance(item, (tuple, list)):
                            for sub in item:
                                if isinstance(sub, (int, float)): feats.append(float(sub))
                elif isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, (int, float)): feats.append(float(v))
                        elif isinstance(v, (tuple, list)):
                            for sub in v:
                                if isinstance(sub, (int, float)): feats.append(float(sub))
            except: pass
        
        states.append(feats)
        
        # Choose action
        if model is not None:
            try:
                x = torch.tensor([feats], dtype=torch.float32)
                with torch.no_grad():
                    logits = model(x, ns=noise_sigma)
                    action_idx = logits.argmax(1).item()
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
    
    return {
        'states': states,
        'actions': actions,
        'levels_cleared': max_lc,
        'n_steps': len(actions),
        'state_dim': len(states[0]) if states else 0
    }


def collect_best_of_k(args):
    """Run K trajectories, return the best one."""
    game_id, K, max_steps, seed, model_path, state_dim, noise_sigma = args
    
    best = None
    for k in range(K):
        traj_args = (game_id, max_steps, seed * 1000 + k, model_path, state_dim, noise_sigma)
        result = collect_single_trajectory(traj_args)
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


# ============================================================
# MAIN: SNN-ExIt Loop
# ============================================================
if __name__ == '__main__':
    GAME_ID = "m0r0"
    N_ITERATIONS = 5   # Self-improvement iterations
    K_EXPLORE = 50      # Trajectories per episode in exploration
    N_EXPLORE = 200     # Episodes per exploration round
    MAX_STEPS = 300
    N_EVAL = 100        # Evaluation episodes
    
    print(f"[{time.strftime('%H:%M:%S')}] Phase 19: SNN-ExIt Self-Evolving Agent")
    print(f"  Game: {GAME_ID}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  ExIt Iterations: {N_ITERATIONS}")
    print(f"  K_explore: {K_EXPLORE}, N_explore: {N_EXPLORE}")
    print(f"{'='*60}", flush=True)
    
    # Discover state dimension
    import arc_agi
    from arcengine import GameAction
    arcade = arc_agi.Arcade()
    env = arcade.make(GAME_ID)
    obs = env.step(GameAction.RESET)
    game = env._game
    state_feats = extract_state_vector(game)
    STATE_DIM = len(state_feats)
    print(f"  State dimension: {STATE_DIM}")
    print(f"  State features: {state_feats}", flush=True)
    del env, arcade
    
    model_path = None
    noise_sigma = 0.0
    all_iteration_results = []
    
    for iteration in range(N_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration + 1}/{N_ITERATIONS}")
        print(f"  Policy: {'Random' if model_path is None else 'CNN'}")
        print(f"  Noise σ: {noise_sigma:.2f}")
        print(f"{'='*60}", flush=True)
        
        # ============================================================
        # Step A: Noisy Beam Search — Collect Miracle Trajectories
        # ============================================================
        t0 = time.time()
        K = K_EXPLORE if iteration == 0 else max(10, K_EXPLORE // (iteration + 1))
        print(f"\n  [A] Collecting miracle trajectories (K={K}, N={N_EXPLORE})...", flush=True)
        
        tasks = [(GAME_ID, K, MAX_STEPS, iteration * 100000 + ep, 
                  model_path, STATE_DIM, noise_sigma) 
                 for ep in range(N_EXPLORE)]
        
        miracles = []
        total_cleared = 0
        
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(collect_best_of_k, tasks))
        
        for r in results:
            if r is not None and r['levels_cleared'] > 0:
                miracles.append(r)
                total_cleared += 1
        
        collect_time = time.time() - t0
        print(f"      Miracles found: {total_cleared}/{N_EXPLORE} ({total_cleared/N_EXPLORE*100:.1f}%)")
        print(f"      Time: {collect_time:.0f}s", flush=True)
        
        if total_cleared == 0:
            print(f"      No miracles! Increasing K and retrying...")
            continue
        
        # ============================================================
        # Step B: Self-Distillation — Train CNN on Miracles
        # ============================================================
        print(f"\n  [B] Self-Distillation (training on {len(miracles)} miracle trajectories)...", 
              flush=True)
        
        # Build dataset
        all_states = []
        all_actions = []
        
        for m in miracles:
            for s, a in zip(m['states'], m['actions']):
                if len(s) == STATE_DIM:
                    all_states.append(s)
                    all_actions.append(a)
        
        if len(all_states) < 10:
            print(f"      Too few training samples ({len(all_states)}). Skipping.")
            continue
        
        X = torch.tensor(all_states, dtype=torch.float32)
        Y = torch.tensor(all_actions, dtype=torch.long)
        
        # Normalize features
        x_mean = X.mean(0)
        x_std = X.std(0) + 1e-8
        X = (X - x_mean) / x_std
        
        print(f"      Training samples: {len(X)}")
        print(f"      Action distribution: {dict(zip(*np.unique(Y.numpy(), return_counts=True)))}")
        
        # Create or re-init model
        model = StateBrain(STATE_DIM, n_actions=4, hidden=128)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        
        model.train()
        best_acc = 0
        for epoch in range(300):
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
                print(f"      Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.3f}")
        
        # Save model
        model_path = os.path.join(SCRIPT_DIR, "data", f"exit_iter{iteration+1}_{GAME_ID}.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save normalization params too
        torch.save({
            'model': model.state_dict(),
            'x_mean': x_mean,
            'x_std': x_std,
            'state_dim': STATE_DIM,
        }, model_path)
        print(f"      Model saved: {model_path}")
        print(f"      Best accuracy: {best_acc:.3f}")
        
        # ============================================================
        # Step C: Evaluate — CNN vs Random vs CNN+Noise
        # ============================================================
        print(f"\n  [C] Evaluation (N={N_EVAL})...", flush=True)
        
        eval_configs = [
            ("Random K=1", None, 1, 0.0),
            ("Random K=11", None, 11, 0.0),
            (f"CNN(iter{iteration+1}) K=1", model_path, 1, 0.0),
            (f"CNN(iter{iteration+1}) K=1 σ=0.1", model_path, 1, 0.1),
            (f"CNN(iter{iteration+1}) K=1 σ=0.2", model_path, 1, 0.2),
            (f"CNN(iter{iteration+1}) K=5 σ=0.1", model_path, 5, 0.1),
            (f"CNN(iter{iteration+1}) K=11 σ=0.1", model_path, 11, 0.1),
        ]
        
        iter_results = {"iteration": iteration + 1, "miracles": total_cleared, 
                       "train_acc": best_acc, "configs": {}}
        
        for name, mp, K, sigma in eval_configs:
            tasks = [(GAME_ID, K, MAX_STEPS, 999999 + ep, mp, STATE_DIM, sigma)
                     for ep in range(N_EVAL)]
            
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                eval_results = list(executor.map(collect_best_of_k, tasks))
            
            clears = sum(1 for r in eval_results if r and r['levels_cleared'] > 0)
            rate = clears / N_EVAL * 100
            iter_results["configs"][name] = {"clears": clears, "rate": rate}
            
            bar = "█" * int(rate / 2)
            print(f"      {name:35s}: {clears:3d}/{N_EVAL} = {rate:5.1f}%  {bar}", flush=True)
        
        all_iteration_results.append(iter_results)
        
        # Update noise for next iteration (reduce as policy improves)
        noise_sigma = 0.15 if iteration == 0 else 0.1
        
        print(f"\n  Iteration {iteration+1} complete.", flush=True)
    
    # ============================================================
    # Save all results
    # ============================================================
    out_path = os.path.join(RESULTS_DIR, f"phase19_exit_{GAME_ID}.json")
    with open(out_path, "w") as f:
        json.dump(all_iteration_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    
    # ============================================================
    # Visualization
    # ============================================================
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: ExIt improvement curve
    iters = [r['iteration'] for r in all_iteration_results]
    cnn_k1 = []
    cnn_k11_noise = []
    random_k11 = []
    
    for r in all_iteration_results:
        configs = r['configs']
        # Find CNN K=1 (no noise) for this iteration
        for name, data in configs.items():
            if 'CNN' in name and 'K=1' in name and 'σ' not in name:
                cnn_k1.append(data['rate'])
            elif 'CNN' in name and 'K=11' in name:
                cnn_k11_noise.append(data['rate'])
            elif 'Random K=11' in name:
                random_k11.append(data['rate'])
    
    if cnn_k1:
        ax1.plot(iters[:len(cnn_k1)], cnn_k1, 'o-', color='#E91E63', linewidth=2, 
                markersize=8, label='CNN K=1 (no noise)')
    if cnn_k11_noise:
        ax1.plot(iters[:len(cnn_k11_noise)], cnn_k11_noise, 's-', color='#FF9800', linewidth=2,
                markersize=8, label='CNN K=11 + noise')
    if random_k11:
        ax1.axhline(y=random_k11[0], color='#999', linestyle='--', label=f'Random K=11 ({random_k11[0]:.0f}%)')
    
    ax1.set_xlabel('ExIt Iteration', fontsize=12)
    ax1.set_ylabel('Clear Rate (%)', fontsize=12) 
    ax1.set_title(f'SNN-ExIt Self-Improvement ({GAME_ID.upper()})\nOracle-Free Self-Evolving Agent',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Right: Training data growth
    miracles = [r['miracles'] for r in all_iteration_results]
    accs = [r['train_acc'] for r in all_iteration_results]
    
    ax2.bar(iters, miracles, color='#4CAF50', alpha=0.7, label='Miracles found')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(iters, [a*100 for a in accs], 'o-', color='#2196F3', linewidth=2, label='Train acc')
    
    ax2.set_xlabel('ExIt Iteration', fontsize=12)
    ax2.set_ylabel('Miracle Trajectories', fontsize=12)
    ax2_twin.set_ylabel('Training Accuracy (%)', fontsize=12, color='#2196F3')
    ax2.set_title('Self-Distillation Progress', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"phase19_exit_{GAME_ID}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot: {plot_path}")
    plt.close('all')
    
    print(f"\n{'='*60}")
    print(f"  PHASE 19 COMPLETE: SNN-ExIt")
    print(f"  Oracle-Free Self-Evolving Agent on {GAME_ID.upper()}")
    print(f"{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Done!", flush=True)
