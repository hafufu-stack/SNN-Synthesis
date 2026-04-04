"""
Phase 25: Enriched State Representation for TR87 ExIt
======================================================
Phase 24 showed: K=1000 gives 72% miracle rate (well above 20% threshold),
but ExIt still fails with dim=7 state. Phase 23 showed pixel input also fails.

Hypothesis: The bottleneck is state REPRESENTATION QUALITY, not miracle count.
If we engineer a richer feature vector from game sprites (positions, distances,
relative coords, velocities, grid encoding), ExIt should work on TR87.

Design:
  - Extract 28+ dimensional state from sprites (matching LS20's dim=28)
  - Re-run ExIt with K=500 bootstrap (46% miracle rate, faster than K=1000)
  - Compare with Phase 24's dim=7 results
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


# ============================================================
# Enriched state extraction for TR87
# ============================================================
def extract_enriched_state(game):
    """Extract rich feature vector from TR87 game state.
    
    Sources:
    1. All scalar attributes (6 dims)
    2. Sprite positions (x, y) for each sprite
    3. Sprite sizes (width, height)  
    4. Pairwise distances between sprites
    5. Relative positions (player-to-each-object)
    6. Grid position encoding (x/128, y/128 normalized)
    """
    features = []
    
    # 1. Scalar attributes
    scalars = []
    for attr in sorted(dir(game)):
        if attr.startswith('_') or attr in ('level_index', 'win_score'):
            continue
        try:
            val = getattr(game, attr)
            if callable(val): continue
            if isinstance(val, (int, float, bool)):
                scalars.append(float(val))
        except:
            pass
    features.extend(scalars)  # ~4-6 dims
    
    # 2. Sprite features
    sprites = []
    for attr in sorted(dir(game)):
        if attr.startswith('_'): continue
        try:
            val = getattr(game, attr)
            if callable(val): continue
            if isinstance(val, list):
                for item in val:
                    if hasattr(item, 'x') and hasattr(item, 'y'):
                        sprites.append(item)
        except:
            pass
    
    # Extract per-sprite features (x, y, w, h, is_visible, is_collidable)
    sprite_data = []
    for s in sprites[:6]:  # Cap at 6 sprites
        sx = getattr(s, 'x', 0)
        sy = getattr(s, 'y', 0)
        sw = getattr(s, 'width', 1)
        sh = getattr(s, 'height', 1)
        sv = float(getattr(s, 'is_visible', True))
        sc = float(getattr(s, 'is_collidable', True))
        sprite_data.append((sx, sy, sw, sh, sv, sc))
        # Normalized positions
        features.extend([sx / 128.0, sy / 128.0, sw / 128.0, sh / 128.0, sv, sc])
    
    # Pad if fewer sprites
    while len(sprite_data) < 6:
        sprite_data.append((0, 0, 0, 0, 0, 0))
        features.extend([0, 0, 0, 0, 0, 0])
    
    # 3. Pairwise distances (first 4 sprites)
    n_sprites = min(len(sprite_data), 4)
    for i in range(n_sprites):
        for j in range(i+1, n_sprites):
            dx = (sprite_data[i][0] - sprite_data[j][0]) / 128.0
            dy = (sprite_data[i][1] - sprite_data[j][1]) / 128.0
            dist = np.sqrt(dx*dx + dy*dy)
            features.extend([dx, dy, dist])
    
    # Pad pairwise if needed (4 sprites = 6 pairs × 3 = 18 dims)
    n_pairs = n_sprites * (n_sprites - 1) // 2
    target_pairs = 6  # C(4,2)
    for _ in range(target_pairs - n_pairs):
        features.extend([0, 0, 0])
    
    # 4. Player-relative features (assume first sprite might be player)
    if len(sprite_data) >= 2:
        px, py = sprite_data[0][0], sprite_data[0][1]
        for i in range(1, min(len(sprite_data), 5)):
            rx = (sprite_data[i][0] - px) / 128.0
            ry = (sprite_data[i][1] - py) / 128.0
            features.extend([rx, ry])
    
    # Pad player-relative (4 objects × 2 = 8 dims)
    while len(features) < 80:  # Ensure minimum
        features.append(0.0)
    
    return np.array(features[:80], dtype=np.float32)  # Cap at 80 dims


def extract_basic_state(game):
    """Original dim=7 extraction (Phase 21/24 baseline)."""
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


# ============================================================
# Worker functions
# ============================================================
def collect_enriched_trajectory(args):
    """Single best-of-K trajectory with enriched state."""
    game_id, K, max_steps, seed, model_path, noise_sigma, use_enriched = args
    
    import os, random, torch, torch.nn as nn
    import numpy as np
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    
    rng = random.Random(seed)
    
    # Load model if available
    model = None
    state_dim = None
    x_mean = None
    x_std = None
    if model_path and os.path.exists(model_path):
        try:
            data = torch.load(model_path, weights_only=True)
            state_dim = data['state_dim']
            x_mean = data.get('x_mean', None)
            x_std = data.get('x_std', None)
            
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
            
            model = SB(sd=state_dim)
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
                if use_enriched:
                    s = extract_enriched_state(game)
                else:
                    s = extract_basic_state(game)
                states.append(s.tolist())
            except:
                dim = state_dim or (80 if use_enriched else 7)
                states.append([0.0] * dim)
            
            if model is not None:
                try:
                    x = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0)
                    if x_mean is not None:
                        x = (x - x_mean) / x_std
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


# ============================================================
# StateBrain MLP
# ============================================================
class StateBrain(nn.Module):
    def __init__(self, state_dim, n_actions=4, hidden=256):
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


if __name__ == '__main__':
    GAME_ID = "tr87"
    MAX_STEPS = 200
    BOOTSTRAP_K = 500  # K=500 gives 46% miracle rate
    N_BOOTSTRAP = 300
    N_EXIT_ITERS = 5
    N_EVAL = 100
    
    print(f"[{time.strftime('%H:%M:%S')}] Phase 25: Enriched State TR87 ExIt")
    print(f"  Game: {GAME_ID}")
    print(f"  Bootstrap K: {BOOTSTRAP_K}")
    print(f"  State: enriched (~80 dim) vs basic (7 dim)")
    print(f"  Workers: {N_WORKERS}")
    print(f"{'='*60}", flush=True)
    
    # ============================================================
    # Step 1: Determine enriched state dim
    # ============================================================
    import logging
    logging.disable(logging.CRITICAL)
    import arc_agi
    from arcengine import GameAction
    
    arcade = arc_agi.Arcade()
    env = arcade.make(GAME_ID)
    obs = env.step(GameAction.RESET)
    game = env._game
    
    enriched = extract_enriched_state(game)
    basic = extract_basic_state(game)
    print(f"\n  Basic state dim: {len(basic)}")
    print(f"  Enriched state dim: {len(enriched)}")
    print(f"  Enriched features sample: {enriched[:10].tolist()}")
    
    ENRICHED_DIM = len(enriched)
    BASIC_DIM = len(basic)
    
    # ============================================================
    # Step 2: Collect bootstrap miracles (reuse for both conditions)
    # ============================================================
    print(f"\n  [Step 2] Collecting {N_BOOTSTRAP} bootstrap episodes at K={BOOTSTRAP_K}...", flush=True)
    t0 = time.time()
    
    # Collect with enriched states
    tasks = [(GAME_ID, BOOTSTRAP_K, MAX_STEPS, 25000000 + ep, None, 0.0, True)
             for ep in range(N_BOOTSTRAP)]
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        bootstrap_results = list(executor.map(collect_enriched_trajectory, tasks))
    
    miracles = [r for r in bootstrap_results if r and r['levels_cleared'] > 0]
    elapsed = time.time() - t0
    print(f"  Bootstrap: {len(miracles)}/{N_BOOTSTRAP} miracles ({len(miracles)/N_BOOTSTRAP*100:.1f}%) in {elapsed:.0f}s")
    
    if len(miracles) < 10:
        print("  ERROR: Not enough miracles. Aborting.")
        sys.exit(1)
    
    # Verify state dim from actual data
    actual_dim = len(miracles[0]['states'][0])
    print(f"  Actual enriched state dim: {actual_dim}")
    ENRICHED_DIM = actual_dim
    
    # ============================================================
    # Step 3: ExIt Loop with enriched states
    # ============================================================
    print(f"\n  [Step 3] ExIt with enriched state (dim={ENRICHED_DIM})...", flush=True)
    
    cumulative_miracles = list(miracles)
    model_path = None
    exit_results_enriched = []
    
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
        X_norm = (X - x_mean) / x_std
        
        print(f"    Training on {len(X)} samples (dim={X.shape[1]})...")
        
        model = StateBrain(state_dim=ENRICHED_DIM, n_actions=4, hidden=256)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        
        model.train()
        best_acc = 0
        for epoch in range(500):
            perm = torch.randperm(len(X_norm))
            batch = perm[:min(256, len(X_norm))]
            loss = criterion(model(X_norm[batch], 0.0), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    acc = (model(X_norm, 0.0).argmax(1) == Y).float().mean().item()
                model.train()
                best_acc = max(best_acc, acc)
        
        print(f"    Train acc: {best_acc:.3f}")
        
        # Save model
        mp = os.path.join(SCRIPT_DIR, "data", f"enriched_tr87_iter{iteration+1}.pt")
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        torch.save({'model': model.state_dict(), 'state_dim': ENRICHED_DIM,
                    'x_mean': x_mean, 'x_std': x_std}, mp)
        model_path = mp
        
        # Collect more miracles with trained CNN
        K_COLLECT = max(50, BOOTSTRAP_K // (2 ** iteration))
        N_COLLECT = 200
        print(f"    Collecting with CNN + K={K_COLLECT} + σ=0.1...")
        
        tasks = [(GAME_ID, K_COLLECT, MAX_STEPS,
                  (iteration+1) * 10000000 + ep, model_path, 0.1, True)
                 for ep in range(N_COLLECT)]
        
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            new_results = list(executor.map(collect_enriched_trajectory, tasks))
        
        new_miracles = [r for r in new_results if r and r['levels_cleared'] > 0]
        cumulative_miracles.extend(new_miracles)
        print(f"    New miracles: {len(new_miracles)}/{N_COLLECT} ({len(new_miracles)/N_COLLECT*100:.1f}%)")
        
        # Evaluate
        eval_configs = [
            ("Random K=11", None, 11, 0.0),
            (f"Enriched(i{iteration+1}) K=1", model_path, 1, 0.0),
            (f"Enriched(i{iteration+1}) K=1 σ=0.1", model_path, 1, 0.1),
            (f"Enriched(i{iteration+1}) K=11 σ=0.1", model_path, 11, 0.1),
        ]
        
        iter_result = {"iteration": iteration+1,
                       "state_dim": ENRICHED_DIM,
                       "miracles_total": len(cumulative_miracles),
                       "train_acc": best_acc, "configs": {}}
        
        for name, mp_eval, K_eval, sigma in eval_configs:
            tasks = [(GAME_ID, K_eval, MAX_STEPS,
                      88000000 + iteration*10000 + ep, mp_eval, sigma, True)
                     for ep in range(N_EVAL)]
            
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                eval_results = list(executor.map(collect_enriched_trajectory, tasks))
            
            clears = sum(1 for r in eval_results if r and r['levels_cleared'] > 0)
            rate = clears / N_EVAL * 100
            iter_result["configs"][name] = {"clears": clears, "rate": rate}
            
            bar = "█" * int(rate / 2)
            print(f"      {name:40s}: {clears:3d}/{N_EVAL} = {rate:5.1f}%  {bar}")
        
        exit_results_enriched.append(iter_result)
        gc.collect()
    
    # ============================================================
    # Save and visualize
    # ============================================================
    output = {
        "experiment": "Phase 25: Enriched State TR87 ExIt",
        "enriched_dim": ENRICHED_DIM,
        "basic_dim": BASIC_DIM,
        "bootstrap_K": BOOTSTRAP_K,
        "bootstrap_miracles": len(miracles),
        "exit_results": exit_results_enriched,
    }
    
    out_path = os.path.join(RESULTS_DIR, "phase25_enriched_exit.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results: {out_path}")
    
    # Visualization
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: ExIt convergence comparison
    ax1 = axes[0]
    iters = [r['iteration'] for r in exit_results_enriched]
    enriched_k11 = [r['configs'].get(f"Enriched(i{r['iteration']}) K=11 σ=0.1", {}).get('rate', 0)
                    for r in exit_results_enriched]
    enriched_k1 = [r['configs'].get(f"Enriched(i{r['iteration']}) K=1 σ=0.1", {}).get('rate', 0)
                   for r in exit_results_enriched]
    
    ax1.plot(iters, enriched_k11, 's-', color='#4CAF50', linewidth=2, markersize=8,
             label=f'Enriched (dim={ENRICHED_DIM}) K=11+σ')
    ax1.plot(iters, enriched_k1, 'o--', color='#2196F3', linewidth=1.5, markersize=6,
             label=f'Enriched (dim={ENRICHED_DIM}) K=1+σ')
    
    # Phase 24 comparison (dim=7, K=1000 bootstrap)
    ax1.axhline(y=0, color='#E91E63', linestyle=':', linewidth=2,
                label='Phase 24: dim=7 K=11+σ (0%)')
    ax1.axhline(y=3, color='#999', linestyle='--', linewidth=1,
                label='Phase 21: dim=7 K=11+σ (3%)')
    
    ax1.set_xlabel('ExIt Iteration', fontsize=12)
    ax1.set_ylabel('Clear Rate (%)', fontsize=12)
    ax1.set_title(f'TR87 ExIt: Enriched State (dim={ENRICHED_DIM})\nvs Basic State (dim=7)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(-5, max(max(enriched_k11) * 1.3, 20))
    
    # Right: Train accuracy comparison
    ax2 = axes[1]
    train_accs = [r['train_acc'] for r in exit_results_enriched]
    ax2.plot(iters, [a * 100 for a in train_accs], 'o-', color='#FF9800', linewidth=2)
    ax2.axhline(y=25, color='#999', linestyle='--', label='Random chance (4 actions)')
    ax2.axhline(y=47, color='#E91E63', linestyle=':', label='Phase 24 dim=7 acc (47%)')
    ax2.set_xlabel('ExIt Iteration', fontsize=12)
    ax2.set_ylabel('Train Accuracy (%)', fontsize=12)
    ax2.set_title(f'CNN Training: Can dim={ENRICHED_DIM}\nLearn Better Than dim=7?',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "phase25_enriched_exit.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot: {plot_path}")
    plt.close('all')
    
    print(f"\n{'='*60}")
    print(f"  PHASE 25 COMPLETE: Enriched State TR87 ExIt")
    print(f"  Enriched dim: {ENRICHED_DIM} vs Basic dim: {BASIC_DIM}")
    if exit_results_enriched:
        last = exit_results_enriched[-1]
        print(f"  Final iteration results:")
        for name, data in last['configs'].items():
            print(f"    {name:40s}: {data['rate']:.1f}%")
    print(f"{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 25 complete!", flush=True)
