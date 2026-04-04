"""
Phase 23: Visual SNN-ExIt on TR87 (Pixel-Level Self-Evolution)
================================================================
TR87 failed with 7 numeric features (Phase 21).
Now we use the 64x64 FRAME observation as input to a proper CNN.
If this works, it proves: "pixels + noise = universal self-evolution"
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
# Visual CNN: 64x64 frame → action
# ============================================================
class VisualBrain(nn.Module):
    """CNN that takes 64x64 frame → action probabilities.
    Small but with enough capacity to learn spatial patterns."""
    def __init__(self, n_actions=4, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            # 32x32 -> 16x16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            # 16x16 -> 8x8
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            # 8x8 -> 4x4
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        )
        # 64 * 4 * 4 = 1024
        self.fc = nn.Sequential(
            nn.Linear(1024, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.output = nn.Linear(hidden // 2, n_actions)
    
    def forward(self, x, noise_sigma=0.0):
        # x: (batch, 1, 64, 64)
        h = self.conv(x)
        h = h.view(h.size(0), -1)  # flatten
        h = self.fc(h)
        if noise_sigma > 0:
            h = h + torch.randn_like(h) * noise_sigma
        return self.output(h)


# ============================================================
# Worker: Collect trajectory using frame observation
# ============================================================
def collect_visual_trajectory(args):
    """Single trajectory using 64x64 frame as input."""
    game_id, max_steps, seed, model_path, noise_sigma = args
    
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi, random, torch, torch.nn as nn
    import numpy as np
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    
    rng = random.Random(seed)
    
    try:
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        obs = env.step(GameAction.RESET)
    except:
        return None
    
    # Load model if available
    model = None
    if model_path and os.path.exists(model_path):
        try:
            class VB(nn.Module):
                def __init__(self, na=4, h=128):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
                        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                        nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(1024, h), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(h, h//2), nn.ReLU(),
                    )
                    self.output = nn.Linear(h//2, na)
                def forward(self, x, ns=0.0):
                    h = self.conv(x)
                    h = h.view(h.size(0), -1)
                    h = self.fc(h)
                    if ns > 0: h = h + torch.randn_like(h) * ns
                    return self.output(h)
            
            data = torch.load(model_path, weights_only=True)
            model = VB()
            model.load_state_dict(data['model'])
            model.eval()
        except:
            model = None
    
    frames = []
    actions = []
    max_lc = 0
    
    for step in range(max_steps):
        # Get frame (64x64 numpy array)
        frame = np.array(obs.frame[0], dtype=np.float32)
        
        # Normalize to [0, 1]
        if frame.max() > 1.0:
            frame = frame / max(frame.max(), 1.0)
        
        frames.append(frame)
        
        if model is not None:
            try:
                x = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
                with torch.no_grad():
                    logits = model(x, ns=noise_sigma)
                    probs = torch.softmax(logits / max(0.5, 1.0 - noise_sigma), dim=1)
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
    
    # Convert frames to compact representation (keep as list of lists for serialization)
    return {
        'frames': [f.tolist() for f in frames],
        'actions': actions,
        'levels_cleared': max_lc,
        'n_steps': len(actions),
    }


def collect_best_of_k_visual(args):
    """Run K trajectories, return the best."""
    game_id, K, max_steps, seed, model_path, noise_sigma = args
    
    best = None
    for k in range(K):
        single_args = (game_id, max_steps, seed * 10000 + k, model_path, noise_sigma)
        result = collect_visual_trajectory(single_args)
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


if __name__ == '__main__':
    GAME_ID = "tr87"
    N_ITERATIONS = 5
    MAX_STEPS = 200  # TR87 is lighter (fps=5)
    
    ITER_CONFIG = [
        {"K": 100, "N": 200, "noise": 0.0,  "desc": "Random bootstrap (visual)"},
        {"K": 50,  "N": 200, "noise": 0.15, "desc": "Visual CNN + noise"},
        {"K": 30,  "N": 200, "noise": 0.10, "desc": "Better visual CNN"},
        {"K": 20,  "N": 200, "noise": 0.08, "desc": "Refined visual CNN"},
        {"K": 15,  "N": 200, "noise": 0.05, "desc": "Fine-tuned visual CNN"},
    ]
    
    print(f"[{time.strftime('%H:%M:%S')}] Phase 23: Visual SNN-ExIt on TR87")
    print(f"  Input: 64x64 FRAME (pixel observation)")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Iterations: {N_ITERATIONS}")
    print(f"{'='*60}", flush=True)
    
    # Verify frame shape
    import arc_agi
    from arcengine import GameAction
    arcade = arc_agi.Arcade()
    env = arcade.make(GAME_ID)
    obs = env.step(GameAction.RESET)
    frame = np.array(obs.frame[0])
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame range: [{frame.min()}, {frame.max()}]")
    print(f"  Frame dtype: {frame.dtype}")
    print(f"  Unique values: {len(np.unique(frame))}")
    del env, arcade
    
    n_params = sum(p.numel() for p in VisualBrain().parameters())
    print(f"  CNN params: {n_params:,}")
    print(f"  (Phase 21 used MLP on 7 numeric features → FAILED)")
    print(f"{'='*60}", flush=True)
    
    model_path = None
    all_iteration_results = []
    cumulative_miracles = []
    
    for iteration in range(N_ITERATIONS):
        cfg = ITER_CONFIG[iteration] if iteration < len(ITER_CONFIG) else ITER_CONFIG[-1]
        K = cfg["K"]
        N_COLLECT = cfg["N"]
        noise = cfg["noise"]
        
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration + 1}/{N_ITERATIONS}: {cfg['desc']}")
        print(f"  K={K}, N={N_COLLECT}, σ={noise}")
        print(f"  Policy: {'Random' if model_path is None else 'Visual CNN'}")
        print(f"{'='*60}", flush=True)
        
        # ============================================================
        # Step A: Collect miracle trajectories
        # ============================================================
        t0 = time.time()
        print(f"\n  [A] Collecting miracle trajectories...", flush=True)
        
        tasks = [(GAME_ID, K, MAX_STEPS, iteration * 100000 + ep,
                  model_path, noise) for ep in range(N_COLLECT)]
        
        new_miracles = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(collect_best_of_k_visual, tasks))
        
        for r in results:
            if r is not None and r['levels_cleared'] > 0:
                cumulative_miracles.append(r)
                new_miracles += 1
        
        collect_time = time.time() - t0
        print(f"      New miracles: {new_miracles}/{N_COLLECT} ({new_miracles/N_COLLECT*100:.1f}%)")
        print(f"      Total miracles (cumulative): {len(cumulative_miracles)}")
        print(f"      Time: {collect_time:.0f}s", flush=True)
        
        if len(cumulative_miracles) < 5:
            print(f"      Not enough miracles. Continuing...")
            iter_results = {"iteration": iteration+1, "miracles_new": new_miracles,
                          "miracles_total": len(cumulative_miracles), "configs": {}}
            all_iteration_results.append(iter_results)
            continue
        
        # ============================================================
        # Step B: Train Visual CNN on miracle frames
        # ============================================================
        print(f"\n  [B] Self-Distillation on {len(cumulative_miracles)} visual miracles...", flush=True)
        
        all_frames = []
        all_actions = []
        for m in cumulative_miracles:
            for f, a in zip(m['frames'], m['actions']):
                all_frames.append(f)
                all_actions.append(a)
        
        X = torch.tensor(all_frames, dtype=torch.float32).unsqueeze(1)  # (N, 1, 64, 64)
        Y = torch.tensor(all_actions, dtype=torch.long)
        
        # Normalize
        x_max = X.max()
        if x_max > 1.0:
            X = X / x_max
        
        print(f"      Training samples: {len(X)}")
        print(f"      X shape: {X.shape}")
        action_dist = dict(zip(*np.unique(Y.numpy(), return_counts=True)))
        print(f"      Action dist: {action_dist}")
        
        model = VisualBrain(n_actions=4, hidden=128)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        
        model.train()
        best_acc = 0
        for epoch in range(500):
            perm = torch.randperm(len(X))
            batch = perm[:min(128, len(X))]  # Smaller batch for CNN memory
            loss = criterion(model(X[batch], 0.0), Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if (epoch + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    # Evaluate in batches to avoid memory issues
                    all_preds = []
                    for i in range(0, len(X), 256):
                        preds = model(X[i:i+256], 0.0).argmax(1)
                        all_preds.append(preds)
                    all_preds = torch.cat(all_preds)
                    acc = (all_preds == Y).float().mean().item()
                model.train()
                best_acc = max(best_acc, acc)
                print(f"      Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.3f}")
        
        # Save model
        model_path = os.path.join(SCRIPT_DIR, "data", f"visual_exit_tr87_iter{iteration+1}.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({'model': model.state_dict()}, model_path)
        print(f"      Model saved: {model_path}")
        print(f"      Best accuracy: {best_acc:.3f}")
        
        gc.collect()
        
        # ============================================================
        # Step C: Evaluate
        # ============================================================
        N_EVAL = 100
        print(f"\n  [C] Evaluation (N={N_EVAL})...", flush=True)
        
        eval_configs = [
            ("Random K=1", None, 1, 0.0),
            ("Random K=11", None, 11, 0.0),
            (f"VisCNN(i{iteration+1}) K=1", model_path, 1, 0.0),
            (f"VisCNN(i{iteration+1}) K=1 σ=0.1", model_path, 1, 0.1),
            (f"VisCNN(i{iteration+1}) K=1 σ=0.2", model_path, 1, 0.2),
            (f"VisCNN(i{iteration+1}) K=5 σ=0.1", model_path, 5, 0.1),
            (f"VisCNN(i{iteration+1}) K=11 σ=0.1", model_path, 11, 0.1),
        ]
        
        iter_results = {"iteration": iteration+1, "miracles_new": new_miracles,
                       "miracles_total": len(cumulative_miracles),
                       "train_acc": best_acc, "configs": {}}
        
        for name, mp, K_eval, sigma in eval_configs:
            tasks = [(GAME_ID, K_eval, MAX_STEPS, 999999 + ep, mp, sigma)
                     for ep in range(N_EVAL)]
            
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                eval_results = list(executor.map(collect_best_of_k_visual, tasks))
            
            clears = sum(1 for r in eval_results if r and r['levels_cleared'] > 0)
            rate = clears / N_EVAL * 100
            iter_results["configs"][name] = {"clears": clears, "rate": rate}
            
            bar = "█" * int(rate / 2)
            print(f"      {name:35s}: {clears:3d}/{N_EVAL} = {rate:5.1f}%  {bar}", flush=True)
        
        all_iteration_results.append(iter_results)
        
        # Save incrementally
        out_path = os.path.join(RESULTS_DIR, "phase23_visual_exit_tr87.json")
        with open(out_path, "w") as f:
            json.dump(all_iteration_results, f, indent=2)
        
        # Free miracle frame memory to prevent OOM
        gc.collect()
    
    # ============================================================
    # Visualization
    # ============================================================
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    iters_data = [r for r in all_iteration_results if 'train_acc' in r]
    if iters_data:
        iter_nums = [r['iteration'] for r in iters_data]
        
        cnn_k1 = [r['configs'].get(f"VisCNN(i{r['iteration']}) K=1", {}).get('rate', 0) for r in iters_data]
        cnn_k11 = [r['configs'].get(f"VisCNN(i{r['iteration']}) K=11 σ=0.1", {}).get('rate', 0) for r in iters_data]
        random_k11 = [r['configs'].get("Random K=11", {}).get('rate', 0) for r in iters_data]
        
        ax1.plot(iter_nums, cnn_k1, 'o-', color='#E91E63', linewidth=2, label='Visual CNN K=1')
        ax1.plot(iter_nums, cnn_k11, 's-', color='#FF9800', linewidth=2, label='Visual CNN K=11+σ')
        if random_k11:
            ax1.axhline(y=random_k11[0], color='#999', linestyle='--', label='Random K=11')
        
        # Phase 21 reference: numeric ExIt achieved 3%
        ax1.axhline(y=3, color='#2196F3', linestyle=':', linewidth=2, label='Numeric ExIt K=11 (3%)')
    
    ax1.set_xlabel('ExIt Iteration', fontsize=12)
    ax1.set_ylabel('Clear Rate (%)', fontsize=12)
    ax1.set_title('Visual SNN-ExIt on TR87\nPixel Observation vs Numeric (Phase 21)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(-5, 105)
    
    if iters_data:
        ax2.bar(iter_nums, [r['miracles_total'] for r in iters_data], color='#4CAF50', alpha=0.7, label='Cumulative miracles')
        ax2_t = ax2.twinx()
        ax2_t.plot(iter_nums, [r['train_acc']*100 for r in iters_data], 'o-', color='#2196F3', linewidth=2, label='Train accuracy')
        ax2_t.set_ylabel('Train Accuracy (%)', color='#2196F3')
        ax2_t.legend(loc='upper right')
    
    ax2.set_xlabel('ExIt Iteration')
    ax2.set_ylabel('Cumulative Miracles')
    ax2.set_title('Visual Self-Distillation Progress', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "phase23_visual_exit_tr87.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot: {plot_path}")
    plt.close('all')
    
    print(f"\n{'='*60}")
    print(f"  PHASE 23 COMPLETE: Visual SNN-ExIt on TR87")
    print(f"  Pixel observation (64x64) vs Numeric (dim=7)")
    print(f"{'='*60}")
    if iters_data:
        last = iters_data[-1]
        for name, data in last['configs'].items():
            print(f"  {name:35s}: {data['rate']:5.1f}%")
    print(f"\n  Reference: Numeric ExIt (Phase 21) K=11 = 3.0%")
    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 23 complete!", flush=True)
