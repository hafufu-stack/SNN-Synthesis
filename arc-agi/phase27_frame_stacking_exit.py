"""
Phase 27: TR87 Frame Stacking ExIt — Memory Breaks the Learnability Wall
=========================================================================
Deep Think's proposal: TR87 fails because CNN sees only 1 frame (Markov).
Frame Stacking (past 3 frames) gives temporal context → may fix learnability.

Two-Condition Theory predicts: if train accuracy rises above chance (25%),
ExIt should start working on TR87.

Control: Phase 21 result (dim=4/7, train_acc ~chance, ExIt = 3%)
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

N_WORKERS = min(cpu_count() - 8, 16)  # Leave cores for GPU track

import torch
import torch.nn as nn
import torch.optim as optim
import logging
logging.disable(logging.CRITICAL)


# ============================================================
# State extraction with Frame Stacking
# ============================================================

def extract_game_state(game):
    """Extract numeric features from game state (same as Phase 20/21)."""
    features = []
    for attr in sorted(dir(game)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(game, attr)
            if callable(val):
                continue
            if isinstance(val, bool):
                features.append(float(val))
            elif isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, str):
                features.append(float(hash(val) % 1000) / 1000.0)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, (int, float)):
                        features.append(float(item))
                    elif isinstance(item, bool):
                        features.append(float(item))
                    elif isinstance(item, (tuple, list)):
                        for sub in item:
                            if isinstance(sub, (int, float)):
                                features.append(float(sub))
                    elif isinstance(item, str):
                        features.append(float(hash(item) % 1000) / 1000.0)
            elif isinstance(val, dict):
                for k, v in sorted(val.items()):
                    if isinstance(v, (int, float)):
                        features.append(float(v))
                    elif isinstance(v, (tuple, list)):
                        for sub in v:
                            if isinstance(sub, (int, float)):
                                features.append(float(sub))
                    elif isinstance(v, str):
                        features.append(float(hash(v) % 1000) / 1000.0)
        except:
            pass
    return features


class FrameStackBrain(nn.Module):
    """MLP with frame stacking: input = [state_t-2, state_t-1, state_t]."""
    def __init__(self, single_dim, n_frames=3, n_actions=4, hidden=256):
        super().__init__()
        total_dim = single_dim * n_frames
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.output = nn.Linear(hidden // 2, n_actions)
        self.single_dim = single_dim
        self.n_frames = n_frames

    def forward(self, x, noise_sigma=0.0):
        h = self.net(x)
        if noise_sigma > 0:
            h = h + torch.randn_like(h) * noise_sigma
        return self.output(h)


# ============================================================
# Worker: collect trajectory with frame stacking
# ============================================================

def collect_single_trajectory(args):
    """Single trajectory with frame stacking for TR87."""
    game_id, max_steps, seed, model_data_path, single_dim, n_frames, noise_sigma = args

    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')

    import arc_agi, random, torch, torch.nn as nn
    import logging
    logging.disable(logging.CRITICAL)
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

    # Load model if provided
    model = None
    if model_data_path and os.path.exists(model_data_path) and single_dim > 0:
        try:
            total_dim = single_dim * n_frames

            class FSB(nn.Module):
                def __init__(self, td, na=4, h=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(td, h), nn.ReLU(), nn.BatchNorm1d(h),
                        nn.Dropout(0.1),
                        nn.Linear(h, h), nn.ReLU(), nn.BatchNorm1d(h),
                        nn.Dropout(0.1),
                        nn.Linear(h, h // 2), nn.ReLU(),
                    )
                    self.output = nn.Linear(h // 2, na)

                def forward(self, x, ns=0.0):
                    h = self.net(x)
                    if ns > 0:
                        h = h + torch.randn_like(h) * ns
                    return self.output(h)

            data = torch.load(model_data_path, weights_only=True)
            model = FSB(total_dim)
            model.load_state_dict(data['model'])
            model.eval()
            x_mean = data['x_mean']
            x_std = data['x_std']
        except:
            model = None

    # Frame buffer for stacking
    frame_buffer = []
    zero_frame = [0.0] * single_dim

    all_stacked_states = []
    all_actions = []
    max_lc = 0

    for step in range(max_steps):
        feats = extract_game_state(game)

        # Pad or truncate to single_dim
        if len(feats) < single_dim:
            feats = feats + [0.0] * (single_dim - len(feats))
        elif len(feats) > single_dim:
            feats = feats[:single_dim]

        frame_buffer.append(feats)
        if len(frame_buffer) > n_frames:
            frame_buffer.pop(0)

        # Build stacked state: [frame_t-(n-1), ..., frame_t-1, frame_t]
        stacked = []
        for i in range(n_frames):
            idx = i - (n_frames - len(frame_buffer))
            if idx < 0:
                stacked.extend(zero_frame)
            else:
                stacked.extend(frame_buffer[idx])

        all_stacked_states.append(stacked)

        if model is not None:
            try:
                x = torch.tensor([stacked], dtype=torch.float32)
                x = (x - x_mean) / x_std
                with torch.no_grad():
                    logits = model(x, ns=noise_sigma)
                    probs = torch.softmax(logits / max(0.5, 1.0 - noise_sigma), dim=1)
                    action_idx = torch.multinomial(probs, 1).item()
                action = ALL_A[action_idx]
            except:
                action = rng.choice(ALL_A)
        else:
            action = rng.choice(ALL_A)

        all_actions.append(ALL_A.index(action))

        try:
            obs = env.step(action)
            if obs.levels_completed > max_lc:
                max_lc = obs.levels_completed
            if obs.state.value in ('GAME_OVER', 'WIN'):
                break
        except:
            break

    return {
        'states': all_stacked_states,
        'actions': all_actions,
        'levels_cleared': max_lc,
        'n_steps': len(all_actions),
        'stacked_dim': len(all_stacked_states[0]) if all_stacked_states else 0
    }


def collect_best_of_k(args):
    """Run K trajectories, return the best."""
    game_id, K, max_steps, seed, model_path, single_dim, n_frames, noise_sigma = args

    best = None
    for k in range(K):
        single_args = (game_id, max_steps, seed * 10000 + k,
                       model_path, single_dim, n_frames, noise_sigma)
        result = collect_single_trajectory(single_args)
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


def main():
    import arc_agi
    from arcengine import GameAction

    GAME_ID = "tr87"
    N_ITERATIONS = 5
    MAX_STEPS = 300
    N_FRAMES = 3  # Stack 3 frames

    ITER_CONFIG = [
        {"K": 100, "N": 300, "noise": 0.0, "desc": "Random bootstrap (frame-stacked)"},
        {"K": 50, "N": 300, "noise": 0.15, "desc": "FS-CNN + noise"},
        {"K": 30, "N": 300, "noise": 0.10, "desc": "Better FS-CNN"},
        {"K": 20, "N": 300, "noise": 0.08, "desc": "Refined FS-CNN"},
        {"K": 15, "N": 300, "noise": 0.05, "desc": "Fine-tuned FS-CNN"},
    ]

    print(f"[{time.strftime('%H:%M:%S')}] Phase 27: TR87 Frame Stacking ExIt")
    print(f"  Frame stacking: {N_FRAMES} frames")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Iterations: {N_ITERATIONS}")
    print(f"  Game: {GAME_ID}")
    print(f"{'=' * 60}", flush=True)

    # Discover single-frame state dimension
    arcade = arc_agi.Arcade()
    env = arcade.make(GAME_ID)
    obs = env.step(GameAction.RESET)
    game = env._game
    state_feats = extract_game_state(game)
    SINGLE_DIM = len(state_feats)
    STACKED_DIM = SINGLE_DIM * N_FRAMES
    print(f"  Single frame dim: {SINGLE_DIM}")
    print(f"  Stacked dim: {STACKED_DIM} ({N_FRAMES} × {SINGLE_DIM})")
    print(f"  State sample (first 10): {state_feats[:10]}", flush=True)
    del env, arcade

    model_path = None
    all_iteration_results = []
    cumulative_miracles = []

    for iteration in range(N_ITERATIONS):
        cfg = ITER_CONFIG[iteration] if iteration < len(ITER_CONFIG) else ITER_CONFIG[-1]
        K = cfg["K"]
        N_COLLECT = cfg["N"]
        noise = cfg["noise"]

        print(f"\n{'=' * 60}")
        print(f"  ITERATION {iteration + 1}/{N_ITERATIONS}: {cfg['desc']}")
        print(f"  K={K}, N={N_COLLECT}, σ={noise}")
        print(f"  Policy: {'Random' if model_path is None else 'FS-CNN'}")
        print(f"{'=' * 60}", flush=True)

        # Step A: Collect miracle trajectories
        t0 = time.time()
        print(f"\n  [A] Collecting miracle trajectories...", flush=True)

        tasks = [(GAME_ID, K, MAX_STEPS, iteration * 100000 + ep,
                  model_path, SINGLE_DIM, N_FRAMES, noise) for ep in range(N_COLLECT)]

        new_miracles = 0
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(collect_best_of_k, tasks))

        for r in results:
            if r is not None and r['levels_cleared'] > 0:
                cumulative_miracles.append(r)
                new_miracles += 1

        collect_time = time.time() - t0
        print(f"      New miracles: {new_miracles}/{N_COLLECT} ({new_miracles / N_COLLECT * 100:.1f}%)")
        print(f"      Total miracles (cumulative): {len(cumulative_miracles)}")
        print(f"      Time: {collect_time:.0f}s", flush=True)

        if len(cumulative_miracles) < 5:
            print(f"      Not enough miracles yet. Continuing...")
            iter_results = {"iteration": iteration + 1, "miracles_new": new_miracles,
                            "miracles_total": len(cumulative_miracles), "configs": {}}
            all_iteration_results.append(iter_results)
            continue

        # Step B: Train FS-CNN on ALL cumulative miracles
        print(f"\n  [B] Self-Distillation on {len(cumulative_miracles)} miracles...", flush=True)

        all_states = []
        all_actions = []
        for m in cumulative_miracles:
            for s, a in zip(m['states'], m['actions']):
                if len(s) == STACKED_DIM:
                    all_states.append(s)
                    all_actions.append(a)

        X = torch.tensor(all_states, dtype=torch.float32)
        Y = torch.tensor(all_actions, dtype=torch.long)

        x_mean = X.mean(0)
        x_std = X.std(0) + 1e-8
        X_norm = (X - x_mean) / x_std

        print(f"      Training samples: {len(X)}")
        action_dist = dict(zip(*np.unique(Y.numpy(), return_counts=True)))
        print(f"      Action dist: {action_dist}")

        model = FrameStackBrain(SINGLE_DIM, n_frames=N_FRAMES, n_actions=4, hidden=256)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"      Model params: {n_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

        model.train()
        best_acc = 0
        for epoch in range(500):
            perm = torch.randperm(len(X_norm))
            batch = perm[:min(512, len(X_norm))]
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
                print(f"      Epoch {epoch + 1}: loss={loss.item():.4f}, acc={acc:.3f}")

        model_path = os.path.join(SCRIPT_DIR, "data", f"exit_tr87_fs_iter{iteration + 1}.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({'model': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std,
                     'single_dim': SINGLE_DIM, 'n_frames': N_FRAMES}, model_path)
        print(f"      Model saved: {model_path}")
        print(f"      Best accuracy: {best_acc:.3f} (chance=25%)")

        # *** KEY METRIC: Did frame stacking improve learnability? ***
        learnability_improved = best_acc > 0.30  # Above chance by 5pp
        print(f"      Learnability improved: {learnability_improved} (acc={best_acc:.3f} vs chance=0.25)")

        # Step C: Evaluate
        N_EVAL = 100
        print(f"\n  [C] Evaluation (N={N_EVAL})...", flush=True)

        eval_configs = [
            ("Random K=1", None, 1, 0.0),
            ("Random K=11", None, 11, 0.0),
            (f"FS-CNN(i{iteration + 1}) K=1", model_path, 1, 0.0),
            (f"FS-CNN(i{iteration + 1}) K=1 σ=0.1", model_path, 1, 0.1),
            (f"FS-CNN(i{iteration + 1}) K=5 σ=0.1", model_path, 5, 0.1),
            (f"FS-CNN(i{iteration + 1}) K=11 σ=0.1", model_path, 11, 0.1),
        ]

        iter_results = {"iteration": iteration + 1, "miracles_new": new_miracles,
                        "miracles_total": len(cumulative_miracles),
                        "train_acc": best_acc,
                        "learnability_improved": learnability_improved,
                        "configs": {}}

        for name, mp, K_eval, sigma in eval_configs:
            tasks = [(GAME_ID, K_eval, MAX_STEPS, 999999 + ep,
                      mp, SINGLE_DIM, N_FRAMES, sigma)
                     for ep in range(N_EVAL)]

            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                eval_results = list(executor.map(collect_best_of_k, tasks))

            clears = sum(1 for r in eval_results if r and r['levels_cleared'] > 0)
            rate = clears / N_EVAL * 100
            iter_results["configs"][name] = {"clears": clears, "rate": rate}

            bar = "█" * int(rate / 2)
            print(f"      {name:35s}: {clears:3d}/{N_EVAL} = {rate:5.1f}%  {bar}", flush=True)

        all_iteration_results.append(iter_results)

        # Save incrementally
        out_path = os.path.join(RESULTS_DIR, "phase27_frame_stacking_exit.json")
        with open(out_path, "w") as f:
            json.dump(all_iteration_results, f, indent=2)

        gc.collect()

    # ============================================================
    # Visualization
    # ============================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    iters_data = [r for r in all_iteration_results if 'train_acc' in r]

    if iters_data:
        iter_nums = [r['iteration'] for r in iters_data]

        # Panel 1: Clear rate across iterations
        ax = axes[0]
        cnn_k11 = [r['configs'].get(f"FS-CNN(i{r['iteration']}) K=11 σ=0.1", {}).get('rate', 0)
                    for r in iters_data]
        cnn_k1 = [r['configs'].get(f"FS-CNN(i{r['iteration']}) K=1", {}).get('rate', 0)
                   for r in iters_data]
        random_k11 = [r['configs'].get("Random K=11", {}).get('rate', 0) for r in iters_data]

        ax.plot(iter_nums, cnn_k11, 's-', color='#FF9800', linewidth=2, label='FS-CNN K=11+σ')
        ax.plot(iter_nums, cnn_k1, 'o-', color='#E91E63', linewidth=2, label='FS-CNN K=1')
        if random_k11:
            ax.axhline(y=random_k11[0], color='#999', linestyle='--', label='Random K=11')
        ax.axhline(y=3, color='#2196F3', linestyle=':', linewidth=2,
                   label='Phase 21 (no stacking) = 3%')
        ax.set_xlabel('ExIt Iteration')
        ax.set_ylabel('Clear Rate (%)')
        ax.set_title('Frame Stacking ExIt on TR87\nvs Phase 21 (No Memory)', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Panel 2: Train accuracy (learnability)
        ax = axes[1]
        accs = [r['train_acc'] * 100 for r in iters_data]
        ax.bar(iter_nums, accs, color='#4CAF50', alpha=0.7)
        ax.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Chance (25%)')
        ax.axhline(y=47, color='#FF9800', linestyle=':', linewidth=2,
                   label='Phase 24 dim=4 (47%, overfitting)')
        ax.set_xlabel('ExIt Iteration')
        ax.set_ylabel('Train Accuracy (%)')
        ax.set_title('Learnability Test\nFrame Stacking vs Baseline', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Panel 3: Cumulative miracles
        ax = axes[2]
        ax.bar(iter_nums, [r['miracles_total'] for r in iters_data],
               color='#2196F3', alpha=0.7)
        ax.set_xlabel('ExIt Iteration')
        ax.set_ylabel('Cumulative Miracles')
        ax.set_title('Training Data Growth', fontweight='bold')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "phase27_frame_stacking_exit.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"\n  Plot: {plot_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  PHASE 27 COMPLETE: TR87 Frame Stacking ExIt")
    print(f"  {'Memory BREAKS the wall!' if any(r.get('learnability_improved', False) for r in all_iteration_results) else 'Memory does NOT help on TR87'}")
    print(f"{'=' * 60}")
    if iters_data:
        last = iters_data[-1]
        for name, data in last['configs'].items():
            print(f"  {name:35s}: {data['rate']:5.1f}%")
        print(f"\n  Train accuracy: {last['train_acc']:.3f} (chance=0.25)")
    print(f"\n  Reference: Phase 21 (no stacking) K=11 = 3.0%")
    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 27 complete!", flush=True)


if __name__ == '__main__':
    main()
