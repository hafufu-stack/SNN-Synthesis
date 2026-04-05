"""
Phase 30: Dynamic K Noisy Beam Search — K Scheduling Ablation on LS20
=====================================================================
Grok's proposal: Does dynamic K (varying across ExIt iterations) outperform fixed K?

Three conditions:
  1. Fixed K=11 (control, Phase 20 reproduction)
  2. Decreasing K: 21→15→11→7→5 (explore-then-exploit)
  3. Increasing K: 5→7→11→15→21 (exploit-then-explore)

Each condition: 5 ExIt iterations on LS20, N=100 eval
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

N_WORKERS = min(cpu_count() - 8, 16)

import torch
import torch.nn as nn
import torch.optim as optim
import logging
logging.disable(logging.CRITICAL)


def extract_game_state(game):
    """Extract ALL numeric features from game state."""
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


class StateBrain(nn.Module):
    def __init__(self, state_dim, n_actions=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.output = nn.Linear(hidden // 2, n_actions)

    def forward(self, x, noise_sigma=0.0):
        h = self.net(x)
        if noise_sigma > 0:
            h = h + torch.randn_like(h) * noise_sigma
        return self.output(h)


def collect_single_trajectory(args):
    game_id, max_steps, seed, model_data_path, state_dim, noise_sigma = args

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

    model = None
    if model_data_path and os.path.exists(model_data_path) and state_dim > 0:
        try:
            class SB(nn.Module):
                def __init__(self, sd, na=4, h=256):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(sd, h), nn.ReLU(), nn.BatchNorm1d(h),
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
            model = SB(state_dim)
            model.load_state_dict(data['model'])
            model.eval()
            x_mean = data['x_mean']
            x_std = data['x_std']
        except:
            model = None

    states, actions, max_lc = [], [], 0

    for step in range(max_steps):
        feats = extract_game_state(game)
        if len(feats) < state_dim:
            feats = feats + [0.0] * (state_dim - len(feats))
        elif len(feats) > state_dim:
            feats = feats[:state_dim]
        states.append(feats)

        if model is not None:
            try:
                x = torch.tensor([feats], dtype=torch.float32)
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
        'states': states, 'actions': actions,
        'levels_cleared': max_lc, 'n_steps': len(actions),
        'state_dim': len(states[0]) if states else 0
    }


def collect_best_of_k(args):
    game_id, K, max_steps, seed, model_path, state_dim, noise_sigma = args
    best = None
    for k in range(K):
        single_args = (game_id, max_steps, seed * 10000 + k, model_path, state_dim, noise_sigma)
        result = collect_single_trajectory(single_args)
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


def run_dynamic_exit(game_id, state_dim, k_schedule, noise_schedule, condition_name):
    """Run ExIt with a given K schedule."""
    print(f"\n{'=' * 60}")
    print(f"  Condition: {condition_name}")
    print(f"  K schedule: {k_schedule}")
    print(f"  σ schedule: {noise_schedule}")
    print(f"{'=' * 60}", flush=True)

    N_ITERATIONS = len(k_schedule)
    N_COLLECT = 200
    N_EVAL = 100

    model_path = None
    iteration_results = []
    cumulative_miracles = []

    for iteration in range(N_ITERATIONS):
        K = k_schedule[iteration]
        noise = noise_schedule[iteration]

        print(f"\n  Iter {iteration + 1}/{N_ITERATIONS}: K={K}, σ={noise}", flush=True)

        # Collect
        t0 = time.time()
        tasks = [(game_id, K, 300, iteration * 100000 + ep,
                  model_path, state_dim, noise) for ep in range(N_COLLECT)]

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(collect_best_of_k, tasks))

        new_miracles = 0
        for r in results:
            if r is not None and r['levels_cleared'] > 0:
                cumulative_miracles.append(r)
                new_miracles += 1

        collect_time = time.time() - t0
        print(f"    Miracles: {new_miracles}/{N_COLLECT} ({new_miracles / N_COLLECT * 100:.1f}%)")
        print(f"    Cumulative: {len(cumulative_miracles)}")
        print(f"    Time: {collect_time:.0f}s", flush=True)

        if len(cumulative_miracles) < 5:
            iteration_results.append({
                "iteration": iteration + 1, "K": K, "noise": noise,
                "miracles_new": new_miracles,
                "miracles_total": len(cumulative_miracles), "configs": {}
            })
            continue

        # Train
        all_states, all_actions = [], []
        for m in cumulative_miracles:
            for s, a in zip(m['states'], m['actions']):
                if len(s) == state_dim:
                    all_states.append(s)
                    all_actions.append(a)

        X = torch.tensor(all_states, dtype=torch.float32)
        Y = torch.tensor(all_actions, dtype=torch.long)
        x_mean = X.mean(0)
        x_std = X.std(0) + 1e-8
        X_norm = (X - x_mean) / x_std

        model = StateBrain(state_dim, n_actions=4, hidden=256)
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

            if (epoch + 1) % 250 == 0:
                model.eval()
                with torch.no_grad():
                    acc = (model(X_norm, 0.0).argmax(1) == Y).float().mean().item()
                model.train()
                best_acc = max(best_acc, acc)

        model_path = os.path.join(SCRIPT_DIR, "data",
                                   f"dynamic_exit_{condition_name}_iter{iteration + 1}.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({'model': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std,
                     'state_dim': state_dim}, model_path)

        # Evaluate
        iter_res = {"iteration": iteration + 1, "K": K, "noise": noise,
                    "miracles_new": new_miracles,
                    "miracles_total": len(cumulative_miracles),
                    "train_acc": best_acc, "configs": {}}

        for name, mp, K_eval, sigma in [
            ("CNN K=1", model_path, 1, 0.0),
            ("CNN K=1 σ=0.1", model_path, 1, 0.1),
            ("CNN K=5 σ=0.1", model_path, 5, 0.1),
            ("CNN K=11 σ=0.1", model_path, 11, 0.1),
        ]:
            tasks = [(game_id, K_eval, 300, 999999 + ep, mp, state_dim, sigma)
                     for ep in range(N_EVAL)]
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                eval_results = list(executor.map(collect_best_of_k, tasks))
            clears = sum(1 for r in eval_results if r and r['levels_cleared'] > 0)
            rate = clears / N_EVAL * 100
            iter_res["configs"][name] = {"clears": clears, "rate": rate}
            bar = "█" * int(rate / 2)
            print(f"    {name:25s}: {clears:3d}/{N_EVAL} = {rate:5.1f}%  {bar}", flush=True)

        iteration_results.append(iter_res)
        gc.collect()

    return iteration_results


def main():
    import arc_agi
    from arcengine import GameAction

    GAME_ID = "ls20"

    print(f"[{time.strftime('%H:%M:%S')}] Phase 30: Dynamic K Noisy Beam Search")
    print(f"  Game: {GAME_ID}")
    print(f"  Workers: {N_WORKERS}")
    print(f"{'=' * 60}", flush=True)

    # Discover state dim
    arcade = arc_agi.Arcade()
    env = arcade.make(GAME_ID)
    obs = env.step(GameAction.RESET)
    game = env._game
    state_feats = extract_game_state(game)
    STATE_DIM = len(state_feats)
    print(f"  State dim: {STATE_DIM}")
    del env, arcade

    # Define K schedules
    conditions = [
        {
            "name": "fixed_K11",
            "k_schedule": [100, 11, 11, 11, 11],  # iter0=bootstrap, rest=K=11
            "noise_schedule": [0.0, 0.10, 0.10, 0.10, 0.10],
        },
        {
            "name": "decreasing_K",
            "k_schedule": [100, 21, 15, 11, 7],  # explore → exploit
            "noise_schedule": [0.0, 0.15, 0.12, 0.10, 0.08],
        },
        {
            "name": "increasing_K",
            "k_schedule": [100, 5, 7, 11, 15],  # exploit → explore
            "noise_schedule": [0.0, 0.05, 0.08, 0.10, 0.12],
        },
    ]

    all_results = {}
    t_total = time.time()

    for cond in conditions:
        t0 = time.time()
        results = run_dynamic_exit(
            GAME_ID, STATE_DIM,
            cond["k_schedule"], cond["noise_schedule"], cond["name"]
        )
        all_results[cond["name"]] = {
            "k_schedule": cond["k_schedule"],
            "noise_schedule": cond["noise_schedule"],
            "elapsed_s": time.time() - t0,
            "iterations": results
        }

        # Save incrementally
        out_path = os.path.join(RESULTS_DIR, "phase30_dynamic_k_exit.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # ============================================================
    # Visualization
    # ============================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {'fixed_K11': '#4CAF50', 'decreasing_K': '#E91E63', 'increasing_K': '#2196F3'}
    markers = {'fixed_K11': 'o', 'decreasing_K': 's', 'increasing_K': '^'}

    # Panel 1: CNN K=11+σ clear rate across iterations
    ax = axes[0]
    for name, data in all_results.items():
        iters = data['iterations']
        iters_with_data = [r for r in iters if 'train_acc' in r]
        if iters_with_data:
            iter_nums = [r['iteration'] for r in iters_with_data]
            rates = [r['configs'].get('CNN K=11 σ=0.1', {}).get('rate', 0) for r in iters_with_data]
            k_labels = [f"K={r['K']}" for r in iters_with_data]
            ax.plot(iter_nums, rates, f'{markers[name]}-', color=colors[name],
                    linewidth=2, markersize=8, label=f'{name} ({data["k_schedule"][1:]})')

    ax.axhline(y=99, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label='Phase 20 target')
    ax.set_xlabel('ExIt Iteration', fontsize=12)
    ax.set_ylabel('Clear Rate (CNN K=11+σ, %)', fontsize=12)
    ax.set_title('Dynamic vs Fixed K Scheduling\n(LS20, CNN K=11 eval)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 105)

    # Panel 2: CNN K=1 (policy quality)
    ax = axes[1]
    for name, data in all_results.items():
        iters = data['iterations']
        iters_with_data = [r for r in iters if 'train_acc' in r]
        if iters_with_data:
            iter_nums = [r['iteration'] for r in iters_with_data]
            rates = [r['configs'].get('CNN K=1', {}).get('rate', 0) for r in iters_with_data]
            ax.plot(iter_nums, rates, f'{markers[name]}-', color=colors[name],
                    linewidth=2, markersize=8, label=name)

    ax.set_xlabel('ExIt Iteration', fontsize=12)
    ax.set_ylabel('Clear Rate (CNN K=1, %)', fontsize=12)
    ax.set_title('Policy Quality (No Beam Search)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "phase30_dynamic_k_exit.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"\n  Plot: {plot_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  PHASE 30 COMPLETE: Dynamic K Noisy Beam Search")
    print(f"  Total time: {time.time() - t_total:.0f}s")
    print(f"{'=' * 60}")
    for name, data in all_results.items():
        iters = data['iterations']
        final = [r for r in iters if 'train_acc' in r]
        if final:
            rate = final[-1]['configs'].get('CNN K=11 σ=0.1', {}).get('rate', 0)
            print(f"  {name:20s}: {rate:.1f}% (K schedule: {data['k_schedule'][1:]})")

    print(f"\n[{time.strftime('%H:%M:%S')}] Phase 30 complete!", flush=True)


if __name__ == '__main__':
    main()
