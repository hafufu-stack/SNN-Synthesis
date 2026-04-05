"""
Phase 28: ExIt on Intermediate Games — Two-Condition Boundary Mapping
=====================================================================
Sonnet's proposal: Map the learnability boundary between LS20 (99%) and TR87 (3%).
Test ExIt on the 4 untested ARC-AGI-3 games: ft09, g50t, wa30, sb26.

For each game:
  1. Random K=100 bootstrap → miracle rate
  2. ExIt 3 iterations (if miracles > 0)
  3. Train accuracy measurement → learnability score

Output: 2D map of (miracle_rate, learnability) → ExIt success
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
    """Game-agnostic MLP: state → action."""
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
    """Single trajectory for any game."""
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

    states = []
    actions = []
    max_lc = 0

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
        'states': states,
        'actions': actions,
        'levels_cleared': max_lc,
        'n_steps': len(actions),
        'state_dim': len(states[0]) if states else 0
    }


def collect_best_of_k(args):
    """Run K trajectories, return the best."""
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


def run_exit_for_game(game_id, state_dim, n_iters=3):
    """Run the full ExIt pipeline for one game."""
    print(f"\n{'=' * 60}")
    print(f"  ExIt on {game_id.upper()} (state_dim={state_dim})")
    print(f"{'=' * 60}", flush=True)

    ITER_CONFIG = [
        {"K": 100, "N": 200, "noise": 0.0, "desc": "Random bootstrap"},
        {"K": 50, "N": 200, "noise": 0.15, "desc": "CNN + noise"},
        {"K": 30, "N": 200, "noise": 0.10, "desc": "Better CNN"},
    ]

    model_path = None
    iteration_results = []
    cumulative_miracles = []

    for iteration in range(n_iters):
        cfg = ITER_CONFIG[iteration] if iteration < len(ITER_CONFIG) else ITER_CONFIG[-1]
        K = cfg["K"]
        N_COLLECT = cfg["N"]
        noise = cfg["noise"]

        print(f"\n  Iter {iteration + 1}/{n_iters}: {cfg['desc']} (K={K}, N={N_COLLECT})", flush=True)

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

        miracle_rate = new_miracles / N_COLLECT * 100
        print(f"    Miracles: {new_miracles}/{N_COLLECT} ({miracle_rate:.1f}%)")
        print(f"    Cumulative: {len(cumulative_miracles)}")
        print(f"    Time: {time.time() - t0:.0f}s", flush=True)

        if len(cumulative_miracles) < 5:
            print(f"    Insufficient miracles, skipping training")
            iteration_results.append({
                "iteration": iteration + 1, "miracles_new": new_miracles,
                "miracles_total": len(cumulative_miracles),
                "miracle_rate": miracle_rate, "configs": {}
            })
            continue

        # Train
        all_states = []
        all_actions = []
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
                print(f"    Epoch {epoch + 1}: loss={loss.item():.4f}, acc={acc:.3f}")

        model_path = os.path.join(SCRIPT_DIR, "data", f"exit_{game_id}_iter{iteration + 1}.pt")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({'model': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std,
                     'state_dim': state_dim}, model_path)

        # Evaluate
        N_EVAL = 100
        iter_res = {"iteration": iteration + 1, "miracles_new": new_miracles,
                    "miracles_total": len(cumulative_miracles),
                    "miracle_rate": miracle_rate,
                    "train_acc": best_acc, "configs": {}}

        for name, mp, K_eval, sigma in [
            ("Random K=11", None, 11, 0.0),
            (f"CNN K=1", model_path, 1, 0.0),
            (f"CNN K=11 σ=0.1", model_path, 11, 0.1),
        ]:
            tasks = [(game_id, K_eval, 300, 999999 + ep, mp, state_dim, sigma)
                     for ep in range(N_EVAL)]
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                eval_results = list(executor.map(collect_best_of_k, tasks))
            clears = sum(1 for r in eval_results if r and r['levels_cleared'] > 0)
            rate = clears / N_EVAL * 100
            iter_res["configs"][name] = {"clears": clears, "rate": rate}
            bar = "█" * int(rate / 2)
            print(f"    {name:30s}: {clears:3d}/{N_EVAL} = {rate:5.1f}%  {bar}", flush=True)

        iteration_results.append(iter_res)
        gc.collect()

    return iteration_results


def main():
    import arc_agi
    from arcengine import GameAction

    GAMES_TO_TEST = ['ft09', 'g50t', 'wa30', 'sb26']
    all_game_results = {}

    print(f"[{time.strftime('%H:%M:%S')}] Phase 28: ExIt on Intermediate Games")
    print(f"  Games: {GAMES_TO_TEST}")
    print(f"  Workers: {N_WORKERS}")
    print(f"{'=' * 60}", flush=True)

    # Discover state dimensions for each game
    game_dims = {}
    for gid in GAMES_TO_TEST:
        try:
            arcade = arc_agi.Arcade()
            env = arcade.make(gid)
            obs = env.step(GameAction.RESET)
            game = env._game
            feats = extract_game_state(game)
            game_dims[gid] = len(feats)
            print(f"  {gid}: state_dim={len(feats)}")
            del env, arcade
        except Exception as e:
            print(f"  {gid}: FAILED to initialize ({e})")
            game_dims[gid] = 0

    # Run ExIt on each game
    t_total = time.time()
    for gid in GAMES_TO_TEST:
        if game_dims[gid] == 0:
            print(f"\n  Skipping {gid} (failed to initialize)")
            all_game_results[gid] = {"error": "initialization failed"}
            continue

        t0 = time.time()
        results = run_exit_for_game(gid, game_dims[gid], n_iters=3)
        all_game_results[gid] = {
            "state_dim": game_dims[gid],
            "elapsed_s": time.time() - t0,
            "iterations": results
        }

        # Save incrementally
        out_path = os.path.join(RESULTS_DIR, "phase28_intermediate_exit.json")
        with open(out_path, "w") as f:
            json.dump(all_game_results, f, indent=2)

    # ============================================================
    # Two-Condition Map visualization
    # ============================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Miracle Rate vs ExIt Clear Rate
    ax = axes[0]
    # Include known data points
    known = {
        'LS20': {'miracle_rate': 21.0, 'exit_rate': 99.0, 'learnable': True},
        'M0R0': {'miracle_rate': 59.0, 'exit_rate': 92.0, 'learnable': True},
        'TR87': {'miracle_rate': 14.5, 'exit_rate': 3.0, 'learnable': False},
    }

    for gid, info in known.items():
        color = '#4CAF50' if info['learnable'] else '#E91E63'
        marker = 'o' if info['learnable'] else 'x'
        ax.scatter(info['miracle_rate'], info['exit_rate'], c=color, marker=marker,
                   s=150, zorder=5, edgecolors='black', linewidths=1.5)
        ax.annotate(gid, (info['miracle_rate'] + 1, info['exit_rate'] + 2), fontsize=10)

    # Add new results
    for gid in GAMES_TO_TEST:
        if gid not in all_game_results or 'iterations' not in all_game_results[gid]:
            continue
        iters = all_game_results[gid]['iterations']
        if not iters:
            continue
        # First iteration miracle rate
        mr = iters[0].get('miracle_rate', 0)
        # Final exit rate
        final = iters[-1]
        exit_rate = final.get('configs', {}).get('CNN K=11 σ=0.1', {}).get('rate', 0)
        learnable = final.get('train_acc', 0) > 0.30

        color = '#4CAF50' if learnable else '#E91E63'
        marker = 's' if learnable else '^'
        ax.scatter(mr, exit_rate, c=color, marker=marker, s=150, zorder=5,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(gid.upper(), (mr + 1, exit_rate + 2), fontsize=10, fontweight='bold')

    ax.set_xlabel('Bootstrap Miracle Rate (%)', fontsize=12)
    ax.set_ylabel('ExIt Clear Rate (CNN K=11+σ, %)', fontsize=12)
    ax.set_title('Two-Condition Theory Map\nMiracle Rate vs ExIt Success', fontweight='bold')
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='20% threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Summary table
    ax = axes[1]
    ax.axis('off')
    summary = "Two-Condition Theory Summary\n" + "=" * 45 + "\n\n"
    summary += f"{'Game':8s} {'Dim':5s} {'MR%':6s} {'Acc':6s} {'ExIt%':7s} {'Learn':6s}\n"
    summary += "-" * 45 + "\n"

    for gid in ['m0r0', 'ls20'] + GAMES_TO_TEST + ['tr87']:
        if gid in ['m0r0', 'ls20', 'tr87']:
            info = known.get(gid.upper(), {})
            summary += f"{gid:8s} {'?':5s} {info.get('miracle_rate', 0):5.1f}% {'---':6s} {info.get('exit_rate', 0):5.1f}%  {'YES' if info.get('learnable', False) else 'NO':6s}\n"
        elif gid in all_game_results and 'iterations' in all_game_results[gid]:
            iters = all_game_results[gid]['iterations']
            if iters:
                dim = all_game_results[gid].get('state_dim', '?')
                mr = iters[0].get('miracle_rate', 0)
                acc = iters[-1].get('train_acc', 0) * 100 if 'train_acc' in iters[-1] else 0
                er = iters[-1].get('configs', {}).get('CNN K=11 σ=0.1', {}).get('rate', 0)
                learn = "YES" if acc > 30 else "NO"
                summary += f"{gid:8s} {dim:5d} {mr:5.1f}% {acc:5.1f}% {er:5.1f}%  {learn:6s}\n"

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "phase28_intermediate_exit.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"\n  Plot: {plot_path}")

    print(f"\n{'=' * 60}")
    print(f"  PHASE 28 COMPLETE: ExIt on Intermediate Games")
    print(f"  Total time: {time.time() - t_total:.0f}s")
    print(f"{'=' * 60}")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 28 complete!", flush=True)


if __name__ == '__main__':
    main()
