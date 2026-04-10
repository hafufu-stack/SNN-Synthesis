"""
SNN-Synthesis Agent for ARC-AGI-3 Kaggle Competition
=====================================================
Noisy Beam Search (NBS) + Expert Iteration (ExIt) Agent

Core idea: Inject calibrated noise (stochastic resonance) into a lightweight
CNN policy to produce "miracle" solutions, then self-distill via ExIt.

Paper: https://doi.org/10.5281/zenodo.15188587
GitHub: https://github.com/hifunsk/snn-synthesis

Author: Hiroto Kyan
License: MIT
"""

# ============================================================
# 0. SETUP: Detect Kaggle vs Local environment
# ============================================================
import os, sys, json, time, random, gc, math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    # Install arc-agi from competition wheels
    WHEELS_DIR = '/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'
    if os.path.exists(WHEELS_DIR):
        os.system(f'pip install --no-index --find-links={WHEELS_DIR} arc-agi arcengine 2>/dev/null')
    ENVIRONMENTS_DIR = '/kaggle/input/arc-prize-2026-arc-agi-3/environment_files'
    RESULTS_DIR = '/kaggle/working'
else:
    # Local development
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    ENVIRONMENTS_DIR = os.path.join(SCRIPT_DIR, 'environment_files')
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = ENVIRONMENTS_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 1. CONFIGURATION
# ============================================================
# Adaptive budget: 6h = 21600s, reserve 600s for overhead
TOTAL_BUDGET_SECONDS = 21000 if IS_KAGGLE else 600  # local: 10min test
N_WORKERS = min(cpu_count() - 2, 12) if IS_KAGGLE else min(cpu_count() - 4, 8)
MAX_STEPS_PER_GAME = 300
N_ACTIONS = 4  # ACTION1-4 (some games support up to ACTION7, start with 4)

# NBS parameters (from v5 research)
BOOTSTRAP_K = 200       # K for initial random exploration
EXIT_K = 50             # K for ExIt collection rounds
NOISE_SIGMA = 0.10      # σ for noise injection during NBS
EXIT_ITERATIONS = 2     # number of ExIt self-improvement loops
TRAIN_EPOCHS = 300      # CNN training epochs per ExIt iteration

print(f"{'='*60}")
print(f"SNN-Synthesis Agent for ARC-AGI-3")
print(f"  Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")
print(f"  Workers: {N_WORKERS}")
print(f"  Budget: {TOTAL_BUDGET_SECONDS}s")
print(f"  NBS: K_bootstrap={BOOTSTRAP_K}, K_exit={EXIT_K}, σ={NOISE_SIGMA}")
print(f"  ExIt: {EXIT_ITERATIONS} iterations, {TRAIN_EPOCHS} epochs")
print(f"{'='*60}", flush=True)


# ============================================================
# 2. STATE EXTRACTION
# ============================================================
def extract_game_state(game):
    """Extract numeric features from game object for CNN input."""
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
                for item in val[:8]:
                    if isinstance(item, (int, float, bool)):
                        features.append(float(item))
                    elif isinstance(item, (list, tuple)):
                        for sub in item[:4]:
                            if isinstance(sub, (int, float)):
                                features.append(float(sub))
            elif isinstance(val, dict):
                for k, v in sorted(val.items())[:8]:
                    if isinstance(v, (int, float)):
                        features.append(float(v))
        except:
            pass
    return features


def hash_state(features, precision=2):
    """Hash state for novelty tracking."""
    import hashlib
    rounded = tuple(round(f, precision) for f in features[:32])
    return hashlib.md5(str(rounded).encode()).hexdigest()[:12]


# ============================================================
# 3. STATEBRAIN: Lightweight CNN Policy
# ============================================================
class StateBrain(nn.Module):
    """
    Lightweight MLP policy that maps game state → action logits.
    Supports noise injection for Noisy Beam Search.
    """
    def __init__(self, state_dim, n_actions=4, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x, noise_sigma=0.0):
        h = self.layers[0](x)  # Linear
        h = self.layers[1](h)  # ReLU
        h = self.layers[2](h)  # Dropout
        if noise_sigma > 0:
            h = h + torch.randn_like(h) * noise_sigma
        h = self.layers[3](h)  # Linear
        h = self.layers[4](h)  # ReLU
        h = self.layers[5](h)  # Dropout
        h = self.layers[6](h)  # Output
        return h


# ============================================================
# 4. WORKER FUNCTIONS (run in separate processes)
# ============================================================
def _worker_play_game(args):
    """Worker: play one game episode. Supports random, model, and curiosity modes."""
    game_id, max_steps, seed, model_bytes, state_dim, noise_sigma, use_curiosity = args

    # Re-set environment variables in worker process
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    if IS_KAGGLE:
        os.environ['ENVIRONMENTS_DIR'] = '/kaggle/input/arc-prize-2026-arc-agi-3/environment_files'
    else:
        os.environ['ENVIRONMENTS_DIR'] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'environment_files')

    import arc_agi, random, hashlib
    import torch, torch.nn as nn
    import numpy as np
    from arcengine import GameAction

    ALL_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2,
                   GameAction.ACTION3, GameAction.ACTION4]
    try:
        _ = GameAction.ACTION5
        ALL_ACTIONS.extend([GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7])
    except:
        pass
    n_actions = len(ALL_ACTIONS)

    rng = random.Random(seed)

    # Load model if provided
    model = None
    if model_bytes is not None:
        try:
            import io
            class SB(nn.Module):
                def __init__(self, sd, na, h=256):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(sd, h), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(h, na))
                def forward(self, x, ns=0.0):
                    h = self.layers[0](x); h = self.layers[1](h); h = self.layers[2](h)
                    if ns > 0: h = h + torch.randn_like(h) * ns
                    h = self.layers[3](h); h = self.layers[4](h)
                    h = self.layers[5](h); h = self.layers[6](h)
                    return h
            buf = io.BytesIO(model_bytes)
            data = torch.load(buf, weights_only=True)
            model = SB(data['state_dim'], data.get('n_actions', 4))
            model.load_state_dict(data['model'])
            model.eval()
        except:
            model = None

    # Initialize game
    try:
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        obs = env.step(GameAction.RESET)
        game = env._game
    except Exception as e:
        return None

    # Play
    states_list = []
    actions_list = []
    max_lc = 0
    state_visit_count = {} if use_curiosity else None

    for step in range(max_steps):
        try:
            feats = extract_game_state(game)
        except:
            feats = [0.0] * max(state_dim or 7, 7)
        states_list.append(feats)

        # Curiosity bonus tracking
        if use_curiosity:
            sh = hash_state(feats)
            state_visit_count[sh] = state_visit_count.get(sh, 0) + 1

        # Choose action
        if model is not None and state_dim is not None:
            try:
                x = torch.tensor(feats[:state_dim], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x, ns=noise_sigma)

                # Add curiosity bonus if enabled
                if use_curiosity and state_visit_count:
                    for a_idx in range(min(n_actions, logits.size(1))):
                        sa_key = f"{hash_state(feats)}_{a_idx}"
                        visit_n = state_visit_count.get(sa_key, 0)
                        logits[0, a_idx] += 3.0 / math.sqrt(visit_n + 1)

                probs = torch.softmax(logits[:, :n_actions] / 0.8, dim=1)
                action_idx = torch.multinomial(probs, 1).item()
                action = ALL_ACTIONS[action_idx]
            except:
                action = rng.choice(ALL_ACTIONS[:4])
                action_idx = ALL_ACTIONS.index(action)
        else:
            # Random with optional curiosity
            if use_curiosity and state_visit_count:
                scores = []
                for a_idx in range(min(4, n_actions)):
                    sa_key = f"{hash_state(feats)}_{a_idx}"
                    visit_n = state_visit_count.get(sa_key, 0)
                    scores.append(3.0 / math.sqrt(visit_n + 1) + rng.gauss(0, 0.1))
                    state_visit_count[sa_key] = visit_n  # will update after action
                action_idx = scores.index(max(scores))
                action = ALL_ACTIONS[action_idx]
            else:
                action = rng.choice(ALL_ACTIONS[:4])
                action_idx = ALL_ACTIONS.index(action)

        actions_list.append(action_idx)

        # Track state-action visits for curiosity
        if use_curiosity:
            sa_key = f"{hash_state(feats)}_{action_idx}"
            state_visit_count[sa_key] = state_visit_count.get(sa_key, 0) + 1

        try:
            obs = env.step(action)
            if obs.levels_completed > max_lc:
                max_lc = obs.levels_completed
            if obs.state.value in ('GAME_OVER', 'WIN'):
                break
        except:
            break

    return {
        'states': states_list,
        'actions': actions_list,
        'levels_cleared': max_lc,
        'n_steps': len(actions_list),
        'unique_states': len(state_visit_count) if state_visit_count else 0,
    }


def best_of_k_worker(args):
    """Run K episodes and return best result."""
    game_id, K, max_steps, seed, model_bytes, state_dim, noise_sigma, use_curiosity = args
    best = None
    for k in range(K):
        result = _worker_play_game(
            (game_id, max_steps, seed * 100000 + k, model_bytes, state_dim, noise_sigma, use_curiosity))
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


# ============================================================
# 5. TRAINING: ExIt Self-Improvement Loop
# ============================================================
def train_statebrain(miracles, state_dim, n_actions=4, epochs=300):
    """Train StateBrain MLP on miracle trajectories."""
    if len(miracles) == 0:
        return None, None

    # Build training data
    all_states = []
    all_actions = []
    for m in miracles:
        for s, a in zip(m['states'], m['actions']):
            if len(s) >= state_dim:
                all_states.append(s[:state_dim])
                all_actions.append(min(a, n_actions - 1))

    if len(all_states) < 10:
        return None, None

    X = torch.tensor(all_states, dtype=torch.float32)
    Y = torch.tensor(all_actions, dtype=torch.long)

    # Normalize
    x_mean = X.mean(0, keepdim=True)
    x_std = X.std(0, keepdim=True).clamp(min=1e-6)
    X = (X - x_mean) / x_std

    model = StateBrain(state_dim, n_actions)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    best_acc = 0
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        batch = perm[:min(256, len(X))]
        logits = model(X[batch])
        loss = criterion(logits, Y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X).argmax(1) == Y).float().mean().item()
            model.train()
            best_acc = max(best_acc, acc)

    model.eval()

    # Serialize model to bytes for multiprocessing
    import io
    buf = io.BytesIO()
    torch.save({'model': model.state_dict(), 'state_dim': state_dim,
                'n_actions': n_actions, 'x_mean': x_mean, 'x_std': x_std}, buf)
    model_bytes = buf.getvalue()

    return model_bytes, best_acc


# ============================================================
# 6. MAIN AGENT: Play All Games
# ============================================================
def get_game_list():
    """Get list of available games."""
    games = []
    env_dir = os.environ.get('ENVIRONMENTS_DIR', ENVIRONMENTS_DIR)
    if os.path.exists(env_dir):
        for name in sorted(os.listdir(env_dir)):
            if os.path.isdir(os.path.join(env_dir, name)):
                games.append(name)
    return games


def quick_scan_game(game_id, scan_budget=30):
    """Quick scan: small K bootstrap to estimate miracle rate."""
    t_start = time.time()
    SCAN_K = 20
    SCAN_N = 30

    tasks = [
        (game_id, SCAN_K, MAX_STEPS_PER_GAME, ep, None, None, 0.0, True)
        for ep in range(SCAN_N)
    ]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(executor.map(best_of_k_worker, tasks))

    miracles = [r for r in results if r and r['levels_cleared'] > 0]
    rate = len(miracles) / SCAN_N if SCAN_N > 0 else 0
    elapsed = time.time() - t_start

    print(f"  [{game_id.upper():6s}] Scan: {len(miracles)}/{SCAN_N} ({rate*100:.0f}%) in {elapsed:.0f}s", flush=True)
    return {'game': game_id, 'miracle_rate': rate, 'miracles': miracles, 'scan_time': elapsed}


def play_single_game(game_id, time_budget, initial_miracles=None):
    """Full NBS + ExIt pipeline for a single game."""
    t_start = time.time()

    print(f"\n  [{game_id.upper()}] Starting ExIt (budget: {time_budget:.0f}s)...", flush=True)

    # Use initial miracles from scan or collect more
    if initial_miracles and len(initial_miracles) > 0:
        miracles = list(initial_miracles)
    else:
        # Full bootstrap
        N_BOOTSTRAP = 100
        tasks = [
            (game_id, BOOTSTRAP_K, MAX_STEPS_PER_GAME, ep, None, None, 0.0, True)
            for ep in range(N_BOOTSTRAP)
        ]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(best_of_k_worker, tasks))
        miracles = [r for r in results if r and r['levels_cleared'] > 0]

    if len(miracles) == 0:
        print(f"  [{game_id.upper()}] No miracles, skipping ExIt", flush=True)
        return {'game': game_id, 'miracle_rate': 0, 'best_clears': 0, 'exit_iters': 0}

    # Determine state dimension
    state_dim = max(len(m['states'][0]) for m in miracles if m['states'])
    miracle_rate = len(miracles) / max(len(miracles), 30)

    # ExIt Loop
    cumulative_miracles = list(miracles)
    model_bytes = None
    best_overall = max(m['levels_cleared'] for m in miracles)

    for exit_iter in range(EXIT_ITERATIONS):
        elapsed = time.time() - t_start
        if elapsed > time_budget * 0.85:
            print(f"  [{game_id.upper()}] Time budget hit at ExIt iter {exit_iter+1}", flush=True)
            break

        # Train
        model_bytes, train_acc = train_statebrain(
            cumulative_miracles, state_dim, N_ACTIONS, TRAIN_EPOCHS)
        if model_bytes is None:
            continue

        # Collect with model + noise
        N_COLLECT = 80
        tasks = [
            (game_id, EXIT_K, MAX_STEPS_PER_GAME,
             (exit_iter+1)*1000000 + ep, model_bytes, state_dim, NOISE_SIGMA, True)
            for ep in range(N_COLLECT)
        ]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(best_of_k_worker, tasks))

        new_miracles = [r for r in results if r and r['levels_cleared'] > 0]
        cumulative_miracles.extend(new_miracles)
        if new_miracles:
            best_overall = max(best_overall, max(r['levels_cleared'] for r in new_miracles))

        new_rate = len(new_miracles) / N_COLLECT * 100
        print(f"  [{game_id.upper()}] ExIt {exit_iter+1}: "
              f"+{len(new_miracles)}/{N_COLLECT} ({new_rate:.1f}%), "
              f"acc={train_acc:.3f}, best={best_overall}", flush=True)

    elapsed = time.time() - t_start
    print(f"  [{game_id.upper()}] DONE: {len(cumulative_miracles)} miracles, "
          f"best={best_overall}, time={elapsed:.0f}s", flush=True)

    return {
        'game': game_id,
        'miracle_rate': miracle_rate,
        'best_clears': best_overall,
        'total_miracles': len(cumulative_miracles),
        'exit_iters': EXIT_ITERATIONS,
        'time_seconds': elapsed,
    }


def main():
    t_global_start = time.time()

    games = get_game_list()
    if not games:
        print("ERROR: No games found!", flush=True)
        return

    print(f"\nFound {len(games)} games: {', '.join(games)}")

    # ============================
    # PASS 1: Quick scan all games
    # ============================
    print(f"\n--- Pass 1: Quick Scan (K={20}, N=30 each) ---")
    scan_results = {}
    for game_id in games:
        elapsed = time.time() - t_global_start
        if elapsed > TOTAL_BUDGET_SECONDS * 0.15:
            print(f"  [{game_id.upper():6s}] Scan skipped (time)", flush=True)
            scan_results[game_id] = {'game': game_id, 'miracle_rate': 0, 'miracles': []}
            continue
        scan_results[game_id] = quick_scan_game(game_id)

    # Sort: solvable games first (higher miracle rate = more ExIt benefit)
    solvable = [(gid, sr) for gid, sr in scan_results.items() if sr['miracle_rate'] > 0]
    unsolvable = [(gid, sr) for gid, sr in scan_results.items() if sr['miracle_rate'] == 0]
    solvable.sort(key=lambda x: x[1]['miracle_rate'], reverse=True)

    print(f"\n  Solvable: {len(solvable)} games")
    print(f"  Unsolvable: {len(unsolvable)} games")

    # ============================
    # PASS 2: ExIt on solvable games
    # ============================
    elapsed_after_scan = time.time() - t_global_start
    remaining = TOTAL_BUDGET_SECONDS - elapsed_after_scan
    n_solvable = max(len(solvable), 1)
    time_per_solvable = remaining * 0.90 / n_solvable  # 90% for solvable, 10% reserve

    print(f"\n--- Pass 2: ExIt on {len(solvable)} solvable games ({time_per_solvable:.0f}s each) ---")

    all_results = []
    for game_id, sr in solvable:
        elapsed = time.time() - t_global_start
        remaining = TOTAL_BUDGET_SECONDS - elapsed
        if remaining < 60:
            all_results.append({'game': game_id, 'skipped': True})
            continue
        budget = min(time_per_solvable, remaining - 30)
        result = play_single_game(game_id, budget, sr.get('miracles', []))
        all_results.append(result)

    # Add unsolvable games (already scanned, no ExIt needed)
    for game_id, sr in unsolvable:
        all_results.append({'game': game_id, 'miracle_rate': 0, 'best_clears': 0, 'exit_iters': 0})

    # Final summary
    total_time = time.time() - t_global_start
    print(f"\n{'='*60}")
    print(f"SNN-SYNTHESIS AGENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total games: {len(games)}")
    print(f"  Total time: {total_time:.0f}s")

    solved = [r for r in all_results if r.get('best_clears', 0) > 0]
    print(f"  Games with miracles: {len(solved)}/{len(games)}")
    for r in sorted(all_results, key=lambda x: x.get('best_clears', 0), reverse=True):
        gid = r.get('game', '?')
        if r.get('skipped'):
            print(f"    {gid:6s}: SKIPPED")
        elif r.get('best_clears', 0) > 0:
            print(f"    {gid:6s}: [OK] clears={r['best_clears']}, "
                  f"miracles={r.get('total_miracles', 0)}, "
                  f"time={r.get('time_seconds', 0):.0f}s")
        else:
            print(f"    {gid:6s}: [--] no miracles")
    print(f"{'='*60}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "snn_synthesis_kaggle_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
