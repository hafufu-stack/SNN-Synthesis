"""
Phase 36: Curiosity-Driven NBS — Breaking the Activation Energy Wall
=====================================================================
4 games (ft09, g50t, wa30, sb26) have miracle rate = 0% with random exploration.
This experiment adds intrinsic motivation (novelty bonus) to NBS:
- Track visited states via hashing
- Bias action selection toward unvisited states
- Test if curiosity-driven exploration achieves miracle rate > 0%
"""
import sys, os, json, time, random, gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import hashlib

os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'environment_files')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_WORKERS = min(cpu_count() - 4, 16)

# ============================================================
# State extraction + hashing
# ============================================================
def extract_game_state(game):
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


def hash_state(features, precision=2):
    """Convert state features to a hash for novelty tracking."""
    rounded = tuple(round(f, precision) for f in features)
    return hashlib.md5(str(rounded).encode()).hexdigest()[:16]


# ============================================================
# Workers
# ============================================================
def random_trajectory(args):
    """Standard random trajectory (baseline)."""
    game_id, max_steps, seed = args
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')

    import arc_agi, random, hashlib
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

    max_lc = 0
    unique_states = set()
    for step in range(max_steps):
        try:
            feats = extract_game_state(game)
            h = hash_state(feats)
            unique_states.add(h)
        except:
            pass
        action = rng.choice(ALL_A)
        try:
            obs = env.step(action)
            if obs.levels_completed > max_lc:
                max_lc = obs.levels_completed
            if obs.state.value in ('GAME_OVER', 'WIN'):
                break
        except:
            break

    return {'levels_cleared': max_lc, 'unique_states': len(unique_states),
            'n_steps': step + 1}


def curiosity_trajectory(args):
    """Curiosity-driven trajectory with novelty bonus."""
    game_id, max_steps, seed, curiosity_coeff = args
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')

    import arc_agi, random, hashlib, math
    import numpy as np
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    rng = random.Random(seed)
    n_actions = len(ALL_A)

    try:
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        obs = env.step(GameAction.RESET)
        game = env._game
    except:
        return None

    state_visit_count = {}  # hash -> count
    max_lc = 0
    states_list = []
    actions_list = []
    unique_states = set()

    for step in range(max_steps):
        try:
            feats = extract_game_state(game)
        except:
            feats = [0.0] * 10

        current_hash = hash_state(feats)
        unique_states.add(current_hash)
        state_visit_count[current_hash] = state_visit_count.get(current_hash, 0) + 1

        states_list.append(feats)

        # Try each action virtually (lookahead) and compute novelty bonus
        action_scores = np.zeros(n_actions)
        for a_idx, a in enumerate(ALL_A):
            try:
                # We can't truly simulate without cloning env, so we use
                # a heuristic: prefer actions we haven't tried from this state
                state_action_key = f"{current_hash}_{a_idx}"
                if state_action_key not in state_visit_count:
                    # Novelty bonus for untried state-action pair
                    action_scores[a_idx] += curiosity_coeff
                else:
                    visit_n = state_visit_count.get(state_action_key, 0)
                    action_scores[a_idx] += curiosity_coeff / math.sqrt(visit_n + 1)
            except:
                pass

            # Add small random noise for tie-breaking
            action_scores[a_idx] += rng.gauss(0, 0.1)

        # Softmax selection with curiosity-weighted scores
        max_score = max(action_scores)
        exp_scores = [math.exp(s - max_score) for s in action_scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        # Weighted random selection
        r = rng.random()
        cumsum = 0
        chosen_idx = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                chosen_idx = i
                break

        action = ALL_A[chosen_idx]
        actions_list.append(chosen_idx)

        # Track state-action visit
        sa_key = f"{current_hash}_{chosen_idx}"
        state_visit_count[sa_key] = state_visit_count.get(sa_key, 0) + 1

        try:
            obs = env.step(action)
            if obs.levels_completed > max_lc:
                max_lc = obs.levels_completed
            if obs.state.value in ('GAME_OVER', 'WIN'):
                break
        except:
            break

    return {'levels_cleared': max_lc, 'unique_states': len(unique_states),
            'n_steps': step + 1, 'states': states_list, 'actions': actions_list}


def best_of_k_random(args):
    game_id, K, max_steps, seed = args
    best = None
    for k in range(K):
        result = random_trajectory((game_id, max_steps, seed * 10000 + k))
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


def best_of_k_curiosity(args):
    game_id, K, max_steps, seed, curiosity_coeff = args
    best = None
    for k in range(K):
        result = curiosity_trajectory((game_id, max_steps, seed * 10000 + k, curiosity_coeff))
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


# ============================================================
# Main
# ============================================================
def main():
    GAMES = ['ft09', 'g50t', 'wa30', 'sb26']  # All miracle-rate-0% games
    MAX_STEPS = 300
    CURIOSITY_COEFFS = [1.0, 3.0, 5.0]

    print(f"[{time.strftime('%H:%M:%S')}] Phase 36: Curiosity-Driven NBS")
    print(f"  Games: {GAMES}")
    print(f"  Curiosity coefficients: {CURIOSITY_COEFFS}")
    print(f"  Workers: {N_WORKERS}")
    print(f"{'='*60}", flush=True)

    all_results = {}

    for game_id in GAMES:
        print(f"\n{'='*60}")
        print(f"  Game: {game_id.upper()}")
        print(f"{'='*60}", flush=True)

        game_results = {}

        # Baseline: Random K=100
        print(f"\n  [Baseline] Random K=100...", flush=True)
        N_TRIALS = 100
        tasks = [(game_id, 100, MAX_STEPS, ep) for ep in range(N_TRIALS)]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(best_of_k_random, tasks))
        clears = sum(1 for r in results if r and r['levels_cleared'] > 0)
        avg_unique = np.mean([r['unique_states'] for r in results if r])
        game_results["random_K100"] = {
            "clears": clears, "rate": clears/N_TRIALS*100,
            "avg_unique_states": float(avg_unique)}
        print(f"    Random K=100: {clears}/{N_TRIALS} ({clears/N_TRIALS*100:.1f}%) "
              f"avg_unique_states={avg_unique:.0f}")

        # Curiosity-driven with different coefficients
        for coeff in CURIOSITY_COEFFS:
            print(f"\n  [Curiosity c={coeff}] K=100...", flush=True)
            tasks = [(game_id, 100, MAX_STEPS, 1000000 + ep, coeff) for ep in range(N_TRIALS)]
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                results = list(executor.map(best_of_k_curiosity, tasks))
            clears = sum(1 for r in results if r and r['levels_cleared'] > 0)
            avg_unique = np.mean([r['unique_states'] for r in results if r])
            label = f"curiosity_c{coeff}_K100"
            game_results[label] = {
                "clears": clears, "rate": clears/N_TRIALS*100,
                "avg_unique_states": float(avg_unique),
                "curiosity_coeff": coeff}
            bar = "█" * int(clears/N_TRIALS*50)
            print(f"    Curiosity c={coeff} K=100: {clears}/{N_TRIALS} ({clears/N_TRIALS*100:.1f}%) "
                  f"avg_unique_states={avg_unique:.0f}  {bar}")

        # Higher K for best curiosity coefficient
        best_coeff = max(CURIOSITY_COEFFS,
                        key=lambda c: game_results.get(f"curiosity_c{c}_K100", {}).get("rate", 0))
        print(f"\n  [Best Curiosity c={best_coeff}] K=200...", flush=True)
        tasks = [(game_id, 200, MAX_STEPS, 2000000 + ep, best_coeff) for ep in range(N_TRIALS)]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            results = list(executor.map(best_of_k_curiosity, tasks))
        clears = sum(1 for r in results if r and r['levels_cleared'] > 0)
        avg_unique = np.mean([r['unique_states'] for r in results if r])
        game_results[f"curiosity_c{best_coeff}_K200"] = {
            "clears": clears, "rate": clears/N_TRIALS*100,
            "avg_unique_states": float(avg_unique),
            "curiosity_coeff": best_coeff}
        bar = "█" * int(clears/N_TRIALS*50)
        print(f"    Curiosity c={best_coeff} K=200: {clears}/{N_TRIALS} ({clears/N_TRIALS*100:.1f}%) "
              f"avg_unique_states={avg_unique:.0f}  {bar}")

        all_results[game_id] = game_results

        # Save incrementally
        out_path = os.path.join(RESULTS_DIR, "phase36_curiosity_nbs.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"PHASE 36 SUMMARY: Curiosity-Driven NBS")
    print(f"{'='*60}")
    for gid, gres in all_results.items():
        print(f"\n  {gid.upper()}:")
        for label, data in gres.items():
            print(f"    {label:30s}: {data['rate']:5.1f}%  unique_states={data['avg_unique_states']:.0f}")
    print(f"{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 36 complete!", flush=True)


if __name__ == '__main__':
    main()
