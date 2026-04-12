"""
Phase 45: Zero-Overhead Intelligence
Follow-up to Phase 44: Can we add intelligence WITHOUT adding overhead?

Key insight from Phase 44: overhead kills. So what if we make the
curiosity computation essentially free (constant-time hash)?

Tests:
1. Random (baseline)
2. XOR-Hash Curiosity (O(1) per action, near-zero overhead)
3. Bloom-Filter Novelty (O(1) per action)
4. Sigma-Diverse Random (no computation, just different RNG seeds)

Author: Hiroto Funasaki
"""
import os, json, math, random, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Same PuzzleGame from Phase 44
# ==============================================================
class PuzzleGame:
    GRID_SIZE = 8
    N_ACTIONS = 7
    N_LEVELS = 3

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.level = 0
        self.steps = 0
        self.total_steps = 0
        self.solved_levels = 0
        self.rules = [self.rng.randint(0, self.N_ACTIONS - 1) for _ in range(self.N_LEVELS)]
        self.level_progress = [0] * self.N_LEVELS
        self.required_correct = [3, 4, 5]
        self._update_grid()

    def _update_grid(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        level = min(self.level, self.N_LEVELS - 1)
        for i in range(min(self.level_progress[level], gs)):
            self.grid[0, i] = 1.0
        rule = self.rules[level]
        self.grid[level + 1, rule] += 0.5
        self.grid[level + 1, (rule + 1) % gs] += 0.2

    def get_state(self):
        return self.grid.flatten()

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        level = min(self.level, self.N_LEVELS - 1)
        if action == self.rules[level]:
            self.level_progress[level] += 1
            if self.level_progress[level] >= self.required_correct[level]:
                self.solved_levels += 1
                self.level += 1
                self.steps = 0
        else:
            self.level_progress[level] = max(0, self.level_progress[level] - 1)
        self._update_grid()
        return self.get_state(), self.solved_levels >= self.N_LEVELS

    def reset_episode(self):
        self.steps = 0


# ==============================================================
# Zero-Overhead Agents
# ==============================================================
class RandomAgent:
    name = "Random"
    overhead = 0

    def choose(self, state_hash, step, n_actions, attempt):
        return random.randint(0, n_actions - 1)


class XORHashAgent:
    """O(1) curiosity: XOR state hash with action to get pseudo-novelty.
    Zero matrix operations, just integer arithmetic."""
    name = "XOR-Hash"
    overhead = 0  # effectively 0: just integer ops

    def __init__(self):
        self.seen = set()

    def choose(self, state_hash, step, n_actions, attempt):
        scores = []
        for a in range(n_actions):
            sa_hash = state_hash ^ (a * 2654435761)  # Fibonacci hashing
            novelty = 0 if sa_hash in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        sa_hash = state_hash ^ (best * 2654435761)
        self.seen.add(sa_hash)
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])
        return best


class BloomFilterAgent:
    """O(1) novelty with a Bloom filter (fixed memory, no hash table growth)."""
    name = "Bloom-Filter"
    overhead = 0

    def __init__(self, size=4096):
        self.bits = np.zeros(size, dtype=np.uint8)
        self.size = size

    def _hashes(self, state_hash, action):
        h1 = (state_hash * 31 + action) % self.size
        h2 = (state_hash * 37 + action * 7) % self.size
        h3 = (state_hash * 41 + action * 13) % self.size
        return h1, h2, h3

    def choose(self, state_hash, step, n_actions, attempt):
        scores = []
        for a in range(n_actions):
            h1, h2, h3 = self._hashes(state_hash, a)
            seen = self.bits[h1] and self.bits[h2] and self.bits[h3]
            novelty = 0.0 if seen else 1.0
            scores.append(novelty + random.gauss(0, 0.3))

        best = max(range(n_actions), key=lambda i: scores[i])
        h1, h2, h3 = self._hashes(state_hash, best)
        self.bits[h1] = 1
        self.bits[h2] = 1
        self.bits[h3] = 1
        return best


class SigmaDiverseAgent:
    """No computation at all: just cycle through different random seeds per attempt.
    Different seed = different exploration trajectory = diverse search."""
    name = "Sigma-Diverse"
    overhead = 0

    def choose(self, state_hash, step, n_actions, attempt):
        # Use attempt number to create different exploration patterns
        seed = state_hash + attempt * 1000003 + step * 7
        rng = random.Random(seed)
        return rng.randint(0, n_actions - 1)


class AdaptiveAgent:
    """Best of all worlds: XOR-Hash + Sigma-Diverse + Miracle Replay.
    Still O(1) overhead."""
    name = "Adaptive-O(1)"
    overhead = 0

    def __init__(self):
        self.xor = XORHashAgent()
        self.miracle_actions = []

    def choose(self, state_hash, step, n_actions, attempt):
        # 30% chance: replay miracle action (if we have any)
        if self.miracle_actions and random.random() < 0.3:
            return random.choice(self.miracle_actions)

        # Even attempts: XOR-Hash curiosity
        if attempt % 2 == 0:
            return self.xor.choose(state_hash, step, n_actions, attempt)
        # Odd attempts: sigma-diverse (different seed)
        else:
            seed = state_hash + attempt * 1000003 + step * 7
            return random.Random(seed).randint(0, n_actions - 1)

    def record_miracle(self, actions):
        self.miracle_actions.extend(actions)


# ==============================================================
# Simulation
# ==============================================================
def simulate(agent, n_games=100, max_actions_per_game=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    total_solved = 0
    total_actions = 0

    for game_idx in range(n_games):
        game = PuzzleGame(seed=seed + game_idx * 100)
        ep_actions = []
        attempt = 0

        for action_count in range(max_actions_per_game):
            state = game.get_state()
            state_hash = hash(state.tobytes()) % (2**31)

            action = agent.choose(state_hash, game.steps, game.N_ACTIONS, attempt)
            ep_actions.append(action)
            total_actions += 1

            _, solved = game.step(action)
            if solved:
                total_solved += 1
                if hasattr(agent, 'record_miracle'):
                    agent.record_miracle(ep_actions)
                break

            if game.total_steps > 0 and game.total_steps % 50 == 0:
                game.reset_episode()
                attempt += 1
                ep_actions = []

    return {
        'agent': agent.name,
        'n_games': n_games,
        'max_actions': max_actions_per_game,
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'total_actions': total_actions,
        'avg_actions': total_actions / n_games,
    }


def main():
    print("=" * 60)
    print("Phase 45: Zero-Overhead Intelligence")
    print("  Can we add intelligence WITHOUT adding overhead?")
    print("=" * 60)

    # Multiple budget levels (measured in max actions per game)
    budgets = [100, 200, 500, 1000, 2000, 5000]

    all_results = {}

    for max_actions in budgets:
        agents = [
            RandomAgent(),
            XORHashAgent(),
            BloomFilterAgent(),
            SigmaDiverseAgent(),
            AdaptiveAgent(),
        ]

        print(f"\n--- Budget: {max_actions} actions/game ---")
        print(f"  {'Agent':20s} | {'Solved':>8s} {'Rate':>8s} {'Avg Actions':>12s}")
        print("  " + "-" * 55)

        budget_results = {}
        for agent in agents:
            r = simulate(agent, n_games=200, max_actions_per_game=max_actions, seed=42)
            print(f"  {r['agent']:20s} | {r['total_solved']:>6d}/200 {r['solve_rate']*100:>6.1f}% "
                  f"{r['avg_actions']:>10.0f}")
            budget_results[r['agent']] = r

        all_results[f"budget_{max_actions}"] = budget_results

    # Cross-budget analysis
    print(f"\n{'='*60}")
    print("CROSS-BUDGET ANALYSIS: Solve Rate (%) by Agent x Budget")
    agent_names = ["Random", "XOR-Hash", "Bloom-Filter", "Sigma-Diverse", "Adaptive-O(1)"]
    header = f"{'Agent':>20s}"
    for b in budgets:
        header += f" {b:>7d}"
    print(header)
    print("-" * (20 + 8 * len(budgets)))

    for name in agent_names:
        row = f"{name:>20s}"
        for b in budgets:
            key = f"budget_{b}"
            rate = all_results[key][name]['solve_rate'] * 100
            row += f" {rate:>6.1f}%"
        print(row)

    # Winner analysis
    print(f"\n{'='*60}")
    print("WINNER AT EACH BUDGET LEVEL:")
    for b in budgets:
        key = f"budget_{b}"
        best = max(all_results[key].values(), key=lambda x: x['solve_rate'])
        improvement = (best['solve_rate'] - all_results[key]['Random']['solve_rate']) * 100
        print(f"  Budget {b:>5d}: {best['agent']:20s} ({best['solve_rate']*100:.1f}%) "
              f"vs Random {all_results[key]['Random']['solve_rate']*100:.1f}% "
              f"[{improvement:+.1f}pp]")

    save_path = os.path.join(RESULTS_DIR, "phase45_zero_overhead.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 45: Zero-Overhead Intelligence',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
