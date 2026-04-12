"""
Phase 46: Crossover Point Analysis
At what overhead level does intelligence start to HELP vs HURT?

Phase 44 found: any overhead -> Random wins
Phase 45 found: zero-overhead intelligence beats Random

This phase precisely maps the crossover: for each overhead level (0..50ms),
what solve rate does a curiosity agent achieve vs Random?

This produces the "optimal overhead" curve for the paper.

Author: Hiroto Funasaki
"""
import os, json, random, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


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


class CuriosityAgent:
    """Parametric curiosity agent: can adjust overhead level."""
    def __init__(self, overhead_ms=0):
        self.overhead = overhead_ms
        self.seen = set()

    def choose(self, state, n_actions):
        state_hash = hash(state.tobytes()) % (2**31)
        scores = []
        for a in range(n_actions):
            sa_hash = state_hash ^ (a * 2654435761)
            novelty = 0 if sa_hash in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        sa_hash = state_hash ^ (best * 2654435761)
        self.seen.add(sa_hash)
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])

        # Simulate overhead by reducing effective actions
        return best


def simulate_with_overhead(overhead_ms, budget_ms, n_games=200, seed=42):
    """Run curiosity agent with given overhead under time budget."""
    random.seed(seed)
    np.random.seed(seed)

    agent = CuriosityAgent(overhead_ms)
    total_solved = 0
    total_actions = 0

    for game_idx in range(n_games):
        game = PuzzleGame(seed=seed + game_idx * 100)
        time_remaining = budget_ms

        while time_remaining > 0:
            state = game.get_state()
            action = agent.choose(state, game.N_ACTIONS)
            time_remaining -= (1 + overhead_ms)  # base + overhead
            total_actions += 1

            _, solved = game.step(action)
            if solved:
                total_solved += 1
                break

            if game.total_steps > 0 and game.total_steps % 50 == 0:
                game.reset_episode()

    return total_solved / n_games, total_actions / n_games


def simulate_random(budget_ms, n_games=200, seed=42):
    """Pure random baseline."""
    random.seed(seed)
    np.random.seed(seed)

    total_solved = 0
    total_actions = 0

    for game_idx in range(n_games):
        game = PuzzleGame(seed=seed + game_idx * 100)
        time_remaining = budget_ms

        while time_remaining > 0:
            action = random.randint(0, game.N_ACTIONS - 1)
            time_remaining -= 1  # base only
            total_actions += 1

            _, solved = game.step(action)
            if solved:
                total_solved += 1
                break

            if game.total_steps > 0 and game.total_steps % 50 == 0:
                game.reset_episode()

    return total_solved / n_games, total_actions / n_games


def main():
    print("=" * 60)
    print("Phase 46: Crossover Point Analysis")
    print("  At what overhead does intelligence start to hurt?")
    print("=" * 60)

    # Test overhead levels: 0, 0.1, 0.2, ..., 2, 3, 5, 10, 20, 50
    overheads = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 50]
    budgets = [2000, 5000, 10000, 50000]

    all_results = {}

    for budget in budgets:
        print(f"\n--- Budget: {budget}ms ---")

        # Random baseline
        random_rate, random_actions = simulate_random(budget, n_games=200, seed=42)
        print(f"  Random: {random_rate*100:.1f}% ({random_actions:.0f} actions/game)")

        results = {'random_rate': random_rate, 'random_actions': random_actions}
        crossover_found = False

        print(f"  {'Overhead':>10s} | {'Rate':>8s} {'Actions':>8s} {'vs Random':>10s} {'Winner':>8s}")
        print("  " + "-" * 52)

        for overhead in overheads:
            cur_rate, cur_actions = simulate_with_overhead(overhead, budget, n_games=200, seed=42)
            diff = cur_rate - random_rate
            winner = "CURIOSITY" if cur_rate > random_rate else "RANDOM"

            print(f"  {overhead:>8.2f}ms | {cur_rate*100:>6.1f}% {cur_actions:>7.0f} "
                  f"{diff*100:>+8.1f}pp {winner:>10s}")

            results[f"overhead_{overhead}"] = {
                'overhead_ms': overhead,
                'solve_rate': cur_rate,
                'avg_actions': cur_actions,
                'diff_vs_random': diff,
            }

            if not crossover_found and cur_rate < random_rate:
                crossover_found = True
                results['crossover_overhead'] = overhead

        all_results[f"budget_{budget}ms"] = results

    # Summary
    print(f"\n{'='*60}")
    print("CROSSOVER POINTS (where curiosity starts to hurt):")
    for budget in budgets:
        key = f"budget_{budget}ms"
        r = all_results[key]
        if 'crossover_overhead' in r:
            print(f"  Budget {budget:>6d}ms: crossover at overhead = {r['crossover_overhead']}ms")
        else:
            print(f"  Budget {budget:>6d}ms: curiosity ALWAYS wins (no crossover)")

    save_path = os.path.join(RESULTS_DIR, "phase46_crossover_point.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 46: Crossover Point Analysis',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
