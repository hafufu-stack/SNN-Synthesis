"""
Phase 49: Test-Time Compute Allocation (ExIt Budgeting)
Find the golden ratio: what % of remaining time should go to
SFT learning vs continued exploration?

Author: Hiroto Funasaki
"""
import os, json, random, time, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Multi-Level Puzzle (from Phase 44)
# ==============================================================
class PuzzleGame:
    GRID_SIZE = 8
    N_ACTIONS = 7
    N_LEVELS = 5  # More levels to make learning more valuable

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.level = 0
        self.steps = 0
        self.total_steps = 0
        self.solved_levels = 0
        self.rules = [self.rng.randint(0, self.N_ACTIONS - 1) for _ in range(self.N_LEVELS)]
        self.level_progress = [0] * self.N_LEVELS
        self.required_correct = [3, 3, 4, 4, 5]
        self._update_grid()

    def _update_grid(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        level = min(self.level, self.N_LEVELS - 1)
        for i in range(min(self.level_progress[level], gs)):
            self.grid[0, i] = 1.0
        rule = self.rules[level]
        self.grid[level + 1, rule] += 0.5

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


# ==============================================================
# N-Gram ExIt Model (from Phase 48)
# ==============================================================
class NGramModel:
    def __init__(self, n_actions=7):
        self.n_actions = n_actions
        self.table = {}
        self.trained = False

    def learn(self, trajectory, learning_time_ms):
        """Learn from miracle trajectory. Simulated learning time."""
        for i in range(len(trajectory)):
            ctx = tuple(trajectory[max(0, i-3):i]) if i > 0 else ()
            key = hash(ctx)
            if key not in self.table:
                self.table[key] = np.zeros(self.n_actions, dtype=np.float32)
            self.table[key][trajectory[i]] += 1.0
        self.trained = True
        return learning_time_ms  # Time consumed

    def predict(self, action_history):
        for n in [3, 2, 1]:
            ctx = tuple(action_history[-n:]) if len(action_history) >= n else tuple(action_history)
            key = hash(ctx)
            if key in self.table and self.table[key].sum() > 0:
                return int(np.argmax(self.table[key]))
        return None


# ==============================================================
# Dual-Process Agent with Configurable Learning Budget
# ==============================================================
def simulate_dual_process(learning_budget_pct, total_budget_ms, n_games=200,
                          learning_overhead_ms=0, seed=42):
    """
    Simulate Dual-Process agent:
    System 1: XOR-hash exploration (until miracle)
    System 2: N-gram learning then exploit (after miracle)

    learning_budget_pct: what % of REMAINING time to spend on learning
    """
    random.seed(seed)
    np.random.seed(seed)

    total_solved = 0
    total_levels = 0

    for game_idx in range(n_games):
        game = PuzzleGame(seed=seed + game_idx * 100)
        time_remaining = total_budget_ms
        seen = set()
        action_history = []
        miracle_found = False
        ngram = NGramModel(n_actions=game.N_ACTIONS)
        miracle_trajectory = []

        while time_remaining > 0:
            state = game.get_state()
            state_hash = hash(state.tobytes()) % (2**31)

            if miracle_found and ngram.trained:
                # System 2: Use learned model
                pred = ngram.predict(action_history)
                if pred is not None:
                    action = pred
                else:
                    action = random.randint(0, game.N_ACTIONS - 1)
                time_remaining -= (1 + learning_overhead_ms * 0.01)  # tiny overhead
            else:
                # System 1: XOR-hash exploration (O(1))
                scores = []
                for a in range(game.N_ACTIONS):
                    sa_hash = state_hash ^ (a * 2654435761)
                    novelty = 0 if sa_hash in seen else 1
                    scores.append(novelty + random.gauss(0, 0.3))
                action = max(range(game.N_ACTIONS), key=lambda i: scores[i])
                sa_hash = state_hash ^ (action * 2654435761)
                seen.add(sa_hash)
                if len(seen) > 10000:
                    seen = set(list(seen)[-5000:])
                time_remaining -= 1  # O(1) overhead

            action_history.append(action)
            _, solved = game.step(action)

            if solved:
                total_solved += 1
                total_levels += game.solved_levels
                break

            # Detect level-up (miracle!)
            if game.solved_levels > 0 and not miracle_found:
                miracle_found = True
                miracle_trajectory = action_history.copy()

                # Spend learning budget
                learn_time = time_remaining * (learning_budget_pct / 100.0)
                actual_learn = ngram.learn(miracle_trajectory, learn_time)
                time_remaining -= actual_learn

            if game.total_steps > 0 and game.total_steps % 50 == 0:
                action_history = []

        total_levels += game.solved_levels

    return {
        'learning_budget_pct': learning_budget_pct,
        'total_budget_ms': total_budget_ms,
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'avg_levels': total_levels / n_games,
    }


def main():
    print("=" * 60)
    print("Phase 49: Test-Time Compute Allocation (ExIt Budgeting)")
    print("  What % of time should go to learning vs exploration?")
    print("=" * 60)

    # Learning budget percentages to test
    pcts = [0, 1, 2, 5, 10, 15, 20, 30, 50, 70, 90]
    budgets = [2000, 5000, 10000, 50000]

    all_results = {}

    for budget in budgets:
        print(f"\n--- Total Budget: {budget}ms ---")
        print(f"  {'Learn%':>8s} | {'Solved':>8s} {'Rate':>8s} {'AvgLvl':>8s}")
        print("  " + "-" * 40)

        budget_results = {}
        for pct in pcts:
            r = simulate_dual_process(pct, budget, n_games=200, seed=42)
            print(f"  {pct:>6d}% | {r['total_solved']:>6d}/200 {r['solve_rate']*100:>6.1f}% "
                  f"{r['avg_levels']:>6.2f}")
            budget_results[f"learn_{pct}pct"] = r

        all_results[f"budget_{budget}ms"] = budget_results

    # Find golden ratio
    print(f"\n{'='*60}")
    print("GOLDEN RATIO: Optimal learning budget at each total budget")
    for budget in budgets:
        key = f"budget_{budget}ms"
        best = max(all_results[key].values(), key=lambda x: x['solve_rate'])
        baseline = all_results[key]['learn_0pct']
        improvement = (best['solve_rate'] - baseline['solve_rate']) * 100
        print(f"  Budget {budget:>6d}ms: optimal = {best['learning_budget_pct']}% "
              f"({best['solve_rate']*100:.1f}% vs baseline {baseline['solve_rate']*100:.1f}%, "
              f"{improvement:+.1f}pp)")

    save_path = os.path.join(RESULTS_DIR, "phase49_compute_allocation.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 49: Test-Time Compute Allocation',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
