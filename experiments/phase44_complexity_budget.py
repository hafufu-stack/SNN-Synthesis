"""
Phase 44: Complexity-Budget Tradeoff Analysis
Scientific investigation of the "complexity kills" phenomenon observed in Kaggle.
(v5 simple=0.13 > v12 complex=0.07)

Hypothesis: Under fixed action budget, overhead per action reduces total attempts,
and simpler agents achieve higher solve rates despite lower per-action quality.

Tests multiple agent complexity levels under identical budget constraints.

Author: Hiroto Funasaki
"""
import os, json, math, random, time, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Environment: Multi-Level Puzzle (ARC-AGI-like)
# ==============================================================
class PuzzleGame:
    """A multi-level puzzle game simulating ARC-AGI structure.
    Each level has a hidden rule. Agent must find it within limited steps.
    """
    GRID_SIZE = 8
    N_ACTIONS = 7
    N_LEVELS = 3

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.level = 0
        self.steps = 0
        self.total_steps = 0
        self.solved_levels = 0
        # Generate hidden rules for each level
        self.rules = [self.rng.randint(0, self.N_ACTIONS - 1) for _ in range(self.N_LEVELS)]
        self.level_progress = [0] * self.N_LEVELS
        self.required_correct = [3, 4, 5]  # increasing difficulty
        self._update_grid()

    def _update_grid(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        # Encode level and progress
        level = min(self.level, self.N_LEVELS - 1)
        for i in range(min(self.level_progress[level], gs)):
            self.grid[0, i] = 1.0
        # Subtle hint about answer
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
        solved = self.solved_levels >= self.N_LEVELS
        return self.get_state(), solved

    def reset_episode(self):
        """Reset to try again but keep level progress."""
        self.steps = 0


# ==============================================================
# Agents with Different Complexity Levels
# ==============================================================
class Agent:
    def __init__(self, name):
        self.name = name
        self.overhead_per_action = 0  # simulated milliseconds

    def choose_action(self, state, n_actions):
        raise NotImplementedError


class RandomAgent(Agent):
    """Level 0: Pure random. Zero overhead."""
    def __init__(self):
        super().__init__("Random")
        self.overhead_per_action = 0

    def choose_action(self, state, n_actions):
        return random.randint(0, n_actions - 1)


class VisitCountAgent(Agent):
    """Level 1: Visit-counting curiosity. Minimal overhead."""
    def __init__(self):
        super().__init__("VisitCount")
        self.overhead_per_action = 1  # ms
        self.visits = {}

    def choose_action(self, state, n_actions):
        sh = hash(state.tobytes()) % 100000
        scores = []
        for a in range(n_actions):
            k = (sh, a)
            v = self.visits.get(k, 0)
            scores.append(1.0 / (v + 1) + random.gauss(0, 0.3))
            self.visits[k] = v
        best = max(range(n_actions), key=lambda i: scores[i])
        self.visits[(sh, best)] = self.visits.get((sh, best), 0) + 1
        return best


class RNDAgent(Agent):
    """Level 2: RND curiosity. Moderate overhead (2 forward passes + 1 backward)."""
    def __init__(self, state_dim=64):
        super().__init__("RND")
        self.overhead_per_action = 5  # ms
        self.state_dim = state_dim
        h = 16
        self.target_w = np.random.randn(state_dim, h).astype(np.float32) * 0.1
        self.pred_w = np.random.randn(state_dim, h).astype(np.float32) * 0.3

    def choose_action(self, state, n_actions):
        x = state[:self.state_dim].reshape(1, -1).astype(np.float32)
        target = x @ self.target_w
        pred = x @ self.pred_w
        novelty = float(np.mean((target - pred) ** 2))

        # Update predictor
        error = pred - target
        self.pred_w -= 0.01 * (x.T @ error)

        # Use novelty to bias exploration
        scores = []
        for a in range(n_actions):
            scores.append(random.gauss(0, 1) * (1 + novelty))
        return max(range(n_actions), key=lambda i: scores[i])


class RNDPlusCNNAgent(Agent):
    """Level 3: RND + Test-Time CNN learning. High overhead."""
    def __init__(self, state_dim=64, n_actions=7):
        super().__init__("RND+CNN")
        self.overhead_per_action = 15  # ms
        self.state_dim = state_dim
        self.n_actions = n_actions
        h = 16
        self.target_w = np.random.randn(state_dim, h).astype(np.float32) * 0.1
        self.pred_w = np.random.randn(state_dim, h).astype(np.float32) * 0.3

        # CNN for learning from miracles
        self.cnn_w1 = np.random.randn(state_dim, 32).astype(np.float32) * 0.1
        self.cnn_w2 = np.random.randn(32, n_actions).astype(np.float32) * 0.1
        self.cnn_trained = False
        self.miracle_data = []

    def choose_action(self, state, n_actions):
        x = state[:self.state_dim].reshape(1, -1).astype(np.float32)

        # RND novelty
        target = x @ self.target_w
        pred = x @ self.pred_w
        novelty = float(np.mean((target - pred) ** 2))
        self.pred_w -= 0.01 * (x.T @ (pred - target))

        if self.cnn_trained:
            # Use CNN
            h = np.maximum(x @ self.cnn_w1, 0)
            logits = h @ self.cnn_w2
            logits = logits[0, :n_actions]
            exp_l = np.exp(logits - logits.max())
            probs = exp_l / exp_l.sum()
            return np.random.choice(n_actions, p=probs)
        else:
            scores = []
            for a in range(n_actions):
                scores.append(random.gauss(0, 1) * (1 + novelty))
            return max(range(n_actions), key=lambda i: scores[i])

    def record_miracle(self, states, actions, n_actions):
        for s, a in zip(states, actions):
            self.miracle_data.append((s, a))
        if len(self.miracle_data) >= 20:
            self._train_cnn(n_actions)

    def _train_cnn(self, n_actions):
        X = np.array([d[0][:self.state_dim] for d in self.miracle_data], dtype=np.float32)
        y = np.array([d[1] for d in self.miracle_data], dtype=np.int64)
        for _ in range(30):
            h = np.maximum(X @ self.cnn_w1, 0)
            logits = h @ self.cnn_w2
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            dl = probs.copy()
            dl[np.arange(len(y)), y] -= 1
            dl /= len(y)
            self.cnn_w2 -= 0.01 * (h.T @ dl)
            dh = (dl @ self.cnn_w2.T) * (h > 0)
            self.cnn_w1 -= 0.01 * (X.T @ dh)
        self.cnn_trained = True


class FullComplexAgent(Agent):
    """Level 4: Everything (RND + CNN + heavy feature extraction). Very high overhead."""
    def __init__(self, state_dim=64, n_actions=7):
        super().__init__("Full-Complex")
        self.overhead_per_action = 30  # ms
        self.rnd_cnn = RNDPlusCNNAgent(state_dim, n_actions)

    def choose_action(self, state, n_actions):
        # Extra overhead: feature extraction, hash computation, etc.
        _ = np.fft.fft(state[:64])  # simulate heavy feature extraction
        _ = hash(state.tobytes())
        return self.rnd_cnn.choose_action(state, n_actions)


# ==============================================================
# Simulation with Budget Constraint
# ==============================================================
def simulate_with_budget(agent, time_budget_ms, n_games=50, seed=42):
    """Run agent on multiple games with a fixed time budget.
    More overhead per action = fewer total actions = potentially lower solve rate.
    """
    random.seed(seed)
    np.random.seed(seed)

    total_solved = 0
    total_actions = 0
    total_miracles = 0

    for game_idx in range(n_games):
        game = PuzzleGame(seed=seed + game_idx * 100)
        time_remaining = time_budget_ms
        ep_states, ep_actions = [], []

        while time_remaining > 0:
            state = game.get_state()
            ep_states.append(state.copy())

            action = agent.choose_action(state, game.N_ACTIONS)
            ep_actions.append(action)

            time_remaining -= (1 + agent.overhead_per_action)  # 1ms base + overhead
            total_actions += 1

            _, solved = game.step(action)
            if solved:
                total_solved += 1
                total_miracles += 1
                # Record miracle for CNN agents
                if hasattr(agent, 'record_miracle'):
                    agent.record_miracle(ep_states, ep_actions, game.N_ACTIONS)
                elif hasattr(agent, 'rnd_cnn') and hasattr(agent.rnd_cnn, 'record_miracle'):
                    agent.rnd_cnn.record_miracle(ep_states, ep_actions, game.N_ACTIONS)
                break

            if game.total_steps > 200:
                # Time to reset episode
                game.reset_episode()
                ep_states, ep_actions = [], []

    return {
        'agent': agent.name,
        'overhead_ms': agent.overhead_per_action,
        'time_budget_ms': time_budget_ms,
        'n_games': n_games,
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'total_actions': total_actions,
        'avg_actions_per_game': total_actions / n_games,
    }


def main():
    print("=" * 60)
    print("Phase 44: Complexity-Budget Tradeoff Analysis")
    print('  "Does simpler always win under time pressure?"')
    print("=" * 60)

    state_dim = PuzzleGame.GRID_SIZE ** 2  # 64
    n_actions = PuzzleGame.N_ACTIONS  # 7

    agents = [
        RandomAgent(),
        VisitCountAgent(),
        RNDAgent(state_dim),
        RNDPlusCNNAgent(state_dim, n_actions),
        FullComplexAgent(state_dim, n_actions),
    ]

    # Test under multiple time budgets
    budgets = [500, 1000, 2000, 5000, 10000, 50000]

    all_results = {}

    for budget in budgets:
        print(f"\n--- Budget: {budget}ms ---")
        print(f"  {'Agent':20s} | {'Overhead':>8s} {'Actions':>8s} {'Solved':>8s} {'Rate':>8s}")
        print("  " + "-" * 60)

        budget_results = {}
        for agent in agents:
            # Reset stateful agents
            if hasattr(agent, 'visits'):
                agent.visits = {}
            if hasattr(agent, 'cnn_trained'):
                agent.cnn_trained = False
                agent.miracle_data = []
            if hasattr(agent, 'rnd_cnn'):
                agent.rnd_cnn.cnn_trained = False
                agent.rnd_cnn.miracle_data = []

            r = simulate_with_budget(agent, budget, n_games=100, seed=42)
            print(f"  {r['agent']:20s} | {r['overhead_ms']:>6d}ms {r['total_actions']:>8,} "
                  f"{r['total_solved']:>6d}/{r['n_games']} {r['solve_rate']*100:>6.1f}%")
            budget_results[r['agent']] = r

        all_results[f"budget_{budget}ms"] = budget_results

    # Cross-budget analysis
    print(f"\n{'='*60}")
    print("CROSS-BUDGET ANALYSIS: Solve Rate (%) by Agent x Budget")
    print(f"{'Agent':>20s}", end="")
    for b in budgets:
        print(f" {b:>8d}ms", end="")
    print()
    print("-" * (20 + 10 * len(budgets)))

    agent_names = [a.name for a in agents]
    for name in agent_names:
        print(f"{name:>20s}", end="")
        for b in budgets:
            key = f"budget_{b}ms"
            rate = all_results[key][name]['solve_rate'] * 100
            print(f" {rate:>8.1f}%", end="")
        print()

    # Find crossover point: where does complexity start to help?
    print(f"\n{'='*60}")
    print("KEY FINDING: Winner at each budget level")
    for b in budgets:
        key = f"budget_{b}ms"
        best = max(all_results[key].values(), key=lambda x: x['solve_rate'])
        print(f"  Budget {b:>6d}ms: Winner = {best['agent']} "
              f"({best['solve_rate']*100:.1f}%, {best['total_actions']:,} actions)")

    # Save
    save_path = os.path.join(RESULTS_DIR, "phase44_complexity_budget.json")
    # Convert for JSON serialization
    serializable = {}
    for bkey, bval in all_results.items():
        serializable[bkey] = {k: v for k, v in bval.items()}

    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 44: Complexity-Budget Tradeoff',
            'timestamp': datetime.now().isoformat(),
            'hypothesis': 'Under fixed time budget, simpler agents win due to more actions',
            'results': serializable,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
