"""
Phase 39: Curiosity-Driven NBS with RND
Use Random Network Distillation as intrinsic reward to break
the 0% miracle rate barrier on hard games.

Tests whether curiosity-driven exploration can produce miracles
where random exploration fails (Condition 1 violation).

Author: Hiroto Funasaki
"""
import os, json, math, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Synthetic "Hard" Game Environment
# ==============================================================
class HardGame:
    """A game designed to have 0% miracle rate with random exploration.

    The game requires a specific sequence of actions to solve.
    Random exploration is extremely unlikely to find the solution,
    but curiosity-driven exploration (visiting novel states) can.
    """
    GRID_SIZE = 6
    N_ACTIONS = 4

    def __init__(self, difficulty=3, seed=None):
        self.difficulty = difficulty  # length of required sequence
        self.rng = random.Random(seed)
        # The "secret" solution sequence
        self.solution = [self.rng.randint(0, self.N_ACTIONS - 1)
                         for _ in range(difficulty)]
        self.reset()

    def reset(self):
        self.progress = 0  # how many correct actions in sequence
        self.steps = 0
        self.max_steps = 100
        self._update_grid()
        return self.get_state()

    def _update_grid(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        # Encode progress in grid
        for i in range(self.progress):
            self.grid[0, i] = 1.0
        # Encode current "hints" (subtle patterns that change with progress)
        self.grid[self.progress % gs, :] += 0.5
        self.grid[:, self.solution[min(self.progress, len(self.solution)-1)]] += 0.3

    def get_state(self):
        return self.grid.flatten()

    def step(self, action):
        self.steps += 1
        if self.progress < len(self.solution):
            if action == self.solution[self.progress]:
                self.progress += 1
                self._update_grid()
            else:
                # Wrong action: partial reset
                self.progress = max(0, self.progress - 1)
                self._update_grid()

        solved = self.progress >= len(self.solution)
        done = solved or self.steps >= self.max_steps
        return self.get_state(), 1.0 if solved else 0.0, done, solved


# ==============================================================
# RND (Random Network Distillation) Module
# ==============================================================
class RNDModule:
    """Curiosity via prediction error on a random target network."""
    def __init__(self, state_dim, hidden=64):
        self.state_dim = state_dim
        # Fixed random target network
        self.target_w1 = np.random.randn(state_dim, hidden).astype(np.float32) * 0.1
        self.target_b1 = np.zeros(hidden, dtype=np.float32)
        self.target_w2 = np.random.randn(hidden, hidden // 2).astype(np.float32) * 0.1
        self.target_b2 = np.zeros(hidden // 2, dtype=np.float32)

        # Learnable predictor network
        scale = np.sqrt(2.0 / state_dim)
        self.pred_w1 = np.random.randn(state_dim, hidden).astype(np.float32) * scale
        self.pred_b1 = np.zeros(hidden, dtype=np.float32)
        self.pred_w2 = np.random.randn(hidden, hidden // 2).astype(np.float32) * scale
        self.pred_b2 = np.zeros(hidden // 2, dtype=np.float32)

    def target_forward(self, x):
        h = np.maximum(x @ self.target_w1 + self.target_b1, 0)
        return h @ self.target_w2 + self.target_b2

    def predictor_forward(self, x):
        h = np.maximum(x @ self.pred_w1 + self.pred_b1, 0)
        return h @ self.pred_w2 + self.pred_b2

    def curiosity_score(self, state):
        """Higher = more novel state."""
        x = state.reshape(1, -1).astype(np.float32)
        target = self.target_forward(x)
        pred = self.predictor_forward(x)
        error = np.mean((target - pred) ** 2)
        return float(error)

    def update_predictor(self, state, lr=0.01):
        """Train predictor to match target (reduces curiosity for seen states)."""
        x = state.reshape(1, -1).astype(np.float32)
        target = self.target_forward(x)

        # Forward
        h1 = x @ self.pred_w1 + self.pred_b1
        a1 = np.maximum(h1, 0)
        pred = a1 @ self.pred_w2 + self.pred_b2

        # Backward (MSE loss)
        d_pred = 2.0 * (pred - target) / pred.shape[1]
        dw2 = a1.T @ d_pred
        db2 = d_pred.sum(0)
        da1 = (d_pred @ self.pred_w2.T) * (h1 > 0)
        dw1 = x.T @ da1
        db1 = da1.sum(0)

        self.pred_w2 -= lr * dw2
        self.pred_b2 -= lr * db2
        self.pred_w1 -= lr * dw1
        self.pred_b1 -= lr * db1


# ==============================================================
# Experiment
# ==============================================================
def play_episode_random(env, max_steps=100):
    state = env.reset()
    for _ in range(max_steps):
        action = random.randint(0, env.N_ACTIONS - 1)
        state, reward, done, solved = env.step(action)
        if done:
            return solved
    return False


def play_episode_rnd(env, rnd, max_steps=100, exploration_weight=3.0):
    state = env.reset()
    for _ in range(max_steps):
        # Score each action by curiosity
        best_action = 0
        best_score = -float('inf')
        for a in range(env.N_ACTIONS):
            # Simulate action (peek)
            env_copy_progress = env.progress
            env_copy_steps = env.steps

            next_state, _, _, _ = env.step(a)
            score = rnd.curiosity_score(next_state) * exploration_weight
            score += random.gauss(0, 0.1)  # small random noise

            if score > best_score:
                best_score = score
                best_action = a

            # Undo (restore state)
            env.progress = env_copy_progress
            env.steps = env_copy_steps
            env._update_grid()

        # Take best action
        state, reward, done, solved = env.step(best_action)
        rnd.update_predictor(state, lr=0.005)

        if done:
            return solved
    return False


def run_experiment(difficulty, n_episodes=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    state_dim = HardGame.GRID_SIZE ** 2  # 36

    # Baseline: pure random
    random_solves = 0
    for ep in range(n_episodes):
        env = HardGame(difficulty=difficulty, seed=seed)
        if play_episode_random(env):
            random_solves += 1

    # RND: curiosity-driven
    rnd = RNDModule(state_dim, hidden=64)
    rnd_solves = 0
    for ep in range(n_episodes):
        env = HardGame(difficulty=difficulty, seed=seed)
        if play_episode_rnd(env, rnd, exploration_weight=3.0):
            rnd_solves += 1

    return {
        'difficulty': difficulty,
        'n_episodes': n_episodes,
        'random_solves': random_solves,
        'random_rate': random_solves / n_episodes,
        'rnd_solves': rnd_solves,
        'rnd_rate': rnd_solves / n_episodes,
        'improvement': (rnd_solves - random_solves) / n_episodes,
    }


def main():
    print("=" * 60)
    print("Phase 39: Curiosity-Driven NBS with RND")
    print("  Can curiosity break the 0% miracle rate barrier?")
    print("=" * 60)

    all_results = {}
    for difficulty in [2, 3, 4, 5, 6]:
        print(f"\n--- Difficulty {difficulty} (requires {difficulty}-step sequence) ---")
        r = run_experiment(difficulty, n_episodes=200, seed=42)
        print(f"  Random: {r['random_solves']}/{r['n_episodes']} ({r['random_rate']*100:.1f}%)")
        print(f"  RND:    {r['rnd_solves']}/{r['n_episodes']} ({r['rnd_rate']*100:.1f}%)")
        print(f"  Improvement: {r['improvement']*100:+.1f}pp")
        all_results[f"difficulty_{difficulty}"] = r

    print(f"\n{'='*60}")
    print("SUMMARY: RND vs Random by Difficulty")
    for key, r in all_results.items():
        d = r['difficulty']
        broke_barrier = r['rnd_rate'] > 0 and r['random_rate'] == 0
        print(f"  D={d}: Random={r['random_rate']*100:.1f}% RND={r['rnd_rate']*100:.1f}% "
              f"{'*** BARRIER BROKEN ***' if broke_barrier else ''}")
    print(f"{'='*60}")

    save_path = os.path.join(RESULTS_DIR, "phase39_curiosity_rnd.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 39: Curiosity-Driven NBS with RND',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
