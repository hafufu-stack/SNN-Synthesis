"""
Phase 43: Test-Time SNN-ExIt Simulation
Quick validation: does learning a CNN *during* gameplay improve performance?

Simulates the Kaggle scenario:
1. Random+curiosity exploration (baseline)
2. When miracle occurs -> train small CNN on miracle data
3. Continue with CNN policy + noise (Test-Time ExIt)

Uses synthetic game environments (no ARC-AGI dependency).
"""
import os, json, time, math, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Synthetic Game Environment
# ==============================================================
class SyntheticGame:
    """A simple grid navigation game with hidden rules.
    Agent must reach goal position. Different game_types have
    different optimal strategies (wall avoidance, shortcuts, etc.)
    """
    GRID_SIZE = 8
    N_ACTIONS = 4  # UP, DOWN, LEFT, RIGHT

    def __init__(self, game_type=0, seed=None):
        self.game_type = game_type
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        gs = self.GRID_SIZE
        self.player = [self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)]
        # Goal depends on game type
        if self.game_type == 0:
            self.goal = [gs-1, gs-1]  # always bottom-right
        elif self.game_type == 1:
            self.goal = [0, gs-1]     # always top-right
        else:
            self.goal = [gs//2, gs//2]  # always center

        # Walls
        self.walls = set()
        for _ in range(self.rng.randint(3, 8)):
            w = (self.rng.randint(0, gs-1), self.rng.randint(0, gs-1))
            if w != tuple(self.player) and w != tuple(self.goal):
                self.walls.add(w)

        self.steps = 0
        self.max_steps = 50
        return self.get_state()

    def get_state(self):
        """Return flattened state vector."""
        gs = self.GRID_SIZE
        grid = np.zeros((3, gs, gs), dtype=np.float32)  # player, goal, walls
        grid[0, self.player[0], self.player[1]] = 1.0
        grid[1, self.goal[0], self.goal[1]] = 1.0
        for wy, wx in self.walls:
            grid[2, wy, wx] = 1.0
        return grid.flatten()

    def step(self, action):
        """Take action. Returns (state, reward, done)."""
        dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        ny, nx = self.player[0] + dy, self.player[1] + dx
        gs = self.GRID_SIZE

        # Bounds and wall check
        if 0 <= ny < gs and 0 <= nx < gs and (ny, nx) not in self.walls:
            self.player = [ny, nx]

        self.steps += 1
        done = (self.player == self.goal) or (self.steps >= self.max_steps)
        solved = (self.player == self.goal)
        return self.get_state(), 1.0 if solved else 0.0, done, solved


# ==============================================================
# Tiny MLP for Test-Time Learning
# ==============================================================
class TinyMLP(nn.Module):
    def __init__(self, state_dim, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x, noise_sigma=0.0):
        h = self.net[0](x)
        h = self.net[1](h)
        if noise_sigma > 0:
            h = h + torch.randn_like(h) * noise_sigma
        h = self.net[2](h)
        h = self.net[3](h)
        return self.net[4](h)


# ==============================================================
# Simulation
# ==============================================================
def play_episode(env, policy=None, noise_sigma=0.15, use_curiosity=True):
    """Play one episode. Returns (states, actions, solved)."""
    state = env.reset()
    states, actions = [], []
    visit_counts = {}

    for _ in range(env.max_steps):
        states.append(state.copy())

        if policy is not None:
            # Use CNN policy with noise
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = policy(x, noise_sigma=noise_sigma)
                probs = torch.softmax(logits / 0.8, dim=1)
                action = torch.multinomial(probs, 1).item()
        elif use_curiosity:
            # Curiosity-guided random
            sh = hash(state.tobytes()) % 100000
            scores = []
            for a in range(env.N_ACTIONS):
                sa_key = (sh, a)
                v = visit_counts.get(sa_key, 0)
                scores.append(1.0 / (v + 1) + random.gauss(0, 0.3))
                visit_counts[sa_key] = v
            action = scores.index(max(scores))
            visit_counts[(sh, action)] = visit_counts.get((sh, action), 0) + 1
        else:
            action = random.randint(0, env.N_ACTIONS - 1)

        actions.append(action)
        state, reward, done, solved = env.step(action)
        if done:
            return states, actions, solved

    return states, actions, False


def train_on_miracles(miracles, state_dim, epochs=50):
    """Quick-train a TinyMLP on miracle trajectories."""
    all_states, all_actions = [], []
    for m in miracles:
        for s, a in zip(m['states'], m['actions']):
            all_states.append(s)
            all_actions.append(a)

    if len(all_states) < 5:
        return None

    X = torch.tensor(np.array(all_states), dtype=torch.float32)
    y = torch.tensor(all_actions, dtype=torch.long)

    model = TinyMLP(state_dim, n_actions=4, hidden=64)
    opt = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))[:min(128, len(X))]
        logits = model(X[perm])
        loss = F.cross_entropy(logits, y[perm])
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    return model


def simulate_kaggle_scenario(n_episodes=200, game_type=0, seed=42):
    """Simulate baseline vs test-time ExIt."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = SyntheticGame(game_type=game_type, seed=seed)
    state_dim = env.GRID_SIZE ** 2 * 3  # 192

    # === Baseline: pure random+curiosity ===
    baseline_solves = 0
    for ep in range(n_episodes):
        env_ep = SyntheticGame(game_type=game_type, seed=seed + ep * 100)
        _, _, solved = play_episode(env_ep, policy=None, use_curiosity=True)
        if solved:
            baseline_solves += 1

    baseline_rate = baseline_solves / n_episodes

    # === Test-Time ExIt ===
    exit_solves = 0
    miracles = []
    model = None

    for ep in range(n_episodes):
        env_ep = SyntheticGame(game_type=game_type, seed=seed + ep * 100)

        if model is not None:
            # Use learned model with noise
            sigma_idx = ep % 10
            sigma = [0.0, 0.05, 0.15, 0.3, 0.5, 0.01, 0.1, 0.2, 0.75, 1.0][sigma_idx]
            states, actions, solved = play_episode(env_ep, policy=model, noise_sigma=sigma)
        else:
            states, actions, solved = play_episode(env_ep, policy=None, use_curiosity=True)

        if solved:
            exit_solves += 1
            miracles.append({'states': states, 'actions': actions})
            # Train/update model on accumulated miracles
            if len(miracles) >= 3:
                model = train_on_miracles(miracles, state_dim, epochs=50)

    exit_rate = exit_solves / n_episodes

    return {
        'game_type': game_type,
        'n_episodes': n_episodes,
        'baseline_rate': baseline_rate,
        'baseline_solves': baseline_solves,
        'exit_rate': exit_rate,
        'exit_solves': exit_solves,
        'improvement': exit_rate - baseline_rate,
        'miracles_collected': len(miracles),
    }


def main():
    print("=" * 60)
    print("Phase 43: Test-Time SNN-ExIt Simulation")
    print("  Does learning during gameplay improve solve rate?")
    print("=" * 60)

    all_results = {}
    for game_type in range(3):
        print(f"\n--- Game Type {game_type} ---")
        r = simulate_kaggle_scenario(n_episodes=300, game_type=game_type, seed=42 + game_type)
        print(f"  Baseline:      {r['baseline_solves']}/{r['n_episodes']} ({r['baseline_rate']*100:.1f}%)")
        print(f"  Test-Time ExIt: {r['exit_solves']}/{r['n_episodes']} ({r['exit_rate']*100:.1f}%)")
        print(f"  Improvement:   {r['improvement']*100:+.1f}pp")
        print(f"  Miracles:      {r['miracles_collected']}")
        all_results[f"game_{game_type}"] = r

    # Summary
    avg_baseline = np.mean([r['baseline_rate'] for r in all_results.values()])
    avg_exit = np.mean([r['exit_rate'] for r in all_results.values()])
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Avg Baseline:  {avg_baseline*100:.1f}%")
    print(f"  Avg ExIt:      {avg_exit*100:.1f}%")
    print(f"  Avg Improve:   {(avg_exit-avg_baseline)*100:+.1f}pp")
    print(f"{'='*60}")

    save_path = os.path.join(RESULTS_DIR, "phase43_test_time_exit.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 43: Test-Time SNN-ExIt Simulation',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
            'summary': {
                'avg_baseline': avg_baseline,
                'avg_exit': avg_exit,
                'improvement': avg_exit - avg_baseline,
            }
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
