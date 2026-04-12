"""
Phase 52: v10 Grand Simulation
The ultimate tournament: all agent architectures compete
under ARC-like conditions (changing maps, noise, time pressure).

Dual-Process O(1) ExIt (System1 + System2) vs CNN vs Random.

Author: Hiroto Funasaki
"""
import os, json, random, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


class ARCLikeGame:
    """ARC-AGI-like grid environment:
    - Multiple levels with different map layouts
    - Noise characters that move randomly
    - Agent must reach goal on each level
    """
    GRID_SIZE = 10
    N_ACTIONS = 4
    N_LEVELS = 5
    N_NOISE = 3

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.level = 0
        self.solved_levels = 0
        self.total_steps = 0
        self._generate_level()

    def _generate_level(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)

        # Random walls (more on higher levels)
        n_walls = 5 + self.level * 3
        self.walls = set()
        for _ in range(n_walls):
            wx, wy = self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)
            self.walls.add((wx, wy))

        # Agent
        while True:
            self.ax, self.ay = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.ax, self.ay) not in self.walls:
                break

        # Goal
        while True:
            self.gx, self.gy = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.gx, self.gy) not in self.walls and (self.gx, self.gy) != (self.ax, self.ay):
                break

        # Noise
        self.noise_pos = []
        for _ in range(self.N_NOISE):
            nx, ny = self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)
            self.noise_pos.append([nx, ny])

        self._render()

    def _render(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        for wx, wy in self.walls:
            self.grid[wy, wx] = -1.0
        self.grid[self.ay, self.ax] = 1.0
        self.grid[self.gy, self.gx] = 2.0
        for nx, ny in self.noise_pos:
            if 0 <= nx < gs and 0 <= ny < gs and self.grid[ny, nx] == 0:
                self.grid[ny, nx] = 0.5

    def _move_noise(self):
        gs = self.GRID_SIZE
        for pos in self.noise_pos:
            dx, dy = [(0,-1),(0,1),(-1,0),(1,0),(0,0)][self.rng.randint(0, 4)]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < gs and 0 <= ny < gs:
                pos[0], pos[1] = nx, ny

    def get_state(self):
        return self.grid.copy()

    def get_local_patch(self, radius=1):
        gs = self.GRID_SIZE
        patch = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = self.ay + dy, self.ax + dx
                if 0 <= ny < gs and 0 <= nx < gs:
                    val = self.grid[ny, nx]
                    # Ignore noise (0.5) for patch -> treat as empty
                    if val == 0.5:
                        val = 0.0
                    patch.append(val)
                else:
                    patch.append(-1)
        return tuple(patch)

    def step(self, action):
        self.total_steps += 1
        gs = self.GRID_SIZE

        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        nx, ny = self.ax + dx, self.ay + dy

        if 0 <= nx < gs and 0 <= ny < gs and (nx, ny) not in self.walls:
            self.ax, self.ay = nx, ny

            if (self.ax, self.ay) == (self.gx, self.gy):
                self.solved_levels += 1
                self.level += 1
                if self.level < self.N_LEVELS:
                    self._generate_level()
                    return self.get_state(), False, True
                else:
                    return self.get_state(), True, True

        self._move_noise()
        self._render()
        return self.get_state(), False, False


# ==============================================================
# Competing Agents
# ==============================================================
class Agent1_Random:
    """Baseline: pure random."""
    name = "Random"

    def __init__(self, state_dim):
        pass

    def step(self, state, patch, n_actions, attempt):
        return random.randint(0, n_actions - 1)

    def on_level_up(self, patch_history, action_history):
        pass


class Agent2_XORHash:
    """Phase 45: XOR-Hash curiosity."""
    name = "XOR-Hash"

    def __init__(self, state_dim):
        self.seen = set()

    def step(self, state, patch, n_actions, attempt):
        sh = hash(state.tobytes()) % (2**31)
        scores = []
        for a in range(n_actions):
            sa_hash = sh ^ (a * 2654435761)
            novelty = 0 if sa_hash in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(sh ^ (best * 2654435761))
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])
        return best

    def on_level_up(self, patch_history, action_history):
        pass


class Agent3_SimHash:
    """Phase 51: SimHash curiosity (noise-robust)."""
    name = "SimHash"

    def __init__(self, state_dim, hash_bits=32):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True)
        self.seen = set()

    def _hash(self, state):
        bits = tuple((state.flatten() @ self.proj > 0).astype(np.int8))
        return hash(bits)

    def step(self, state, patch, n_actions, attempt):
        sh = self._hash(state)
        scores = []
        for a in range(n_actions):
            sa_hash = sh ^ (a * 2654435761)
            novelty = 0 if sa_hash in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(sh ^ (best * 2654435761))
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])
        return best

    def on_level_up(self, patch_history, action_history):
        pass


class Agent4_DualProcess:
    """THE v10 AGENT: Dual-Process O(1) ExIt.
    System 1: SimHash curiosity + sigma-diverse noise
    System 2: Patch-Hash ExIt (learn from miracles)
    Learning budget: 5% of remaining time.
    """
    name = "Dual-Process v10"

    def __init__(self, state_dim, hash_bits=32):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True)
        self.seen = set()
        # Patch ExIt dictionary
        self.patch_table = {}
        self.has_learned = False

    def _simhash(self, state):
        bits = tuple((state.flatten() @ self.proj > 0).astype(np.int8))
        return hash(bits)

    def step(self, state, patch, n_actions, attempt):
        # System 2: if we've learned, try patch lookup first
        if self.has_learned and patch is not None:
            key = hash(patch)
            if key in self.patch_table:
                counts = self.patch_table[key]
                if counts.sum() > 0:
                    return int(np.argmax(counts))

        # System 1: SimHash curiosity + sigma-diverse
        sigmas = [0.05, 0.15, 0.3, 0.5, 0.01]
        sigma = sigmas[attempt % len(sigmas)]

        sh = self._simhash(state)
        scores = []
        for a in range(n_actions):
            sa_hash = sh ^ (a * 2654435761)
            novelty = 0 if sa_hash in self.seen else 1
            scores.append(novelty + random.gauss(0, sigma))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(sh ^ (best * 2654435761))
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])
        return best

    def on_level_up(self, patch_history, action_history):
        """Learn patch -> action from miracle trajectory."""
        for patch, action in zip(patch_history, action_history):
            key = hash(patch)
            if key not in self.patch_table:
                self.patch_table[key] = np.zeros(4, dtype=np.float32)
            self.patch_table[key][action] += 1.0
        self.has_learned = True


class Agent5_HeavyCNN:
    """Simulated heavy CNN agent (like v12): high overhead."""
    name = "Heavy-CNN (v12)"

    def __init__(self, state_dim, overhead_per_action=5):
        self.overhead = overhead_per_action
        self.seen = set()

    def step(self, state, patch, n_actions, attempt):
        # Simulate overhead: do a redundant matrix op
        _ = np.random.randn(64, 32) @ np.random.randn(32, 16)
        sh = hash(state.tobytes()) % (2**31)
        scores = [random.gauss(0, 1) for _ in range(n_actions)]
        return max(range(n_actions), key=lambda i: scores[i])

    def on_level_up(self, patch_history, action_history):
        pass


# ==============================================================
# Grand Tournament
# ==============================================================
def tournament(agent_class, n_games=200, time_budget_ms=10000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    state_dim = ARCLikeGame.GRID_SIZE ** 2
    total_solved = 0
    total_levels = 0
    total_actions = 0

    for game_idx in range(n_games):
        game = ARCLikeGame(seed=seed + game_idx * 100)
        agent = agent_class(state_dim)
        time_remaining = time_budget_ms
        patch_history = []
        action_history = []
        attempt = 0

        # Determine overhead per action
        if hasattr(agent, 'overhead'):
            overhead = agent.overhead
        else:
            overhead = 0  # O(1) agents

        # Measure actual overhead
        if agent_class == Agent5_HeavyCNN:
            action_cost = 6  # 1ms base + 5ms overhead
        else:
            action_cost = 1  # O(1)

        while time_remaining > 0:
            state = game.get_state()
            patch = game.get_local_patch()

            action = agent.step(state, patch, game.N_ACTIONS, attempt)
            patch_history.append(patch)
            action_history.append(action)
            total_actions += 1
            time_remaining -= action_cost

            _, all_solved, level_up = game.step(action)

            if level_up:
                agent.on_level_up(patch_history, action_history)
                patch_history = []
                action_history = []
                attempt += 1

            if all_solved:
                total_solved += 1
                break

            if game.total_steps > 0 and game.total_steps % 100 == 0:
                patch_history = []
                action_history = []

    return {
        'agent': agent_class.name,
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'avg_levels': total_levels / n_games if n_games > 0 else 0,
        'total_actions': total_actions,
        'avg_actions': total_actions / n_games,
    }


def main():
    print("=" * 60)
    print("Phase 52: v10 Grand Simulation")
    print("  THE ULTIMATE TOURNAMENT: All agents under ARC conditions")
    print("=" * 60)

    agents = [Agent1_Random, Agent2_XORHash, Agent3_SimHash,
              Agent4_DualProcess, Agent5_HeavyCNN]
    budgets = [2000, 5000, 10000, 50000]

    all_results = {}

    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"  BUDGET: {budget}ms")
        print(f"{'='*60}")
        print(f"  {'Agent':>22s} | {'Solved':>8s} {'Rate':>8s} {'Actions':>10s}")
        print("  " + "-" * 55)

        budget_results = {}
        for agent_cls in agents:
            r = tournament(agent_cls, n_games=200, time_budget_ms=budget, seed=42)
            print(f"  {r['agent']:>22s} | {r['total_solved']:>6d}/200 "
                  f"{r['solve_rate']*100:>6.1f}% {r['total_actions']:>10,}")
            budget_results[agent_cls.name] = r

        all_results[f"budget_{budget}ms"] = budget_results

    # Grand Summary
    print(f"\n{'='*60}")
    print("GRAND TOURNAMENT RESULTS")
    print(f"{'='*60}")
    print(f"{'Agent':>22s}", end="")
    for b in budgets:
        print(f" {b:>8d}ms", end="")
    print()
    print("-" * (22 + 10 * len(budgets)))

    for agent_cls in agents:
        print(f"{agent_cls.name:>22s}", end="")
        for b in budgets:
            key = f"budget_{b}ms"
            rate = all_results[key][agent_cls.name]['solve_rate'] * 100
            print(f" {rate:>7.1f}%", end="")
        print()

    # Winner
    print(f"\n{'='*60}")
    print("CHAMPION AT EACH BUDGET:")
    for b in budgets:
        key = f"budget_{b}ms"
        best = max(all_results[key].values(), key=lambda x: x['solve_rate'])
        print(f"  Budget {b:>6d}ms: CHAMPION = {best['agent']} ({best['solve_rate']*100:.1f}%)")

    # v10 vs v12 comparison
    print(f"\n{'='*60}")
    print("v10 (Dual-Process) vs v12 (Heavy-CNN):")
    for b in budgets:
        key = f"budget_{b}ms"
        v10 = all_results[key]['Dual-Process v10']['solve_rate'] * 100
        v12 = all_results[key]['Heavy-CNN (v12)']['solve_rate'] * 100
        print(f"  Budget {b:>6d}ms: v10={v10:.1f}% vs v12={v12:.1f}% "
              f"({v10-v12:+.1f}pp)")

    save_path = os.path.join(RESULTS_DIR, "phase52_grand_simulation.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 52: v10 Grand Simulation',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
