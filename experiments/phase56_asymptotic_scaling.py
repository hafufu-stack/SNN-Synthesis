"""
Phase 56: Asymptotic Scaling Law (10M Actions - The Deep Abyss)
Can brute-force compute + O(1) curiosity reach 100% solve rate?
Or does an activation energy wall exist that no random walk can cross?

Budgets: 5K, 10K, 100K, 1M, 10M actions
Agents: SimHash curiosity, XOR-Hash curiosity, Random

Author: Hiroto Funasaki
"""
import os, json, random, time, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


class ARCLikeGame:
    """ARC-like grid puzzle with walls, noise, and multiple levels."""
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
        n_walls = 5 + self.level * 3
        self.walls = set()
        for _ in range(n_walls):
            self.walls.add((self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)))
        while True:
            self.ax, self.ay = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.ax, self.ay) not in self.walls: break
        while True:
            self.gx, self.gy = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.gx, self.gy) not in self.walls and (self.gx, self.gy) != (self.ax, self.ay): break
        self.noise_pos = [[self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)] for _ in range(self.N_NOISE)]
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
                    return False, True  # not_all_done, level_up
                return True, True  # all_done, level_up
        self._move_noise()
        self._render()
        return False, False


# ==============================================================
# Agents
# ==============================================================
class RandomAgent:
    name = "Random"
    def __init__(self, state_dim):
        pass
    def choose(self, state, n_actions):
        return random.randint(0, n_actions - 1)


class XORHashAgent:
    name = "XOR-Hash"
    def __init__(self, state_dim):
        self.seen = set()
    def choose(self, state, n_actions):
        sh = hash(state.tobytes()) % (2**31)
        scores = []
        for a in range(n_actions):
            sa = sh ^ (a * 2654435761)
            novelty = 0 if sa in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(sh ^ (best * 2654435761))
        # Memory management: cap at 50K for large budgets
        if len(self.seen) > 50000:
            self.seen = set(list(self.seen)[-25000:])
        return best


class SimHashAgent:
    name = "SimHash"
    def __init__(self, state_dim, hash_bits=32):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True)
        self.seen = set()
    def _hash(self, state):
        bits = tuple((state.flatten() @ self.proj > 0).astype(np.int8))
        return hash(bits)
    def choose(self, state, n_actions):
        sh = self._hash(state)
        scores = []
        for a in range(n_actions):
            sa = sh ^ (a * 2654435761)
            novelty = 0 if sa in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(sh ^ (best * 2654435761))
        if len(self.seen) > 50000:
            self.seen = set(list(self.seen)[-25000:])
        return best


class SimHashDiverseAgent:
    """SimHash + sigma-diverse noise (Kaggle v13 agent)."""
    name = "SimHash-σDiv"
    def __init__(self, state_dim, hash_bits=32):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True)
        self.seen = set()
        self.sigmas = [0.01, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
        self.step_count = 0
    def _hash(self, state):
        bits = tuple((state.flatten() @ self.proj > 0).astype(np.int8))
        return hash(bits)
    def choose(self, state, n_actions):
        sh = self._hash(state)
        sigma = self.sigmas[self.step_count % len(self.sigmas)]
        self.step_count += 1
        scores = []
        for a in range(n_actions):
            sa = sh ^ (a * 2654435761)
            novelty = 0 if sa in self.seen else 1
            scores.append(novelty + random.gauss(0, sigma))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(sh ^ (best * 2654435761))
        if len(self.seen) > 50000:
            self.seen = set(list(self.seen)[-25000:])
        return best


# ==============================================================
# Simulation with progress tracking
# ==============================================================
def simulate_scaling(agent_class, n_games, max_actions, seed=42, progress_interval=None):
    random.seed(seed)
    np.random.seed(seed)
    state_dim = ARCLikeGame.GRID_SIZE ** 2

    total_solved = 0
    total_levels = 0
    steps_to_solve = []  # track when each game is solved

    for game_idx in range(n_games):
        game = ARCLikeGame(seed=seed + game_idx * 100)
        agent = agent_class(state_dim)

        for step in range(max_actions):
            state = game.get_state()
            action = agent.choose(state.flatten(), game.N_ACTIONS)
            all_done, level_up = game.step(action)

            if all_done:
                total_solved += 1
                steps_to_solve.append(step + 1)
                break

        total_levels += game.solved_levels

        if progress_interval and (game_idx + 1) % progress_interval == 0:
            print(f"    [{agent_class.name}] {game_idx+1}/{n_games} games, "
                  f"solved={total_solved} ({total_solved/(game_idx+1)*100:.1f}%)")

    median_steps = float(np.median(steps_to_solve)) if steps_to_solve else float('inf')
    mean_steps = float(np.mean(steps_to_solve)) if steps_to_solve else float('inf')

    return {
        'agent': agent_class.name,
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'avg_levels': total_levels / n_games,
        'median_steps_to_solve': median_steps,
        'mean_steps_to_solve': mean_steps,
        'max_budget': max_actions,
    }


def main():
    print("=" * 70)
    print("Phase 56: Asymptotic Scaling Law (The Deep Abyss)")
    print("  Does brute-force compute + O(1) curiosity reach 100%?")
    print("  Or is there an activation energy wall?")
    print("=" * 70)

    agents = [RandomAgent, XORHashAgent, SimHashAgent, SimHashDiverseAgent]
    # Exponential scaling: 5K -> 10M
    budgets = [5_000, 10_000, 100_000, 1_000_000, 10_000_000]
    # Reduce game count for large budgets to keep runtime manageable
    n_games_map = {
        5_000: 200,
        10_000: 200,
        100_000: 100,
        1_000_000: 50,
        10_000_000: 20,
    }

    all_results = {}

    for budget in budgets:
        n_games = n_games_map[budget]
        print(f"\n{'='*70}")
        print(f"  BUDGET: {budget:,} actions | N_GAMES: {n_games}")
        print(f"{'='*70}")
        print(f"  {'Agent':>18s} | {'Solved':>8s} {'Rate':>8s} {'MedianSteps':>12s} {'AvgLvl':>8s}")
        print("  " + "-" * 62)

        budget_results = {}
        for agent_cls in agents:
            # Show progress for large budgets
            prog = max(1, n_games // 5) if budget >= 100_000 else None
            t0 = time.time()
            r = simulate_scaling(agent_cls, n_games=n_games,
                                max_actions=budget, seed=42,
                                progress_interval=prog)
            elapsed = time.time() - t0
            r['wall_time_sec'] = elapsed

            med = f"{r['median_steps_to_solve']:,.0f}" if r['median_steps_to_solve'] != float('inf') else "N/A"
            print(f"  {r['agent']:>18s} | {r['total_solved']:>5d}/{n_games:<3d} "
                  f"{r['solve_rate']*100:>6.1f}% {med:>12s} {r['avg_levels']:>6.2f} "
                  f"[{elapsed:.1f}s]")
            budget_results[r['agent']] = r

        all_results[f"budget_{budget}"] = budget_results

    # Scaling curve summary
    print(f"\n{'='*70}")
    print("SCALING CURVE: Solve Rate (%) by Budget")
    print(f"{'Agent':>18s}", end="")
    for b in budgets:
        print(f" {b:>10,}", end="")
    print()
    print("-" * (18 + 11 * len(budgets)))

    for agent_cls in agents:
        print(f"{agent_cls.name:>18s}", end="")
        for b in budgets:
            key = f"budget_{b}"
            rate = all_results[key][agent_cls.name]['solve_rate'] * 100
            print(f" {rate:>9.1f}%", end="")
        print()

    # Key analysis: Is there a wall?
    print(f"\n{'='*70}")
    print("ANALYSIS: Asymptotic Behavior")
    for agent_cls in agents:
        rates = []
        for b in budgets:
            key = f"budget_{b}"
            rates.append(all_results[key][agent_cls.name]['solve_rate'])

        # Check if rate plateaus or keeps growing
        if len(rates) >= 3:
            last_3 = rates[-3:]
            improvement = last_3[-1] - last_3[0]
            if rates[-1] >= 0.99:
                verdict = "→ CONVERGES to ~100% (no wall)"
            elif improvement < 0.01:
                verdict = f"→ PLATEAUS at {rates[-1]*100:.1f}% (WALL DETECTED)"
            else:
                verdict = f"→ STILL GROWING (+{improvement*100:.1f}pp over last 3)"
        else:
            verdict = "→ insufficient data"
        print(f"  {agent_cls.name:>18s}: {verdict}")
        print(f"    Rates: {' → '.join(f'{r*100:.1f}%' for r in rates)}")

    # Wall time summary
    print(f"\n{'='*70}")
    print("WALL TIME (seconds):")
    for agent_cls in agents:
        print(f"  {agent_cls.name:>18s}:", end="")
        for b in budgets:
            key = f"budget_{b}"
            wt = all_results[key][agent_cls.name].get('wall_time_sec', 0)
            print(f"  {wt:>8.1f}s", end="")
        print()

    # Clean up inf values for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float) and (obj == float('inf') or obj == float('-inf')):
            return None
        return obj

    save_path = os.path.join(RESULTS_DIR, "phase56_asymptotic_scaling.json")
    with open(save_path, 'w') as f:
        json.dump(clean_for_json({
            'experiment': 'Phase 56: Asymptotic Scaling Law',
            'timestamp': datetime.now().isoformat(),
            'budgets': budgets,
            'results': all_results,
        }), f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
