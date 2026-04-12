"""
Phase 51: SimHash Curiosity
Locality-Sensitive Hashing (SimHash) for noise-robust, O(1) curiosity.
One random matrix multiplication -> sign -> hash.
Similar states get similar hashes (unlike XOR-Hash).

Author: Hiroto Funasaki
"""
import os, json, random, time, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


class NoisyPuzzle:
    """Grid puzzle with noise characters that move randomly."""
    GRID_SIZE = 8
    N_ACTIONS = 4
    N_LEVELS = 3

    def __init__(self, n_noise=3, seed=None):
        self.rng = random.Random(seed)
        self.n_noise = n_noise
        self.level = 0
        self.solved_levels = 0
        self.total_steps = 0
        self._generate_level()

    def _generate_level(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)

        while True:
            self.ax, self.ay = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if self.grid[self.ay, self.ax] == 0:
                break

        while True:
            self.gx, self.gy = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.gx, self.gy) != (self.ax, self.ay):
                break

        # Noise characters (value=0.5)
        self.noise_pos = []
        for _ in range(self.n_noise):
            while True:
                nx, ny = self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)
                if (nx, ny) != (self.ax, self.ay) and (nx, ny) != (self.gx, self.gy):
                    self.noise_pos.append([nx, ny])
                    break

        self._render()

    def _render(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        self.grid[self.ay, self.ax] = 1.0  # agent
        self.grid[self.gy, self.gx] = 2.0  # goal
        for nx, ny in self.noise_pos:
            if self.grid[ny, nx] == 0:
                self.grid[ny, nx] = 0.5  # noise

    def _move_noise(self):
        gs = self.GRID_SIZE
        for pos in self.noise_pos:
            dx, dy = [(0,-1),(0,1),(-1,0),(1,0),(0,0)][self.rng.randint(0, 4)]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < gs and 0 <= ny < gs:
                pos[0], pos[1] = nx, ny

    def get_state(self):
        return self.grid.flatten()

    def step(self, action):
        self.total_steps += 1
        gs = self.GRID_SIZE

        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        nx, ny = self.ax + dx, self.ay + dy

        if 0 <= nx < gs and 0 <= ny < gs:
            self.ax, self.ay = nx, ny

        if (self.ax, self.ay) == (self.gx, self.gy):
            self.solved_levels += 1
            self.level += 1
            if self.level < self.N_LEVELS:
                self._generate_level()
                return self.get_state(), False
            else:
                return self.get_state(), True

        self._move_noise()
        self._render()
        return self.get_state(), False


# ==============================================================
# Curiosity Agents
# ==============================================================
class RandomAgent:
    name = "Random"
    def __init__(self, state_dim):
        pass

    def choose(self, state, n_actions):
        return random.randint(0, n_actions - 1)

    def update(self, state):
        pass


class XORHashCuriosity:
    """Phase 45: exact hash, noise-sensitive."""
    name = "XOR-Hash"

    def __init__(self, state_dim):
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
        return best

    def update(self, state):
        pass


class SimHashCuriosity:
    """SimHash: Locality-Sensitive Hashing for noise-robust novelty.
    One matrix multiply + sign = similar states get same hash.
    """
    name = "SimHash"

    def __init__(self, state_dim, hash_bits=32):
        self.hash_bits = hash_bits
        # Fixed random projection matrix (computed once, never changes)
        np.random.seed(42)
        self.projection = np.random.randn(state_dim, hash_bits).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        self.seen = set()

    def _hash(self, state):
        """O(state_dim * hash_bits) hash that's noise-robust."""
        projected = state @ self.projection  # (hash_bits,)
        bits = tuple((projected > 0).astype(np.int8))
        return hash(bits)

    def choose(self, state, n_actions):
        sh = self._hash(state)
        scores = []
        for a in range(n_actions):
            sa_hash = sh ^ (a * 2654435761)
            novelty = 0 if sa_hash in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        sa_hash = sh ^ (best * 2654435761)
        self.seen.add(sa_hash)
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])
        return best

    def update(self, state):
        pass


class MultiResSimHash:
    """Multi-resolution SimHash: hash at multiple bit widths.
    Low bits = coarse (more noise-robust), high bits = fine (more precise).
    """
    name = "MultiRes-SimHash"

    def __init__(self, state_dim, bit_levels=[8, 16, 32]):
        self.bit_levels = bit_levels
        np.random.seed(42)
        max_bits = max(bit_levels)
        self.projection = np.random.randn(state_dim, max_bits).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        self.seen = {b: set() for b in bit_levels}

    def _hash(self, state, n_bits):
        projected = state @ self.projection[:, :n_bits]
        bits = tuple((projected > 0).astype(np.int8))
        return hash(bits)

    def choose(self, state, n_actions):
        scores = []
        for a in range(n_actions):
            novelty = 0.0
            for bits in self.bit_levels:
                sh = self._hash(state, bits) ^ (a * 2654435761)
                if sh not in self.seen[bits]:
                    novelty += 1.0 / len(self.bit_levels)
            scores.append(novelty + random.gauss(0, 0.2))

        best = max(range(n_actions), key=lambda i: scores[i])
        for bits in self.bit_levels:
            sh = self._hash(state, bits) ^ (best * 2654435761)
            self.seen[bits].add(sh)
            if len(self.seen[bits]) > 10000:
                self.seen[bits] = set(list(self.seen[bits])[-5000:])
        return best

    def update(self, state):
        pass


# ==============================================================
# Simulation
# ==============================================================
def simulate(agent_class, n_noise, n_games=200, max_actions=2000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    state_dim = NoisyPuzzle.GRID_SIZE ** 2
    total_solved = 0
    total_levels = 0

    for game_idx in range(n_games):
        game = NoisyPuzzle(n_noise=n_noise, seed=seed + game_idx * 100)
        agent = agent_class(state_dim)

        for step in range(max_actions):
            state = game.get_state()
            action = agent.choose(state, game.N_ACTIONS)
            _, solved = game.step(action)
            if solved:
                total_solved += 1
                break

        total_levels += game.solved_levels

    return {
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'avg_levels': total_levels / n_games,
    }


def measure_hash_latency(agent_class, state_dim=64, n_runs=10000):
    agent = agent_class(state_dim)
    state = np.random.randn(state_dim).astype(np.float32)

    # Warmup
    for _ in range(100):
        agent.choose(state, 4)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        agent.choose(state, 4)
    t1 = time.perf_counter()

    return (t1 - t0) / n_runs * 1000  # ms


def main():
    print("=" * 60)
    print("Phase 51: SimHash Curiosity")
    print("  Noise-robust O(1) curiosity via Locality-Sensitive Hashing")
    print("=" * 60)

    agents = [RandomAgent, XORHashCuriosity, SimHashCuriosity, MultiResSimHash]
    state_dim = NoisyPuzzle.GRID_SIZE ** 2

    # Latency test
    print("\n--- Latency Benchmark ---")
    for agent_cls in agents:
        lat = measure_hash_latency(agent_cls, state_dim)
        print(f"  {agent_cls.name:>20s}: {lat:.4f}ms/action")

    # Test with varying noise levels
    noise_levels = [0, 1, 3, 5, 8]
    all_results = {}

    for n_noise in noise_levels:
        print(f"\n--- Noise: {n_noise} characters ---")
        print(f"  {'Agent':>20s} | {'Solved':>8s} {'Rate':>8s} {'AvgLvl':>8s}")
        print("  " + "-" * 50)

        noise_results = {}
        for agent_cls in agents:
            r = simulate(agent_cls, n_noise=n_noise, n_games=200,
                         max_actions=2000, seed=42)
            print(f"  {agent_cls.name:>20s} | {r['total_solved']:>6d}/200 "
                  f"{r['solve_rate']*100:>6.1f}% {r['avg_levels']:>6.2f}")
            noise_results[agent_cls.name] = r

        all_results[f"noise_{n_noise}"] = noise_results

    # Robustness analysis
    print(f"\n{'='*60}")
    print("NOISE ROBUSTNESS: Performance degradation from noise=0 to noise=8")
    for agent_cls in agents:
        r0 = all_results['noise_0'][agent_cls.name]['solve_rate']
        r8 = all_results['noise_8'][agent_cls.name]['solve_rate']
        drop = (r8 - r0) * 100
        print(f"  {agent_cls.name:>20s}: {r0*100:.1f}% -> {r8*100:.1f}% ({drop:+.1f}pp)")

    save_path = os.path.join(RESULTS_DIR, "phase51_simhash_curiosity.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 51: SimHash Curiosity',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
