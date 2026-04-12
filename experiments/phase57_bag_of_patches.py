"""
Phase 57: Bag-of-Patches Curiosity (Visual Cortex V1 Mimicry)
Emulate CNN's translation invariance with O(1) hash operations:
  - Extract all 3x3 local patches via sliding window
  - Hash each patch exactly (no fuzzy matching)
  - Novelty = fraction of unseen local patches in current state

"Same shape at different position = same hash" → spatial generalization
without Convolution, without fuzzy matching, without any NN overhead.

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
                    return False, True
                return True, True
        self._move_noise()
        self._render()
        return False, False


# ==============================================================
# Patch Extraction Utilities
# ==============================================================
def extract_patches(grid, patch_size=3, ignore_noise=True):
    """Extract all patch_size x patch_size patches from grid.
    Returns a set of patch hashes (translation-invariant).
    Noise (0.5 values) is zeroed out for robustness.
    """
    gs = grid.shape[0]
    patches = set()
    r = patch_size // 2
    for y in range(r, gs - r):
        for x in range(r, gs - r):
            patch = grid[y-r:y+r+1, x-r:x+r+1].copy()
            if ignore_noise:
                patch[patch == 0.5] = 0.0  # ignore noise characters
            patches.add(hash(patch.tobytes()))
    return patches


def extract_patches_multisize(grid, sizes=[2, 3, 4], ignore_noise=True):
    """Multi-scale patch extraction (like multi-scale V1 receptive fields)."""
    all_patches = set()
    for s in sizes:
        patches = extract_patches(grid, patch_size=s, ignore_noise=ignore_noise)
        # Prefix each hash with the patch size to avoid cross-scale collisions
        all_patches.update((s, h) for h in patches)
    return all_patches


# ==============================================================
# Agents
# ==============================================================
class RandomAgent:
    name = "Random"
    def __init__(self, state_dim):
        pass
    def choose(self, state_grid, state_flat, n_actions):
        return random.randint(0, n_actions - 1)


class SimHashAgent:
    """Baseline: Phase 51 SimHash curiosity on full state."""
    name = "SimHash (Full)"
    def __init__(self, state_dim, hash_bits=32):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True)
        self.seen = set()
    def _hash(self, state_flat):
        bits = tuple((state_flat @ self.proj > 0).astype(np.int8))
        return hash(bits)
    def choose(self, state_grid, state_flat, n_actions):
        sh = self._hash(state_flat)
        scores = []
        for a in range(n_actions):
            sa = sh ^ (a * 2654435761)
            novelty = 0 if sa in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(sh ^ (best * 2654435761))
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])
        return best


class BagOfPatchesAgent:
    """Bag-of-Patches curiosity: translation-invariant local pattern matching.
    Novelty = fraction of patches in current state not seen before.
    """
    name = "Bag-of-Patches"
    def __init__(self, state_dim, patch_size=3):
        self.patch_size = patch_size
        self.seen_patches = set()  # global set of all seen patch hashes

    def choose(self, state_grid, state_flat, n_actions):
        current_patches = extract_patches(state_grid, self.patch_size)
        # Novelty: fraction of patches that are new
        n_total = len(current_patches)
        if n_total == 0:
            return random.randint(0, n_actions - 1)
        n_new = len(current_patches - self.seen_patches)
        novelty_score = n_new / n_total

        # Add action-dependent exploration noise
        scores = []
        for a in range(n_actions):
            # Use action hash for tiebreaking
            action_key = hash((frozenset(current_patches), a))
            seen_key = action_key in self.seen_patches
            action_novelty = 0 if seen_key else novelty_score
            scores.append(action_novelty + random.gauss(0, 0.3))

        best = max(range(n_actions), key=lambda i: scores[i])

        # Register current patches as seen
        self.seen_patches.update(current_patches)
        if len(self.seen_patches) > 50000:
            self.seen_patches = set(list(self.seen_patches)[-25000:])

        return best


class BagOfPatchesXORAgent:
    """Bag-of-Patches + XOR-Hash state curiosity combined.
    Uses both local translation-invariance AND global state novelty.
    """
    name = "BoP+XOR"
    def __init__(self, state_dim, patch_size=3):
        self.patch_size = patch_size
        self.seen_patches = set()
        self.seen_states = set()

    def choose(self, state_grid, state_flat, n_actions):
        # Local curiosity: bag-of-patches
        current_patches = extract_patches(state_grid, self.patch_size)
        n_total = max(len(current_patches), 1)
        n_new = len(current_patches - self.seen_patches)
        local_novelty = n_new / n_total

        # Global curiosity: XOR-Hash
        sh = hash(state_flat.tobytes()) % (2**31)
        scores = []
        for a in range(n_actions):
            sa = sh ^ (a * 2654435761)
            global_novelty = 0 if sa in self.seen_states else 1
            # Combine local + global novelty
            combined = 0.5 * local_novelty + 0.5 * global_novelty
            scores.append(combined + random.gauss(0, 0.3))

        best = max(range(n_actions), key=lambda i: scores[i])

        # Update memory
        self.seen_patches.update(current_patches)
        self.seen_states.add(sh ^ (best * 2654435761))
        if len(self.seen_patches) > 50000:
            self.seen_patches = set(list(self.seen_patches)[-25000:])
        if len(self.seen_states) > 10000:
            self.seen_states = set(list(self.seen_states)[-5000:])

        return best


class MultiScaleBoPAgent:
    """Multi-scale Bag-of-Patches (2x2, 3x3, 4x4).
    Mimics multi-scale V1 receptive fields.
    """
    name = "MultiScale-BoP"
    def __init__(self, state_dim, sizes=[2, 3, 4]):
        self.sizes = sizes
        self.seen_patches = set()

    def choose(self, state_grid, state_flat, n_actions):
        current_patches = extract_patches_multisize(state_grid, self.sizes)
        n_total = max(len(current_patches), 1)
        n_new = len(current_patches - self.seen_patches)
        novelty = n_new / n_total

        scores = []
        for a in range(n_actions):
            action_key = hash((frozenset(current_patches), a))
            action_seen = action_key in self.seen_patches
            action_novelty = 0 if action_seen else novelty
            scores.append(action_novelty + random.gauss(0, 0.3))

        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen_patches.update(current_patches)
        if len(self.seen_patches) > 50000:
            self.seen_patches = set(list(self.seen_patches)[-25000:])
        return best


# ==============================================================
# Simulation
# ==============================================================
def simulate(agent_class, n_games=200, max_actions=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    state_dim = ARCLikeGame.GRID_SIZE ** 2

    total_solved = 0
    total_levels = 0

    for game_idx in range(n_games):
        game = ARCLikeGame(seed=seed + game_idx * 100)
        agent = agent_class(state_dim)

        for step in range(max_actions):
            state_grid = game.get_state()
            state_flat = state_grid.flatten()
            action = agent.choose(state_grid, state_flat, game.N_ACTIONS)
            all_done, level_up = game.step(action)
            if all_done:
                total_solved += 1
                break

        total_levels += game.solved_levels

    return {
        'agent': agent_class.name,
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'avg_levels': total_levels / n_games,
    }


def measure_latency(agent_class, state_dim=100, n_runs=5000):
    """Measure per-action latency."""
    agent = agent_class(state_dim)
    grid = np.random.randn(10, 10).astype(np.float32)
    flat = grid.flatten()

    # Warmup
    for _ in range(100):
        agent.choose(grid, flat, 4)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        agent.choose(grid, flat, 4)
    t1 = time.perf_counter()

    return (t1 - t0) / n_runs * 1000  # ms


def main():
    print("=" * 70)
    print("Phase 57: Bag-of-Patches Curiosity (Visual Cortex V1 Mimicry)")
    print("  Translation-invariant local pattern matching via hash sets")
    print("  Can we mimic CNN spatial generalization with O(1) operations?")
    print("=" * 70)

    agents = [RandomAgent, SimHashAgent, BagOfPatchesAgent,
              BagOfPatchesXORAgent, MultiScaleBoPAgent]
    state_dim = ARCLikeGame.GRID_SIZE ** 2

    # Latency benchmark
    print("\n--- Latency Benchmark ---")
    for agent_cls in agents:
        lat = measure_latency(agent_cls, state_dim)
        print(f"  {agent_cls.name:>20s}: {lat:.4f} ms/action")

    # Test with varying budgets
    budgets = [1000, 2000, 5000, 10000]
    all_results = {}

    for budget in budgets:
        print(f"\n--- Budget: {budget:,} actions ---")
        print(f"  {'Agent':>20s} | {'Solved':>8s} {'Rate':>8s} {'AvgLvl':>8s}")
        print("  " + "-" * 50)

        budget_results = {}
        for agent_cls in agents:
            t0 = time.time()
            r = simulate(agent_cls, n_games=200, max_actions=budget, seed=42)
            elapsed = time.time() - t0
            r['wall_time_sec'] = elapsed
            print(f"  {r['agent']:>20s} | {r['total_solved']:>5d}/200 "
                  f"{r['solve_rate']*100:>6.1f}% {r['avg_levels']:>6.2f} [{elapsed:.1f}s]")
            budget_results[r['agent']] = r

        all_results[f"budget_{budget}"] = budget_results

    # Noise robustness test (the real value proposition)
    print(f"\n{'='*70}")
    print("NOISE ROBUSTNESS TEST: Same maps, varying noise levels")
    noise_results = {}
    for n_noise in [0, 3, 5, 8]:
        print(f"\n--- Noise: {n_noise} characters (budget=5000) ---")
        nr = {}
        for agent_cls in agents:
            # Modify noise count inline
            original_noise = ARCLikeGame.N_NOISE
            ARCLikeGame.N_NOISE = n_noise
            r = simulate(agent_cls, n_games=200, max_actions=5000, seed=42)
            ARCLikeGame.N_NOISE = original_noise
            nr[agent_cls.name] = r
            print(f"  {agent_cls.name:>20s}: {r['solve_rate']*100:.1f}%")
        noise_results[f"noise_{n_noise}"] = nr

    all_results['noise_robustness'] = noise_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Solve Rate (%) @ budget=5000")
    for agent_cls in agents:
        r = all_results['budget_5000'][agent_cls.name]
        print(f"  {r['agent']:>20s}: {r['solve_rate']*100:.1f}%")

    # Key analysis
    print(f"\n{'='*70}")
    print("KEY FINDING: Does Bag-of-Patches beat SimHash?")
    sim_rate = all_results['budget_5000']['SimHash (Full)']['solve_rate']
    bop_rate = all_results['budget_5000']['Bag-of-Patches']['solve_rate']
    hybrid_rate = all_results['budget_5000']['BoP+XOR']['solve_rate']
    multi_rate = all_results['budget_5000']['MultiScale-BoP']['solve_rate']
    print(f"  SimHash (Full):     {sim_rate*100:.1f}%")
    print(f"  Bag-of-Patches:     {bop_rate*100:.1f}% ({(bop_rate-sim_rate)*100:+.1f}pp vs SimHash)")
    print(f"  BoP+XOR hybrid:    {hybrid_rate*100:.1f}% ({(hybrid_rate-sim_rate)*100:+.1f}pp vs SimHash)")
    print(f"  MultiScale-BoP:    {multi_rate*100:.1f}% ({(multi_rate-sim_rate)*100:+.1f}pp vs SimHash)")

    save_path = os.path.join(RESULTS_DIR, "phase57_bag_of_patches.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 57: Bag-of-Patches Curiosity',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
