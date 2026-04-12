"""
Phase 53: Associative SimHash ExIt (Hippocampal Associative Memory)
Break the "exact match curse" of Patch-Hash using SimHash + Hamming distance.
Fuzzy matching via bit operations: XOR + popcount = O(1) associative recall.

Author: Hiroto Funasaki
"""
import os, json, random, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


class SpatialPuzzle:
    GRID_SIZE = 10
    N_ACTIONS = 4
    def __init__(self, n_levels=5, seed=None):
        self.rng = random.Random(seed)
        self.n_levels = n_levels
        self.level = 0
        self.solved_levels = 0
        self.total_steps = 0
        self._generate_level()

    def _generate_level(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        n_walls = 5 + self.level * 2
        self.walls = set()
        for _ in range(n_walls):
            self.walls.add((self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)))
        while True:
            self.ax, self.ay = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.ax, self.ay) not in self.walls: break
        while True:
            self.gx, self.gy = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.gx, self.gy) not in self.walls and (self.gx, self.gy) != (self.ax, self.ay): break
        # Add noise characters
        self.noise = []
        for _ in range(3):
            self.noise.append([self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)])
        self._render()

    def _render(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        for wx, wy in self.walls:
            self.grid[wy, wx] = -1.0
        self.grid[self.ay, self.ax] = 1.0
        self.grid[self.gy, self.gx] = 2.0
        for nx, ny in self.noise:
            if 0 <= nx < gs and 0 <= ny < gs and self.grid[ny, nx] == 0:
                self.grid[ny, nx] = 0.5

    def get_local_patch(self, radius=2):
        """5x5 patch around agent (bigger for more context)."""
        gs = self.GRID_SIZE
        patch = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = self.ay + dy, self.ax + dx
                if 0 <= ny < gs and 0 <= nx < gs:
                    val = self.grid[ny, nx]
                    if val == 0.5: val = 0.0  # ignore noise
                    patch.append(val)
                else:
                    patch.append(-1)
        return np.array(patch, dtype=np.float32)

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
                if self.level < self.n_levels:
                    self._generate_level()
                    return True, False  # level_up, not all_done
                return True, True
        # Move noise
        for pos in self.noise:
            dx2, dy2 = [(0,-1),(0,1),(-1,0),(1,0),(0,0)][self.rng.randint(0,4)]
            nnx, nny = pos[0]+dx2, pos[1]+dy2
            if 0 <= nnx < gs and 0 <= nny < gs:
                pos[0], pos[1] = nnx, nny
        self._render()
        return False, False


# ==============================================================
# ExIt Agents
# ==============================================================
class ExactPatchExIt:
    """Phase 50: Exact hash match (the one that failed on cross-map)."""
    name = "Exact-Patch"
    def __init__(self, n_actions=4):
        self.table = {}
        self.n_actions = n_actions
        self.trained = False

    def learn(self, patches, actions):
        for p, a in zip(patches, actions):
            key = hash(p.tobytes())
            if key not in self.table:
                self.table[key] = np.zeros(self.n_actions, dtype=np.float32)
            self.table[key][a] += 1.0
        self.trained = True

    def predict(self, patch):
        key = hash(patch.tobytes())
        if key in self.table and self.table[key].sum() > 0:
            return int(np.argmax(self.table[key]))
        return None


class AssociativeSimHashExIt:
    """NEW: SimHash + Hamming distance fuzzy matching (hippocampal recall)."""
    name = "Associative-SimHash"

    def __init__(self, n_actions=4, hash_bits=32, hamming_threshold=4):
        self.n_actions = n_actions
        self.hash_bits = hash_bits
        self.threshold = hamming_threshold
        np.random.seed(42)
        self.projection = np.random.randn(25, hash_bits).astype(np.float32)  # 5x5=25
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        # Memory: list of (simhash_int, action_counts)
        self.memory = []
        self.trained = False

    def _simhash(self, patch):
        """Convert patch to SimHash integer."""
        projected = patch @ self.projection
        bits = (projected > 0).astype(np.uint32)
        result = 0
        for i in range(self.hash_bits):
            result |= (int(bits[i]) << i)
        return result

    def _hamming(self, a, b):
        """Hamming distance via XOR + popcount. O(1)."""
        x = a ^ b
        count = 0
        while x:
            count += 1
            x &= (x - 1)  # Kernighan's trick
        return count

    def learn(self, patches, actions):
        for p, a in zip(patches, actions):
            sh = self._simhash(p)
            # Check if similar hash already exists
            found = False
            for i, (stored_hash, counts) in enumerate(self.memory):
                if self._hamming(sh, stored_hash) <= 1:
                    counts[a] += 1.0
                    found = True
                    break
            if not found:
                counts = np.zeros(self.n_actions, dtype=np.float32)
                counts[a] = 1.0
                self.memory.append((sh, counts))
        self.trained = True

    def predict(self, patch):
        if not self.memory:
            return None
        sh = self._simhash(patch)

        best_dist = self.hash_bits + 1
        best_counts = None

        for stored_hash, counts in self.memory:
            dist = self._hamming(sh, stored_hash)
            if dist < best_dist:
                best_dist = dist
                best_counts = counts

        if best_dist <= self.threshold and best_counts is not None and best_counts.sum() > 0:
            return int(np.argmax(best_counts))
        return None


class MultiThresholdSimHash:
    """SimHash with adaptive threshold: try strict first, then relax."""
    name = "Adaptive-SimHash"

    def __init__(self, n_actions=4, hash_bits=32):
        self.inner = AssociativeSimHashExIt(n_actions, hash_bits, hamming_threshold=8)
        self.thresholds = [0, 1, 2, 4, 6, 8]

    def learn(self, patches, actions):
        self.inner.learn(patches, actions)

    @property
    def trained(self):
        return self.inner.trained

    def predict(self, patch):
        if not self.inner.memory:
            return None
        sh = self.inner._simhash(patch)

        best_dist = self.inner.hash_bits + 1
        best_counts = None
        for stored_hash, counts in self.inner.memory:
            dist = self.inner._hamming(sh, stored_hash)
            if dist < best_dist:
                best_dist = dist
                best_counts = counts

        # Try thresholds from strictest to most relaxed
        for t in self.thresholds:
            if best_dist <= t and best_counts is not None and best_counts.sum() > 0:
                return int(np.argmax(best_counts))
        return None


# ==============================================================
# Simulation
# ==============================================================
def simulate(exit_class, n_games=200, max_actions=3000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    total_solved = 0
    total_levels = 0
    total_hits = 0
    total_misses = 0

    for game_idx in range(n_games):
        game = SpatialPuzzle(n_levels=5, seed=seed + game_idx * 100)
        agent = exit_class(n_actions=game.N_ACTIONS)
        patch_history = []
        action_history = []

        for step in range(max_actions):
            patch = game.get_local_patch(radius=2)

            # Try learned prediction
            pred = None
            if agent.trained:
                pred = agent.predict(patch)

            if pred is not None:
                action = pred
                total_hits += 1
            else:
                # XOR-hash exploration fallback
                state_hash = hash(game.grid.tobytes()) % (2**31)
                scores = [random.gauss(0, 1) for _ in range(game.N_ACTIONS)]
                action = max(range(game.N_ACTIONS), key=lambda i: scores[i])
                total_misses += 1

            patch_history.append(patch.copy())
            action_history.append(action)

            level_up, all_done = game.step(action)

            if level_up:
                agent.learn(patch_history, action_history)
                patch_history = []
                action_history = []

            if all_done:
                total_solved += 1
                break

        total_levels += game.solved_levels

    hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

    return {
        'agent': exit_class.name,
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'avg_levels': total_levels / n_games,
        'hit_rate': hit_rate,
    }


def main():
    print("=" * 60)
    print("Phase 53: Associative SimHash ExIt")
    print("  Fuzzy matching via Hamming distance (hippocampal recall)")
    print("=" * 60)

    agents = [ExactPatchExIt, AssociativeSimHashExIt, MultiThresholdSimHash]
    budgets = [500, 1000, 2000, 5000]

    all_results = {}

    for max_actions in budgets:
        print(f"\n--- Budget: {max_actions} actions ---")
        print(f"  {'Agent':>22s} | {'Solved':>8s} {'Rate':>8s} {'AvgLvl':>8s} {'HitRate':>8s}")
        print("  " + "-" * 60)

        budget_results = {}
        for agent_cls in agents:
            r = simulate(agent_cls, n_games=200, max_actions=max_actions, seed=42)
            print(f"  {r['agent']:>22s} | {r['total_solved']:>6d}/200 "
                  f"{r['solve_rate']*100:>6.1f}% {r['avg_levels']:>6.2f} "
                  f"{r['hit_rate']*100:>6.1f}%")
            budget_results[r['agent']] = r

        all_results[f"budget_{max_actions}"] = budget_results

    # Key comparison
    print(f"\n{'='*60}")
    print("EXACT vs ASSOCIATIVE SimHash @ budget=5000:")
    exact = all_results['budget_5000']['Exact-Patch']
    assoc = all_results['budget_5000']['Associative-SimHash']
    adapt = all_results['budget_5000']['Adaptive-SimHash']
    print(f"  Exact-Patch:       {exact['solve_rate']*100:.1f}% (hit={exact['hit_rate']*100:.1f}%)")
    print(f"  Associative:       {assoc['solve_rate']*100:.1f}% (hit={assoc['hit_rate']*100:.1f}%)")
    print(f"  Adaptive-SimHash:  {adapt['solve_rate']*100:.1f}% (hit={adapt['hit_rate']*100:.1f}%)")
    diff = (assoc['solve_rate'] - exact['solve_rate']) * 100
    print(f"  Improvement:       {diff:+.1f}pp")

    save_path = os.path.join(RESULTS_DIR, "phase53_associative_simhash.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 53: Associative SimHash ExIt',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
