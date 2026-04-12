"""
Phase 58: Curiosity-Diverse Swarm (Hash-Diverse Beam Search)
Extension of σ-diverse NBS (Phase 37a): instead of only varying noise σ,
also vary the curiosity algorithm itself across beams.

K=11 beams, each with different:
  - Hash function: SimHash, XOR-Hash, Bloom-Filter, Random
  - σ noise levels

Different hash functions have different "blind spots" (collision patterns).
Diversity across curiosity types may let at least one beam slip through
where others get stuck.

Author: Hiroto Funasaki
"""
import os, json, random, time, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


class ARCLikeGame:
    """ARC-like grid puzzle."""
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

    def clone(self):
        """Deep copy for beam search."""
        c = ARCLikeGame.__new__(ARCLikeGame)
        c.rng = random.Random()
        c.rng.setstate(self.rng.getstate())
        c.level = self.level
        c.solved_levels = self.solved_levels
        c.total_steps = self.total_steps
        c.walls = set(self.walls)
        c.ax, c.ay = self.ax, self.ay
        c.gx, c.gy = self.gx, self.gy
        c.noise_pos = [list(p) for p in self.noise_pos]
        c.grid = self.grid.copy()
        return c

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
# Curiosity Modules (pluggable into beams)
# ==============================================================
class SimHashCuriosity:
    name = "SimHash"
    def __init__(self, state_dim, seed=42):
        rng = np.random.RandomState(seed)
        self.proj = rng.randn(state_dim, 32).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True)
        self.seen = set()

    def novelty(self, state_flat, action):
        bits = tuple((state_flat @ self.proj > 0).astype(np.int8))
        sh = hash(bits)
        sa = sh ^ (action * 2654435761)
        return 0 if sa in self.seen else 1

    def update(self, state_flat, action):
        bits = tuple((state_flat @ self.proj > 0).astype(np.int8))
        sh = hash(bits)
        self.seen.add(sh ^ (action * 2654435761))
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])


class XORCuriosity:
    name = "XOR"
    def __init__(self, state_dim, seed=42):
        self.seen = set()

    def novelty(self, state_flat, action):
        sh = hash(state_flat.tobytes()) % (2**31)
        sa = sh ^ (action * 2654435761)
        return 0 if sa in self.seen else 1

    def update(self, state_flat, action):
        sh = hash(state_flat.tobytes()) % (2**31)
        self.seen.add(sh ^ (action * 2654435761))
        if len(self.seen) > 10000:
            self.seen = set(list(self.seen)[-5000:])


class BloomCuriosity:
    """Lightweight Bloom-filter-based novelty.
    Uses multiple hash functions to reduce false positives.
    """
    name = "Bloom"
    def __init__(self, state_dim, seed=42, n_hashes=4, table_size=65536):
        self.n_hashes = n_hashes
        self.table_size = table_size
        self.bits = np.zeros(table_size, dtype=np.bool_)
        rng = np.random.RandomState(seed)
        # Random projection seeds for different hash functions
        self.salts = [rng.randint(0, 2**31) for _ in range(n_hashes)]

    def _hash_indices(self, state_flat, action):
        base = hash(state_flat.tobytes()) ^ (action * 2654435761)
        return [(base ^ salt) % self.table_size for salt in self.salts]

    def novelty(self, state_flat, action):
        indices = self._hash_indices(state_flat, action)
        # If all bits are set, we've likely seen this before
        if all(self.bits[idx] for idx in indices):
            return 0
        return 1

    def update(self, state_flat, action):
        indices = self._hash_indices(state_flat, action)
        for idx in indices:
            self.bits[idx] = True


class RandomCuriosity:
    """No curiosity: pure random. Acts as control within swarm."""
    name = "Random"
    def __init__(self, state_dim, seed=42):
        pass
    def novelty(self, state_flat, action):
        return random.random()  # uniform noise
    def update(self, state_flat, action):
        pass


# ==============================================================
# Beam Search Agents
# ==============================================================
class SingleCuriosityAgent:
    """Single agent with one curiosity module (baseline)."""
    def __init__(self, curiosity_class, state_dim, sigma=0.3, seed=42):
        self.curiosity = curiosity_class(state_dim, seed=seed)
        self.sigma = sigma
        self.name = f"Single-{curiosity_class.name}"

    def choose(self, state_flat, n_actions):
        scores = []
        for a in range(n_actions):
            nov = self.curiosity.novelty(state_flat, a)
            scores.append(nov + random.gauss(0, self.sigma))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.curiosity.update(state_flat, best)
        return best


class SigmaDiverseBeam:
    """Phase 37a: σ-diverse NBS with single curiosity type."""
    name = "σ-Diverse (SimHash)"
    def __init__(self, state_dim, K=11, seed=42):
        self.K = K
        self.sigmas = np.linspace(0.01, 3.0, K).tolist()
        self.curiosities = [SimHashCuriosity(state_dim, seed=seed + i) for i in range(K)]

    def choose(self, state_flat, n_actions):
        best_action = None
        best_score = float('-inf')
        best_k = 0
        for k in range(self.K):
            scores = []
            for a in range(n_actions):
                nov = self.curiosities[k].novelty(state_flat, a)
                scores.append(nov + random.gauss(0, self.sigmas[k]))
            action = max(range(n_actions), key=lambda i: scores[i])
            score = max(scores)
            if score > best_score:
                best_score = score
                best_action = action
                best_k = k
        # Update only the winning beam's curiosity
        self.curiosities[best_k].update(state_flat, best_action)
        return best_action


class CuriosityDiverseSwarm:
    """THE NEW AGENT: Curiosity-diverse beam search.
    Each beam has a DIFFERENT curiosity algorithm + different σ.
    """
    name = "Curiosity-Diverse Swarm"
    def __init__(self, state_dim, K=11, seed=42):
        self.K = K
        # Assign curiosity types round-robin across beams
        curiosity_types = [SimHashCuriosity, XORCuriosity, BloomCuriosity, RandomCuriosity]
        self.beams = []
        self.sigmas = np.linspace(0.01, 3.0, K).tolist()
        for k in range(K):
            ct = curiosity_types[k % len(curiosity_types)]
            cur = ct(state_dim, seed=seed + k * 100)
            self.beams.append(cur)

    def choose(self, state_flat, n_actions):
        best_action = None
        best_score = float('-inf')
        best_k = 0
        for k in range(self.K):
            scores = []
            for a in range(n_actions):
                nov = self.beams[k].novelty(state_flat, a)
                scores.append(nov + random.gauss(0, self.sigmas[k]))
            action = max(range(n_actions), key=lambda i: scores[i])
            score = max(scores)
            if score > best_score:
                best_score = score
                best_action = action
                best_k = k
        self.beams[best_k].update(state_flat, best_action)
        return best_action


class CuriosityDiverseWithVoting:
    """Variant: majority vote instead of max score."""
    name = "Diverse-Vote"
    def __init__(self, state_dim, K=11, seed=42):
        self.K = K
        curiosity_types = [SimHashCuriosity, XORCuriosity, BloomCuriosity, RandomCuriosity]
        self.beams = []
        self.sigmas = np.linspace(0.01, 3.0, K).tolist()
        for k in range(K):
            ct = curiosity_types[k % len(curiosity_types)]
            self.beams.append(ct(state_dim, seed=seed + k * 100))

    def choose(self, state_flat, n_actions):
        votes = np.zeros(n_actions, dtype=np.float32)
        for k in range(self.K):
            scores = []
            for a in range(n_actions):
                nov = self.beams[k].novelty(state_flat, a)
                scores.append(nov + random.gauss(0, self.sigmas[k]))
            action = max(range(n_actions), key=lambda i: scores[i])
            votes[action] += 1.0
        best = int(np.argmax(votes))
        for beam in self.beams:
            beam.update(state_flat, best)
        return best


# ==============================================================
# Simulation
# ==============================================================
def simulate_beam(agent, n_games=200, max_actions=5000, seed=42):
    total_solved = 0
    total_levels = 0
    state_dim = ARCLikeGame.GRID_SIZE ** 2

    for game_idx in range(n_games):
        random.seed(seed + game_idx * 1000)
        np.random.seed(seed + game_idx * 1000)
        game = ARCLikeGame(seed=seed + game_idx * 100)

        for step in range(max_actions):
            state = game.get_state()
            state_flat = state.flatten()
            action = agent.choose(state_flat, game.N_ACTIONS)
            all_done, level_up = game.step(action)
            if all_done:
                total_solved += 1
                break
        total_levels += game.solved_levels

    return {
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'avg_levels': total_levels / n_games,
    }


def main():
    print("=" * 70)
    print("Phase 58: Curiosity-Diverse Swarm (Hash-Diverse Beam Search)")
    print("  σ-diversity + curiosity-algorithm diversity = Swarm Intelligence")
    print("=" * 70)

    state_dim = ARCLikeGame.GRID_SIZE ** 2
    budgets = [1000, 2000, 5000, 10000]

    # Define all agents to test
    agents_configs = [
        ("Random", lambda: SingleCuriosityAgent(RandomCuriosity, state_dim, sigma=1.0)),
        ("Single-SimHash", lambda: SingleCuriosityAgent(SimHashCuriosity, state_dim, sigma=0.3)),
        ("Single-XOR", lambda: SingleCuriosityAgent(XORCuriosity, state_dim, sigma=0.3)),
        ("Single-Bloom", lambda: SingleCuriosityAgent(BloomCuriosity, state_dim, sigma=0.3)),
        ("σ-Diverse-SimHash", lambda: SigmaDiverseBeam(state_dim, K=11)),
        ("Curiosity-Diverse", lambda: CuriosityDiverseSwarm(state_dim, K=11)),
        ("Diverse-Vote", lambda: CuriosityDiverseWithVoting(state_dim, K=11)),
    ]

    all_results = {}

    for budget in budgets:
        print(f"\n{'='*70}")
        print(f"  BUDGET: {budget:,} actions")
        print(f"{'='*70}")
        print(f"  {'Agent':>25s} | {'Solved':>8s} {'Rate':>8s} {'AvgLvl':>8s}")
        print("  " + "-" * 55)

        budget_results = {}
        for name, make_agent in agents_configs:
            agent = make_agent()
            t0 = time.time()
            random.seed(42)
            np.random.seed(42)
            r = simulate_beam(agent, n_games=200, max_actions=budget, seed=42)
            elapsed = time.time() - t0
            r['agent'] = name
            r['wall_time_sec'] = elapsed
            print(f"  {name:>25s} | {r['total_solved']:>5d}/200 "
                  f"{r['solve_rate']*100:>6.1f}% {r['avg_levels']:>6.2f} [{elapsed:.1f}s]")
            budget_results[name] = r

        all_results[f"budget_{budget}"] = budget_results

    # Summary
    print(f"\n{'='*70}")
    print("SCALING SUMMARY: Solve Rate (%)")
    print(f"{'Agent':>25s}", end="")
    for b in budgets:
        print(f" {b:>8,}", end="")
    print()
    print("-" * (25 + 9 * len(budgets)))
    for name, _ in agents_configs:
        print(f"{name:>25s}", end="")
        for b in budgets:
            key = f"budget_{b}"
            rate = all_results[key][name]['solve_rate'] * 100
            print(f" {rate:>7.1f}%", end="")
        print()

    # Key analysis
    print(f"\n{'='*70}")
    print("KEY FINDING: Does curiosity diversity beat σ-diversity alone?")
    sigma_rate = all_results['budget_5000']['σ-Diverse-SimHash']['solve_rate']
    diverse_rate = all_results['budget_5000']['Curiosity-Diverse']['solve_rate']
    vote_rate = all_results['budget_5000']['Diverse-Vote']['solve_rate']
    single_sim = all_results['budget_5000']['Single-SimHash']['solve_rate']
    print(f"  Single-SimHash:        {single_sim*100:.1f}%")
    print(f"  σ-Diverse (SimHash):   {sigma_rate*100:.1f}%")
    print(f"  Curiosity-Diverse:     {diverse_rate*100:.1f}% ({(diverse_rate-sigma_rate)*100:+.1f}pp vs σ-Diverse)")
    print(f"  Diverse-Vote:          {vote_rate*100:.1f}% ({(vote_rate-sigma_rate)*100:+.1f}pp vs σ-Diverse)")

    # Per-curiosity-type analysis at budget=5000
    print(f"\n{'='*70}")
    print("INDIVIDUAL CURIOSITY TYPES @ budget=5000:")
    for name in ['Single-SimHash', 'Single-XOR', 'Single-Bloom', 'Random']:
        r = all_results['budget_5000'][name]
        print(f"  {name:>20s}: {r['solve_rate']*100:.1f}%")

    save_path = os.path.join(RESULTS_DIR, "phase58_curiosity_diverse_swarm.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 58: Curiosity-Diverse Swarm',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
