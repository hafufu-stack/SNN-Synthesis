"""
Phase 55: Macro-Action Chunking (Cerebellar Motor Memory)
Extract frequently-occurring action sequences from miracles
and execute them as "muscle memory" without looking at state.

State-independent Open-Loop control via action chunks.

Author: Hiroto Funasaki
"""
import os, json, random, numpy as np
from collections import Counter
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
        self.walls = set()
        for _ in range(5 + self.level * 2):
            self.walls.add((self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)))
        while True:
            self.ax, self.ay = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.ax, self.ay) not in self.walls: break
        while True:
            self.gx, self.gy = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if (self.gx, self.gy) not in self.walls and (self.gx, self.gy) != (self.ax, self.ay): break
        self.noise = [[self.rng.randint(0,gs-1), self.rng.randint(0,gs-1)] for _ in range(3)]
        self._render()

    def _render(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        for wx, wy in self.walls: self.grid[wy, wx] = -1
        self.grid[self.ay, self.ax] = 1
        self.grid[self.gy, self.gx] = 2
        for nx, ny in self.noise:
            if 0 <= nx < gs and 0 <= ny < gs and self.grid[ny, nx] == 0:
                self.grid[ny, nx] = 0.5

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
                    return True, False
                return True, True
        for pos in self.noise:
            d = [(0,-1),(0,1),(-1,0),(1,0),(0,0)][self.rng.randint(0,4)]
            p0, p1 = pos[0]+d[0], pos[1]+d[1]
            if 0 <= p0 < gs and 0 <= p1 < gs: pos[0], pos[1] = p0, p1
        self._render()
        return False, False


# ==============================================================
# Macro-Action Extraction
# ==============================================================
def extract_chunks(trajectories, chunk_sizes=[2, 3, 4], top_k=10):
    """Extract most frequent action subsequences from miracle trajectories."""
    chunk_counts = Counter()
    for traj in trajectories:
        for cs in chunk_sizes:
            for i in range(len(traj) - cs + 1):
                chunk = tuple(traj[i:i+cs])
                chunk_counts[chunk] += 1
    return [chunk for chunk, _ in chunk_counts.most_common(top_k)]


# ==============================================================
# Agents
# ==============================================================
class RandomAgent:
    name = "Random"
    def __init__(self):
        self.macro_queue = []
    def step(self, n_actions):
        return random.randint(0, n_actions - 1)
    def on_miracle(self, trajectory): pass


class MacroChunkAgent:
    """Execute extracted macro-actions (open-loop, state-independent)."""
    name = "Macro-Chunk"

    def __init__(self, chunk_prob=0.3):
        self.chunks = []
        self.macro_queue = []
        self.chunk_prob = chunk_prob

    def step(self, n_actions):
        # If we're in the middle of a macro, continue
        if self.macro_queue:
            return self.macro_queue.pop(0)

        # Chance to start a macro
        if self.chunks and random.random() < self.chunk_prob:
            chunk = random.choice(self.chunks)
            self.macro_queue = list(chunk[1:])  # remaining actions
            return chunk[0]

        return random.randint(0, n_actions - 1)

    def on_miracle(self, trajectory):
        new_chunks = extract_chunks([trajectory], chunk_sizes=[2,3,4], top_k=20)
        for c in new_chunks:
            if c not in self.chunks:
                self.chunks.append(c)
        # Keep top-30
        self.chunks = self.chunks[:30]


class MacroPlusCuriosityAgent:
    """Macro-chunks + XOR-Hash curiosity for unexplored states."""
    name = "Macro+Curiosity"

    def __init__(self, chunk_prob=0.3):
        self.chunks = []
        self.macro_queue = []
        self.chunk_prob = chunk_prob
        self.seen = set()

    def step(self, n_actions, state_hash=None):
        # Continue macro
        if self.macro_queue:
            return self.macro_queue.pop(0)

        # If state is novel, start a macro
        if state_hash is not None:
            is_novel = state_hash not in self.seen
            self.seen.add(state_hash)
            if len(self.seen) > 10000:
                self.seen = set(list(self.seen)[-5000:])

        if self.chunks and random.random() < self.chunk_prob:
            chunk = random.choice(self.chunks)
            self.macro_queue = list(chunk[1:])
            return chunk[0]

        # Curiosity-guided single action
        if state_hash is not None:
            scores = []
            for a in range(n_actions):
                sa = state_hash ^ (a * 2654435761)
                novelty = 0 if sa in self.seen else 1
                scores.append(novelty + random.gauss(0, 0.3))
            return max(range(n_actions), key=lambda i: scores[i])

        return random.randint(0, n_actions - 1)

    def on_miracle(self, trajectory):
        new_chunks = extract_chunks([trajectory], chunk_sizes=[2,3,4], top_k=20)
        for c in new_chunks:
            if c not in self.chunks:
                self.chunks.append(c)
        self.chunks = self.chunks[:30]


class AdaptiveChunkAgent:
    """Macro-chunks with adaptive probability (increase after miracles)."""
    name = "Adaptive-Chunk"

    def __init__(self):
        self.chunks = []
        self.macro_queue = []
        self.chunk_prob = 0.1  # start low
        self.n_miracles = 0
        self.seen = set()

    def step(self, n_actions, state_hash=None):
        if self.macro_queue:
            return self.macro_queue.pop(0)

        if self.chunks and random.random() < self.chunk_prob:
            chunk = random.choice(self.chunks)
            self.macro_queue = list(chunk[1:])
            return chunk[0]

        if state_hash is not None:
            scores = []
            for a in range(n_actions):
                sa = state_hash ^ (a * 2654435761)
                novelty = 0 if sa in self.seen else 1
                scores.append(novelty + random.gauss(0, 0.3))
            self.seen.add(state_hash)
            if len(self.seen) > 10000:
                self.seen = set(list(self.seen)[-5000:])
            return max(range(n_actions), key=lambda i: scores[i])

        return random.randint(0, n_actions - 1)

    def on_miracle(self, trajectory):
        self.n_miracles += 1
        self.chunk_prob = min(0.5, 0.1 + self.n_miracles * 0.1)
        new_chunks = extract_chunks([trajectory], chunk_sizes=[2,3,4], top_k=20)
        for c in new_chunks:
            if c not in self.chunks:
                self.chunks.append(c)
        self.chunks = self.chunks[:30]


# ==============================================================
# Simulation
# ==============================================================
def simulate(agent_class, n_games=200, max_actions=3000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    total_solved = 0
    total_levels = 0

    for game_idx in range(n_games):
        game = SpatialPuzzle(n_levels=5, seed=seed + game_idx * 100)
        agent = agent_class()
        action_history = []

        for step in range(max_actions):
            state_hash = hash(game.grid.tobytes()) % (2**31)

            if hasattr(agent, 'seen'):
                action = agent.step(game.N_ACTIONS, state_hash=state_hash)
            else:
                action = agent.step(game.N_ACTIONS)

            action_history.append(action)

            level_up, all_done = game.step(action)

            if level_up:
                agent.on_miracle(action_history)
                action_history = []

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


def main():
    print("=" * 60)
    print("Phase 55: Macro-Action Chunking (Cerebellar Motor Memory)")
    print("  State-independent open-loop control via action chunks")
    print("=" * 60)

    agents = [RandomAgent, MacroChunkAgent, MacroPlusCuriosityAgent, AdaptiveChunkAgent]
    budgets = [500, 1000, 2000, 5000]

    all_results = {}

    for max_actions in budgets:
        print(f"\n--- Budget: {max_actions} actions ---")
        print(f"  {'Agent':>22s} | {'Solved':>8s} {'Rate':>8s} {'AvgLvl':>8s}")
        print("  " + "-" * 50)

        budget_results = {}
        for agent_cls in agents:
            r = simulate(agent_cls, n_games=200, max_actions=max_actions, seed=42)
            print(f"  {r['agent']:>22s} | {r['total_solved']:>6d}/200 "
                  f"{r['solve_rate']*100:>6.1f}% {r['avg_levels']:>6.2f}")
            budget_results[r['agent']] = r

        all_results[f"budget_{max_actions}"] = budget_results

    # Key finding
    print(f"\n{'='*60}")
    print("MACRO-CHUNK vs RANDOM @ budget=5000:")
    for agent_cls in agents:
        r = all_results['budget_5000'][agent_cls.name]
        print(f"  {r['agent']:>22s}: {r['solve_rate']*100:.1f}% ({r['avg_levels']:.2f} levels)")

    save_path = os.path.join(RESULTS_DIR, "phase55_macro_chunks.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 55: Macro-Action Chunking',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
