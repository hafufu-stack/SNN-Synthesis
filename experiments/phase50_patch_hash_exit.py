"""
Phase 50: Relative-Patch Hash ExIt
Instead of memorizing action sequences (N-Gram), memorize
"local 3x3 patch around change -> optimal action" mappings.
This generalizes across different starting positions/maps.

O(1) learning, O(1) inference - CNN replacement via dictionary.

Author: Hiroto Funasaki
"""
import os, json, random, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


class SpatialPuzzle:
    """Grid puzzle where agent position and map change each level."""
    GRID_SIZE = 10
    N_ACTIONS = 4  # UP, DOWN, LEFT, RIGHT

    def __init__(self, n_levels=5, seed=None):
        self.rng = random.Random(seed)
        self.n_levels = n_levels
        self.level = 0
        self.steps = 0
        self.total_steps = 0
        self.solved_levels = 0
        self._generate_level()

    def _generate_level(self):
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        # Random walls
        for _ in range(gs):
            wx, wy = self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)
            self.grid[wy, wx] = -1  # wall

        # Agent position (random, not on wall)
        while True:
            self.ax, self.ay = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if self.grid[self.ay, self.ax] != -1:
                break

        # Goal position (random, not on wall, not on agent)
        while True:
            self.gx, self.gy = self.rng.randint(1, gs-2), self.rng.randint(1, gs-2)
            if self.grid[self.gy, self.gx] != -1 and (self.gx, self.gy) != (self.ax, self.ay):
                break

        self.grid[self.ay, self.ax] = 1   # agent
        self.grid[self.gy, self.gx] = 2   # goal
        self.steps = 0

    def get_state(self):
        return self.grid.copy()

    def get_local_patch(self, radius=1):
        """Get local patch around agent position."""
        gs = self.GRID_SIZE
        patch = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                ny, nx = self.ay + dy, self.ax + dx
                if 0 <= ny < gs and 0 <= nx < gs:
                    patch.append(self.grid[ny, nx])
                else:
                    patch.append(-1)  # out of bounds = wall
        return tuple(patch)

    def get_relative_goal_direction(self):
        """Get relative direction to goal (for oracle)."""
        dx = self.gx - self.ax
        dy = self.gy - self.ay
        # Prefer larger displacement axis
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 2  # RIGHT or LEFT
        else:
            return 1 if dy > 0 else 0  # DOWN or UP

    def step(self, action):
        self.steps += 1
        self.total_steps += 1
        gs = self.GRID_SIZE

        # Move: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        nx, ny = self.ax + dx, self.ay + dy

        # Check bounds and walls
        if 0 <= nx < gs and 0 <= ny < gs and self.grid[ny, nx] != -1:
            self.grid[self.ay, self.ax] = 0  # clear old pos
            self.ax, self.ay = nx, ny

            # Check goal
            if (self.ax, self.ay) == (self.gx, self.gy):
                self.solved_levels += 1
                self.level += 1
                if self.level < self.n_levels:
                    self._generate_level()
                    return self.get_state(), False, True  # level_up
                else:
                    return self.get_state(), True, True   # all solved

            self.grid[self.ay, self.ax] = 1  # set new pos

        return self.get_state(), False, False


# ==============================================================
# ExIt Agents
# ==============================================================
class ActionNGramExIt:
    """Phase 48 approach: memorize action sequences."""
    name = "Action-NGram"

    def __init__(self, n_actions=4):
        self.table = {}
        self.n_actions = n_actions
        self.trained = False

    def learn(self, trajectory):
        for i in range(len(trajectory)):
            for n in [1, 2, 3]:
                ctx = tuple(trajectory[max(0, i-n):i]) if i > 0 else ()
                key = hash(ctx)
                if key not in self.table:
                    self.table[key] = np.zeros(self.n_actions, dtype=np.float32)
                self.table[key][trajectory[i]] += 1.0
        self.trained = True

    def predict(self, action_history, state=None):
        for n in [3, 2, 1]:
            ctx = tuple(action_history[-n:]) if len(action_history) >= n else tuple(action_history)
            key = hash(ctx)
            if key in self.table and self.table[key].sum() > 0:
                return int(np.argmax(self.table[key]))
        return None


class PatchHashExIt:
    """NEW: memorize local_patch -> optimal_action."""
    name = "Patch-Hash"

    def __init__(self, n_actions=4, patch_radius=1):
        self.table = {}
        self.n_actions = n_actions
        self.patch_radius = patch_radius
        self.trained = False

    def learn(self, trajectory_patches, trajectory_actions):
        """Learn patch -> action mappings from miracle."""
        for patch, action in zip(trajectory_patches, trajectory_actions):
            key = hash(patch)
            if key not in self.table:
                self.table[key] = np.zeros(self.n_actions, dtype=np.float32)
            self.table[key][action] += 1.0
        self.trained = True

    def predict(self, action_history=None, state=None, patch=None):
        if patch is not None:
            key = hash(patch)
            if key in self.table and self.table[key].sum() > 0:
                return int(np.argmax(self.table[key]))
        return None


class HybridExIt:
    """Combine Patch-Hash + Action-NGram."""
    name = "Hybrid"

    def __init__(self, n_actions=4):
        self.patch = PatchHashExIt(n_actions)
        self.ngram = ActionNGramExIt(n_actions)

    def learn(self, trajectory_patches, trajectory_actions):
        self.patch.learn(trajectory_patches, trajectory_actions)
        self.ngram.learn(trajectory_actions)

    @property
    def trained(self):
        return self.patch.trained or self.ngram.trained

    def predict(self, action_history, state=None, patch=None):
        # Try patch first (spatial), then N-gram (temporal)
        r = self.patch.predict(patch=patch)
        if r is not None:
            return r
        return self.ngram.predict(action_history)


# ==============================================================
# Simulation
# ==============================================================
def simulate(agent_class, n_games=200, max_actions=2000, n_levels=5, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    total_solved = 0
    total_levels = 0

    for game_idx in range(n_games):
        game = SpatialPuzzle(n_levels=n_levels, seed=seed + game_idx * 100)
        agent = agent_class(n_actions=game.N_ACTIONS)
        action_history = []
        patch_history = []
        miracle_patches = []
        miracle_actions = []

        for step in range(max_actions):
            state = game.get_state()
            patch = game.get_local_patch(radius=1)
            patch_history.append(patch)

            # Try learned model
            pred = None
            if agent.trained:
                if hasattr(agent, 'patch'):
                    pred = agent.predict(action_history, patch=patch)
                else:
                    pred = agent.predict(action_history)

            if pred is not None:
                action = pred
            else:
                # Random exploration with XOR-hash novelty
                state_hash = hash(state.tobytes()) % (2**31)
                scores = []
                for a in range(game.N_ACTIONS):
                    sa_hash = state_hash ^ (a * 2654435761)
                    scores.append(random.gauss(0, 1))
                action = max(range(game.N_ACTIONS), key=lambda i: scores[i])

            action_history.append(action)
            _, all_solved, level_up = game.step(action)

            if level_up:
                # Record miracle
                miracle_patches = patch_history.copy()
                miracle_actions = action_history.copy()

                if isinstance(agent, (PatchHashExIt, HybridExIt)):
                    agent.learn(miracle_patches, miracle_actions)
                elif isinstance(agent, ActionNGramExIt):
                    agent.learn(miracle_actions)

                action_history = []
                patch_history = []

            if all_solved:
                total_solved += 1
                break

        total_levels += game.solved_levels

    return {
        'agent': agent_class.__name__ if hasattr(agent_class, '__name__') else str(agent_class),
        'total_solved': total_solved,
        'solve_rate': total_solved / n_games,
        'avg_levels': total_levels / n_games,
    }


def main():
    print("=" * 60)
    print("Phase 50: Relative-Patch Hash ExIt")
    print("  Can local patches replace CNNs for spatial generalization?")
    print("=" * 60)

    agents = [ActionNGramExIt, PatchHashExIt, HybridExIt]
    budgets = [500, 1000, 2000, 5000]

    all_results = {}

    for max_actions in budgets:
        print(f"\n--- Budget: {max_actions} actions ---")
        print(f"  {'Agent':>15s} | {'Solved':>8s} {'Rate':>8s} {'AvgLvl':>8s}")
        print("  " + "-" * 45)

        budget_results = {}
        for agent_cls in agents:
            name = agent_cls.__name__
            if name == 'ActionNGramExIt':
                display = 'Action-NGram'
            elif name == 'PatchHashExIt':
                display = 'Patch-Hash'
            else:
                display = 'Hybrid'

            r = simulate(agent_cls, n_games=200, max_actions=max_actions, seed=42)
            print(f"  {display:>15s} | {r['total_solved']:>6d}/200 "
                  f"{r['solve_rate']*100:>6.1f}% {r['avg_levels']:>6.2f}")
            budget_results[display] = r

        all_results[f"budget_{max_actions}"] = budget_results

    # Summary
    print(f"\n{'='*60}")
    print("WINNER AT EACH BUDGET:")
    for b in budgets:
        key = f"budget_{b}"
        best = max(all_results[key].values(), key=lambda x: x['solve_rate'])
        print(f"  Budget {b:>5d}: {best['agent']:>20s} "
              f"({best['solve_rate']*100:.1f}%, {best['avg_levels']:.2f} levels)")

    save_path = os.path.join(RESULTS_DIR, "phase50_patch_hash_exit.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 50: Relative-Patch Hash ExIt',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
