"""
Phase 54: Temporal SimHash (Short-Term Memory via Bit Compression)
Emulate synaptic fading memory with O(1) bit operations:
  Temporal_Hash_t = SimHash(state_t) ^ (Temporal_Hash_{t-1} >> 1)

Past information gradually shifts out (decays), creating a
temporal context window without any RNN or matrix operations.

Author: Hiroto Funasaki
"""
import os, json, random, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


class TemporalGame:
    """Game with rules that depend on action history."""
    N_ACTIONS = 4
    def __init__(self, rule_type=0, seed=None):
        self.rng = random.Random(seed)
        self.rule_type = rule_type
        self.history = []
        self.steps = 0
        self.max_steps = 30
        self.grid = np.random.randn(36).astype(np.float32) * 0.01

    def get_optimal_action(self):
        if self.rule_type == 0:  # repeat-2-ago
            return self.history[-2] % self.N_ACTIONS if len(self.history) >= 2 else 0
        elif self.rule_type == 1:  # alternate
            return len(self.history) % 2
        elif self.rule_type == 2:  # cycle-4
            return len(self.history) % self.N_ACTIONS
        else:  # opposite-last
            return (self.history[-1] + 2) % self.N_ACTIONS if self.history else 0

    def get_state(self):
        # Encode step and last few actions into grid
        state = self.grid.copy()
        if self.history:
            state[0] = self.history[-1] / self.N_ACTIONS
        if len(self.history) >= 2:
            state[1] = self.history[-2] / self.N_ACTIONS
        state[2] = self.steps / self.max_steps
        return state


# ==============================================================
# Curiosity Agents with Temporal Context
# ==============================================================
class MemorylessAgent:
    """No memory at all: pure state hash."""
    name = "Memoryless"
    def __init__(self, state_dim=36, hash_bits=32):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32) * 0.1
        self.seen = set()

    def choose(self, state, n_actions, step):
        sh = hash(tuple((state @ self.proj > 0).astype(np.int8)))
        scores = []
        for a in range(n_actions):
            sa = sh ^ (a * 2654435761)
            novelty = 0 if sa in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(sh ^ (best * 2654435761))
        return best


class TemporalSimHashAgent:
    """NEW: Temporal SimHash with decaying memory.
    temporal_hash = simhash(state) ^ (prev_temporal_hash >> 1)
    """
    name = "Temporal-SimHash"

    def __init__(self, state_dim=36, hash_bits=32):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32) * 0.1
        self.hash_bits = hash_bits
        self.temporal_hash = 0
        self.seen = set()

    def _simhash_int(self, state):
        projected = state @ self.proj
        result = 0
        for i in range(self.hash_bits):
            if projected[i] > 0:
                result |= (1 << i)
        return result

    def choose(self, state, n_actions, step):
        # Current state hash
        current_sh = self._simhash_int(state)
        # Temporal: fold in decayed past
        self.temporal_hash = current_sh ^ (self.temporal_hash >> 1)

        scores = []
        for a in range(n_actions):
            sa = self.temporal_hash ^ (a * 2654435761)
            novelty = 0 if sa in self.seen else 1
            scores.append(novelty + random.gauss(0, 0.3))
        best = max(range(n_actions), key=lambda i: scores[i])
        self.seen.add(self.temporal_hash ^ (best * 2654435761))
        return best


class MultiShiftTemporalAgent:
    """Multiple shift amounts = multiple temporal scales."""
    name = "MultiShift-Temporal"

    def __init__(self, state_dim=36, hash_bits=32):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32) * 0.1
        self.hash_bits = hash_bits
        self.temporal_hashes = {1: 0, 2: 0, 4: 0, 8: 0}  # multiple decay rates
        self.seen = {s: set() for s in self.temporal_hashes}

    def _simhash_int(self, state):
        projected = state @ self.proj
        result = 0
        for i in range(self.hash_bits):
            if projected[i] > 0:
                result |= (1 << i)
        return result

    def choose(self, state, n_actions, step):
        current_sh = self._simhash_int(state)

        # Update all temporal scales
        for shift in self.temporal_hashes:
            self.temporal_hashes[shift] = current_sh ^ (self.temporal_hashes[shift] >> shift)

        # Score actions across all temporal scales
        scores = []
        for a in range(n_actions):
            novelty = 0.0
            for shift, th in self.temporal_hashes.items():
                sa = th ^ (a * 2654435761)
                if sa not in self.seen[shift]:
                    novelty += 1.0 / len(self.temporal_hashes)
            scores.append(novelty + random.gauss(0, 0.2))

        best = max(range(n_actions), key=lambda i: scores[i])
        for shift, th in self.temporal_hashes.items():
            self.seen[shift].add(th ^ (best * 2654435761))
        return best


# ==============================================================
# N-Gram ExIt Oracle (using temporal hash as key)
# ==============================================================
class TemporalSimHashExIt:
    """Temporal SimHash as dictionary key for ExIt."""
    name = "Temporal-SimHash-ExIt"

    def __init__(self, state_dim=36, hash_bits=32, n_actions=4):
        np.random.seed(42)
        self.proj = np.random.randn(state_dim, hash_bits).astype(np.float32) * 0.1
        self.hash_bits = hash_bits
        self.n_actions = n_actions
        self.temporal_hash = 0
        self.table = {}
        self.trained = False

    def _simhash_int(self, state):
        projected = state @ self.proj
        result = 0
        for i in range(self.hash_bits):
            if projected[i] > 0:
                result |= (1 << i)
        return result

    def learn(self, states, actions):
        th = 0
        for s, a in zip(states, actions):
            sh = self._simhash_int(s)
            th = sh ^ (th >> 1)
            if th not in self.table:
                self.table[th] = np.zeros(self.n_actions, dtype=np.float32)
            self.table[th][a] += 1.0
        self.trained = True

    def choose(self, state, n_actions, step):
        sh = self._simhash_int(state)
        self.temporal_hash = sh ^ (self.temporal_hash >> 1)

        if self.trained and self.temporal_hash in self.table:
            counts = self.table[self.temporal_hash]
            if counts.sum() > 0:
                return int(np.argmax(counts))

        scores = [random.gauss(0, 1) for _ in range(n_actions)]
        return max(range(n_actions), key=lambda i: scores[i])


# ==============================================================
# Simulation
# ==============================================================
def evaluate_temporal(agent_class, rule_type, n_episodes=200,
                      max_steps=30, n_miracles=5, seed=42):
    """Evaluate temporal agent on rule-based game.
    First generate miracles, then test."""
    random.seed(seed)
    np.random.seed(seed)

    state_dim = 36
    agent = agent_class(state_dim)

    # Generate and teach miracles (for ExIt agents)
    if hasattr(agent, 'learn'):
        for m in range(n_miracles):
            game = TemporalGame(rule_type=rule_type, seed=seed + m * 1000)
            states, actions = [], []
            for step in range(max_steps):
                state = game.get_state()
                optimal = game.get_optimal_action()
                states.append(state.copy())
                actions.append(optimal)
                game.history.append(optimal)
                game.steps += 1
            agent.learn(states, actions)

    # Evaluate
    correct = 0
    total = 0
    for ep in range(n_episodes):
        game = TemporalGame(rule_type=rule_type, seed=seed + 10000 + ep * 100)
        if hasattr(agent, 'temporal_hash'):
            agent.temporal_hash = 0  # reset per episode

        for step in range(max_steps):
            state = game.get_state()
            optimal = game.get_optimal_action()
            chosen = agent.choose(state, game.N_ACTIONS, step)

            if chosen == optimal:
                correct += 1
            total += 1

            game.history.append(optimal)  # assume correct action taken
            game.steps += 1

    return correct / total if total > 0 else 0


def main():
    print("=" * 60)
    print("Phase 54: Temporal SimHash (Short-Term Memory via Bits)")
    print("  Can bit-shift XOR replace RNN for temporal tasks?")
    print("=" * 60)

    rule_names = ["Repeat-2-ago", "Alternate", "Cycle-4", "Opposite-last"]
    agents = [MemorylessAgent, TemporalSimHashAgent,
              MultiShiftTemporalAgent, TemporalSimHashExIt]

    all_results = {}

    for rule_type in range(4):
        print(f"\n--- Rule: {rule_names[rule_type]} ---")
        print(f"  {'Agent':>25s} | {'Accuracy':>10s}")
        print("  " + "-" * 40)

        rule_results = {}
        for agent_cls in agents:
            acc = evaluate_temporal(agent_cls, rule_type, n_episodes=200, seed=42)
            print(f"  {agent_cls.name:>25s} | {acc*100:>8.1f}%")
            rule_results[agent_cls.name] = {'accuracy': acc}

        all_results[rule_names[rule_type]] = rule_results

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Accuracy (%) by Agent x Rule")
    print(f"{'Agent':>25s}", end="")
    for r in rule_names:
        print(f" {r:>15s}", end="")
    print()
    print("-" * (25 + 16 * len(rule_names)))
    for agent_cls in agents:
        print(f"{agent_cls.name:>25s}", end="")
        for r in rule_names:
            acc = all_results[r][agent_cls.name]['accuracy'] * 100
            print(f" {acc:>13.1f}%", end="")
        print()

    # Compare with Phase 40-DT and Phase 48
    print(f"\n{'='*60}")
    print("COMPARISON with Prior Phases:")
    print(f"{'Method':>25s}  Repeat  Altern  Cycle4  Oppost")
    print(f"{'CNN (Phase 40-DT)':>25s}   100.0   50.8   27.5   49.8")
    print(f"{'Transformer (40-DT)':>25s}   100.0  100.0  100.0  100.0")
    print(f"{'N-Gram ExIt (Phase 48)':>25s}   100.0  100.0  100.0  100.0")
    exit_name = TemporalSimHashExIt.name
    print(f"{'Temporal-SimHash-ExIt':>25s}", end="")
    for r in rule_names:
        acc = all_results[r][exit_name]['accuracy'] * 100
        print(f"  {acc:>5.1f}", end="")
    print()

    save_path = os.path.join(RESULTS_DIR, "phase54_temporal_simhash.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 54: Temporal SimHash',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
