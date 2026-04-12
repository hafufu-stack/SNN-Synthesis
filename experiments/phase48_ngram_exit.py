"""
Phase 48: Non-Parametric Zero-Overhead ExIt
N-gram dictionary: learn from miracles WITHOUT backprop.
O(1) learning time, O(1) inference time.

Key idea: Store "action_history_hash -> next_action" transitions
from miracle trajectories. No neural network needed.

Author: Hiroto Funasaki
"""
import os, json, random, numpy as np
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Temporal Game (from Phase 40-DT)
# ==============================================================
class TemporalGame:
    GRID_SIZE = 6
    N_ACTIONS = 4

    def __init__(self, rule_type=0, seed=None):
        self.rng = random.Random(seed)
        self.rule_type = rule_type
        self.reset()

    def reset(self):
        self.steps = 0
        self.max_steps = 30
        self.history = []
        return None

    def get_optimal_action(self):
        if self.rule_type == 0:
            return self.history[-2] % self.N_ACTIONS if len(self.history) >= 2 else 0
        elif self.rule_type == 1:
            return len(self.history) % 2
        elif self.rule_type == 2:
            return len(self.history) % self.N_ACTIONS
        else:
            return (self.history[-1] + 2) % self.N_ACTIONS if self.history else 0

    def step(self, action):
        self.steps += 1
        optimal = self.get_optimal_action()
        correct = (action == optimal)
        self.history.append(action)
        done = self.steps >= self.max_steps
        return correct, done


# ==============================================================
# N-Gram Dictionary ExIt
# ==============================================================
class NGramExIt:
    """Non-parametric ExIt: learns from miracle trajectories using N-gram hashing.
    Learning cost: O(L) where L = trajectory length
    Inference cost: O(1)
    Memory: O(unique_ngrams)
    """
    def __init__(self, n_values=[1, 2, 3, 4, 5], n_actions=4):
        self.n_values = n_values
        self.n_actions = n_actions
        # Dictionary: hash(last_n_actions) -> action_counts[n_actions]
        self.tables = {n: {} for n in n_values}
        self.n_miracles = 0

    def learn_from_miracle(self, trajectory):
        """Learn transitions from a miracle trajectory in O(len(trajectory))."""
        self.n_miracles += 1
        for n in self.n_values:
            for i in range(len(trajectory)):
                if i < n:
                    # Use padding for start of sequence
                    context = tuple([-1] * (n - i) + list(trajectory[:i]))
                else:
                    context = tuple(trajectory[i-n:i])

                key = hash(context)
                if key not in self.tables[n]:
                    self.tables[n][key] = np.zeros(self.n_actions, dtype=np.float32)
                self.tables[n][key][trajectory[i]] += 1.0

    def predict(self, action_history):
        """Predict next action using longest matching N-gram. O(1)."""
        # Try longest N-gram first (most specific)
        for n in reversed(self.n_values):
            if len(action_history) < n:
                context = tuple([-1] * (n - len(action_history)) + list(action_history[-n:]))
            else:
                context = tuple(action_history[-n:])

            key = hash(context)
            if key in self.tables[n]:
                counts = self.tables[n][key]
                if counts.sum() > 0:
                    # Return action with highest count
                    return int(np.argmax(counts)), n
        return None, 0

    def choose_action(self, action_history, n_actions):
        """Choose action: use N-gram if available, else random."""
        pred, n_used = self.predict(action_history)
        if pred is not None:
            return pred, 'ngram', n_used
        return random.randint(0, n_actions - 1), 'random', 0


# ==============================================================
# Simulation
# ==============================================================
def generate_miracle(rule_type, seed=42):
    """Generate a perfect trajectory (all correct actions)."""
    env = TemporalGame(rule_type=rule_type, seed=seed)
    env.reset()
    trajectory = []
    for _ in range(env.max_steps):
        optimal = env.get_optimal_action()
        trajectory.append(optimal)
        env.history.append(optimal)
    return trajectory


def evaluate_ngram(exit_model, rule_type, n_episodes=200, seed=42):
    """Evaluate N-gram model on fresh episodes."""
    correct_total = 0
    total_steps = 0
    ngram_used = 0
    random_used = 0

    for ep in range(n_episodes):
        env = TemporalGame(rule_type=rule_type, seed=seed + ep * 100)
        env.reset()
        action_history = []

        for step in range(env.max_steps):
            action, source, n_used = exit_model.choose_action(
                action_history, env.N_ACTIONS)

            optimal = env.get_optimal_action()
            if action == optimal:
                correct_total += 1
            total_steps += 1

            if source == 'ngram':
                ngram_used += 1
            else:
                random_used += 1

            action_history.append(action)
            env.history.append(action)

    return {
        'accuracy': correct_total / total_steps if total_steps > 0 else 0,
        'ngram_usage': ngram_used / total_steps if total_steps > 0 else 0,
        'random_usage': random_used / total_steps if total_steps > 0 else 0,
    }


def main():
    print("=" * 60)
    print("Phase 48: Non-Parametric Zero-Overhead ExIt")
    print("  Can N-gram dictionaries replace neural network ExIt?")
    print("=" * 60)

    rule_names = ["Repeat-2-ago", "Alternate", "Cycle-4", "Opposite-last"]
    all_results = {}

    for rule_type in range(4):
        print(f"\n--- Rule: {rule_names[rule_type]} ---")
        rule_results = {}

        # Test with different numbers of miracle trajectories
        for n_miracles in [0, 1, 3, 5, 10, 20]:
            ngram = NGramExIt(n_values=[1, 2, 3, 4, 5], n_actions=4)

            # Learn from miracles
            for m in range(n_miracles):
                miracle = generate_miracle(rule_type, seed=42 + m * 100)
                ngram.learn_from_miracle(miracle)

            # Evaluate
            r = evaluate_ngram(ngram, rule_type, n_episodes=200, seed=42)
            table_size = sum(len(t) for t in ngram.tables.values())

            print(f"  Miracles={n_miracles:>3d}: acc={r['accuracy']*100:.1f}%  "
                  f"ngram_use={r['ngram_usage']*100:.0f}%  "
                  f"table_size={table_size}")

            rule_results[f"miracles_{n_miracles}"] = {
                'n_miracles': n_miracles,
                'accuracy': r['accuracy'],
                'ngram_usage': r['ngram_usage'],
                'table_size': table_size,
            }

        all_results[rule_names[rule_type]] = rule_results

    # Compare with neural network ExIt (Phase 43: +37pp, Phase 40-DT: 100%)
    print(f"\n{'='*60}")
    print("COMPARISON: N-Gram ExIt vs Neural Network ExIt")
    print(f"{'Rule':>18s} | {'NGram@20':>10s} {'CNN(40-DT)':>10s} {'TF(40-DT)':>10s}")
    print("-" * 55)

    for rule in rule_names:
        ngram_acc = all_results[rule]['miracles_20']['accuracy'] * 100
        # From Phase 40-DT results
        cnn_accs = {'Repeat-2-ago': 100, 'Alternate': 50.8, 'Cycle-4': 27.5, 'Opposite-last': 49.8}
        tf_accs = {'Repeat-2-ago': 100, 'Alternate': 100, 'Cycle-4': 100, 'Opposite-last': 100}
        print(f"{rule:>18s} | {ngram_acc:>8.1f}% {cnn_accs[rule]:>8.1f}% {tf_accs[rule]:>8.1f}%")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDINGS:")
    for rule in rule_names:
        r20 = all_results[rule]['miracles_20']
        r0 = all_results[rule]['miracles_0']
        gain = (r20['accuracy'] - r0['accuracy']) * 100
        print(f"  {rule}: {r0['accuracy']*100:.1f}% -> {r20['accuracy']*100:.1f}% "
              f"({gain:+.1f}pp) with ZERO computation overhead!")

    save_path = os.path.join(RESULTS_DIR, "phase48_ngram_exit.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 48: Non-Parametric Zero-Overhead ExIt',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
