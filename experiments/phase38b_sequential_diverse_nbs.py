"""
Phase 38b: Sequential Sigma-Diverse NBS (Kaggle Simulation)
Simulates Kaggle's sequential RESET mechanism: each attempt
uses a different sigma from a schedule, rotating through
diverse noise levels across retries.

Author: Hiroto Funasaki
Direct application: v9/v10 Kaggle agent choose_action logic.
"""
import os, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from collections import Counter
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"

# Sigma rotation schedule for sequential attempts
SIGMA_SCHEDULE = [0.0, 0.05, 0.15, 0.30, 0.50, 0.01, 0.10, 0.20, 0.40, 0.75]


class ReasoningModel(nn.Module):
    def __init__(self, in_dim=16, hidden=32, n_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)
        self.hidden = hidden

    def forward(self, x, noise=None):
        h = F.relu(self.fc1(x))
        if noise is not None:
            h = h + noise
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class SimulatedGame:
    """Simulates an ARC-like game with actions and win/lose states.

    The game has a hidden "correct action sequence" of length L.
    Agent must guess actions sequentially. On wrong action -> GAME_OVER.
    On correct action -> advance. Complete all L -> WIN.
    """
    def __init__(self, difficulty, rng):
        self.difficulty = difficulty
        # Correct action sequence (harder = longer)
        seq_len = {1: 2, 2: 3, 3: 5, 4: 7}[difficulty]
        self.correct_seq = [rng.randint(0, 4) for _ in range(seq_len)]
        self.progress = 0

    def step(self, action):
        """Returns: 'CORRECT', 'WIN', or 'GAME_OVER'"""
        if action == self.correct_seq[self.progress]:
            self.progress += 1
            if self.progress >= len(self.correct_seq):
                return 'WIN'
            return 'CORRECT'
        else:
            return 'GAME_OVER'

    def reset(self):
        self.progress = 0


def simulate_agent_fixed_sigma(model, game, sigma, max_actions=80, hidden=32):
    """Simulate agent with fixed sigma across all attempts."""
    game.reset()
    actions_used = 0
    n_attempts = 0

    while actions_used < max_actions:
        game.reset()
        n_attempts += 1

        # Play one episode
        for step in range(len(game.correct_seq) + 5):
            if actions_used >= max_actions:
                return False, actions_used, n_attempts

            # Generate action using model + noise
            x = torch.randn(1, 16)  # state embedding (simplified)
            noise = torch.randn(1, hidden) * sigma
            with torch.no_grad():
                logits = model(x, noise=noise)
                action = logits.argmax(1).item()

            actions_used += 1
            result = game.step(action)

            if result == 'WIN':
                return True, actions_used, n_attempts
            elif result == 'GAME_OVER':
                actions_used += 1  # RESET costs 1 action
                break

    return False, actions_used, n_attempts


def simulate_agent_diverse_sigma(model, game, sigma_schedule, max_actions=80, hidden=32):
    """Simulate agent with rotating sigma per attempt."""
    game.reset()
    actions_used = 0
    n_attempts = 0

    while actions_used < max_actions:
        game.reset()
        # Pick sigma from schedule based on attempt number
        sigma = sigma_schedule[n_attempts % len(sigma_schedule)]
        n_attempts += 1

        for step in range(len(game.correct_seq) + 5):
            if actions_used >= max_actions:
                return False, actions_used, n_attempts

            x = torch.randn(1, 16)
            noise = torch.randn(1, hidden) * sigma
            with torch.no_grad():
                logits = model(x, noise=noise)
                action = logits.argmax(1).item()

            actions_used += 1
            result = game.step(action)

            if result == 'WIN':
                return True, actions_used, n_attempts
            elif result == 'GAME_OVER':
                actions_used += 1
                break

    return False, actions_used, n_attempts


def main():
    print("=" * 60)
    print("Phase 38b: Sequential Sigma-Diverse NBS (Kaggle Sim)")
    print("=" * 60)

    torch.manual_seed(42)

    # Create a simple model (untrained - mimics blank-slate Kaggle agent)
    model = ReasoningModel(in_dim=16, hidden=32, n_classes=4)

    n_games = 200
    results = {}

    for difficulty in [1, 2, 3, 4]:
        print(f"\n--- Difficulty {difficulty} (seq_len={[2,3,5,7][difficulty-1]}) ---")

        # Fixed sigma tests
        fixed_results = {}
        for sigma in [0.0, 0.05, 0.10, 0.15, 0.30, 0.50, 1.0]:
            wins = 0
            total_actions = 0
            for g in range(n_games):
                rng = np.random.RandomState(g + difficulty * 1000)
                game = SimulatedGame(difficulty, rng)
                won, acts, _ = simulate_agent_fixed_sigma(model, game, sigma)
                if won:
                    wins += 1
                total_actions += acts
            rate = wins / n_games
            fixed_results[sigma] = rate

        best_fixed_sigma = max(fixed_results, key=fixed_results.get)
        best_fixed_rate = fixed_results[best_fixed_sigma]

        # Diverse sigma (rotating)
        wins_diverse = 0
        total_attempts_diverse = 0
        for g in range(n_games):
            rng = np.random.RandomState(g + difficulty * 1000)
            game = SimulatedGame(difficulty, rng)
            won, acts, attempts = simulate_agent_diverse_sigma(
                model, game, SIGMA_SCHEDULE
            )
            if won:
                wins_diverse += 1
            total_attempts_diverse += attempts
        diverse_rate = wins_diverse / n_games

        print(f"  Fixed sigma results:")
        for sigma, rate in sorted(fixed_results.items()):
            marker = " <-- best" if sigma == best_fixed_sigma else ""
            print(f"    s={sigma:.2f}: {rate:.3f}{marker}")
        print(f"  Diverse sigma: {diverse_rate:.3f}")
        print(f"  Best fixed:    {best_fixed_rate:.3f} (s={best_fixed_sigma})")

        gap = diverse_rate - best_fixed_rate
        if abs(gap) < 0.02:
            verdict = "EQUIVALENT"
        elif gap > 0:
            verdict = "DIVERSE WINS"
        else:
            verdict = "Fixed wins"
        print(f"  Gap: {gap:+.3f} -> {verdict}")

        results[f'difficulty_{difficulty}'] = {
            'fixed_results': {str(k): v for k, v in fixed_results.items()},
            'best_fixed_sigma': best_fixed_sigma,
            'best_fixed_rate': best_fixed_rate,
            'diverse_rate': diverse_rate,
            'gap': gap,
            'verdict': verdict,
        }

    # Global summary
    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY: Sequential Sigma-Diverse NBS")
    print("=" * 60)
    print(f"\n{'Difficulty':>12s} {'Best Fixed':>12s} {'Diverse':>10s} {'Gap':>8s} {'Verdict'}")
    print("-" * 60)

    n_ok = 0
    for key, r in results.items():
        d = key.split('_')[1]
        print(f"{'D'+d:>12s} {r['best_fixed_rate']:>12.3f} {r['diverse_rate']:>10.3f} "
              f"{r['gap']:>+8.3f} {r['verdict']}")
        if r['verdict'] in ['EQUIVALENT', 'DIVERSE WINS']:
            n_ok += 1

    print(f"\nDiverse matches or beats fixed: {n_ok}/{len(results)}")

    if n_ok >= len(results) - 1:
        print("\nCONCLUSION: Sigma rotation in sequential retries is VIABLE for Kaggle!")
        print("-> Integrate into v9 agent: attempt_sigma = SIGMA_SCHEDULE[attempt % len(SIGMA_SCHEDULE)]")
    else:
        print("\nCONCLUSION: Mixed results. Need more investigation.")

    save_path = os.path.join(RESULTS_DIR, "phase38b_sequential_diverse_nbs.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 38b: Sequential Sigma-Diverse NBS',
            'timestamp': datetime.now().isoformat(),
            'sigma_schedule': SIGMA_SCHEDULE,
            'results': results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
