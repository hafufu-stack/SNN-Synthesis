"""
Phase 37b: Multi-Game Knowledge Multiplexing
Test if a tiny CNN (MicroBrain ~6K params) can learn two completely
different game strategies simultaneously without catastrophic forgetting,
using condition_id gating (proven in Phase 35c).

Author: Hiroto Funasaki
Theory: Phase 35c gating mechanism applied to ARC-like game strategies.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"

N_COLORS = 10
GRID_SIZE = 8
N_ACTIONS = 7


# ==============================================================
# MicroBrain with Game-ID Gating
# ==============================================================
class MicroBrainMultiGame(nn.Module):
    """Tiny CNN with game-ID gating for multi-game knowledge.

    Architecture matches Kaggle's MicroBrain but adds
    condition_embed for game-specific routing.
    ~8K params total (6K base + 2K gating).
    """
    def __init__(self, n_colors=N_COLORS, n_actions=N_ACTIONS,
                 n_games=2, hidden=32):
        super().__init__()
        # Game-ID gating (Phase 35c mechanism)
        self.game_embed = nn.Embedding(n_games, hidden)

        # CNN encoder
        self.conv1 = nn.Conv2d(n_colors, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Policy head
        self.fc1 = nn.Linear(16, hidden)
        self.fc_action = nn.Linear(hidden, n_actions)

    def forward(self, grid_tensor, game_id=None):
        # CNN encode
        x = F.relu(self.conv1(grid_tensor))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)  # (B, 16)

        h = F.relu(self.fc1(x))  # (B, hidden)

        # Apply game-specific gating
        if game_id is not None:
            gate = torch.sigmoid(self.game_embed(game_id))  # (B, hidden)
            h = h * gate  # multiplicative modulation

        return self.fc_action(h)  # (B, n_actions)


class MicroBrainNoGating(nn.Module):
    """Control: same architecture WITHOUT gating."""
    def __init__(self, n_colors=N_COLORS, n_actions=N_ACTIONS, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv2d(n_colors, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(16, hidden)
        self.fc_action = nn.Linear(hidden, n_actions)

    def forward(self, grid_tensor, game_id=None):
        x = F.relu(self.conv1(grid_tensor))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        h = F.relu(self.fc1(x))
        return self.fc_action(h)


# ==============================================================
# Synthetic "Game" Miracle Trajectories
# ==============================================================
def generate_game_data(game_type, n_samples=300, seed=42):
    """Generate synthetic miracle trajectories for different game types.

    Game A (maze-like): Correct action depends on where non-zero cells are
    Game B (sorting-like): Correct action depends on color distribution
    """
    rng = np.random.RandomState(seed)
    grids = []
    actions = []

    for _ in range(n_samples):
        grid = np.zeros((N_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        if game_type == "A":
            # Game A: Maze navigation
            # Place a "player" (color 1) and "goal" (color 2)
            py, px = rng.randint(0, GRID_SIZE, 2)
            gy, gx = rng.randint(0, GRID_SIZE, 2)
            grid[1, py, px] = 1.0  # player
            grid[2, gy, gx] = 1.0  # goal

            # Add some walls (color 3)
            n_walls = rng.randint(3, 8)
            for _ in range(n_walls):
                wy, wx = rng.randint(0, GRID_SIZE, 2)
                grid[3, wy, wx] = 1.0

            # Correct action: move toward goal
            dx = gx - px
            dy = gy - py
            if abs(dx) > abs(dy):
                action = 0 if dx > 0 else 1  # right / left
            else:
                action = 2 if dy > 0 else 3  # down / up

        elif game_type == "B":
            # Game B: Color sorting
            # Fill grid with random colors
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    color = rng.randint(0, 5)
                    grid[color, i, j] = 1.0

            # Correct action: based on dominant color
            color_counts = grid[:5].sum(axis=(1, 2))
            dominant = np.argmax(color_counts)
            action = dominant % N_ACTIONS

        grids.append(grid)
        actions.append(action)

    return (
        torch.tensor(np.array(grids)),
        torch.tensor(np.array(actions), dtype=torch.long)
    )


# ==============================================================
# Training
# ==============================================================
def train_multi_game(model, X_a, y_a, X_b, y_b, game_ids_a, game_ids_b,
                     use_gating=True, epochs=100, lr=0.01):
    """Train on mixed data from two games."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    # Combine data
    X = torch.cat([X_a, X_b])
    y = torch.cat([y_a, y_b])
    game_ids = torch.cat([game_ids_a, game_ids_b])

    for epoch in range(epochs):
        model.train()
        if use_gating:
            logits = model(X, game_id=game_ids)
        else:
            logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if use_gating:
                pred = model(X, game_id=game_ids).argmax(1)
            else:
                pred = model(X).argmax(1)

            mask_a = game_ids == 0
            mask_b = game_ids == 1
            acc_a = (pred[mask_a] == y[mask_a]).float().mean().item()
            acc_b = (pred[mask_b] == y[mask_b]).float().mean().item()

            # Cross-game test
            if use_gating:
                wrong_ids = 1 - game_ids  # swap 0↔1
                pred_wrong = model(X, game_id=wrong_ids).argmax(1)
                acc_a_wrong = (pred_wrong[mask_a] == y[mask_a]).float().mean().item()
                acc_b_wrong = (pred_wrong[mask_b] == y[mask_b]).float().mean().item()
            else:
                acc_a_wrong = acc_a  # no gating = same result
                acc_b_wrong = acc_b

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}  "
                  f"A={acc_a:.3f} B={acc_b:.3f}  "
                  f"A(wrong)={acc_a_wrong:.3f} B(wrong)={acc_b_wrong:.3f}")

        history.append({
            'epoch': epoch,
            'loss': loss.item(),
            'game_a_correct': acc_a,
            'game_b_correct': acc_b,
            'game_a_wrong': acc_a_wrong,
            'game_b_wrong': acc_b_wrong,
        })

    return history


def main():
    print("=" * 60)
    print("Phase 37b: Multi-Game Knowledge Multiplexing")
    print("Can a 8K-param CNN hold 2 game strategies simultaneously?")
    print("=" * 60)

    torch.manual_seed(42)

    # Generate game data
    print("\nGenerating game data...")
    X_a, y_a = generate_game_data("A", n_samples=400, seed=42)
    X_b, y_b = generate_game_data("B", n_samples=400, seed=123)

    game_ids_a = torch.zeros(len(X_a), dtype=torch.long)
    game_ids_b = torch.ones(len(X_b), dtype=torch.long)

    print(f"Game A: {X_a.shape}, actions: {y_a.unique().tolist()}")
    print(f"Game B: {X_b.shape}, actions: {y_b.unique().tolist()}")

    # Count params
    model_test = MicroBrainMultiGame()
    n_params = sum(p.numel() for p in model_test.parameters())
    print(f"Model params (with gating): {n_params:,}")

    # ---- Exp 1: Without gating (catastrophic forgetting?) ----
    print("\n--- Exp 1: No Gating (expect interference) ---")
    model_no_gate = MicroBrainNoGating()
    hist_no_gate = train_multi_game(
        model_no_gate, X_a, y_a, X_b, y_b,
        game_ids_a, game_ids_b,
        use_gating=False, epochs=100
    )

    # ---- Exp 2: With gating (knowledge separation) ----
    print("\n--- Exp 2: Game-ID Gating (expect separation) ---")
    model_gated = MicroBrainMultiGame()
    hist_gated = train_multi_game(
        model_gated, X_a, y_a, X_b, y_b,
        game_ids_a, game_ids_b,
        use_gating=True, epochs=100
    )

    # ---- Exp 3: Single-game baseline ----
    print("\n--- Exp 3: Single-game baselines ---")
    model_a_only = MicroBrainNoGating()
    opt_a = optim.Adam(model_a_only.parameters(), lr=0.01)
    for epoch in range(100):
        model_a_only.train()
        loss = F.cross_entropy(model_a_only(X_a), y_a)
        opt_a.zero_grad()
        loss.backward()
        opt_a.step()
    model_a_only.eval()
    with torch.no_grad():
        acc_a_only = (model_a_only(X_a).argmax(1) == y_a).float().mean().item()

    model_b_only = MicroBrainNoGating()
    opt_b = optim.Adam(model_b_only.parameters(), lr=0.01)
    for epoch in range(100):
        model_b_only.train()
        loss = F.cross_entropy(model_b_only(X_b), y_b)
        opt_b.zero_grad()
        loss.backward()
        opt_b.step()
    model_b_only.eval()
    with torch.no_grad():
        acc_b_only = (model_b_only(X_b).argmax(1) == y_b).float().mean().item()
    print(f"  Game A only: {acc_a_only:.3f}")
    print(f"  Game B only: {acc_b_only:.3f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    final_ng = hist_no_gate[-1]
    final_g = hist_gated[-1]

    print(f"\n{'Method':<25s} {'Game A':>8s} {'Game B':>8s} {'Avg':>8s}")
    print("-" * 55)
    print(f"{'Single-game baseline':<25s} {acc_a_only:>8.3f} {acc_b_only:>8.3f} "
          f"{(acc_a_only + acc_b_only)/2:>8.3f}")
    print(f"{'Mixed (no gating)':<25s} {final_ng['game_a_correct']:>8.3f} "
          f"{final_ng['game_b_correct']:>8.3f} "
          f"{(final_ng['game_a_correct'] + final_ng['game_b_correct'])/2:>8.3f}")
    print(f"{'Mixed (ID gating)':<25s} {final_g['game_a_correct']:>8.3f} "
          f"{final_g['game_b_correct']:>8.3f} "
          f"{(final_g['game_a_correct'] + final_g['game_b_correct'])/2:>8.3f}")

    # Catastrophic forgetting score
    cf_no_gate = (acc_a_only - final_ng['game_a_correct'] +
                  acc_b_only - final_ng['game_b_correct']) / 2
    cf_gated = (acc_a_only - final_g['game_a_correct'] +
                acc_b_only - final_g['game_b_correct']) / 2

    print(f"\nCatastrophic Forgetting (lower=better):")
    print(f"  No gating:  {cf_no_gate:+.3f}")
    print(f"  ID gating:  {cf_gated:+.3f}")

    # Knowledge separation
    sep = ((final_g['game_a_correct'] + final_g['game_b_correct']) / 2 -
           (final_g['game_a_wrong'] + final_g['game_b_wrong']) / 2)
    print(f"\nKnowledge Separation (gated): {sep:.3f}")

    # Save
    save_path = os.path.join(RESULTS_DIR, "phase37b_multi_game_multiplex.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 37b: Multi-Game Knowledge Multiplexing',
            'timestamp': datetime.now().isoformat(),
            'single_game': {'A': acc_a_only, 'B': acc_b_only},
            'no_gating_final': final_ng,
            'gated_final': final_g,
            'catastrophic_forgetting_no_gate': cf_no_gate,
            'catastrophic_forgetting_gated': cf_gated,
            'knowledge_separation': sep,
            'model_params': n_params,
        }, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
