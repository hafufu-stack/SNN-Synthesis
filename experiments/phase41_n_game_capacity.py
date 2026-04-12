"""
Phase 41: N-Game Capacity Limit
Measure how many tasks a single 115K gated CNN can multiplex
before knowledge separation degrades.

Extends Phase 38a (2 games) to 2, 5, 7, 10, 15, 20 games.
Measures: Knowledge Separation Score, per-game accuracy, capacity scaling law.

Author: Hiroto Funasaki
"""
import os, json, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"
N_COLORS, GRID_SIZE, N_ACTIONS = 10, 8, 7

# XLarge config from Phase 38a (115K params)
CH1, CH2, HIDDEN = 64, 128, 256


def generate_diverse_games(n_games, n_samples_per_game=400, seed=42):
    """Generate N diverse synthetic games with different optimal strategies.

    Each game has a unique state→action mapping:
    - Games 0-2: Position-based (move toward colored target)
    - Games 3-5: Pattern-based (count dominant patterns)
    - Games 6-9: Region-based (act on densest quadrant)
    - Games 10+: Hybrid (combination strategies)
    """
    rng = np.random.RandomState(seed)
    all_grids, all_actions, all_game_ids = [], [], []

    for game_id in range(n_games):
        grids, actions = [], []
        # Use different seed per game for variety
        game_rng = np.random.RandomState(seed + game_id * 1000)

        for _ in range(n_samples_per_game):
            grid = np.zeros((N_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

            if game_id % 5 == 0:
                # Type A: Navigate to target (direction-based)
                py, px = game_rng.randint(0, GRID_SIZE, 2)
                gy, gx = game_rng.randint(0, GRID_SIZE, 2)
                color_p = 1 + (game_id % 4)
                color_g = 2 + (game_id % 3)
                grid[color_p, py, px] = 1.0
                grid[color_g, gy, gx] = 1.0
                for _ in range(game_rng.randint(2, 6)):
                    wy, wx = game_rng.randint(0, GRID_SIZE, 2)
                    grid[3 + game_id % 4, wy, wx] = 1.0
                dx, dy = gx - px, gy - py
                action = (0 if dx > 0 else 1) if abs(dx) > abs(dy) else (2 if dy > 0 else 3)

            elif game_id % 5 == 1:
                # Type B: Dominant color (count-based)
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        c = game_rng.randint(0, min(5 + game_id // 5, N_COLORS))
                        grid[c, i, j] = 1.0
                counts = grid[:N_ACTIONS].sum(axis=(1, 2))
                action = int(np.argmax(counts))

            elif game_id % 5 == 2:
                # Type C: Quadrant density (region-based)
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        if game_rng.random() < 0.3 + 0.05 * game_id:
                            grid[game_rng.randint(0, N_COLORS), i, j] = 1.0
                half = GRID_SIZE // 2
                quads = [
                    grid[:, :half, :half].sum(),
                    grid[:, :half, half:].sum(),
                    grid[:, half:, :half].sum(),
                    grid[:, half:, half:].sum(),
                ]
                action = int(np.argmax(quads)) % N_ACTIONS

            elif game_id % 5 == 3:
                # Type D: Edge count (border-based)
                for _ in range(game_rng.randint(5, 15)):
                    y, x = game_rng.randint(0, GRID_SIZE, 2)
                    c = game_rng.randint(0, N_COLORS)
                    grid[c, y, x] = 1.0
                edge_counts = [
                    grid[:, 0, :].sum(),   # top
                    grid[:, -1, :].sum(),  # bottom
                    grid[:, :, 0].sum(),   # left
                    grid[:, :, -1].sum(),  # right
                ]
                action = int(np.argmax(edge_counts)) % N_ACTIONS

            else:
                # Type E: Diagonal pattern (geometry-based)
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        if game_rng.random() < 0.25:
                            grid[game_rng.randint(0, N_COLORS), i, j] = 1.0
                # Count along main diagonal vs anti-diagonal
                main_diag = sum(grid[:, i, i].sum() for i in range(GRID_SIZE))
                anti_diag = sum(grid[:, i, GRID_SIZE-1-i].sum() for i in range(GRID_SIZE))
                center = grid[:, 2:6, 2:6].sum()
                scores = [main_diag, anti_diag, center, grid.sum() - center]
                action = int(np.argmax(scores)) % N_ACTIONS

            grids.append(grid)
            actions.append(action)

        all_grids.append(torch.tensor(np.array(grids)))
        all_actions.append(torch.tensor(actions, dtype=torch.long))
        all_game_ids.append(torch.full((n_samples_per_game,), game_id, dtype=torch.long))

    return all_grids, all_actions, all_game_ids


class ScalableCNN(nn.Module):
    """CNN with configurable size + optional N-game ID gating."""
    def __init__(self, ch1=64, ch2=128, hidden=256, n_games=2, use_gating=True):
        super().__init__()
        self.use_gating = use_gating
        self.conv1 = nn.Conv2d(N_COLORS, ch1, 3, padding=1)
        self.conv2 = nn.Conv2d(ch1, ch2, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch2, hidden)
        self.fc_action = nn.Linear(hidden, N_ACTIONS)
        if use_gating:
            self.game_embed = nn.Embedding(n_games, hidden)

    def forward(self, x, game_id=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        h = F.relu(self.fc1(x))
        if self.use_gating and game_id is not None:
            h = h * torch.sigmoid(self.game_embed(game_id))
        return self.fc_action(h)


def run_n_game_experiment(n_games, all_grids, all_actions, all_game_ids, epochs=150):
    """Run gated vs ungated comparison with N games."""
    # Combine data
    X = torch.cat(all_grids[:n_games])
    y = torch.cat(all_actions[:n_games])
    gids = torch.cat(all_game_ids[:n_games])

    results = {}
    for use_gating in [False, True]:
        tag = "gated" if use_gating else "no_gate"
        model = ScalableCNN(CH1, CH2, HIDDEN, n_games=n_games, use_gating=use_gating)
        n_params = sum(p.numel() for p in model.parameters())
        opt = optim.Adam(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for epoch in range(epochs):
            model.train()
            # Mini-batch for larger datasets
            perm = torch.randperm(len(X))[:min(1024, len(X))]
            logits = model(X[perm], game_id=gids[perm]) if use_gating else model(X[perm])
            loss = F.cross_entropy(logits, y[perm])
            opt.zero_grad(); loss.backward(); opt.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_correct = (model(X, game_id=gids) if use_gating else model(X)).argmax(1)
            per_game_acc = {}
            per_game_acc_wrong = {}

            for gid in range(n_games):
                mask = gids == gid
                acc = (pred_correct[mask] == y[mask]).float().mean().item()
                per_game_acc[f"game_{gid}"] = acc

                if use_gating:
                    # Test with wrong game ID (random other)
                    wrong_id = (gid + 1) % n_games
                    wrong_gids = torch.full_like(gids[mask], wrong_id)
                    pred_wrong = model(X[mask], game_id=wrong_gids).argmax(1)
                    acc_wrong = (pred_wrong == y[mask]).float().mean().item()
                    per_game_acc_wrong[f"game_{gid}_wrong"] = acc_wrong
                else:
                    per_game_acc_wrong[f"game_{gid}_wrong"] = acc

            avg_correct = np.mean(list(per_game_acc.values()))
            avg_wrong = np.mean(list(per_game_acc_wrong.values()))
            separation = avg_correct - avg_wrong

        results[tag] = {
            'params': n_params,
            'avg_correct': avg_correct,
            'avg_wrong': avg_wrong,
            'separation': separation,
            'per_game': per_game_acc,
            'per_game_wrong': per_game_acc_wrong,
        }

    return results


def main():
    print("=" * 60)
    print("Phase 41: N-Game Capacity Limit")
    print("  Model: XLarge (~115K params, ch1=64, ch2=128, hidden=256)")
    print("  Testing: N = {2, 5, 7, 10, 15, 20} games")
    print("=" * 60)

    torch.manual_seed(42)

    # Generate 20 diverse games
    print("\nGenerating 20 synthetic games (400 samples each)...")
    all_grids, all_actions, all_game_ids = generate_diverse_games(20, 400, seed=42)
    print(f"  Total data: {sum(len(g) for g in all_grids)} samples")

    # Test configs
    n_games_list = [2, 5, 7, 10, 15, 20]
    all_results = {}

    print(f"\n{'N_games':>8s} | {'NoGate Avg':>10s} {'Gated Avg':>10s} {'Separation':>10s} | {'Params/Game':>11s}")
    print("-" * 70)

    for n_games in n_games_list:
        print(f"\n--- Testing N={n_games} games ---")
        r = run_n_game_experiment(n_games, all_grids, all_actions, all_game_ids, epochs=150)
        ng, g = r['no_gate'], r['gated']
        params_per_game = g['params'] / n_games

        print(f"{n_games:>8d} | {ng['avg_correct']:>10.3f} {g['avg_correct']:>10.3f} "
              f"{g['separation']:>+10.3f} | {params_per_game:>10,.0f}")

        all_results[f"{n_games}_games"] = {
            'n_games': n_games,
            **r,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SCALING LAW ANALYSIS")
    print(f"{'='*60}")
    for key, r in all_results.items():
        n = r['n_games']
        g = r['gated']
        ng = r['no_gate']
        gated_wins = g['avg_correct'] > ng['avg_correct']
        sep_ok = g['separation'] > 0.05
        print(f"  N={n:2d}: Gated={g['avg_correct']:.3f} NoGate={ng['avg_correct']:.3f} "
              f"Sep={g['separation']:+.3f}  "
              f"[{'GATED WINS' if gated_wins else 'NOGATE WINS'}] "
              f"[{'SEPARATED' if sep_ok else 'mixed'}]")

    # Save
    save_path = os.path.join(RESULTS_DIR, "phase41_n_game_capacity.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 41: N-Game Capacity Limit',
            'timestamp': datetime.now().isoformat(),
            'model_config': {'ch1': CH1, 'ch2': CH2, 'hidden': HIDDEN},
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
