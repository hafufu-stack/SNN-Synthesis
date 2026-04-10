"""
Phase 38a: Multi-Game Gating - Capacity Scaling
Scale up model size from 2.7K -> 60K -> 240K params to find
the threshold where ID-gating successfully prevents
catastrophic forgetting in multi-game learning.

Author: Hiroto Funasaki
"""
import os, json, numpy as np, math, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"
N_COLORS, GRID_SIZE, N_ACTIONS = 10, 8, 7


def generate_game_data(game_type, n_samples=400, seed=42):
    rng = np.random.RandomState(seed)
    grids, actions = [], []
    for _ in range(n_samples):
        grid = np.zeros((N_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        if game_type == "A":
            py, px = rng.randint(0, GRID_SIZE, 2)
            gy, gx = rng.randint(0, GRID_SIZE, 2)
            grid[1, py, px] = 1.0
            grid[2, gy, gx] = 1.0
            for _ in range(rng.randint(3, 8)):
                wy, wx = rng.randint(0, GRID_SIZE, 2)
                grid[3, wy, wx] = 1.0
            dx, dy = gx - px, gy - py
            action = (0 if dx > 0 else 1) if abs(dx) > abs(dy) else (2 if dy > 0 else 3)
        else:
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    grid[rng.randint(0, 5), i, j] = 1.0
            action = int(np.argmax(grid[:5].sum(axis=(1, 2)))) % N_ACTIONS
        grids.append(grid)
        actions.append(action)
    return torch.tensor(np.array(grids)), torch.tensor(actions, dtype=torch.long)


class ScalableCNN(nn.Module):
    """CNN with configurable size + optional game-ID gating."""
    def __init__(self, ch1=8, ch2=16, hidden=32, n_games=2, use_gating=True):
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


def run_experiment(ch1, ch2, hidden, label, X_a, y_a, X_b, y_b):
    """Run gated vs ungated comparison at a given model scale."""
    X = torch.cat([X_a, X_b])
    y = torch.cat([y_a, y_b])
    gids = torch.cat([torch.zeros(len(X_a)), torch.ones(len(X_b))]).long()

    results = {}
    for use_gating in [False, True]:
        tag = "gated" if use_gating else "no_gate"
        model = ScalableCNN(ch1, ch2, hidden, use_gating=use_gating)
        n_params = sum(p.numel() for p in model.parameters())
        opt = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(120):
            model.train()
            logits = model(X, game_id=gids) if use_gating else model(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred = (model(X, game_id=gids) if use_gating else model(X)).argmax(1)
            ma, mb = gids == 0, gids == 1
            acc_a = (pred[ma] == y[ma]).float().mean().item()
            acc_b = (pred[mb] == y[mb]).float().mean().item()
            if use_gating:
                pred_w = model(X, game_id=1-gids).argmax(1)
                acc_aw = (pred_w[ma] == y[ma]).float().mean().item()
                acc_bw = (pred_w[mb] == y[mb]).float().mean().item()
            else:
                acc_aw, acc_bw = acc_a, acc_b

        results[tag] = {
            'params': n_params, 'acc_a': acc_a, 'acc_b': acc_b,
            'acc_a_wrong': acc_aw, 'acc_b_wrong': acc_bw,
            'avg': (acc_a + acc_b) / 2,
            'separation': (acc_a + acc_b) / 2 - (acc_aw + acc_bw) / 2,
        }
    return results


def main():
    print("=" * 60)
    print("Phase 38a: Multi-Game Gating - Capacity Scaling")
    print("=" * 60)

    torch.manual_seed(42)
    X_a, y_a = generate_game_data("A", 400, seed=42)
    X_b, y_b = generate_game_data("B", 400, seed=123)

    # Single-game baselines
    baselines = {}
    for name, Xg, yg in [("A", X_a, y_a), ("B", X_b, y_b)]:
        m = ScalableCNN(16, 32, 64, use_gating=False)
        o = optim.Adam(m.parameters(), lr=0.01)
        for _ in range(120):
            m.train(); l = F.cross_entropy(m(Xg), yg); o.zero_grad(); l.backward(); o.step()
        m.eval()
        with torch.no_grad():
            baselines[name] = (m(Xg).argmax(1) == yg).float().mean().item()
    print(f"Single-game baselines: A={baselines['A']:.3f}, B={baselines['B']:.3f}")

    # Scale configs: (ch1, ch2, hidden, label)
    configs = [
        (4,  8,   16,  "Tiny (~0.5K)"),
        (8,  16,  32,  "Small (~2.7K)"),
        (16, 32,  64,  "Medium (~10K)"),
        (32, 64,  128, "Large (~60K)"),
        (64, 128, 256, "XLarge (~240K)"),
    ]

    all_results = {}
    print(f"\n{'Scale':<18s} {'Params':>8s} | {'NoGate Avg':>10s} {'Gated Avg':>10s} {'Sep':>6s}")
    print("-" * 65)
    for ch1, ch2, hidden, label in configs:
        r = run_experiment(ch1, ch2, hidden, label, X_a, y_a, X_b, y_b)
        ng, g = r['no_gate'], r['gated']
        print(f"{label:<18s} {g['params']:>8,} | {ng['avg']:>10.3f} {g['avg']:>10.3f} {g['separation']:>+6.3f}")
        all_results[label] = r

    # Summary
    print(f"\n{'='*60}")
    print("Scaling Analysis:")
    for label, r in all_results.items():
        g = r['gated']
        verdict = "SEPARATED" if g['separation'] > 0.05 else "mixed"
        print(f"  {label}: Sep={g['separation']:+.3f}  "
              f"A={g['acc_a']:.3f} B={g['acc_b']:.3f}  [{verdict}]")

    save_path = os.path.join(RESULTS_DIR, "phase38a_capacity_scaling.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 38a: Capacity Scaling',
            'timestamp': datetime.now().isoformat(),
            'baselines': baselines, 'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
