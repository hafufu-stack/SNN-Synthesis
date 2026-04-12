"""
Phase 47: Temporal Architecture Latency Benchmark
Compare Transformer vs GRU vs CNN: accuracy AND inference latency.
Find the architecture that achieves 100% on temporal tasks while staying under 0.5ms.

Author: Hiroto Funasaki
"""
import os, json, time, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
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
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        self.grid[self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)] = 1.0
        return self.get_state()

    def get_state(self):
        return self.grid.flatten()

    def get_optimal_action(self):
        if self.rule_type == 0:
            return self.history[-2] % self.N_ACTIONS if len(self.history) >= 2 else 0
        elif self.rule_type == 1:
            return len(self.history) % 2
        elif self.rule_type == 2:
            return len(self.history) % self.N_ACTIONS
        else:
            return (self.history[-1] + 2) % self.N_ACTIONS if self.history else 0


# ==============================================================
# Three Architectures
# ==============================================================
class TinyCNN(nn.Module):
    """CNN: sees only current state (no history)."""
    def __init__(self, state_dim, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, x):
        # x: (batch, seq, state_dim) -> use last state only
        if x.dim() == 3:
            x = x[:, -1, :]
        return self.net(x)


class TinyGRU(nn.Module):
    """GRU (RNN): O(1) per step with hidden state. Sees full history."""
    def __init__(self, state_dim, n_actions=4, hidden=64, n_layers=1):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, n_layers, batch_first=True)
        self.output = nn.Linear(hidden, n_actions)

    def forward(self, x):
        # x: (batch, seq, state_dim)
        out, _ = self.gru(x)
        return self.output(out[:, -1, :])

    def forward_step(self, x, hidden=None):
        """Single-step inference: O(1) per step."""
        # x: (batch, 1, state_dim)
        out, hidden = self.gru(x, hidden)
        return self.output(out[:, -1, :]), hidden


class TinyTransformer(nn.Module):
    """Transformer: full attention. O(N^2) per step."""
    def __init__(self, state_dim, n_actions=4, d_model=64, nhead=4,
                 n_layers=2, context_len=10):
        super().__init__()
        self.state_proj = nn.Linear(state_dim, d_model)
        self.pos_embed = nn.Embedding(context_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, n_actions)
        self.context_len = context_len

    def forward(self, x):
        bs, seq_len, _ = x.shape
        h = self.state_proj(x)
        pos = torch.arange(seq_len, device=x.device)
        h = h + self.pos_embed(pos).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.transformer(h, mask=mask)
        return self.output(h[:, -1, :])


# ==============================================================
# Data Generation
# ==============================================================
def generate_data(rule_type, n_episodes=500, context_len=10, seed=42):
    random.seed(seed)
    state_dim = TemporalGame.GRID_SIZE ** 2
    X_all, y_all = [], []

    for ep in range(n_episodes):
        env = TemporalGame(rule_type=rule_type, seed=seed + ep)
        ep_states = []
        state = env.reset()

        for step in range(env.max_steps):
            optimal = env.get_optimal_action()
            ep_states.append(state.copy())

            # Build context
            ctx = ep_states[-context_len:]
            while len(ctx) < context_len:
                ctx.insert(0, np.zeros(state_dim, dtype=np.float32))
            ctx = ctx[-context_len:]

            X_all.append(np.array(ctx))
            y_all.append(optimal)

            env.history.append(optimal)
            state = env.get_state()

    return (torch.tensor(np.array(X_all), dtype=torch.float32),
            torch.tensor(y_all, dtype=torch.long))


def train_model(model, X, y, epochs=100, lr=0.001):
    n = len(y)
    split = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx, test_idx = perm[:split], perm[split:]

    opt = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        batch = train_idx[torch.randperm(len(train_idx))[:min(256, len(train_idx))]]
        logits = model(X[batch])
        loss = F.cross_entropy(logits, y[batch])
        opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(X[test_idx])
                acc = (test_logits.argmax(1) == y[test_idx]).float().mean().item()
                best_acc = max(best_acc, acc)

    return best_acc


def measure_latency(model, X_sample, n_runs=1000):
    """Measure single-sample inference latency in milliseconds."""
    model.eval()
    x = X_sample[:1]  # single sample

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(x)

    # Measure
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    return {
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'min_ms': np.min(times),
    }


def measure_gru_step_latency(model, state_dim, n_runs=1000):
    """Measure GRU single-step (recurrent) latency."""
    model.eval()
    x = torch.randn(1, 1, state_dim)
    hidden = None

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _, hidden = model.forward_step(x, hidden)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _, hidden = model.forward_step(x, hidden)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'min_ms': np.min(times),
    }


def main():
    print("=" * 60)
    print("Phase 47: Temporal Architecture Latency Benchmark")
    print("  Accuracy + Latency: which model survives 0.5ms?")
    print("=" * 60)

    torch.manual_seed(42)
    state_dim = TemporalGame.GRID_SIZE ** 2  # 36
    context_len = 10
    rule_names = ["Repeat-2-ago", "Alternate", "Cycle-4", "Opposite-last"]

    all_results = {}

    for rule_type in range(4):
        print(f"\n--- Rule: {rule_names[rule_type]} ---")
        X, y = generate_data(rule_type, n_episodes=500, context_len=context_len)

        # Build models
        models = {
            'CNN': TinyCNN(state_dim, n_actions=4, hidden=64),
            'GRU': TinyGRU(state_dim, n_actions=4, hidden=64),
            'Transformer': TinyTransformer(state_dim, n_actions=4, d_model=64,
                                           nhead=4, n_layers=2, context_len=context_len),
        }

        rule_results = {}
        print(f"  {'Model':>15s} | {'Params':>8s} {'Acc':>8s} {'Mean':>8s} {'P95':>8s} {'P99':>8s} {'<0.5ms':>7s}")
        print("  " + "-" * 65)

        for name, model in models.items():
            n_params = sum(p.numel() for p in model.parameters())

            # Train
            acc = train_model(model, X, y, epochs=100)

            # Measure latency (full-context forward)
            lat = measure_latency(model, X, n_runs=1000)

            # GRU also gets single-step latency
            if name == 'GRU':
                step_lat = measure_gru_step_latency(model, state_dim, n_runs=1000)
                lat['step_mean_ms'] = step_lat['mean_ms']
                lat['step_p95_ms'] = step_lat['p95_ms']

            under_05 = "YES" if lat['p95_ms'] < 0.5 else "NO"
            print(f"  {name:>15s} | {n_params:>8,} {acc*100:>6.1f}% "
                  f"{lat['mean_ms']:>6.3f}ms {lat['p95_ms']:>6.3f}ms {lat['p99_ms']:>6.3f}ms "
                  f"{under_05:>6s}")

            if name == 'GRU':
                print(f"  {'GRU (step)':>15s} | {'':>8s} {'':>8s} "
                      f"{lat['step_mean_ms']:>6.3f}ms {lat['step_p95_ms']:>6.3f}ms")

            rule_results[name] = {
                'n_params': n_params,
                'accuracy': acc,
                'latency': lat,
                'under_0_5ms': under_05 == "YES",
            }

        all_results[rule_names[rule_type]] = rule_results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Best model for Kaggle (accuracy + speed)")
    print(f"{'Rule':>18s} | {'Best Accurate':>20s} {'Best Fast':>20s}")
    print("-" * 65)

    for rule in rule_names:
        r = all_results[rule]
        best_acc_name = max(r, key=lambda k: r[k]['accuracy'])
        best_fast_name = min(r, key=lambda k: r[k]['latency']['p95_ms'])
        print(f"{rule:>18s} | {best_acc_name:>10s} ({r[best_acc_name]['accuracy']*100:.0f}%) "
              f"{best_fast_name:>10s} ({r[best_fast_name]['latency']['p95_ms']:.3f}ms)")

    # Is GRU the sweet spot?
    gru_wins = 0
    for rule in rule_names:
        r = all_results[rule]
        if r['GRU']['accuracy'] > 0.95 and r['GRU']['latency']['p95_ms'] < 0.5:
            gru_wins += 1
    print(f"\nGRU under 0.5ms with >95% accuracy: {gru_wins}/4 rules")

    save_path = os.path.join(RESULTS_DIR, "phase47_latency_benchmark.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 47: Temporal Architecture Latency Benchmark',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
