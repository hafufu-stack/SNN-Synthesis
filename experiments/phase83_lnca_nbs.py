"""
Phase 83: L-NCA + Temporal NBS Simulator

Final integration test: L-NCA with temporal noise beam search
on ARC-like puzzles under time constraints.

Simulates Kaggle ARC scenario:
  - Given 2 demo pairs (input->output)
  - Must predict output for test input
  - K=11 beams with diverse temporal noise on tau gates
  - Time budget: simulate solve-rate under 0.5ms latency constraint

Tasks: expand, gravity, mirror_h (synthetic ARC-like)

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

HIDDEN_CH = 8
NCA_STEPS = 10
GRID_SIZE = 8
EPOCHS = 40
K = 11
LR = 1e-3
BATCH_SIZE = 32
N_TRAIN = 2000
N_TEST = 200

TAU_NOISE_SIGMAS = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]


# Task generators
def gen_expand(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.float32)
        for _ in range(rng.randint(1, 4)):
            y, x = rng.randint(1, gs-1, size=2)
            grid[y, x] = 1.0
        result = grid.copy()
        for y in range(gs):
            for x in range(gs):
                if grid[y, x] > 0:
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < gs and 0 <= nx < gs:
                            result[ny, nx] = 1.0
        inputs.append(grid); targets.append(result)
    return (torch.tensor(np.array(inputs)).unsqueeze(1),
            torch.tensor(np.array(targets)).unsqueeze(1))

def gen_gravity(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.float32)
        for _ in range(rng.randint(2, gs)):
            r, c = rng.randint(0, gs, size=2)
            grid[r, c] = 1.0
        result = np.zeros_like(grid)
        for c in range(gs):
            count = int(grid[:, c].sum())
            for r in range(count):
                result[gs-1-r, c] = 1.0
        inputs.append(grid); targets.append(result)
    return (torch.tensor(np.array(inputs)).unsqueeze(1),
            torch.tensor(np.array(targets)).unsqueeze(1))


class LiquidNCA(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=8, out_ch=1):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.perceive = nn.Conv2d(in_ch + hidden_ch, hidden_ch * 2, 3, padding=1)
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch * 2, hidden_ch, 1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau_gate = nn.Conv2d(in_ch + hidden_ch * 2, hidden_ch, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hidden_ch, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hidden_ch, out_ch, 1)
        self.temporal_sigma = 0.0

    def step(self, x_input, state):
        combined = torch.cat([x_input, state], dim=1)
        perception = self.perceive(combined)
        delta = self.update(perception)
        tau_input = torch.cat([x_input, state, delta], dim=1)
        tau_logit = self.tau_gate(tau_input) + self.b_tau
        if self.temporal_sigma > 0:
            tau_logit = tau_logit + torch.randn_like(tau_logit) * self.temporal_sigma
        beta = torch.sigmoid(tau_logit)
        beta = torch.clamp(beta, 0.01, 0.99)
        state = beta * state + (1 - beta) * delta
        return state

    def forward(self, x, n_steps=NCA_STEPS):
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hidden_ch, h, w, device=x.device)
        for _ in range(n_steps):
            state = self.step(x, state)
        return torch.sigmoid(self.readout(state))


def train_model(model, x_train, y_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n = x_train.size(0)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb, yb = x_train[idx].to(DEVICE), y_train[idx].to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                p = model(x_train[:200].to(DEVICE))
                acc = ((p > 0.5).float() == y_train[:200].to(DEVICE)).float().mean()
            print(f"      Epoch {epoch+1}: pixel_acc={acc*100:.1f}%")


def eval_single(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x.to(DEVICE))
        pred_b = (pred > 0.5).float()
        target = y.to(DEVICE)
        pixel = (pred_b == target).float().mean().item()
        exact = (pred_b.reshape(pred_b.size(0), -1) ==
                 target.reshape(target.size(0), -1)).all(dim=1).float().mean().item()
    return pixel, exact


def eval_temporal_nbs(model, x, y, sigmas):
    """NBS with temporal noise diversity on L-NCA."""
    model.eval()
    b = x.size(0)
    gs = x.size(-1)
    target = y.to(DEVICE)
    vote_sum = torch.zeros(b, 1, gs, gs, device=DEVICE)

    with torch.no_grad():
        for sigma in sigmas:
            model.temporal_sigma = sigma
            pred = model(x.to(DEVICE))
            vote_sum += pred

    model.temporal_sigma = 0.0
    avg_pred = vote_sum / len(sigmas)
    pred_b = (avg_pred > 0.5).float()
    pixel = (pred_b == target).float().mean().item()
    exact = (pred_b.reshape(b, -1) == target.reshape(b, -1)).all(dim=1).float().mean().item()
    return pixel, exact


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase83_lnca_temporal_nbs.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 83: L-NCA + Temporal NBS',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 83: L-NCA + Temporal NBS Simulator")
    print("  Final ARC-agent integration test")
    print(f"  K={K} beams, NCA steps={NCA_STEPS}")
    print("=" * 70)

    TASKS = {'expand': gen_expand, 'gravity': gen_gravity}
    all_results = {}

    for task_name, task_fn in TASKS.items():
        print(f"\n  --- Task: {task_name} ---")
        x_train, y_train = task_fn(N_TRAIN, GRID_SIZE, seed=SEED)
        x_test, y_test = task_fn(N_TEST, GRID_SIZE, seed=SEED+1)

        model = LiquidNCA().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")

        print("    Training...")
        train_model(model, x_train, y_train)

        # A) Baseline (single pass)
        px_base, ex_base = eval_single(model, x_test, y_test)
        print(f"    A) Baseline: pixel={px_base*100:.1f}%, exact={ex_base*100:.1f}%")

        # B) Temporal NBS (K=11, sigma-diverse)
        px_nbs, ex_nbs = eval_temporal_nbs(model, x_test, y_test, TAU_NOISE_SIGMAS[:K])
        print(f"    B) Temporal NBS (K={K}): pixel={px_nbs*100:.1f}%, exact={ex_nbs*100:.1f}%")

        # C) Latency benchmark
        model.eval()
        x_single = x_test[:1].to(DEVICE)
        # Warmup
        for _ in range(5):
            model(x_single)
        # Benchmark
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x_single)
            times.append((time.perf_counter() - t0) * 1000)
        avg_latency = np.mean(times)
        nbs_latency = avg_latency * K  # K sequential passes

        print(f"    C) Latency: single={avg_latency:.2f}ms, "
              f"NBS(K={K})={nbs_latency:.2f}ms")

        all_results[task_name] = {
            'n_params': n_params,
            'baseline': {'pixel': px_base, 'exact': ex_base},
            'temporal_nbs': {'pixel': px_nbs, 'exact': ex_nbs},
            'latency_ms': avg_latency,
            'nbs_latency_ms': nbs_latency,
        }
        _save(all_results)

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: L-NCA + Temporal NBS")
    print(f"{'='*70}")
    for task, r in all_results.items():
        ex_delta = (r['temporal_nbs']['exact'] - r['baseline']['exact']) * 100
        print(f"  {task}:")
        print(f"    Baseline:     exact={r['baseline']['exact']*100:.1f}%")
        print(f"    Temporal NBS: exact={r['temporal_nbs']['exact']*100:.1f}% "
              f"({ex_delta:+.1f}pp)")
        print(f"    Latency:      {r['latency_ms']:.2f}ms single, "
              f"{r['nbs_latency_ms']:.2f}ms NBS")

    _generate_figure(all_results)
    print("\nPhase 83 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tasks = list(results.keys())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Exact match comparison
        ax = axes[0]
        x_pos = np.arange(len(tasks))
        w = 0.35
        base_vals = [results[t]['baseline']['exact']*100 for t in tasks]
        nbs_vals = [results[t]['temporal_nbs']['exact']*100 for t in tasks]
        b1 = ax.bar(x_pos - w/2, base_vals, w, label='Baseline', color='#9CA3AF')
        b2 = ax.bar(x_pos + w/2, nbs_vals, w, label='Temporal NBS', color='#EC4899')
        for b, v in zip(b1, base_vals):
            ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=9)
        for b, v in zip(b2, nbs_vals):
            ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.0f}%', ha='center', fontsize=9)
        ax.set_xticks(x_pos); ax.set_xticklabels(tasks)
        ax.set_ylabel('Exact Match (%)'); ax.set_title('Solve Rate', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 110)

        # Right: Latency
        ax = axes[1]
        single = [results[t]['latency_ms'] for t in tasks]
        nbs = [results[t]['nbs_latency_ms'] for t in tasks]
        b1 = ax.bar(x_pos - w/2, single, w, label='Single pass', color='#3B82F6')
        b2 = ax.bar(x_pos + w/2, nbs, w, label=f'NBS (K={K})', color='#F59E0B')
        for b, v in zip(b1, single):
            ax.text(b.get_x()+b.get_width()/2, v+0.2, f'{v:.1f}ms', ha='center', fontsize=8)
        for b, v in zip(b2, nbs):
            ax.text(b.get_x()+b.get_width()/2, v+0.2, f'{v:.1f}ms', ha='center', fontsize=8)
        ax.set_xticks(x_pos); ax.set_xticklabels(tasks)
        ax.set_ylabel('Latency (ms)'); ax.set_title('Inference Speed', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)

        fig.suptitle('Phase 83: L-NCA + Temporal NBS for ARC\n'
                    'Temporal noise beam search on cellular automata',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase83_lnca_nbs.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
