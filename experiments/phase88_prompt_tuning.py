"""
Phase 88: Cellular Prompt Tuning

Freeze L-NCA backbone, add learnable "task context" channels.
TTT only updates the context tensor (not weights) -> no collapse.

Protocol:
  1. Load Foundation Backbone (Phase 87)
  2. Freeze all weights
  3. Add task_context tensor (learnable, shape: 1, ctx_ch, 1, 1)
  4. TTT: Backprop on demos updates ONLY task_context
  5. Measure latency and accuracy for 10-20 steps

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy
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

N_COLORS = 10
HIDDEN_CH = 16
CTX_CH = 4  # Context channels for prompt tuning
NCA_STEPS = 10
GRID_SIZE = 8
N_DEMO = 3
N_TEST = 200
TTT_LR = 0.1
STEP_COUNTS = [0, 5, 10, 15, 20, 30, 50]


def to_onehot(grid, nc=N_COLORS):
    h, w = grid.shape
    oh = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): oh[c] = (grid == c).astype(np.float32)
    return oh


# Reuse task generators
def gen_gravity(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 5)):
            r, c = rng.randint(0, gs, size=2); grid[r, c] = 1
        for _ in range(rng.randint(1, 4)):
            r, c = rng.randint(0, gs, size=2)
            if grid[r, c] == 0: grid[r, c] = 2
        result = np.zeros_like(grid)
        for r in range(gs):
            for c in range(gs):
                if grid[r, c] == 2: result[r, c] = 2
        for c in range(gs):
            cnt = sum(1 for r in range(gs) if grid[r, c] == 1)
            row, placed = gs-1, 0
            while placed < cnt and row >= 0:
                if result[row, c] == 0: result[row, c] = 1; placed += 1
                row -= 1
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def gen_move_right(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(2, 6)):
            r, c = rng.randint(0, gs, size=2)
            grid[r, c] = rng.randint(1, 5)
        result = np.zeros_like(grid)
        for r in range(gs):
            for c in range(gs):
                if grid[r, c] > 0: result[r, (c+1) % gs] = grid[r, c]
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


def gen_color_invert(n, gs=8, seed=None):
    rng = np.random.RandomState(seed)
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((gs, gs), dtype=np.int64)
        for _ in range(rng.randint(3, 8)):
            r, c = rng.randint(0, gs, size=2); grid[r, c] = rng.randint(1, 4)
        result = grid.copy()
        result[grid == 1] = 2; result[grid == 2] = 1
        inputs.append(to_onehot(grid)); targets.append(result)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets))


class PromptLNCA(nn.Module):
    """L-NCA with learnable task context (prompt tuning)."""
    def __init__(self, nc=10, hc=16, ctx_ch=4):
        super().__init__()
        self.hc = hc
        self.ctx_ch = ctx_ch
        # Context projection (maps ctx_ch to additional input channels)
        self.ctx_proj = nn.Conv2d(ctx_ch, nc, 1, bias=False)
        # Main NCA (will be frozen during TTT)
        self.perceive = nn.Conv2d(nc + hc, hc * 2, 3, padding=1)
        self.update = nn.Sequential(
            nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(nc + hc * 2, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)
        # Task context (the only thing updated during TTT)
        self.task_context = nn.Parameter(torch.zeros(1, ctx_ch, 1, 1))

    def forward(self, x, n_steps=NCA_STEPS):
        b, c, h, w = x.shape
        # Add context signal to input
        ctx = self.ctx_proj(self.task_context.expand(b, -1, h, w))
        x_aug = x + ctx
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        for _ in range(n_steps):
            combined = torch.cat([x_aug, state], dim=1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x_aug, state, delta], dim=1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)

    def freeze_backbone(self):
        """Freeze everything except task_context."""
        for name, p in self.named_parameters():
            if name != 'task_context':
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def load_foundation(model):
    """Load Phase 87 foundation weights (compatible keys)."""
    path = os.path.join(RESULTS_DIR, "foundation_lnca.pt")
    if os.path.exists(path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        # Load matching keys
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"    Loaded {loaded} params from foundation backbone")
        return True
    else:
        print(f"    WARNING: Foundation not found, training from scratch")
        return False


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase88_prompt_tuning.json"), 'w',
              encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 88: Cellular Prompt Tuning',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 88: Cellular Prompt Tuning")
    print("  Freeze backbone, learn ONLY task_context")
    print("=" * 70)

    NOVEL_TASKS = {
        'gravity': gen_gravity,
        'move_right': gen_move_right,
        'color_invert': gen_color_invert,
    }

    results = {}

    for task_name, gen_fn in NOVEL_TASKS.items():
        print(f"\n  --- Novel task: {task_name} ---")
        x_all, y_all = gen_fn(N_TEST + N_DEMO + 10, GRID_SIZE, seed=SEED+20)
        demo_x, demo_y = x_all[:N_DEMO].to(DEVICE), y_all[:N_DEMO].to(DEVICE)
        test_x, test_y = x_all[N_DEMO+10:], y_all[N_DEMO+10:]

        task_results = []

        for n_steps in STEP_COUNTS:
            # Fresh model each time
            model = PromptLNCA().to(DEVICE)
            load_foundation(model)
            model.freeze_backbone()

            # TTT: only update task_context
            if n_steps > 0:
                optimizer = torch.optim.Adam([model.task_context], lr=TTT_LR)
                t0 = time.perf_counter()
                for _ in range(n_steps):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(demo_x), demo_y)
                    loss.backward()
                    optimizer.step()
                latency = (time.perf_counter() - t0) * 1000
            else:
                latency = 0.0

            # Evaluate
            model.eval()
            with torch.no_grad():
                preds = model(test_x.to(DEVICE)).argmax(1)
                target = test_y.to(DEVICE)
                pixel = (preds == target).float().mean().item()
                exact = (preds.reshape(-1, GRID_SIZE**2) ==
                         target.reshape(-1, GRID_SIZE**2)).all(1).float().mean().item()

            entry = {'steps': n_steps, 'latency_ms': latency,
                     'pixel_acc': pixel, 'exact_match': exact}
            task_results.append(entry)
            print(f"    steps={n_steps:3d}: {latency:7.1f}ms "
                  f"pixel={pixel*100:.1f}% exact={exact*100:.1f}%")

            del model; gc.collect()

        results[task_name] = task_results
        _save(results)

    # Compare: Full weight TTT vs Prompt-only TTT at 15 steps
    print(f"\n  --- Stability comparison (15 steps) ---")
    x_all, y_all = gen_gravity(N_TEST + N_DEMO + 10, GRID_SIZE, seed=SEED+20)
    demo_x, demo_y = x_all[:N_DEMO].to(DEVICE), y_all[:N_DEMO].to(DEVICE)
    test_x, test_y = x_all[N_DEMO+10:], y_all[N_DEMO+10:]

    # Full weight TTT
    model_full = PromptLNCA().to(DEVICE)
    load_foundation(model_full)
    model_full.unfreeze_all()
    opt = torch.optim.Adam(model_full.parameters(), lr=TTT_LR)
    for _ in range(15):
        opt.zero_grad()
        F.cross_entropy(model_full(demo_x), demo_y).backward()
        opt.step()
    model_full.eval()
    with torch.no_grad():
        p = model_full(test_x.to(DEVICE)).argmax(1)
        px_full = (p == test_y.to(DEVICE)).float().mean().item()
    print(f"    Full-weight TTT (15 steps): pixel={px_full*100:.1f}%")

    # Prompt-only TTT
    model_prompt = PromptLNCA().to(DEVICE)
    load_foundation(model_prompt)
    model_prompt.freeze_backbone()
    opt = torch.optim.Adam([model_prompt.task_context], lr=TTT_LR)
    for _ in range(15):
        opt.zero_grad()
        F.cross_entropy(model_prompt(demo_x), demo_y).backward()
        opt.step()
    model_prompt.eval()
    with torch.no_grad():
        p = model_prompt(test_x.to(DEVICE)).argmax(1)
        px_prompt = (p == test_y.to(DEVICE)).float().mean().item()
    print(f"    Prompt-only TTT (15 steps): pixel={px_prompt*100:.1f}%")

    results['stability'] = {
        'full_weight_15': px_full, 'prompt_only_15': px_prompt
    }
    _save(results)

    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Cellular Prompt Tuning")
    print(f"{'='*70}")
    for task, entries in results.items():
        if isinstance(entries, list):
            best = max(entries, key=lambda e: e['pixel_acc'])
            print(f"  {task}: best pixel={best['pixel_acc']*100:.1f}% "
                  f"at {best['steps']} steps ({best['latency_ms']:.0f}ms)")

    _generate_figure(results)
    print("\nPhase 88 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tasks = [k for k, v in results.items() if isinstance(v, list)]
        fig, axes = plt.subplots(1, len(tasks), figsize=(5*len(tasks), 5))
        if len(tasks) == 1: axes = [axes]
        for i, task in enumerate(tasks):
            ax = axes[i]
            entries = results[task]
            steps = [e['steps'] for e in entries]
            pixel = [e['pixel_acc']*100 for e in entries]
            lat = [e['latency_ms'] for e in entries]
            ax.plot(steps, pixel, 'o-', color='#EC4899', linewidth=2, markersize=7)
            ax.set_xlabel('TTT Steps'); ax.set_ylabel('Pixel Acc (%)')
            ax.set_title(f'{task}', fontweight='bold')
            ax.grid(alpha=0.3); ax.set_ylim(0, 110)
            ax2 = ax.twinx()
            ax2.plot(steps[1:], lat[1:], 's--', color='#3B82F6', alpha=0.5, markersize=5)
            ax2.set_ylabel('Latency (ms)', color='#3B82F6')
        fig.suptitle('Phase 88: Cellular Prompt Tuning\nFreeze backbone, learn context only',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase88_prompt_tuning.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
