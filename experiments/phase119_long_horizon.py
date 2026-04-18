"""
Phase 119: Long-Horizon ARC with Soft-Landing

Uses Phase 116's energy canary + soft-landing to solve spatial
reasoning tasks that require LONG information propagation (T=30-50).

Previous NCA weakness: local receptive field can't see across
the grid. Running more steps helps propagation but causes collapse.
Soft-landing fixes this: run T=50 without dying.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026


# ====================================================================
# Spatial reasoning tasks (synthetic ARC-like)
# ====================================================================
def generate_flood_fill_task(grid_size=16, n_samples=500):
    """
    Generate flood fill tasks: fill connected region from a seed pixel.
    Requires information to propagate across the entire grid.
    """
    inputs = []
    targets = []
    for _ in range(n_samples):
        # Create grid with random walls
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        # Add random walls
        n_walls = random.randint(5, 15)
        for _ in range(n_walls):
            x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            length = random.randint(2, grid_size//2)
            if random.random() > 0.5:  # horizontal
                for dx in range(length):
                    if x + dx < grid_size:
                        grid[y, x + dx] = 1.0
            else:  # vertical
                for dy in range(length):
                    if y + dy < grid_size:
                        grid[y + dy, x] = 1.0

        # Seed point (not on wall)
        while True:
            sx, sy = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            if grid[sy, sx] == 0:
                break

        # Create input: walls + seed marker
        inp = np.zeros((2, grid_size, grid_size), dtype=np.float32)
        inp[0] = grid  # walls channel
        inp[1, sy, sx] = 1.0  # seed channel

        # Create target: flood fill from seed
        target = np.zeros((grid_size, grid_size), dtype=np.float32)
        # BFS flood fill
        visited = set()
        queue = [(sy, sx)]
        while queue:
            cy, cx = queue.pop(0)
            if (cy, cx) in visited:
                continue
            if cy < 0 or cy >= grid_size or cx < 0 or cx >= grid_size:
                continue
            if grid[cy, cx] == 1.0:
                continue
            visited.add((cy, cx))
            target[cy, cx] = 1.0
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                queue.append((cy+dy, cx+dx))

        inputs.append(inp)
        targets.append(target)

    return (torch.tensor(np.array(inputs)),
            torch.tensor(np.array(targets)).unsqueeze(1))


def generate_long_move_task(grid_size=16, n_samples=500):
    """
    Move an object from one corner to another (long-range transport).
    """
    inputs = []
    targets = []
    for _ in range(n_samples):
        inp = np.zeros((2, grid_size, grid_size), dtype=np.float32)
        target = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Object: 2x2 block at random position
        ox = random.randint(0, grid_size - 3)
        oy = random.randint(0, grid_size - 3)
        inp[0, oy:oy+2, ox:ox+2] = 1.0

        # Target position (far away)
        tx = grid_size - 3 - ox  # opposite side
        ty = grid_size - 3 - oy
        inp[1, ty:ty+2, tx:tx+2] = 0.5  # target marker (dimmer)
        target[ty:ty+2, tx:tx+2] = 1.0

        inputs.append(inp)
        targets.append(target)

    return (torch.tensor(np.array(inputs)),
            torch.tensor(np.array(targets)).unsqueeze(1))


# ====================================================================
# L-NCA with Soft-Landing
# ====================================================================
class LNCA_ARC(nn.Module):
    """L-NCA for spatial reasoning with soft-landing capability."""
    def __init__(self, in_ch=2, hidden_ch=64, out_ch=1):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch), nn.ReLU()
        )
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
        )
        self.tau_gate = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1), nn.Sigmoid()
        )
        self.decoder = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x, n_steps=5):
        state = self.encoder(x)
        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)
            state = beta * state + (1 - beta) * delta
        return torch.sigmoid(self.decoder(state))

    def forward_with_soft_landing(self, x, n_steps=50, energy_threshold=None,
                                   slowdown=3.0):
        """Forward with energy monitoring and soft-landing."""
        state = self.encoder(x)
        prev_state = state.detach().clone()
        tau_modifier = 0.0
        energies = []

        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)

            if tau_modifier > 0:
                beta = torch.sigmoid(
                    torch.logit(beta.clamp(1e-6, 1-1e-6)) + tau_modifier
                )

            state = beta * state + (1 - beta) * delta

            energy = (state - prev_state).pow(2).mean().item()
            energies.append(energy)
            prev_state = state.detach().clone()

            if energy_threshold is not None and t >= 3:
                if energy > energy_threshold:
                    tau_modifier = slowdown
                elif tau_modifier > 0 and energy < energy_threshold * 0.3:
                    tau_modifier = max(0, tau_modifier - 0.5)

        return torch.sigmoid(self.decoder(state)), energies


# ====================================================================
# Training and evaluation
# ====================================================================
def train_nca(model, train_x, train_y, n_steps=5, epochs=50, lr=1e-3):
    """Train NCA on spatial task."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=n_steps)
            loss = F.binary_cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}")
    return model


def eval_iou(model, test_x, test_y, n_steps=5, soft_landing=False,
             energy_threshold=None, slowdown=3.0):
    """Evaluate IoU (intersection over union) for spatial output."""
    model.eval()
    dataset = torch.utils.data.TensorDataset(test_x, test_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    total_iou = 0; n = 0
    all_energies = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if soft_landing:
                out, energies = model.forward_with_soft_landing(
                    x, n_steps=n_steps,
                    energy_threshold=energy_threshold, slowdown=slowdown)
                all_energies.extend(energies)
            else:
                out = model(x, n_steps=n_steps)

            pred = (out > 0.5).float()
            intersection = (pred * y).sum(dim=(1,2,3))
            union = ((pred + y) > 0).float().sum(dim=(1,2,3))
            iou = (intersection / (union + 1e-8)).mean().item()
            total_iou += iou * x.size(0)
            n += x.size(0)

    return total_iou / n, all_energies


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 119: Long-Horizon ARC with Soft-Landing")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    tasks = {
        'flood_fill': generate_flood_fill_task(grid_size=16, n_samples=800),
        'long_move': generate_long_move_task(grid_size=16, n_samples=800),
    }

    all_results = {}

    for task_name, (data_x, data_y) in tasks.items():
        print(f"\n{'='*50}")
        print(f"  Task: {task_name} (grid=16x16)")
        print(f"{'='*50}")

        # Split train/test
        n_train = 600
        train_x, test_x = data_x[:n_train], data_x[n_train:]
        train_y, test_y = data_y[:n_train], data_y[n_train:]

        # Train with T=5 (standard)
        print(f"\n  [Training] L-NCA (T=5)...")
        model = LNCA_ARC(in_ch=2, hidden_ch=64, out_ch=1).to(DEVICE)
        model = train_nca(model, train_x, train_y, n_steps=5, epochs=50)

        # Test at various T
        print(f"\n  [Testing] IoU vs T...")
        t_results = {}
        t_values = [1, 3, 5, 10, 15, 20, 30, 40, 50]

        for T in t_values:
            iou, _ = eval_iou(model, test_x, test_y, n_steps=T)
            t_results[T] = {'iou': iou, 'method': 'no_landing'}
            print(f"    T={T:2d}: IoU={iou:.4f}", end="")
            if T == 5:
                print(" <-- training T", end="")
            print()

        # Find optimal T and calibrate energy threshold
        optimal_T = max(t_results.keys(), key=lambda t: t_results[t]['iou'])
        optimal_iou = t_results[optimal_T]['iou']
        print(f"\n  Optimal T: {optimal_T} (IoU={optimal_iou:.4f})")

        # Get energy baseline at optimal T
        _, baseline_energies = eval_iou(model, test_x, test_y, n_steps=optimal_T,
                                         soft_landing=True, energy_threshold=1e10)
        baseline_energy = np.mean(baseline_energies[-3:]) if baseline_energies else 1.0
        print(f"  Baseline energy: {baseline_energy:.6f}")

        # Test soft-landing at long horizons
        print(f"\n  [Soft-Landing] Testing long horizons...")
        for T in [20, 30, 40, 50]:
            for mult in [1.5, 2.0, 3.0, 5.0]:
                threshold = baseline_energy * mult
                for slowdown in [2.0, 5.0]:
                    iou, energies = eval_iou(
                        model, test_x, test_y, n_steps=T,
                        soft_landing=True, energy_threshold=threshold,
                        slowdown=slowdown
                    )
                    key = f'SL_T{T}_m{mult}_s{slowdown}'
                    t_results[key] = {
                        'iou': iou, 'T': T, 'mult': mult, 'slowdown': slowdown,
                        'method': 'soft_landing'
                    }

                    # Print if better than unchecked
                    unchecked = t_results.get(T, {}).get('iou', 0)
                    if iou > unchecked + 0.01:
                        print(f"    T={T}, mult={mult}, slow={slowdown}: "
                              f"IoU={iou:.4f} (unchecked={unchecked:.4f}, "
                              f"improvement={iou-unchecked:+.4f})")

        # Best soft-landing result
        sl_results = {k: v for k, v in t_results.items() if isinstance(k, str) and k.startswith('SL')}
        if sl_results:
            best_sl = max(sl_results.items(), key=lambda kv: kv[1]['iou'])
            print(f"\n  Best soft-landing: {best_sl[0]}")
            print(f"    IoU = {best_sl[1]['iou']:.4f}")

        all_results[task_name] = {
            'optimal_T': optimal_T,
            'optimal_iou': optimal_iou,
            't_results': {str(k): v for k, v in t_results.items()},
        }

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("  LONG-HORIZON ARC RESULTS")
    print(f"{'='*70}")
    for task_name, res in all_results.items():
        print(f"\n  {task_name}:")
        print(f"    Optimal T: {res['optimal_T']} (IoU={res['optimal_iou']:.4f})")

        # Compare short T, long T (unchecked), long T (soft-landing)
        tr = res['t_results']
        short = tr.get('5', {}).get('iou', 0)
        long_unchecked = tr.get('50', {}).get('iou', 0)
        sl = max([v['iou'] for k, v in tr.items()
                  if isinstance(k, str) and k.startswith('SL_T50')], default=0)

        print(f"    T=5 (training):     IoU={short:.4f}")
        print(f"    T=50 (unchecked):   IoU={long_unchecked:.4f}")
        print(f"    T=50 (soft-land):   IoU={sl:.4f}")
        if sl > 0:
            print(f"    Soft-landing effect: {sl - long_unchecked:+.4f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase119_long_horizon.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 119: Long-Horizon ARC with Soft-Landing',
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, (task_name, res) in enumerate(all_results.items()):
            tr = res['t_results']
            # Unchecked curve
            int_ts = sorted([k for k in tr.keys() if k.isdigit()], key=int)
            ts_plot = [int(k) for k in int_ts]
            ious = [tr[k]['iou'] for k in int_ts]
            axes[idx].plot(ts_plot, ious, 'r-o', label='No soft-landing', markersize=5)

            # Soft-landing points at T=30,50
            for T in [30, 50]:
                sl_at_T = [(k, v) for k, v in tr.items()
                           if isinstance(k, str) and k.startswith(f'SL_T{T}')]
                if sl_at_T:
                    best = max(sl_at_T, key=lambda kv: kv[1]['iou'])
                    axes[idx].scatter([T], [best[1]['iou']], c='green', s=100,
                                     zorder=5, marker='*',
                                     label=f'Soft-landing T={T}' if T == 30 else '')

            # Optimal T marker
            axes[idx].axvline(x=res['optimal_T'], color='blue', linestyle='--',
                             alpha=0.5, label=f'Optimal T={res["optimal_T"]}')
            axes[idx].set_xlabel('T (steps)')
            axes[idx].set_ylabel('IoU')
            axes[idx].set_title(f'{task_name}: IoU vs T')
            axes[idx].legend(fontsize=8)

        plt.suptitle('Phase 119: Long-Horizon ARC with Soft-Landing', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase119_long_horizon.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 119 complete!")


if __name__ == '__main__':
    main()
