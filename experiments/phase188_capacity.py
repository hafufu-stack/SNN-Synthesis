"""
Phase 188: NCA Capacity Law - The Memory Equation

How many random input-output pairs can an NCA memorize?
Measure M (memorizable pairs) vs P (parameters) to derive:
    M = alpha * P^beta

Uses completely random noise grids (no learnable patterns),
forcing pure memorization (no generalization possible).

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
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
N_COLORS = 10
GRID_SIZE = 8


class CapacityNCA(nn.Module):
    """Minimal NCA for capacity measurement. No task embedding."""
    def __init__(self, n_colors=10, hidden_ch=32, n_steps=5):
        super().__init__()
        self.n_steps = n_steps
        C = hidden_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(n_colors, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )
        self.update = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 1),
        )
        self.tau_gate = nn.Sequential(
            nn.Conv2d(C, C, 1), nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

    def forward(self, x):
        state = self.encoder(x)
        for t in range(self.n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)
            state = beta * state + (1 - beta) * delta
        return self.decoder(state)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def generate_random_dataset(n_pairs, grid_size=8, n_colors=10):
    """Generate completely random input-output pairs (no learnable pattern)."""
    inputs, outputs = [], []
    for _ in range(n_pairs):
        inp = torch.randint(0, n_colors, (grid_size, grid_size))
        out = torch.randint(0, n_colors, (grid_size, grid_size))
        # One-hot encode
        inp_oh = F.one_hot(inp, n_colors).permute(2, 0, 1).float()
        out_oh = out  # class indices for CE loss
        inputs.append(inp_oh)
        outputs.append(out_oh)
    return torch.stack(inputs), torch.stack(outputs)


def can_memorize(hidden_ch, n_pairs, n_steps=5, max_epochs=500, device=DEVICE):
    """Test if NCA with given hidden_ch can memorize n_pairs random pairs."""
    torch.manual_seed(SEED + n_pairs + hidden_ch)
    model = CapacityNCA(N_COLORS, hidden_ch, n_steps).to(device)
    inputs, targets = generate_random_dataset(n_pairs, GRID_SIZE, N_COLORS)
    inputs, targets = inputs.to(device), targets.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)

    for epoch in range(max_epochs):
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                pa = (pred == targets).float().mean().item()
                if pa >= 0.999:
                    return True, epoch + 1, pa

    with torch.no_grad():
        logits = model(inputs)
        pred = logits.argmax(dim=1)
        pa = (pred == targets).float().mean().item()
    return pa >= 0.999, max_epochs, pa


def binary_search_capacity(hidden_ch, n_steps=5, low=1, high=200):
    """Binary search for max memorizable pairs."""
    best_m = 0
    while low <= high:
        mid = (low + high) // 2
        success, epochs, pa = can_memorize(hidden_ch, mid, n_steps)
        if success:
            best_m = mid
            low = mid + 1
        else:
            high = mid - 1
    return best_m


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 188: NCA Capacity Law - The Memory Equation")
    print(f"  Random noise memorization: M = alpha * P^beta")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Colors: {N_COLORS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Test different hidden channel sizes
    channel_sizes = [4, 8, 16, 32, 64, 128]
    results = {}

    for C in channel_sizes:
        print(f"\n[C={C}] Measuring capacity...", flush=True)
        model_tmp = CapacityNCA(N_COLORS, C, 5)
        n_params = model_tmp.count_params()
        del model_tmp

        # Quick probe to set binary search range
        # Start with small test
        success_1, _, pa_1 = can_memorize(C, 1)
        if not success_1:
            print(f"  Cannot even memorize 1 pair! (PA={pa_1:.2f})")
            results[C] = {'params': n_params, 'capacity': 0}
            continue

        # Binary search
        max_try = min(500, n_params // 10)  # rough upper bound
        capacity = binary_search_capacity(C, n_steps=5, low=1, high=max_try)

        results[C] = {'params': n_params, 'capacity': capacity}
        print(f"  C={C}: P={n_params:,}, M={capacity} "
              f"(ratio M/P={capacity/max(1,n_params):.4f})")

    # Fit power law: M = alpha * P^beta
    import scipy.optimize as opt
    params_list = [results[C]['params'] for C in channel_sizes if results[C]['capacity'] > 0]
    caps_list = [results[C]['capacity'] for C in channel_sizes if results[C]['capacity'] > 0]

    if len(params_list) >= 3:
        log_p = np.log(params_list)
        log_m = np.log(np.array(caps_list) + 1e-10)
        # Linear fit in log space: log(M) = log(alpha) + beta * log(P)
        beta, log_alpha = np.polyfit(log_p, log_m, 1)
        alpha = np.exp(log_alpha)
        print(f"\n  SCALING LAW: M = {alpha:.4f} * P^{beta:.4f}")
        r_squared = 1 - np.sum((log_m - (log_alpha + beta * log_p))**2) / np.sum((log_m - np.mean(log_m))**2)
        print(f"  R-squared: {r_squared:.4f}")
    else:
        alpha, beta, r_squared = 0, 0, 0
        print("\n  Not enough data points for regression")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 188 Complete ({elapsed:.0f}s)")
    for C in channel_sizes:
        r = results[C]
        print(f"  C={C:4d}: Params={r['params']:>8,}, Capacity={r['capacity']:>4}")
    if alpha > 0:
        print(f"\n  THE EQUATION: M = {alpha:.4f} * P^{beta:.4f}  (R2={r_squared:.4f})")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase188_capacity.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 188: NCA Capacity Law',
            'timestamp': datetime.now().isoformat(),
            'grid_size': GRID_SIZE, 'n_colors': N_COLORS,
            'results': {str(C): r for C, r in results.items()},
            'scaling_law': {'alpha': alpha, 'beta': beta, 'r_squared': r_squared},
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        ps = [results[C]['params'] for C in channel_sizes]
        ms = [results[C]['capacity'] for C in channel_sizes]

        # Log-log plot with fit
        valid = [(p, m) for p, m in zip(ps, ms) if m > 0]
        if valid:
            vp, vm = zip(*valid)
            axes[0].scatter(vp, vm, s=80, color='#e74c3c', zorder=5, edgecolor='black')
            if alpha > 0:
                fit_p = np.logspace(np.log10(min(vp)*0.5), np.log10(max(vp)*2), 100)
                fit_m = alpha * fit_p ** beta
                axes[0].plot(fit_p, fit_m, '--', color='#2ecc71', linewidth=2,
                           label=f'M = {alpha:.2f} * P^{beta:.2f}')
                axes[0].legend(fontsize=10)
        axes[0].set_xscale('log'); axes[0].set_yscale('log')
        axes[0].set_xlabel('Parameters (P)'); axes[0].set_ylabel('Memory Capacity (M)')
        axes[0].set_title('NCA Capacity Law', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Bar chart
        labels = [f'C={C}' for C in channel_sizes]
        axes[1].bar(labels, ms, color='#3498db', alpha=0.85, edgecolor='black')
        axes[1].set_ylabel('Memorizable Pairs (M)')
        axes[1].set_title('Capacity vs Channel Size', fontweight='bold')

        # Efficiency (M/P ratio)
        ratios = [m/max(1, p) for m, p in zip(ms, ps)]
        axes[2].bar(labels, ratios, color='#f39c12', alpha=0.85, edgecolor='black')
        axes[2].set_ylabel('M/P Ratio')
        axes[2].set_title('Memory Efficiency', fontweight='bold')

        fig.suptitle('Phase 188: The Intelligence Equation -- NCA Memory Capacity',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase188_capacity_law.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    main()
