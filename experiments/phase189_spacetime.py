"""
Phase 189: Space-Time Memory Law - Does Time Increase Memory?

Fix parameters P, vary NCA steps T.
If M doesn't increase with T -> "Time is for reasoning, Space is for memory."

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase188_capacity import CapacityNCA, generate_random_dataset


def can_memorize_t(hidden_ch, n_pairs, n_steps, max_epochs=500, device=DEVICE):
    """Test memorization with specific step count T."""
    torch.manual_seed(SEED + n_pairs + hidden_ch + n_steps)
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


def binary_search_capacity_t(hidden_ch, n_steps, low=1, high=200):
    """Binary search for max memorizable pairs at given step count."""
    best_m = 0
    while low <= high:
        mid = (low + high) // 2
        success, epochs, pa = can_memorize_t(hidden_ch, mid, n_steps)
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
    print("Phase 189: Space-Time Memory Law")
    print(f"  Does time (T) increase memory capacity?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Fix parameters, vary steps
    fixed_channels = [16, 32, 64]
    step_counts = [1, 2, 3, 5, 8, 12, 20]
    results = {}

    for C in fixed_channels:
        model_tmp = CapacityNCA(N_COLORS, C, 1)
        n_params = model_tmp.count_params()
        del model_tmp

        print(f"\n[C={C}, P={n_params:,}] Testing T = {step_counts}...", flush=True)
        capacities = {}

        for T in step_counts:
            max_try = min(300, n_params // 5)
            cap = binary_search_capacity_t(C, T, low=1, high=max_try)
            capacities[T] = cap
            print(f"  T={T:2d}: M={cap}")

        results[C] = {'params': n_params, 'capacities': capacities}

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 189 Complete ({elapsed:.0f}s)")
    for C in fixed_channels:
        r = results[C]
        caps = r['capacities']
        cap_str = ", ".join(f"T={t}:{caps[t]}" for t in step_counts)
        print(f"  C={C} (P={r['params']:,}): {cap_str}")

    # Analysis: does M increase with T?
    for C in fixed_channels:
        caps = list(results[C]['capacities'].values())
        if len(caps) >= 2:
            slope = (caps[-1] - caps[0]) / max(1, len(caps) - 1)
            verdict = "TIME HELPS MEMORY" if slope > 1 else "TIME IS FOR REASONING ONLY"
            print(f"  C={C}: slope={slope:.2f} -> {verdict}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase189_spacetime.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 189: Space-Time Memory Law',
            'timestamp': datetime.now().isoformat(),
            'results': {str(C): r for C, r in results.items()},
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        colors_plot = ['#e74c3c', '#2ecc71', '#3498db']

        for idx, C in enumerate(fixed_channels):
            caps = results[C]['capacities']
            ts = list(caps.keys())
            ms = list(caps.values())
            axes[0].plot(ts, ms, 'o-', color=colors_plot[idx], linewidth=2,
                        markersize=6, label=f'C={C} (P={results[C]["params"]:,})')
        axes[0].set_xlabel('NCA Steps (T)'); axes[0].set_ylabel('Memory Capacity (M)')
        axes[0].set_title('M vs T (Fixed P)', fontweight='bold')
        axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

        # Normalized (M/M_at_T=1)
        for idx, C in enumerate(fixed_channels):
            caps = results[C]['capacities']
            ts = list(caps.keys())
            ms = list(caps.values())
            base = max(1, ms[0])
            axes[1].plot(ts, [m/base for m in ms], 'o-', color=colors_plot[idx],
                        linewidth=2, label=f'C={C}')
        axes[1].set_xlabel('NCA Steps (T)'); axes[1].set_ylabel('M / M(T=1)')
        axes[1].set_title('Normalized Capacity', fontweight='bold')
        axes[1].axhline(1.0, color='black', linestyle='--', alpha=0.5)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        # Space vs Time contribution
        for idx, C in enumerate(fixed_channels):
            caps = results[C]['capacities']
            t1_cap = caps.get(1, 0)
            max_cap = max(caps.values())
            space_contrib = t1_cap
            time_contrib = max_cap - t1_cap
            axes[2].bar(idx - 0.15, space_contrib, 0.3, color=colors_plot[idx],
                       alpha=0.85, label='Space (P)' if idx == 0 else '')
            axes[2].bar(idx + 0.15, time_contrib, 0.3, color=colors_plot[idx],
                       alpha=0.4, hatch='//', label='Time (+T)' if idx == 0 else '')
        axes[2].set_xticks(range(len(fixed_channels)))
        axes[2].set_xticklabels([f'C={C}' for C in fixed_channels])
        axes[2].set_ylabel('Memory Capacity'); axes[2].set_title('Space vs Time', fontweight='bold')
        axes[2].legend()

        fig.suptitle('Phase 189: Space-Time Memory Law -- Is Time for Reasoning or Memory?',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase189_spacetime.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    main()
