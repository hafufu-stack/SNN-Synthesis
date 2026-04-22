"""
Phase 190: The Leviathan NCA - CPU + Virtual Memory Giant

Break VRAM limits by running on CPU with massive hidden channels.
Test if enormous parameter count enables Grokking on ARC tasks.

WARNING: C=1024+ will consume many GB of RAM. Training is SLOW on CPU.
This is intentional -- we're probing the limits of physics.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime
import psutil

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
SEED = 2026
N_COLORS = 10
GRID_SIZE = 8

# Force CPU for this experiment
DEVICE = "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase188_capacity import CapacityNCA, generate_random_dataset


def get_memory_usage_gb():
    """Get current process memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def train_leviathan(hidden_ch, n_pairs, n_steps=5, max_epochs=200):
    """Train a giant CPU-based NCA on random memorization task."""
    torch.manual_seed(SEED)
    model = CapacityNCA(N_COLORS, hidden_ch, n_steps).to(DEVICE)
    n_params = model.count_params()
    param_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)

    print(f"    Model: C={hidden_ch}, P={n_params:,}, Size={param_mb:.1f}MB")
    print(f"    RAM before: {get_memory_usage_gb():.2f}GB")

    inputs, targets = generate_random_dataset(n_pairs, GRID_SIZE, N_COLORS)
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []

    for epoch in range(max_epochs):
        t0 = time.time()
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        dt = time.time() - t0

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            pa = (pred == targets).float().mean().item()

        history.append({'epoch': epoch, 'loss': loss.item(), 'pa': pa, 'time': dt})

        if (epoch + 1) % 20 == 0:
            ram = get_memory_usage_gb()
            print(f"    Epoch {epoch+1}/{max_epochs}: Loss={loss.item():.4f}, "
                  f"PA={pa*100:.1f}%, {dt:.2f}s/step, RAM={ram:.1f}GB")

        if pa >= 0.999:
            print(f"    MEMORIZED at epoch {epoch+1}!")
            break

    final_pa = history[-1]['pa']
    final_loss = history[-1]['loss']
    avg_time = np.mean([h['time'] for h in history])
    ram_peak = get_memory_usage_gb()

    del model, inputs, targets, opt
    gc.collect()

    return {
        'hidden_ch': hidden_ch, 'n_params': n_params,
        'n_pairs': n_pairs, 'final_pa': final_pa,
        'final_loss': final_loss, 'avg_time_per_step': avg_time,
        'ram_peak_gb': ram_peak, 'memorized': final_pa >= 0.999,
        'history': history
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 190: The Leviathan NCA - CPU Giant")
    print(f"  Breaking VRAM limits with CPU + virtual memory")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Device: {DEVICE}")
    print(f"  Available RAM: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    print(f"  Total RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print("=" * 70)

    # Progressive sizes: small -> giant
    # Each test: memorize 20 random pairs (fixed) with increasing C
    n_pairs = 20
    channel_sizes = [32, 64, 128, 256, 512, 1024]
    results = {}

    for C in channel_sizes:
        print(f"\n[Leviathan C={C}] Summoning...", flush=True)

        # Safety check: estimate memory before creating
        estimated_mb = (C * C * 9 * 4 * 4) / (1024**2)  # rough estimate
        avail_gb = psutil.virtual_memory().available / (1024**3)
        if estimated_mb > avail_gb * 1000 * 0.8:
            print(f"  SKIP: would need ~{estimated_mb:.0f}MB, only {avail_gb:.1f}GB available")
            break

        result = train_leviathan(C, n_pairs, n_steps=5, max_epochs=200)
        results[C] = result

        # Summary
        status = "MEMORIZED" if result['memorized'] else f"PA={result['final_pa']*100:.1f}%"
        print(f"  Result: {status}, {result['avg_time_per_step']:.3f}s/step, "
              f"RAM={result['ram_peak_gb']:.1f}GB")

        # Abort if taking too long per step
        if result['avg_time_per_step'] > 30:
            print(f"  ABORT: too slow ({result['avg_time_per_step']:.0f}s/step), skipping larger")
            break

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 190 Complete ({elapsed:.0f}s)")
    for C, r in results.items():
        status = "MEMORIZED" if r['memorized'] else f"PA={r['final_pa']*100:.1f}%"
        print(f"  C={C:5d}: P={r['n_params']:>10,}, {status}, "
              f"{r['avg_time_per_step']:.3f}s/step, RAM={r['ram_peak_gb']:.1f}GB")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Save without full history to keep file small
    save_results = {}
    for C, r in results.items():
        save_r = {k: v for k, v in r.items() if k != 'history'}
        # Keep only milestone epochs from history
        save_r['history_milestones'] = [h for h in r['history'] if (h['epoch']+1) % 20 == 0]
        save_results[str(C)] = save_r

    with open(os.path.join(RESULTS_DIR, "phase190_leviathan.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 190: The Leviathan NCA',
            'timestamp': datetime.now().isoformat(),
            'n_pairs': n_pairs,
            'results': save_results,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        cs = list(results.keys())
        params = [results[c]['n_params'] for c in cs]
        pas = [results[c]['final_pa']*100 for c in cs]
        times = [results[c]['avg_time_per_step'] for c in cs]
        rams = [results[c]['ram_peak_gb'] for c in cs]

        # PA vs Params
        axes[0].semilogx(params, pas, 'o-', color='#2ecc71', linewidth=2, markersize=8)
        axes[0].set_xlabel('Parameters'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title(f'Leviathan PA ({n_pairs} random pairs)', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Time per step
        axes[1].bar([f'C={c}' for c in cs], times, color='#e74c3c', alpha=0.85)
        axes[1].set_ylabel('Time/Step (s)'); axes[1].set_title('CPU Speed', fontweight='bold')

        # RAM usage
        axes[2].bar([f'C={c}' for c in cs], rams, color='#3498db', alpha=0.85)
        axes[2].set_ylabel('RAM (GB)'); axes[2].set_title('Memory Usage', fontweight='bold')

        fig.suptitle('Phase 190: The Leviathan NCA (CPU Giant)', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase190_leviathan.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results

if __name__ == '__main__':
    main()
