"""
Phase 143: The Intrinsic Hourglass — Autonomous Internal Clock for NCA

Phase 117 showed that external time injection (t/T) destroys NCA's
time-invariance and degrades performance. Solution: let the cell
carry its own "hourglass" — an internal channel that decays autonomously.

Architecture:
  - Hidden state has N+1 channels: N for computation + 1 "clock" channel
  - Clock channel initialized to 1.0, decays each step via learned dynamics
  - Other channels can "read" the clock to know their progress
  - NCA rule itself remains time-invariant (same weights every step)

This enables time-varying behavior WITHOUT external time injection.

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


# ================================================================
# NCA with Intrinsic Hourglass
# ================================================================
class HourglassNCA(nn.Module):
    """
    NCA with an internal clock channel.
    Channel layout: [computation_channels..., clock_channel]
    """
    def __init__(self, in_channels=1, hidden=33, out_channels=10, steps=8):
        super().__init__()
        self.steps = steps
        self.hidden = hidden  # includes clock channel
        self.compute_channels = hidden - 1  # 32 compute + 1 clock

        self.proj_in = nn.Conv2d(in_channels, hidden, 3, padding=1)
        # NCA update rule (time-invariant: same weights used every step)
        self.rule = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(self.compute_channels, out_channels))  # only compute channels

    def forward(self, x):
        h = self.proj_in(x)

        # Initialize clock channel to 1.0 (full hourglass)
        h[:, -1:, :, :] = 1.0

        clock_trajectory = []

        for t in range(self.steps):
            delta = self.rule(h)
            h = F.relu(h + delta)

            # Record clock channel value
            clock_val = h[:, -1:, :, :].mean().item()
            clock_trajectory.append(clock_val)

        self._clock_trajectory = clock_trajectory

        # Classify using only compute channels (not clock)
        return self.classifier(h[:, :-1, :, :])


# ================================================================
# Baseline NCA (no clock)
# ================================================================
class BaselineNCA(nn.Module):
    def __init__(self, in_channels=1, hidden=32, out_channels=10, steps=8):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, out_channels))

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        for _ in range(self.steps):
            h = F.relu(h + self.rule(h))
        return self.classifier(h)


# ================================================================
# NCA with External Clock (Phase 117 approach — expected to be worse)
# ================================================================
class ExternalClockNCA(nn.Module):
    """Injects t/T as an extra constant channel (Phase 117 approach)."""
    def __init__(self, in_channels=1, hidden=32, out_channels=10, steps=8):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, 3, padding=1),  # +1 for time channel
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, out_channels))

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        for t in range(self.steps):
            # External clock: constant channel with value t/T
            time_ch = torch.full((h.size(0), 1, h.size(2), h.size(3)),
                                t / self.steps, device=h.device)
            h_with_time = torch.cat([h, time_ch], dim=1)
            h = F.relu(h + self.rule(h_with_time))
        return self.classifier(h)


# ================================================================
# Training and evaluation
# ================================================================
def train_model(model, train_loader, epochs=15, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    accs = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def test_temporal_generalization(model, loader, step_counts):
    """Test model at different T values (temporal generalization)."""
    model.eval()
    results = {}
    original_steps = model.steps
    for T in step_counts:
        model.steps = T
        acc = evaluate(model, loader)
        results[T] = acc
    model.steps = original_steps
    return results


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 143: The Intrinsic Hourglass")
    print("  Autonomous internal clock for NCA")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    TRAIN_STEPS = 8
    results = {}

    # Train all 3 architectures
    architectures = {
        'Baseline NCA': BaselineNCA(steps=TRAIN_STEPS).to(DEVICE),
        'External Clock': ExternalClockNCA(steps=TRAIN_STEPS).to(DEVICE),
        'Hourglass NCA': HourglassNCA(steps=TRAIN_STEPS).to(DEVICE),
    }

    for name, model in architectures.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] Training ({n_params:,} params, T={TRAIN_STEPS})...")
        train_model(model, train_loader, epochs=15)
        acc = evaluate(model, test_loader)
        results[name] = {'accuracy': acc, 'params': n_params}
        print(f"  Accuracy: {acc*100:.2f}%")

    # Temporal generalization test
    print("\n[Temporal Generalization] Testing at different T values...")
    test_steps = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    temporal_results = {}

    for name, model in architectures.items():
        gen_results = test_temporal_generalization(model, test_loader, test_steps)
        temporal_results[name] = gen_results
        print(f"\n  {name}:")
        for T, acc in sorted(gen_results.items()):
            marker = " ★" if T == TRAIN_STEPS else ""
            print(f"    T={T:3d}: {acc*100:.2f}%{marker}")

    # Analyze hourglass clock trajectory
    print("\n[Clock Analysis] Hourglass internal clock dynamics:")
    hg_model = architectures['Hourglass NCA']
    hg_model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        _ = hg_model(x.to(DEVICE))
    clock_traj = hg_model._clock_trajectory
    print(f"  Clock trajectory (T={TRAIN_STEPS}):")
    for t, v in enumerate(clock_traj):
        bar = "#" * int(v * 30) if v > 0 else ""
        print(f"    t={t}: {v:.4f} {bar}")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 143 Complete ({elapsed:.0f}s)")
    for name in architectures:
        r = results[name]
        # Robustness: accuracy at T=32 vs T=8
        gen = temporal_results[name]
        t32 = gen.get(32, 0)
        t8 = gen.get(8, 0)
        print(f"  {name:20s}: {r['accuracy']*100:.2f}% | T=32: {t32*100:.2f}% | "
              f"T64: {gen.get(64,0)*100:.2f}%")

    # Key finding: which architecture generalizes best?
    best_gen = max(architectures.keys(),
                   key=lambda n: temporal_results[n].get(32, 0))
    print(f"\n  Best temporal generalization: {best_gen}")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase143_intrinsic_hourglass.json"), 'w',
              encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 143: The Intrinsic Hourglass',
            'timestamp': datetime.now().isoformat(),
            'train_steps': TRAIN_STEPS,
            'results': results,
            'temporal_results': {k: {str(kk): vv for kk, vv in v.items()}
                                for k, v in temporal_results.items()},
            'clock_trajectory': clock_traj,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        # Panel 1: Accuracy comparison
        names = list(results.keys())
        accs = [results[n]['accuracy']*100 for n in names]
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        bars = axes[0].bar(range(3), accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{acc:.1f}%', ha='center', fontweight='bold')
        axes[0].set_xticks(range(3))
        axes[0].set_xticklabels(['Baseline\nNCA', 'External\nClock', 'Hourglass\nNCA'])
        axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Clock Comparison', fontweight='bold')

        # Panel 2: Temporal generalization
        for name, color in zip(names, colors):
            gen = temporal_results[name]
            ts = sorted(gen.keys())
            axes[1].plot(ts, [gen[t]*100 for t in ts], 'o-', color=color,
                        label=name, markersize=4)
        axes[1].axvline(x=TRAIN_STEPS, color='red', linestyle=':', alpha=0.5,
                       label=f'T={TRAIN_STEPS} (train)')
        axes[1].set_xlabel('T (steps)'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Temporal Generalization', fontweight='bold')
        axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

        # Panel 3: Clock trajectory
        axes[2].plot(range(len(clock_traj)), clock_traj, 'o-', color='#2ecc71',
                    markersize=8, linewidth=2)
        axes[2].fill_between(range(len(clock_traj)), clock_traj, alpha=0.2, color='#2ecc71')
        axes[2].set_xlabel('NCA Step'); axes[2].set_ylabel('Clock Channel Value')
        axes[2].set_title('Intrinsic Hourglass Dynamics', fontweight='bold')
        axes[2].grid(alpha=0.3)

        # Panel 4: Robustness comparison (accuracy drop at T>>8)
        for name, color in zip(names, colors):
            gen = temporal_results[name]
            t8 = gen.get(8, 0)*100
            drops = [(t, gen[t]*100 - t8) for t in sorted(gen.keys()) if t >= 8]
            ts, ds = zip(*drops)
            axes[3].plot(ts, ds, 'o-', color=color, label=name, markersize=4)
        axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[3].set_xlabel('T (steps)'); axes[3].set_ylabel('Accuracy Change vs T=8 (%)')
        axes[3].set_title('Robustness to Extended Steps', fontweight='bold')
        axes[3].legend(fontsize=7); axes[3].grid(alpha=0.3)

        plt.suptitle('Phase 143: The Intrinsic Hourglass — Autonomous Internal Clock for NCA',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase143_intrinsic_hourglass.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
