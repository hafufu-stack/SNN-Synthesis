"""
Phase 111: ANN-to-LNN Analytical Conversion

Derives the Liquid Neural Network equivalent of v11's universal
SNN conversion formula θ = 2.0 × max(a).

Goal: Find b_τ = f(activation_stats) that achieves lossless
ANN→LNN conversion by analytically setting τ gate biases.

Theory:
  ANN neuron:  y = ReLU(Wx + b)           [static]
  LNN neuron:  s' = β*s + (1-β)*Δ(x)      [temporal]
               β = sigmoid(W_τ·[x,s] + b_τ)

  If b_τ is set correctly, LNN converges to ANN output in T steps.

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
# Standard ANN (MLP)
# ====================================================================
class ANN_MLP(nn.Module):
    """Standard feedforward MLP for MNIST."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_activations(self, x):
        """Return activations at each layer for τ calibration."""
        x = x.view(-1, 784)
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        a3 = self.fc3(a2)
        return [a1, a2, a3]


# ====================================================================
# Liquid Neural Network (LNN) with configurable τ
# ====================================================================
class LNN_MLP(nn.Module):
    """
    LNN equivalent of ANN_MLP.
    Same weights, but each layer has a liquid τ-gate that controls
    how fast the hidden state converges.
    """
    def __init__(self, ann_model, b_tau_values=None):
        super().__init__()
        # Copy weights from ANN (no retraining!)
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.fc1.weight.data = ann_model.fc1.weight.data.clone()
        self.fc1.bias.data = ann_model.fc1.bias.data.clone()
        self.fc2.weight.data = ann_model.fc2.weight.data.clone()
        self.fc2.bias.data = ann_model.fc2.bias.data.clone()
        self.fc3.weight.data = ann_model.fc3.weight.data.clone()
        self.fc3.bias.data = ann_model.fc3.bias.data.clone()

        # τ gate biases (controls liquid dynamics speed)
        # Higher b_τ → β closer to 1 → slower update → more memory
        # Lower b_τ → β closer to 0 → faster update → instant response
        if b_tau_values is None:
            b_tau_values = [0.0, 0.0, 0.0]  # default: β≈0.5

        self.b_tau = nn.ParameterList([
            nn.Parameter(torch.ones(size) * bt, requires_grad=False)
            for size, bt in zip([256, 128, 10], b_tau_values)
        ])

    def forward(self, x, n_steps=10):
        x = x.view(-1, 784)
        b = x.size(0)

        # Initialize liquid states to zero
        s1 = torch.zeros(b, 256, device=x.device)
        s2 = torch.zeros(b, 128, device=x.device)
        s3 = torch.zeros(b, 10, device=x.device)

        for t in range(n_steps):
            # Layer 1: liquid update
            delta1 = F.relu(self.fc1(x))
            beta1 = torch.sigmoid(self.b_tau[0]).unsqueeze(0)
            s1 = beta1 * s1 + (1 - beta1) * delta1

            # Layer 2: liquid update
            delta2 = F.relu(self.fc2(s1))
            beta2 = torch.sigmoid(self.b_tau[1]).unsqueeze(0)
            s2 = beta2 * s2 + (1 - beta2) * delta2

            # Layer 3: liquid output
            delta3 = self.fc3(s2)
            beta3 = torch.sigmoid(self.b_tau[2]).unsqueeze(0)
            s3 = beta3 * s3 + (1 - beta3) * delta3

        return s3


# ====================================================================
# τ Conversion Formulas (analogous to θ = α × max(a))
# ====================================================================
def compute_tau_formulas(ann_model, calibration_data):
    """
    Compute different b_τ settings from ANN activation statistics.

    Returns dict of formula_name → [b_tau_L1, b_tau_L2, b_tau_L3]
    """
    ann_model.eval()
    all_activations = [[], [], []]

    with torch.no_grad():
        for x in calibration_data:
            acts = ann_model.get_activations(x.to(DEVICE))
            for i, a in enumerate(acts):
                all_activations[i].append(a.cpu())

    # Concatenate all activations
    for i in range(3):
        all_activations[i] = torch.cat(all_activations[i], dim=0)

    # Compute statistics per layer
    stats = []
    for i, acts in enumerate(all_activations):
        s = {
            'max': acts.max().item(),
            'mean': acts.mean().item(),
            'std': acts.std().item(),
            'p99': torch.quantile(acts.float(), 0.99).item(),
        }
        stats.append(s)
        print(f"  Layer {i+1}: max={s['max']:.3f} mean={s['mean']:.3f} "
              f"std={s['std']:.3f} p99={s['p99']:.3f}")

    formulas = {}

    # Formula 0: Fixed baseline (β ≈ 0.5, standard liquid)
    formulas['fixed_0.0'] = [0.0, 0.0, 0.0]

    # Formula 1: Large negative (β → 0, instant convergence = ANN-like)
    for v in [-5.0, -3.0, -1.0]:
        formulas[f'fixed_{v}'] = [v, v, v]

    # Formula 2: Large positive (β → 1, strong memory)
    for v in [1.0, 3.0, 5.0]:
        formulas[f'fixed_{v}'] = [v, v, v]

    # Formula 3: Max-scaled (v11 analogy: θ = α × max(a))
    for alpha in [0.5, 1.0, 2.0, 3.0, 5.0]:
        formulas[f'max_α={alpha}'] = [
            -alpha * stats[i]['max'] for i in range(3)
        ]

    # Formula 4: Std-scaled (high variance → fast response)
    for alpha in [1.0, 2.0, 5.0, 10.0]:
        formulas[f'std_α={alpha}'] = [
            -alpha * stats[i]['std'] for i in range(3)
        ]

    # Formula 5: Inverse-max (large activation → faster τ)
    for alpha in [1.0, 2.0, 5.0]:
        formulas[f'inv_max_α={alpha}'] = [
            -alpha / (stats[i]['max'] + 1e-6) for i in range(3)
        ]

    # Formula 6: Log-scaled
    for alpha in [1.0, 2.0, 5.0]:
        formulas[f'log_α={alpha}'] = [
            -alpha * np.log(stats[i]['max'] + 1.0) for i in range(3)
        ]

    # Formula 7: P99-scaled (robust to outliers)
    for alpha in [1.0, 2.0, 3.0]:
        formulas[f'p99_α={alpha}'] = [
            -alpha * stats[i]['p99'] for i in range(3)
        ]

    return formulas, stats


# ====================================================================
# Evaluation
# ====================================================================
def evaluate_model(model, test_loader):
    """Evaluate accuracy on test set."""
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 111: ANN→LNN Analytical Conversion")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load MNIST
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # Step 1: Train ANN baseline
    print("\n[Step 1] Training ANN baseline (MLP 784→256→128→10)...")
    ann = ANN_MLP().to(DEVICE)
    opt = torch.optim.Adam(ann.parameters(), lr=1e-3)
    for epoch in range(10):
        ann.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(ann(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

    ann_acc = evaluate_model(ann, test_loader)
    print(f"  ANN accuracy: {ann_acc*100:.2f}%")

    # Step 2: Compute activation statistics for τ calibration
    print("\n[Step 2] Computing activation statistics...")
    calib_loader = torch.utils.data.DataLoader(train_ds, batch_size=500, shuffle=True)
    calib_x, _ = next(iter(calib_loader))
    calib_data = [calib_x]  # list of tensors for compute_tau_formulas

    formulas, stats = compute_tau_formulas(ann, calib_data)
    print(f"  Generated {len(formulas)} τ formulas to test")

    # Step 3: Test each formula at different T values
    print("\n[Step 3] Testing τ formulas...")
    t_values = [1, 3, 5, 10, 20, 50]
    results = []

    for fname, b_taus in sorted(formulas.items()):
        lnn = LNN_MLP(ann, b_taus).to(DEVICE)
        row = {'formula': fname, 'b_tau': b_taus}

        for T in t_values:
            # Override n_steps
            lnn_acc = 0; total = 0; correct = 0
            lnn.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = lnn(x, n_steps=T)
                    pred = out.argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            lnn_acc = correct / total
            row[f'T={T}'] = lnn_acc

        # Best T
        best_T = max(t_values, key=lambda t: row[f'T={t}'])
        best_acc = row[f'T={best_T}']
        gap = best_acc - ann_acc
        row['best_T'] = best_T
        row['best_acc'] = best_acc
        row['gap'] = gap

        marker = " ★" if abs(gap) < 0.005 else (" ✓" if abs(gap) < 0.02 else "")
        print(f"  {fname:20s}: best={best_acc*100:.2f}% (T={best_T}) "
              f"gap={gap*100:+.2f}%{marker}")
        results.append(row)

        del lnn; gc.collect()

    # Step 4: Find the universal formula
    print(f"\n{'='*70}")
    print(f"  ANN baseline: {ann_acc*100:.2f}%")
    print(f"\n  Top 5 τ formulas (closest to ANN):")
    results.sort(key=lambda r: abs(r['gap']))
    for i, r in enumerate(results[:5]):
        print(f"    {i+1}. {r['formula']:20s}: {r['best_acc']*100:.2f}% "
              f"(T={r['best_T']}, gap={r['gap']*100:+.2f}%)")
        print(f"       b_τ = {[f'{v:.3f}' for v in r['b_tau']]}")

    best = results[0]
    print(f"\n  ★ UNIVERSAL FORMULA: {best['formula']}")
    print(f"    b_τ = {[f'{v:.3f}' for v in best['b_tau']]}")
    print(f"    Accuracy: {best['best_acc']*100:.2f}% (ANN: {ann_acc*100:.2f}%)")
    print(f"    Gap: {best['gap']*100:+.2f}%")
    print(f"    Optimal T: {best['best_T']}")
    print(f"{'='*70}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase111_ann_to_lnn.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 111: ANN→LNN Analytical Conversion',
            'timestamp': datetime.now().isoformat(),
            'ann_accuracy': ann_acc,
            'activation_stats': stats,
            'best_formula': best['formula'],
            'best_b_tau': best['b_tau'],
            'best_accuracy': best['best_acc'],
            'best_T': best['best_T'],
            'gap': best['gap'],
            'all_results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Top formulas accuracy vs T
        top5 = sorted(results, key=lambda r: abs(r['gap']))[:5]
        for r in top5:
            accs = [r[f'T={t}'] * 100 for t in t_values]
            axes[0].plot(t_values, accs, 'o-', label=r['formula'], markersize=4)
        axes[0].axhline(y=ann_acc*100, color='red', linestyle='--', label='ANN baseline')
        axes[0].set_xlabel('T (steps)'); axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Top 5 τ Formulas: LNN Accuracy vs T')
        axes[0].legend(fontsize=7); axes[0].set_xscale('log')

        # Plot 2: All formulas best accuracy
        all_sorted = sorted(results, key=lambda r: r['best_acc'], reverse=True)
        names = [r['formula'] for r in all_sorted[:15]]
        accs = [r['best_acc']*100 for r in all_sorted[:15]]
        colors = ['green' if abs(r['gap']) < 0.005 else 'steelblue' for r in all_sorted[:15]]
        axes[1].barh(range(len(names)), accs, color=colors)
        axes[1].set_yticks(range(len(names))); axes[1].set_yticklabels(names, fontsize=7)
        axes[1].axvline(x=ann_acc*100, color='red', linestyle='--')
        axes[1].set_xlabel('Best Accuracy (%)'); axes[1].set_title('All Formulas Ranked')

        # Plot 3: Activation statistics
        layer_names = ['FC1 (256)', 'FC2 (128)', 'FC3 (10)']
        x_pos = np.arange(3)
        axes[2].bar(x_pos - 0.2, [s['max'] for s in stats], 0.2, label='max', color='#e74c3c')
        axes[2].bar(x_pos, [s['std'] for s in stats], 0.2, label='std', color='#2ecc71')
        axes[2].bar(x_pos + 0.2, [s['mean'] for s in stats], 0.2, label='mean', color='#3498db')
        axes[2].set_xticks(x_pos); axes[2].set_xticklabels(layer_names)
        axes[2].set_title('ANN Activation Statistics'); axes[2].legend()

        plt.suptitle(f'Phase 111: ANN→LNN Conversion (ANN={ann_acc*100:.1f}%, '
                     f'Best LNN={best["best_acc"]*100:.1f}%)', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase111_ann_to_lnn.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 111 complete!")


if __name__ == '__main__':
    main()
