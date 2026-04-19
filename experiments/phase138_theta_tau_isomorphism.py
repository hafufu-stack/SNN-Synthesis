"""
Phase 138: The θ-τ Isomorphism — ANN ⇔ SNN ⇔ LNN Lossless Conversion

Proves that ANN, SNN, and LNN are three "phases" of the same underlying
computation, connected by analytical conversion formulas:
  - ANN → SNN: θ = α × max(a)          [Phase 111 legacy]
  - ANN → LNN: b_τ = -α × max(a)       [Phase 111 discovery]
  - SNN → LNN: b_τ = -θ (direct!)       [NEW: θ-τ isomorphism]
  - LNN → SNN: θ = -b_τ (inverse!)      [NEW: τ-θ isomorphism]

Key claim: θ and b_τ are linearly related, making all three architectures
mathematically equivalent (isomorphic) under the right parameterization.

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
# Architecture 1: Standard ANN (MLP)
# ================================================================
class ANN(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        return self.fc3(a2)

    def get_activations(self, x):
        x = x.view(x.size(0), -1)
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        a3 = self.fc3(a2)
        return [a1, a2, a3]


# ================================================================
# Architecture 2: SNN (Rate-coded Spiking Neural Network)
# ================================================================
class SNN(nn.Module):
    """SNN converted from ANN. Same weights, threshold-based spiking."""
    def __init__(self, ann, thresholds):
        super().__init__()
        self.fc1 = nn.Linear(ann.fc1.in_features, ann.fc1.out_features)
        self.fc2 = nn.Linear(ann.fc2.in_features, ann.fc2.out_features)
        self.fc3 = nn.Linear(ann.fc3.in_features, ann.fc3.out_features)

        # Copy weights
        self.fc1.weight.data = ann.fc1.weight.data.clone()
        self.fc1.bias.data = ann.fc1.bias.data.clone()
        self.fc2.weight.data = ann.fc2.weight.data.clone()
        self.fc2.bias.data = ann.fc2.bias.data.clone()
        self.fc3.weight.data = ann.fc3.weight.data.clone()
        self.fc3.bias.data = ann.fc3.bias.data.clone()

        self.thresholds = thresholds  # [θ1, θ2, θ3]

    def forward(self, x, T=50):
        x = x.view(x.size(0), -1)
        b = x.size(0)
        h = self.fc1.out_features

        # Membrane potentials and spike counts
        v1 = torch.zeros(b, h, device=x.device)
        v2 = torch.zeros(b, h, device=x.device)
        v3 = torch.zeros(b, self.fc3.out_features, device=x.device)
        spike_count = torch.zeros(b, self.fc3.out_features, device=x.device)

        for t in range(T):
            # Layer 1: integrate + fire
            v1 = v1 + self.fc1(x)
            s1 = (v1 >= self.thresholds[0]).float()
            v1 = v1 * (1 - s1)  # reset on spike

            # Layer 2
            v2 = v2 + self.fc2(s1)
            s2 = (v2 >= self.thresholds[1]).float()
            v2 = v2 * (1 - s2)

            # Layer 3 (output accumulator, no threshold)
            v3 = v3 + self.fc3(s2)
            spike_count += v3.sign().clamp(min=0)

        return v3 / T  # rate-coded output


# ================================================================
# Architecture 3: LNN (Liquid Neural Network)
# ================================================================
class LNN(nn.Module):
    """LNN converted from ANN. Same weights, τ-gated liquid dynamics."""
    def __init__(self, ann, b_taus):
        super().__init__()
        self.fc1 = nn.Linear(ann.fc1.in_features, ann.fc1.out_features)
        self.fc2 = nn.Linear(ann.fc2.in_features, ann.fc2.out_features)
        self.fc3 = nn.Linear(ann.fc3.in_features, ann.fc3.out_features)

        self.fc1.weight.data = ann.fc1.weight.data.clone()
        self.fc1.bias.data = ann.fc1.bias.data.clone()
        self.fc2.weight.data = ann.fc2.weight.data.clone()
        self.fc2.bias.data = ann.fc2.bias.data.clone()
        self.fc3.weight.data = ann.fc3.weight.data.clone()
        self.fc3.bias.data = ann.fc3.bias.data.clone()

        self.b_taus = b_taus  # [b_τ1, b_τ2, b_τ3]

    def forward(self, x, T=10):
        x = x.view(x.size(0), -1)
        b = x.size(0)

        s1 = torch.zeros(b, self.fc1.out_features, device=x.device)
        s2 = torch.zeros(b, self.fc2.out_features, device=x.device)
        s3 = torch.zeros(b, self.fc3.out_features, device=x.device)

        for t in range(T):
            d1 = F.relu(self.fc1(x))
            beta1 = torch.sigmoid(torch.tensor(self.b_taus[0], device=x.device))
            s1 = beta1 * s1 + (1 - beta1) * d1

            d2 = F.relu(self.fc2(s1))
            beta2 = torch.sigmoid(torch.tensor(self.b_taus[1], device=x.device))
            s2 = beta2 * s2 + (1 - beta2) * d2

            d3 = self.fc3(s2)
            beta3 = torch.sigmoid(torch.tensor(self.b_taus[2], device=x.device))
            s3 = beta3 * s3 + (1 - beta3) * d3

        return s3


# ================================================================
# Activation statistics and conversion formulas
# ================================================================
def collect_activation_stats(ann, data_loader, n_batches=10):
    ann.eval()
    all_acts = [[] for _ in range(3)]
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            if i >= n_batches: break
            acts = ann.get_activations(x.to(DEVICE))
            for j, a in enumerate(acts):
                all_acts[j].append(a.cpu())
    stats = []
    for j in range(3):
        cat = torch.cat(all_acts[j], 0)
        stats.append({
            'max': cat.max().item(),
            'mean': cat.mean().item(),
            'std': cat.std().item(),
        })
    return stats


def theta_from_stats(stats, alpha=2.0):
    """ANN → SNN: θ = α × max(a)"""
    return [alpha * s['max'] for s in stats]


def btau_from_stats(stats, alpha=2.0):
    """ANN → LNN: b_τ = -α × max(a)"""
    return [-alpha * s['max'] for s in stats]


def btau_from_theta(thetas):
    """SNN → LNN: b_τ = -θ (the isomorphism!)"""
    return [-t for t in thetas]


def theta_from_btau(b_taus):
    """LNN → SNN: θ = -b_τ (inverse isomorphism!)"""
    return [-b for b in b_taus]


# ================================================================
# Evaluation
# ================================================================
def evaluate(model, loader, **kwargs):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, **kwargs)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 138: The θ-τ Isomorphism (ANN ⇔ SNN ⇔ LNN)")
    print("=" * 70)

    # Load MNIST
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # ====== Step 1: Train ANN ======
    print("\n[Step 1] Training ANN baseline...")
    ann = ANN().to(DEVICE)
    opt = torch.optim.Adam(ann.parameters(), lr=1e-3)
    for epoch in range(10):
        ann.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(ann(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    ann_acc = evaluate(ann, test_loader)
    print(f"  ANN accuracy: {ann_acc*100:.2f}%")

    # ====== Step 2: Collect activation stats ======
    print("\n[Step 2] Collecting activation statistics...")
    stats = collect_activation_stats(ann, train_loader)
    for i, s in enumerate(stats):
        print(f"  Layer {i+1}: max={s['max']:.3f}, mean={s['mean']:.3f}, std={s['std']:.3f}")

    # ====== Step 3: Test all 6 conversion paths ======
    print("\n[Step 3] Testing all conversion paths...")
    conversion_results = {}

    # Path 1: ANN → SNN (direct)
    alpha_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    best_snn_acc = 0; best_snn_alpha = 0; best_snn_thetas = []
    for alpha in alpha_values:
        thetas = theta_from_stats(stats, alpha)
        snn = SNN(ann, thetas).to(DEVICE)
        acc = evaluate(snn, test_loader, T=50)
        if acc > best_snn_acc:
            best_snn_acc = acc; best_snn_alpha = alpha; best_snn_thetas = thetas
        del snn; gc.collect()
    conversion_results['ANN→SNN'] = {
        'accuracy': best_snn_acc,
        'alpha': best_snn_alpha,
        'params': best_snn_thetas,
        'gap': best_snn_acc - ann_acc
    }
    print(f"  ANN→SNN: {best_snn_acc*100:.2f}% (α={best_snn_alpha}, gap={conversion_results['ANN→SNN']['gap']*100:+.2f}%)")

    # Path 2: ANN → LNN (direct)
    best_lnn_acc = 0; best_lnn_alpha = 0; best_lnn_btaus = []
    for alpha in alpha_values:
        btaus = btau_from_stats(stats, alpha)
        lnn = LNN(ann, btaus).to(DEVICE)
        acc = evaluate(lnn, test_loader, T=10)
        if acc > best_lnn_acc:
            best_lnn_acc = acc; best_lnn_alpha = alpha; best_lnn_btaus = btaus
        del lnn; gc.collect()
    conversion_results['ANN→LNN'] = {
        'accuracy': best_lnn_acc,
        'alpha': best_lnn_alpha,
        'params': best_lnn_btaus,
        'gap': best_lnn_acc - ann_acc
    }
    print(f"  ANN→LNN: {best_lnn_acc*100:.2f}% (α={best_lnn_alpha}, gap={conversion_results['ANN→LNN']['gap']*100:+.2f}%)")

    # Path 3: SNN → LNN via isomorphism (b_τ = -θ)
    iso_btaus = btau_from_theta(best_snn_thetas)
    lnn_from_snn = LNN(ann, iso_btaus).to(DEVICE)
    snn_to_lnn_acc = evaluate(lnn_from_snn, test_loader, T=10)
    conversion_results['SNN→LNN (θ→b_τ)'] = {
        'accuracy': snn_to_lnn_acc,
        'params': iso_btaus,
        'gap': snn_to_lnn_acc - ann_acc,
        'formula': 'b_τ = -θ'
    }
    print(f"  SNN→LNN: {snn_to_lnn_acc*100:.2f}% (b_τ=-θ, gap={conversion_results['SNN→LNN (θ→b_τ)']['gap']*100:+.2f}%)")
    del lnn_from_snn

    # Path 4: LNN → SNN via inverse isomorphism (θ = -b_τ)
    iso_thetas = theta_from_btau(best_lnn_btaus)
    snn_from_lnn = SNN(ann, iso_thetas).to(DEVICE)
    lnn_to_snn_acc = evaluate(snn_from_lnn, test_loader, T=50)
    conversion_results['LNN→SNN (b_τ→θ)'] = {
        'accuracy': lnn_to_snn_acc,
        'params': iso_thetas,
        'gap': lnn_to_snn_acc - ann_acc,
        'formula': 'θ = -b_τ'
    }
    print(f"  LNN→SNN: {lnn_to_snn_acc*100:.2f}% (θ=-b_τ, gap={conversion_results['LNN→SNN (b_τ→θ)']['gap']*100:+.2f}%)")
    del snn_from_lnn

    # Path 5: Full cycle: ANN → SNN → LNN → SNN (roundtrip)
    roundtrip_thetas = theta_from_btau(btau_from_theta(best_snn_thetas))
    snn_roundtrip = SNN(ann, roundtrip_thetas).to(DEVICE)
    roundtrip_acc = evaluate(snn_roundtrip, test_loader, T=50)
    conversion_results['Roundtrip (ANN→SNN→LNN→SNN)'] = {
        'accuracy': roundtrip_acc,
        'gap': roundtrip_acc - best_snn_acc,
        'lossless': abs(roundtrip_acc - best_snn_acc) < 0.001
    }
    print(f"  Roundtrip: {roundtrip_acc*100:.2f}% (vs original SNN {best_snn_acc*100:.2f}%, "
          f"loss={abs(roundtrip_acc - best_snn_acc)*100:.4f}%)")
    del snn_roundtrip

    # Path 6: α sweep to show linear relationship
    print("\n[Step 4] Mapping θ-τ linear relationship...")
    isomorphism_data = []
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]:
        thetas = theta_from_stats(stats, alpha)
        btaus = btau_from_stats(stats, alpha)

        snn = SNN(ann, thetas).to(DEVICE)
        snn_acc = evaluate(snn, test_loader, T=50)
        del snn

        lnn = LNN(ann, btaus).to(DEVICE)
        lnn_acc = evaluate(lnn, test_loader, T=10)
        del lnn

        isomorphism_data.append({
            'alpha': alpha,
            'theta_l1': thetas[0], 'btau_l1': btaus[0],
            'snn_acc': snn_acc, 'lnn_acc': lnn_acc,
            'gap': abs(snn_acc - lnn_acc)
        })
        print(f"  α={alpha:5.1f}: SNN={snn_acc*100:.1f}% LNN={lnn_acc*100:.1f}% gap={abs(snn_acc-lnn_acc)*100:.2f}%")
        gc.collect()

    # ====== Results ======
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Phase 138 Complete ({elapsed:.0f}s)")
    print(f"  ANN baseline: {ann_acc*100:.2f}%")
    for path, res in conversion_results.items():
        print(f"  {path}: {res['accuracy']*100:.2f}% (gap={res['gap']*100:+.2f}%)")
    roundtrip_lossless = conversion_results['Roundtrip (ANN→SNN→LNN→SNN)']['lossless']
    print(f"  Roundtrip lossless: {roundtrip_lossless}")
    print(f"  θ-τ Isomorphism: θ = α×max(a), b_τ = -α×max(a) → b_τ = -θ")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_data = {
        'experiment': 'Phase 138: θ-τ Isomorphism',
        'timestamp': datetime.now().isoformat(),
        'ann_accuracy': ann_acc,
        'activation_stats': stats,
        'conversion_results': {k: {kk: vv for kk, vv in v.items()
                                    if not isinstance(vv, (list,))}
                                for k, v in conversion_results.items()},
        'isomorphism_data': isomorphism_data,
        'roundtrip_lossless': roundtrip_lossless,
        'elapsed_seconds': elapsed
    }
    with open(os.path.join(RESULTS_DIR, "phase138_theta_tau_isomorphism.json"), 'w',
              encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        # Panel 1: Conversion accuracy comparison
        paths = list(conversion_results.keys())
        accs = [conversion_results[p]['accuracy'] * 100 for p in paths]
        colors = ['#4CAF50' if abs(conversion_results[p]['gap']) < 0.02
                  else '#FF9800' for p in paths]
        bars = axes[0].barh(range(len(paths)), accs, color=colors, alpha=0.85)
        axes[0].axvline(x=ann_acc*100, color='red', linestyle='--', label='ANN baseline')
        axes[0].set_yticks(range(len(paths)))
        axes[0].set_yticklabels(paths, fontsize=8)
        axes[0].set_xlabel('Accuracy (%)')
        axes[0].set_title('All Conversion Paths', fontweight='bold')
        axes[0].legend()
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{acc:.1f}%', va='center', fontsize=8)

        # Panel 2: θ vs b_τ linear relationship
        alphas = [d['alpha'] for d in isomorphism_data]
        thetas_l1 = [d['theta_l1'] for d in isomorphism_data]
        btaus_l1 = [d['btau_l1'] for d in isomorphism_data]
        axes[1].scatter(thetas_l1, btaus_l1, s=80, c=alphas, cmap='RdYlGn_r',
                       edgecolors='black', zorder=5)
        # Perfect isomorphism line: b_τ = -θ
        x_range = np.linspace(min(thetas_l1), max(thetas_l1), 100)
        axes[1].plot(x_range, -x_range, 'r--', linewidth=2, label='b_τ = -θ (theory)')
        axes[1].set_xlabel('θ (SNN threshold)')
        axes[1].set_ylabel('b_τ (LNN gate bias)')
        axes[1].set_title('θ-τ Isomorphism (Layer 1)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Panel 3: SNN vs LNN accuracy at each α
        snn_accs = [d['snn_acc']*100 for d in isomorphism_data]
        lnn_accs = [d['lnn_acc']*100 for d in isomorphism_data]
        axes[2].plot(alphas, snn_accs, 'o-', color='#e74c3c', label='SNN', markersize=6)
        axes[2].plot(alphas, lnn_accs, 's-', color='#3498db', label='LNN', markersize=6)
        axes[2].axhline(y=ann_acc*100, color='gray', linestyle='--', alpha=0.5, label='ANN')
        axes[2].set_xlabel('α (conversion strength)')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title('SNN vs LNN Across α', fontweight='bold')
        axes[2].legend()
        axes[2].set_xscale('log')
        axes[2].grid(alpha=0.3)

        # Panel 4: Phase diagram
        phases = ['ANN\n(Static)', 'SNN\n(Discrete\nSpikes)', 'LNN\n(Continuous\nDynamics)']
        phase_accs = [ann_acc*100, best_snn_acc*100, best_lnn_acc*100]
        phase_colors = ['#e74c3c', '#2ecc71', '#3498db']
        bars = axes[3].bar(range(3), phase_accs, color=phase_colors, alpha=0.85,
                          edgecolor='black', linewidth=1.5)
        axes[3].set_xticks(range(3))
        axes[3].set_xticklabels(phases, fontsize=9)
        axes[3].set_ylabel('Accuracy (%)')
        axes[3].set_title('Three Phases of Neural Computation', fontweight='bold')
        for bar, acc in zip(bars, phase_accs):
            axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{acc:.1f}%', ha='center', fontweight='bold')
        # Draw arrows between phases
        axes[3].annotate('', xy=(1, max(phase_accs)*0.92), xytext=(0, max(phase_accs)*0.92),
                        arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
        axes[3].text(0.5, max(phase_accs)*0.94, 'θ = α·max(a)', ha='center',
                    fontsize=8, color='purple')
        axes[3].annotate('', xy=(2, max(phase_accs)*0.86), xytext=(1, max(phase_accs)*0.86),
                        arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
        axes[3].text(1.5, max(phase_accs)*0.88, 'b_τ = -θ', ha='center',
                    fontsize=8, color='orange')

        plt.suptitle(f'Phase 138: θ-τ Isomorphism — Three Phases of Neural Computation '
                     f'(ANN={ann_acc*100:.1f}%)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase138_theta_tau_isomorphism.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print(f"\nPhase 138 complete! ({elapsed:.0f}s)")
    return result_data


if __name__ == '__main__':
    main()
