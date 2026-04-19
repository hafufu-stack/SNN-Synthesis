"""
Phase 140: The Universal Neural Compiler

Train ONE ANN, then compile it into 4 different architectures at inference time:
  - ANN mode:  Static feedforward (fastest)
  - SNN mode:  Discrete spikes via θ = α × max(a)
  - LNN mode:  Continuous liquid dynamics via b_τ = -α × max(a)
  - LSNN mode: Hybrid liquid+spike via θ-τ isomorphism

All modes use the SAME learned weights — zero retraining.
This is the "Universal Phase Transition Compiler" for neural networks.

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
# Surrogate gradient for spike function
# ================================================================
class SurrogateSpike(torch.autograd.Function):
    scale = 25.0
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sig = torch.sigmoid(SurrogateSpike.scale * x)
        return grad_output * sig * (1 - sig) * SurrogateSpike.scale

spike_fn = SurrogateSpike.apply


# ================================================================
# The Universal Neural Compiler
# ================================================================
class UniversalNeuralCompiler(nn.Module):
    """
    A single model that can run in 4 different modes using the same weights.
    """
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

        # Activation stats (computed after training)
        self._max_acts = None
        self._alpha_snn = 2.0
        self._alpha_lnn = 3.0

    def _compute_activation_stats(self, data_loader):
        """Calibrate conversion parameters from training data."""
        self.eval()
        all_max = [0.0, 0.0, 0.0]
        with torch.no_grad():
            for i, (x, _) in enumerate(data_loader):
                if i >= 20: break
                x = x.to(DEVICE).view(x.size(0), -1)
                a1 = F.relu(self.fc1(x))
                a2 = F.relu(self.fc2(a1))
                a3 = self.fc3(a2)
                for j, a in enumerate([a1, a2, a3]):
                    m = a.max().item()
                    if m > all_max[j]: all_max[j] = m
        self._max_acts = all_max
        print(f"  Calibrated max activations: {[f'{m:.2f}' for m in all_max]}")

    def compile(self, mode='ann'):
        """Set the inference mode. Returns self for chaining."""
        assert mode in ['ann', 'snn', 'lnn', 'lsnn']
        self._mode = mode
        return self

    def forward(self, x, T=None):
        mode = getattr(self, '_mode', 'ann')
        if mode == 'ann':
            return self._forward_ann(x)
        elif mode == 'snn':
            return self._forward_snn(x, T or 50)
        elif mode == 'lnn':
            return self._forward_lnn(x, T or 10)
        elif mode == 'lsnn':
            return self._forward_lsnn(x, T or 20)

    def _forward_ann(self, x):
        """Standard feedforward — no temporal dynamics."""
        x = x.view(x.size(0), -1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def _forward_snn(self, x, T):
        """SNN mode: threshold-based spiking, θ = α × max(a)."""
        x = x.view(x.size(0), -1)
        b = x.size(0)
        thresholds = [self._alpha_snn * m for m in self._max_acts]

        v1 = torch.zeros(b, 128, device=x.device)
        v2 = torch.zeros(b, 128, device=x.device)
        v3 = torch.zeros(b, 10, device=x.device)

        for t in range(T):
            v1 = v1 + self.fc1(x)
            s1 = (v1 >= thresholds[0]).float()
            v1 = v1 * (1 - s1)

            v2 = v2 + self.fc2(s1)
            s2 = (v2 >= thresholds[1]).float()
            v2 = v2 * (1 - s2)

            v3 = v3 + self.fc3(s2)

        return v3 / T

    def _forward_lnn(self, x, T):
        """LNN mode: liquid τ-gated dynamics, b_τ = -α × max(a)."""
        x = x.view(x.size(0), -1)
        b = x.size(0)
        b_taus = [-self._alpha_lnn * m for m in self._max_acts]

        s1 = torch.zeros(b, 128, device=x.device)
        s2 = torch.zeros(b, 128, device=x.device)
        s3 = torch.zeros(b, 10, device=x.device)

        for t in range(T):
            beta1 = torch.sigmoid(torch.tensor(b_taus[0], device=x.device))
            s1 = beta1 * s1 + (1 - beta1) * F.relu(self.fc1(x))

            beta2 = torch.sigmoid(torch.tensor(b_taus[1], device=x.device))
            s2 = beta2 * s2 + (1 - beta2) * F.relu(self.fc2(s1))

            beta3 = torch.sigmoid(torch.tensor(b_taus[2], device=x.device))
            s3 = beta3 * s3 + (1 - beta3) * self.fc3(s2)

        return s3

    def _forward_lsnn(self, x, T):
        """LSNN mode: liquid internals + discrete spike output."""
        x = x.view(x.size(0), -1)
        b = x.size(0)

        # Use θ-τ isomorphism: same α for both threshold and τ
        alpha = (self._alpha_snn + self._alpha_lnn) / 2
        thresholds = [alpha * m for m in self._max_acts]
        decay_rates = [torch.exp(torch.tensor(-1.0 / max(alpha * m, 0.1)))
                      for m in self._max_acts]

        v1 = torch.zeros(b, 128, device=x.device)
        v2 = torch.zeros(b, 128, device=x.device)
        spike_accum = torch.zeros(b, 128, device=x.device)

        for t in range(T):
            # Layer 1: LNN dynamics + spike output
            v1 = decay_rates[0].to(x.device) * v1 + (1 - decay_rates[0].to(x.device)) * self.fc1(x)
            s1 = spike_fn(v1 - thresholds[0])
            v1 = v1 - s1 * thresholds[0]  # reset

            # Layer 2: LNN dynamics + spike output
            v2 = decay_rates[1].to(x.device) * v2 + (1 - decay_rates[1].to(x.device)) * self.fc2(s1)
            s2 = spike_fn(v2 - thresholds[1])
            v2 = v2 - s2 * thresholds[1]
            spike_accum += s2

        return self.fc3(spike_accum / T)


# ================================================================
# Training and evaluation
# ================================================================
def train_ann(model, train_loader, epochs=10, lr=1e-3):
    model.compile('ann')  # Train in ANN mode
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()


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
    print("Phase 140: The Universal Neural Compiler")
    print("  One model, four architectures, zero retraining")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # ====== Step 1: Train single ANN ======
    print("\n[Step 1] Training universal model in ANN mode...")
    compiler = UniversalNeuralCompiler().to(DEVICE)
    n_params = sum(p.numel() for p in compiler.parameters())
    print(f"  Parameters: {n_params:,}")
    train_ann(compiler, train_loader, epochs=10)

    # ====== Step 2: Calibrate ======
    print("\n[Step 2] Calibrating activation statistics...")
    compiler._compute_activation_stats(train_loader)

    # ====== Step 3: Compile and test all 4 modes ======
    print("\n[Step 3] Testing all compilation modes...")
    modes = ['ann', 'snn', 'lnn', 'lsnn']
    mode_labels = {
        'ann': 'ANN (Static Feedforward)',
        'snn': 'SNN (Discrete Spikes)',
        'lnn': 'LNN (Liquid Dynamics)',
        'lsnn': 'LSNN (Liquid + Spikes)'
    }
    results = {}

    for mode in modes:
        compiler.compile(mode)
        t_start = time.time()
        acc = evaluate(compiler, test_loader)
        latency = (time.time() - t_start) * 1000  # ms
        results[mode] = {
            'accuracy': acc,
            'latency_ms': latency,
            'label': mode_labels[mode]
        }
        print(f"  {mode_labels[mode]:35s}: {acc*100:.2f}%  ({latency:.0f}ms)")

    # ====== Step 4: α sweep ======
    print("\n[Step 4] α sensitivity analysis...")
    alpha_sweep = {}
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0]:
        compiler._alpha_snn = alpha
        compiler._alpha_lnn = alpha
        compiler.compile('snn')
        snn_acc = evaluate(compiler, test_loader)
        compiler.compile('lnn')
        lnn_acc = evaluate(compiler, test_loader)
        compiler.compile('lsnn')
        lsnn_acc = evaluate(compiler, test_loader)
        alpha_sweep[alpha] = {'snn': snn_acc, 'lnn': lnn_acc, 'lsnn': lsnn_acc}
        print(f"  α={alpha:5.1f}: SNN={snn_acc*100:.1f}% LNN={lnn_acc*100:.1f}% LSNN={lsnn_acc*100:.1f}%")

    # Restore defaults
    compiler._alpha_snn = 2.0
    compiler._alpha_lnn = 3.0

    # ====== Summary ======
    elapsed = time.time() - t0
    ann_acc = results['ann']['accuracy']
    print(f"\n{'='*70}")
    print(f"Phase 140: Universal Neural Compiler ({elapsed:.0f}s)")
    print(f"  Trained ONCE in ANN mode, compiled to 4 architectures:")
    for mode in modes:
        r = results[mode]
        gap = r['accuracy'] - ann_acc
        print(f"    {r['label']:35s}: {r['accuracy']*100:.2f}% (gap={gap*100:+.2f}%)")
    print(f"\n  Key insight: All architectures share the SAME weights.")
    print(f"  The θ-τ isomorphism enables zero-shot mode switching.")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase140_universal_compiler.json"), 'w',
              encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 140: Universal Neural Compiler',
            'timestamp': datetime.now().isoformat(),
            'n_params': n_params,
            'results': results,
            'alpha_sweep': {str(k): v for k, v in alpha_sweep.items()},
            'max_activations': compiler._max_acts,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        # Panel 1: Accuracy comparison
        mode_names = ['ANN\n(Static)', 'SNN\n(Spike)', 'LNN\n(Liquid)', 'LSNN\n(Hybrid)']
        accs = [results[m]['accuracy']*100 for m in modes]
        colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
        bars = axes[0].bar(range(4), accs, color=colors, alpha=0.85,
                          edgecolor='black', linewidth=1.5)
        axes[0].set_xticks(range(4)); axes[0].set_xticklabels(mode_names)
        axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Same Weights, 4 Modes', fontweight='bold')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=10)

        # Panel 2: Latency comparison
        lats = [results[m]['latency_ms'] for m in modes]
        axes[1].bar(range(4), lats, color=colors, alpha=0.85, edgecolor='black')
        axes[1].set_xticks(range(4)); axes[1].set_xticklabels(mode_names)
        axes[1].set_ylabel('Inference Time (ms)'); axes[1].set_title('Latency Comparison', fontweight='bold')

        # Panel 3: α sweep
        alphas = sorted(alpha_sweep.keys())
        for arch, color, label in [('snn', '#2ecc71', 'SNN'),
                                    ('lnn', '#3498db', 'LNN'),
                                    ('lsnn', '#9b59b6', 'LSNN')]:
            vals = [alpha_sweep[a][arch]*100 for a in alphas]
            axes[2].plot(alphas, vals, 'o-', color=color, label=label, markersize=5)
        axes[2].axhline(y=ann_acc*100, color='red', linestyle='--', label='ANN', alpha=0.7)
        axes[2].set_xlabel('α (conversion strength)'); axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title('α Sensitivity', fontweight='bold')
        axes[2].legend(); axes[2].set_xscale('log'); axes[2].grid(alpha=0.3)

        # Panel 4: Compiler pipeline diagram
        axes[3].set_xlim(0, 10); axes[3].set_ylim(0, 10)
        axes[3].axis('off')
        axes[3].set_title('Universal Neural Compiler', fontweight='bold', fontsize=12)

        # Central ANN box
        axes[3].add_patch(plt.Rectangle((3.5, 4), 3, 2, linewidth=2,
                         edgecolor='black', facecolor='#f1c40f', alpha=0.7))
        axes[3].text(5, 5, f'Trained ANN\n{n_params:,} params\n{ann_acc*100:.1f}%',
                    ha='center', va='center', fontsize=9, fontweight='bold')

        # Output modes
        mode_positions = [(0.5, 8.5), (3.5, 8.5), (6.5, 8.5), (3.5, 0.5)]
        mode_colors_map = {'ann': '#e74c3c', 'snn': '#2ecc71', 'lnn': '#3498db', 'lsnn': '#9b59b6'}
        for (mx, my), mode, label in zip(mode_positions, modes, mode_names):
            c = mode_colors_map[mode]
            axes[3].add_patch(plt.Rectangle((mx, my), 2.5, 1.5, linewidth=1.5,
                             edgecolor='black', facecolor=c, alpha=0.5))
            acc = results[mode]['accuracy']*100
            axes[3].text(mx+1.25, my+0.75, f'{label}\n{acc:.1f}%',
                        ha='center', va='center', fontsize=8, fontweight='bold')
            # Arrow from center
            axes[3].annotate('', xy=(mx+1.25, my+0.1 if my > 5 else my+1.4),
                           xytext=(5, 6 if my > 5 else 4),
                           arrowprops=dict(arrowstyle='->', color=c, lw=2))

        plt.suptitle('Phase 140: The Universal Neural Compiler — One Training, Four Architectures',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase140_universal_compiler.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print(f"\nPhase 140 complete! ({elapsed:.0f}s)")
    return results


if __name__ == '__main__':
    main()
