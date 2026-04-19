"""
Phase 139: Liquid Spiking Neural Networks (LSNN) — A New Architecture

Creates a novel hybrid neuron combining:
  - Internal state: LNN continuous ODE dynamics (perfect gradient flow)
  - Output: SNN discrete spikes with surrogate gradient (clean discretization)

This is the biological solution to the VQ Paradox:
  - VQ (v11) failed because STE kills gradients over iterative NCA steps
  - Surrogate gradients (from SNN literature) solve this exact problem
  - "Continuous Thought, Discrete Action" implemented at the NEURON level

Architecture comparison:
  ANN:  continuous in → continuous out    (blurry outputs)
  SNN:  discrete in → discrete out        (poor gradients)
  LNN:  continuous in → continuous out     (temporal dynamics, still blurry)
  LSNN: continuous in → DISCRETE out      (best of both worlds!)

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
# Surrogate Gradient Functions
# ================================================================
class SurrogateSpike(torch.autograd.Function):
    """Spike function with sigmoid surrogate gradient."""
    scale = 25.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sigmoid = torch.sigmoid(SurrogateSpike.scale * x)
        return grad_output * sigmoid * (1 - sigmoid) * SurrogateSpike.scale

spike_fn = SurrogateSpike.apply


# ================================================================
# Architecture: LSNN Cell (Liquid Spiking Neural Network)
# ================================================================
class LSNNCell(nn.Module):
    """
    Single LSNN layer:
      - Membrane potential evolves via LNN ODE: dv/dt = -v/τ + I(t)
      - Output is discrete spike: s = H(v - θ), reset on spike
      - Gradient flows through surrogate gradient (not STE!)
    """
    def __init__(self, in_features, out_features, tau_init=2.0, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        # Learnable time constant (LNN component)
        self.log_tau = nn.Parameter(torch.ones(out_features) * np.log(tau_init))
        self.threshold = threshold

    @property
    def tau(self):
        return self.log_tau.exp().clamp(min=0.1, max=100.0)

    def forward(self, x, v_prev):
        """
        x: input (batch, in_features)
        v_prev: membrane potential from previous step (batch, out_features)
        Returns: (spike, v_new)
        """
        # LNN dynamics: v(t+1) = v(t) * decay + input_current
        decay = torch.exp(-1.0 / self.tau).unsqueeze(0)  # (1, out)
        current = self.fc(x)

        # Continuous state update (LNN-like)
        v_new = decay * v_prev + (1 - decay) * current

        # Discrete spike output (SNN-like) with surrogate gradient
        spike = spike_fn(v_new - self.threshold)

        # Soft reset (subtract threshold on spike)
        v_new = v_new - spike * self.threshold

        return spike, v_new


# ================================================================
# Full LSNN Network
# ================================================================
class LSNN(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10, T=20):
        super().__init__()
        self.T = T
        self.cell1 = LSNNCell(in_dim, hidden)
        self.cell2 = LSNNCell(hidden, hidden)
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        b = x.size(0)

        v1 = torch.zeros(b, self.cell1.fc.out_features, device=x.device)
        v2 = torch.zeros(b, self.cell2.fc.out_features, device=x.device)
        spike_sum = torch.zeros(b, self.cell2.fc.out_features, device=x.device)

        for t in range(self.T):
            s1, v1 = self.cell1(x, v1)
            s2, v2 = self.cell2(s1, v2)
            spike_sum += s2

        # Rate-coded readout
        return self.readout(spike_sum / self.T)


# ================================================================
# Comparison models
# ================================================================
class PureANN(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class PureSNN(nn.Module):
    """Standard SNN with surrogate gradients (no liquid dynamics)."""
    def __init__(self, in_dim=784, hidden=128, out_dim=10, T=20, threshold=1.0):
        super().__init__()
        self.T = T; self.threshold = threshold
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        b = x.size(0)
        v1 = torch.zeros(b, 128, device=x.device)
        v2 = torch.zeros(b, 128, device=x.device)
        spk_sum = torch.zeros(b, 128, device=x.device)

        for t in range(self.T):
            v1 = 0.9 * v1 + self.fc1(x)
            s1 = spike_fn(v1 - self.threshold)
            v1 = v1 - s1 * self.threshold

            v2 = 0.9 * v2 + self.fc2(s1)
            s2 = spike_fn(v2 - self.threshold)
            v2 = v2 - s2 * self.threshold
            spk_sum += s2

        return self.readout(spk_sum / self.T)


class PureLNN(nn.Module):
    """Pure LNN (continuous, no spikes)."""
    def __init__(self, in_dim=784, hidden=128, out_dim=10, T=10):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.log_tau1 = nn.Parameter(torch.ones(hidden) * np.log(2.0))
        self.log_tau2 = nn.Parameter(torch.ones(hidden) * np.log(2.0))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        b = x.size(0)
        s1 = torch.zeros(b, 128, device=x.device)
        s2 = torch.zeros(b, 128, device=x.device)

        for t in range(self.T):
            d1 = torch.exp(-1.0 / self.log_tau1.exp().clamp(0.1, 100))
            s1 = d1 * s1 + (1-d1) * F.relu(self.fc1(x))
            d2 = torch.exp(-1.0 / self.log_tau2.exp().clamp(0.1, 100))
            s2 = d2 * s2 + (1-d2) * F.relu(self.fc2(s1))

        return self.fc3(s2)


# ================================================================
# Training loop
# ================================================================
def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0; n_batch = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batch += 1
        train_losses.append(total_loss / n_batch)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_accs.append(correct / total)

    return train_losses, test_accs


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 139: Liquid Spiking Neural Networks (LSNN)")
    print("  'Continuous Thought, Discrete Action' at the neuron level")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    EPOCHS = 10
    results = {}

    # ====== Train all 4 architectures ======
    architectures = {
        'ANN': PureANN().to(DEVICE),
        'SNN': PureSNN(T=20).to(DEVICE),
        'LNN': PureLNN(T=10).to(DEVICE),
        'LSNN': LSNN(T=20).to(DEVICE),
    }

    all_histories = {}
    for name, model in architectures.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] Training ({n_params:,} params)...")
        losses, accs = train_model(model, train_loader, test_loader, epochs=EPOCHS)
        results[name] = {
            'final_accuracy': accs[-1],
            'best_accuracy': max(accs),
            'n_params': n_params,
            'final_loss': losses[-1],
        }
        all_histories[name] = {'losses': losses, 'accs': accs}
        print(f"  Final: {accs[-1]*100:.2f}%, Best: {max(accs)*100:.2f}%")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ====== Test LSNN τ dynamics ======
    print("\n[Analysis] LSNN learned τ values:")
    lsnn = architectures['LSNN']
    for i, cell in enumerate([lsnn.cell1, lsnn.cell2]):
        tau = cell.tau.detach().cpu()
        print(f"  Layer {i+1}: τ mean={tau.mean():.3f}, std={tau.std():.3f}, "
              f"min={tau.min():.3f}, max={tau.max():.3f}")

    # ====== Summary ======
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 139 Results ({elapsed:.0f}s):")
    print(f"{'Architecture':<12} {'Accuracy':>10} {'Params':>10} {'Type':>25}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*25}")
    for name in ['ANN', 'SNN', 'LNN', 'LSNN']:
        r = results[name]
        types = {'ANN': 'static/continuous',
                 'SNN': 'temporal/discrete',
                 'LNN': 'temporal/continuous',
                 'LSNN': 'temporal/continuous+discrete'}
        print(f"  {name:<10} {r['final_accuracy']*100:>8.2f}% {r['n_params']:>9,} {types[name]:>25}")

    lsnn_vs_ann = results['LSNN']['final_accuracy'] - results['ANN']['final_accuracy']
    lsnn_vs_snn = results['LSNN']['final_accuracy'] - results['SNN']['final_accuracy']
    print(f"\n  LSNN vs ANN: {lsnn_vs_ann*100:+.2f}%")
    print(f"  LSNN vs SNN: {lsnn_vs_snn*100:+.2f}%")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase139_lsnn.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 139: Liquid Spiking Neural Networks',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'histories': {k: {'accs': v['accs'], 'losses': v['losses']}
                         for k, v in all_histories.items()},
            'lsnn_tau': {
                'layer1': {'mean': lsnn.cell1.tau.mean().item(),
                           'std': lsnn.cell1.tau.std().item()},
                'layer2': {'mean': lsnn.cell2.tau.mean().item(),
                           'std': lsnn.cell2.tau.std().item()},
            },
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        # Panel 1: Training curves
        for name, hist in all_histories.items():
            axes[0].plot(range(1, EPOCHS+1), [a*100 for a in hist['accs']],
                        'o-', label=name, markersize=4)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('Training Progress', fontweight='bold')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        # Panel 2: Final accuracy comparison
        names = ['ANN', 'SNN', 'LNN', 'LSNN']
        accs = [results[n]['final_accuracy']*100 for n in names]
        colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
        bars = axes[1].bar(range(4), accs, color=colors, alpha=0.85,
                          edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(range(4))
        axes[1].set_xticklabels(['ANN\n(Static)', 'SNN\n(Spike)', 'LNN\n(Liquid)', 'LSNN\n(Hybrid)'])
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Architecture Comparison', fontweight='bold')
        for bar, acc in zip(bars, accs):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{acc:.1f}%', ha='center', fontweight='bold')

        # Panel 3: LSNN τ distribution
        tau1 = lsnn.cell1.tau.detach().cpu().numpy()
        tau2 = lsnn.cell2.tau.detach().cpu().numpy()
        axes[2].hist(tau1, bins=30, alpha=0.6, label='Layer 1', color='#e74c3c')
        axes[2].hist(tau2, bins=30, alpha=0.6, label='Layer 2', color='#3498db')
        axes[2].set_xlabel('τ (time constant)'); axes[2].set_ylabel('Count')
        axes[2].set_title('LSNN Learned τ Distribution', fontweight='bold')
        axes[2].legend()

        # Panel 4: Architecture taxonomy
        arch_data = {
            'ANN': (0, 0), 'SNN': (1, 1), 'LNN': (0, 1), 'LSNN': (1, 1)
        }
        # Discrete output vs Temporal dynamics
        for name in names:
            x_val = 1 if name in ['SNN', 'LSNN'] else 0
            y_val = 1 if name in ['SNN', 'LNN', 'LSNN'] else 0
            color = colors[names.index(name)]
            acc = results[name]['final_accuracy']*100
            axes[3].scatter(x_val, y_val, s=acc*8, c=color, alpha=0.7,
                           edgecolors='black', linewidth=2, zorder=5)
            offset = 0.08 if name != 'SNN' else -0.08
            axes[3].annotate(f'{name}\n{acc:.1f}%', (x_val, y_val+offset),
                           ha='center', fontsize=9, fontweight='bold')

        axes[3].set_xticks([0, 1])
        axes[3].set_xticklabels(['Continuous\nOutput', 'Discrete\nOutput (Spike)'])
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(['Static', 'Temporal\nDynamics'])
        axes[3].set_title('Architecture Phase Space', fontweight='bold')
        axes[3].set_xlim(-0.5, 1.5); axes[3].set_ylim(-0.5, 1.5)
        axes[3].grid(alpha=0.3)

        plt.suptitle('Phase 139: LSNN — Continuous Thought, Discrete Action at the Neuron Level',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase139_lsnn.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print(f"\nPhase 139 complete! ({elapsed:.0f}s)")
    return results


if __name__ == '__main__':
    main()
