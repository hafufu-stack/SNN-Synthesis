"""
Phase 72: Liquid-LIF Cell - Full-Scratch Implementation and Benchmark

Creates a new neural architecture: Liquid Leaky Integrate-and-Fire (Liquid-LIF).
- SNN's spike-based binary communication (energy efficient)
- LNN's input-dependent dynamic time constant tau (temporal adaptivity)
- Surrogate gradient (ATan) for backpropagation through spikes

Benchmark: Sequential MNIST (28 steps x 28 features)
Compare: Standard LIF, GRU, Liquid-LIF
Metrics: Accuracy, spike sparsity, training speed

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

# Hyperparameters
HIDDEN_DIM = 128
N_CLASSES = 10
SEQ_LEN = 28       # rows of MNIST image
INPUT_DIM = 28      # columns (pixels per row)
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3


# ==============================================================
# Surrogate Gradient Functions
# ==============================================================
class ATanSurrogate(torch.autograd.Function):
    """ATan surrogate gradient for spike generation."""
    alpha = 2.0

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = ATanSurrogate.alpha / 2 / (1 + (np.pi / 2 * ATanSurrogate.alpha * input).pow(2))
        return grad_output * grad


spike_fn = ATanSurrogate.apply


# ==============================================================
# Standard LIF Cell (baseline SNN)
# ==============================================================
class LIFCell(nn.Module):
    """Standard Leaky Integrate-and-Fire neuron with FIXED time constant."""
    def __init__(self, input_dim, hidden_dim, beta=0.9, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta          # Fixed decay (exp(-dt/tau))
        self.threshold = threshold
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, state=None):
        """x: (batch, input_dim), state: (mem,) or None"""
        if state is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            mem = state

        # Leaky integration
        mem = self.beta * mem + self.fc(x)

        # Spike generation
        spk = spike_fn(mem - self.threshold)

        # Reset by subtraction
        mem = mem - spk * self.threshold

        return spk, mem


# ==============================================================
# Liquid-LIF Cell (NEW ARCHITECTURE)
# ==============================================================
class LiquidLIFCell(nn.Module):
    """Liquid Leaky Integrate-and-Fire neuron.
    
    The key innovation: tau (time constant) is NOT fixed but dynamically
    computed from input and membrane state, like a Liquid Neural Network.
    
    beta = sigmoid(W_tau @ [x, mem] + b_tau)  <-- LIQUID gate
    mem = beta * mem + W_in @ x               <-- LIF integration  
    spk = Heaviside(mem - threshold)           <-- Spike
    mem = mem - spk * threshold                <-- Reset
    """
    def __init__(self, input_dim, hidden_dim, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        # Input projection
        self.fc_in = nn.Linear(input_dim, hidden_dim)

        # Liquid tau gate: computes dynamic beta from [x, mem]
        self.fc_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.b_tau = nn.Parameter(torch.ones(hidden_dim) * 1.5)  # Init ~sigmoid(1.5)=0.82

    def forward(self, x, state=None):
        """x: (batch, input_dim), state: (mem,) or None"""
        if state is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            mem = state

        # LIQUID GATE: dynamic time constant
        tau_input = torch.cat([x, mem], dim=-1)
        beta = torch.sigmoid(self.fc_tau(tau_input) + self.b_tau)
        beta = torch.clamp(beta, 0.01, 0.99)

        # Leaky integration with DYNAMIC beta
        mem = beta * mem + self.fc_in(x)

        # Spike generation
        spk = spike_fn(mem - self.threshold)

        # Reset by subtraction
        mem = mem - spk * self.threshold

        return spk, mem


# ==============================================================
# Network Wrappers
# ==============================================================
class SNNClassifier(nn.Module):
    """Wraps a recurrent cell (LIF or LiquidLIF) into a classifier."""
    def __init__(self, cell, hidden_dim, n_classes):
        super().__init__()
        self.cell = cell
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """x: (batch, seq_len, input_dim)"""
        batch_size, seq_len, _ = x.shape
        state = None
        spike_count = 0
        total_neurons = 0

        for t in range(seq_len):
            spk, state = self.cell(x[:, t, :], state)
            spike_count += spk.sum().item()
            total_neurons += spk.numel()

        # Readout from final membrane state (not spikes, for stability)
        out = self.readout(state)
        sparsity = 1.0 - (spike_count / max(1, total_neurons))
        return out, sparsity


class GRUClassifier(nn.Module):
    """GRU baseline."""
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """x: (batch, seq_len, input_dim)"""
        out, _ = self.gru(x)
        out = self.readout(out[:, -1, :])
        return out, 0.0  # GRU has no sparsity


# ==============================================================
# Training and Evaluation
# ==============================================================
def get_sequential_mnist(batch_size):
    """Load MNIST as sequential data (28 steps x 28 features)."""
    from torchvision import datasets, transforms
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "mnist")
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    try:
        train_ds = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
        test_ds = datasets.MNIST(data_dir, train=False, download=False, transform=transform)
    except RuntimeError:
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_sparsity = 0
    n_batches = 0

    for images, labels in train_loader:
        # Reshape: (batch, 1, 28, 28) -> (batch, 28, 28) = (batch, seq, features)
        x = images.squeeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, sparsity = model(x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += x.size(0)
        total_sparsity += sparsity
        n_batches += 1

    return (total_loss / total_samples,
            total_correct / total_samples,
            total_sparsity / max(1, n_batches))


def evaluate(model, test_loader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    total_sparsity = 0
    n_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            logits, sparsity = model(x)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += x.size(0)
            total_sparsity += sparsity
            n_batches += 1

    return total_correct / total_samples, total_sparsity / max(1, n_batches)


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase72_liquid_lif.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 72: Liquid-LIF Cell',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 72: Liquid-LIF Cell - New Architecture Benchmark")
    print("  Task: Sequential MNIST (28 steps x 28 features)")
    print(f"  Device: {DEVICE}, Epochs: {EPOCHS}, Hidden: {HIDDEN_DIM}")
    print("=" * 70)

    train_loader, test_loader = get_sequential_mnist(BATCH_SIZE)
    print(f"  Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}\n")

    MODELS = {
        'LIF': lambda: SNNClassifier(
            LIFCell(INPUT_DIM, HIDDEN_DIM, beta=0.9), HIDDEN_DIM, N_CLASSES),
        'GRU': lambda: GRUClassifier(INPUT_DIM, HIDDEN_DIM, N_CLASSES),
        'LiquidLIF': lambda: SNNClassifier(
            LiquidLIFCell(INPUT_DIM, HIDDEN_DIM), HIDDEN_DIM, N_CLASSES),
    }

    all_results = {}

    for model_name, model_fn in MODELS.items():
        print(f"\n  === {model_name} ===")
        model = model_fn().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        history = []
        best_acc = 0
        t_start = time.time()

        for epoch in range(EPOCHS):
            train_loss, train_acc, train_sparsity = train_one_epoch(
                model, train_loader, optimizer, DEVICE)
            test_acc, test_sparsity = evaluate(model, test_loader, DEVICE)
            scheduler.step()

            if test_acc > best_acc:
                best_acc = test_acc

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'sparsity': test_sparsity,
            })

            print(f"  Epoch {epoch+1:2d}: train_acc={train_acc*100:.1f}%  "
                  f"test_acc={test_acc*100:.1f}%  sparsity={test_sparsity:.3f}  "
                  f"loss={train_loss:.4f}")

        elapsed = time.time() - t_start
        all_results[model_name] = {
            'best_test_acc': best_acc,
            'final_test_acc': history[-1]['test_acc'],
            'final_sparsity': history[-1]['sparsity'],
            'n_params': n_params,
            'train_time_sec': elapsed,
            'history': history,
        }
        print(f"  Best: {best_acc*100:.1f}%, Time: {elapsed:.0f}s, Params: {n_params:,}")

        del model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _save(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Sequential MNIST Benchmark")
    print(f"{'='*70}")
    print(f"  {'Model':<15s} {'Best Acc':>10s} {'Sparsity':>10s} {'Params':>10s} {'Time':>8s}")
    for name, r in all_results.items():
        print(f"  {name:<15s} {r['best_test_acc']*100:>9.1f}% {r['final_sparsity']:>9.3f} "
              f"{r['n_params']:>10,} {r['train_time_sec']:>7.0f}s")

    _generate_figure(all_results)
    print("\nPhase 72 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = {'LIF': '#3B82F6', 'GRU': '#9CA3AF', 'LiquidLIF': '#EC4899'}

        # 1) Accuracy over epochs
        ax = axes[0]
        for name, r in results.items():
            epochs = [h['epoch'] for h in r['history']]
            accs = [h['test_acc'] * 100 for h in r['history']]
            ax.plot(epochs, accs, '-o', color=colors.get(name, '#333'), label=name, markersize=4)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Learning Curves', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)

        # 2) Best accuracy comparison
        ax = axes[1]
        names = list(results.keys())
        accs = [results[n]['best_test_acc'] * 100 for n in names]
        bars = ax.bar(names, accs, color=[colors.get(n, '#333') for n in names],
                     edgecolor='white', linewidth=1.5)
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width()/2, a + 0.3, f'{a:.1f}%',
                   ha='center', fontweight='bold', fontsize=11)
        ax.set_ylabel('Best Test Accuracy (%)')
        ax.set_title('Peak Performance', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 3) Sparsity vs Accuracy (SNN advantage)
        ax = axes[2]
        for name, r in results.items():
            ax.scatter(r['final_sparsity'] * 100, r['best_test_acc'] * 100,
                      c=colors.get(name, '#333'), s=150, label=name,
                      edgecolors='white', linewidth=1.5, zorder=5)
        ax.set_xlabel('Spike Sparsity (%)')
        ax.set_ylabel('Best Test Accuracy (%)')
        ax.set_title('Efficiency Frontier', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)

        fig.suptitle('Phase 72: Liquid-LIF Cell - Sequential MNIST Benchmark',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase72_liquid_lif.png"),
                   bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
