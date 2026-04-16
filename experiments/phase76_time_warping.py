"""
Phase 76: Zero-Shot Time-Warping

Tests whether Liquid-LIF can generalize to unseen time scales
WITHOUT retraining (zero-shot), while GRU and standard LIF fail.

Setup:
  - Train on normal Sequential MNIST (28 steps x 28 features)
  - Test at multiple time warpings:
    - 0.5x (compress: 14 steps, average consecutive rows)
    - 1.0x (normal: 28 steps - baseline)
    - 2.0x (stretch: 56 steps, repeat each row)
    - 3.0x (stretch: 84 steps, repeat each row 3x)

Models: LIF (fixed tau), GRU, Liquid-LIF (dynamic tau)
Expected: Liquid-LIF maintains accuracy under time warping.

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

HIDDEN_DIM = 16
N_CLASSES = 10
INPUT_DIM = 28
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
WARP_FACTORS = [0.5, 1.0, 2.0, 3.0]


class ATanSurrogate(torch.autograd.Function):
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


class LIFCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.9, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.threshold = threshold
        self.fc = nn.Linear(input_dim, hidden_dim)
    def forward(self, x, state=None):
        if state is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            mem = state
        mem = self.beta * mem + self.fc(x)
        spk = spike_fn(mem - self.threshold)
        mem = mem - spk * self.threshold
        return spk, mem


class LiquidLIFCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.b_tau = nn.Parameter(torch.ones(hidden_dim) * 1.5)
    def forward(self, x, state=None):
        if state is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            mem = state
        tau_input = torch.cat([x, mem], dim=-1)
        beta = torch.sigmoid(self.fc_tau(tau_input) + self.b_tau)
        beta = torch.clamp(beta, 0.01, 0.99)
        mem = beta * mem + self.fc_in(x)
        spk = spike_fn(mem - self.threshold)
        mem = mem - spk * self.threshold
        return spk, mem


class SNNClassifier(nn.Module):
    def __init__(self, cell, hidden_dim, n_classes):
        super().__init__()
        self.cell = cell
        self.readout = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        state = None
        for t in range(seq_len):
            _, state = self.cell(x[:, t, :], state)
        return self.readout(state)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.readout(out[:, -1, :])


def get_mnist():
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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def time_warp(x, factor):
    """Warp the time axis of a sequence.
    x: (batch, seq_len, features)
    factor < 1: compress, factor > 1: stretch
    """
    batch_size, seq_len, features = x.shape
    if factor == 1.0:
        return x
    elif factor > 1.0:
        # Stretch: repeat each timestep
        repeat_n = int(factor)
        warped = x.repeat_interleave(repeat_n, dim=1)
        return warped
    else:
        # Compress: average consecutive rows
        stride = int(1.0 / factor)
        new_len = seq_len // stride
        warped = x[:, :new_len * stride, :].reshape(batch_size, new_len, stride, features).mean(dim=2)
        return warped


def train_model(model, train_loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        correct, total = 0, 0
        for images, labels in train_loader:
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += x.size(0)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:2d}: train_acc={correct/total*100:.1f}%")


def evaluate_warped(model, test_loader, device, warp_factor):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.squeeze(1).to(device)  # (batch, 28, 28)
            labels = labels.to(device)
            x_warped = time_warp(x, warp_factor)
            logits = model(x_warped)
            correct += (logits.argmax(1) == labels).sum().item()
            total += x.size(0)
    return correct / total


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase76_time_warping.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 76: Zero-Shot Time-Warping',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 76: Zero-Shot Time-Warping")
    print("  Can Liquid-LIF generalize to unseen time scales?")
    print(f"  Hidden: {HIDDEN_DIM}, Warp factors: {WARP_FACTORS}")
    print("=" * 70)

    train_loader, test_loader = get_mnist()

    MODELS = {
        'LIF': lambda: SNNClassifier(LIFCell(INPUT_DIM, HIDDEN_DIM), HIDDEN_DIM, N_CLASSES),
        'GRU': lambda: GRUClassifier(INPUT_DIM, HIDDEN_DIM, N_CLASSES),
        'LiquidLIF': lambda: SNNClassifier(LiquidLIFCell(INPUT_DIM, HIDDEN_DIM), HIDDEN_DIM, N_CLASSES),
    }

    all_results = {}

    for model_name, model_fn in MODELS.items():
        print(f"\n  === {model_name} ===")
        model = model_fn().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,}")

        # Train on normal time scale (28 steps)
        print(f"    Training (28 steps)...")
        train_model(model, train_loader, DEVICE)

        # Evaluate at each warp factor
        model_results = {}
        for wf in WARP_FACTORS:
            acc = evaluate_warped(model, test_loader, DEVICE, wf)
            model_results[f"warp_{wf}"] = acc
            new_steps = int(28 * wf)
            print(f"    Warp {wf}x ({new_steps:3d} steps): {acc*100:.1f}%")

        # Compute degradation from 1.0x
        baseline = model_results['warp_1.0']
        for wf in WARP_FACTORS:
            if wf != 1.0:
                delta = (model_results[f'warp_{wf}'] - baseline) * 100
                print(f"      Delta from 1.0x: {delta:+.1f}pp")

        all_results[model_name] = {
            'n_params': n_params,
            'warp_results': model_results,
        }
        _save(all_results)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Zero-Shot Time-Warping")
    print(f"{'='*70}")
    print(f"  {'Model':15s}", end="")
    for wf in WARP_FACTORS:
        print(f"  {wf}x", end="")
    print()
    for name, r in all_results.items():
        print(f"  {name:15s}", end="")
        for wf in WARP_FACTORS:
            acc = r['warp_results'][f'warp_{wf}'] * 100
            print(f"  {acc:5.1f}%", end="")
        print()

    _generate_figure(all_results)
    print("\nPhase 76 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = {'LIF': '#3B82F6', 'GRU': '#9CA3AF', 'LiquidLIF': '#EC4899'}
        markers = {'LIF': 's', 'GRU': 'D', 'LiquidLIF': 'o'}

        for name, r in results.items():
            wfs = WARP_FACTORS
            accs = [r['warp_results'][f'warp_{wf}'] * 100 for wf in wfs]
            ax.plot(wfs, accs, '-', marker=markers.get(name, 'o'),
                   color=colors.get(name, '#333'), label=name,
                   linewidth=2, markersize=8)

        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Training scale')
        ax.set_xlabel('Time Warp Factor', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Phase 76: Zero-Shot Time-Warping\n'
                    'Can Liquid-LIF generalize to unseen time scales?',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(WARP_FACTORS)

        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase76_time_warping.png"),
                   bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
