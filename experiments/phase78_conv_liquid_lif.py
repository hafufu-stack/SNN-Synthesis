"""
Phase 78: Conv-Liquid-LIF - Spatial Liquid Neurons for 2D Vision

Upgrades Liquid-LIF from 1D (Linear) to 2D (Conv2d) for spatial reasoning.
Each pixel location has its own dynamic tau - "active areas" speed up,
"static background" slows down. Event-driven vision.

Task: Synthetic 2D grid task (predict next position of moving pixel on 8x8 grid)
Compare: ConvGRU vs ConvLIF vs ConvLiquidLIF

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

GRID_SIZE = 8
N_CHANNELS = 1
HIDDEN_CHANNELS = 8
SEQ_LEN = 5        # 5 frames of moving pixel
N_TRAIN = 5000
N_TEST = 1000
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3


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


# ==============================================================
# Synthetic 2D Task: Moving Pixel Prediction
# ==============================================================
class MovingPixelDataset(Dataset):
    """A pixel moves on an 8x8 grid with constant velocity.
    Input: 5 frames. Target: position in frame 6.
    Velocities: {-1, 0, +1} in each axis (with wrapping)."""
    def __init__(self, n_samples, grid_size=8, seq_len=5, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        self.data = []
        self.targets = []
        for _ in range(n_samples):
            # Random start position
            y, x = rng.randint(0, grid_size, size=2)
            # Random velocity (exclude 0,0)
            while True:
                vy, vx = rng.randint(-1, 2, size=2)
                if vy != 0 or vx != 0:
                    break
            frames = []
            for t in range(seq_len + 1):
                frame = np.zeros((1, grid_size, grid_size), dtype=np.float32)
                py = (y + vy * t) % grid_size
                px = (x + vx * t) % grid_size
                frame[0, py, px] = 1.0
                frames.append(frame)
            self.data.append(np.stack(frames[:seq_len]))  # (seq, 1, H, W)
            # Target: position in frame 6
            target_y = (y + vy * seq_len) % grid_size
            target_x = (x + vx * seq_len) % grid_size
            self.targets.append(target_y * grid_size + target_x)  # Flatten to class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx], dtype=torch.long)


# ==============================================================
# ConvLiquidLIF Cell
# ==============================================================
class ConvLiquidLIFCell(nn.Module):
    """2D Convolutional Liquid-LIF: per-pixel dynamic tau."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3, threshold=1.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.threshold = threshold
        pad = kernel_size // 2
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad)
        self.conv_tau = nn.Conv2d(in_channels + hidden_channels, hidden_channels,
                                  kernel_size, padding=pad)
        self.b_tau = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 1.5)

    def forward(self, x, state=None):
        """x: (B, C_in, H, W), state: (B, C_hid, H, W) or None"""
        if state is None:
            b, _, h, w = x.shape
            mem = torch.zeros(b, self.hidden_channels, h, w, device=x.device)
        else:
            mem = state

        tau_input = torch.cat([x, mem], dim=1)
        beta = torch.sigmoid(self.conv_tau(tau_input) + self.b_tau)
        beta = torch.clamp(beta, 0.01, 0.99)

        mem = beta * mem + self.conv_in(x)
        spk = spike_fn(mem - self.threshold)
        mem = mem - spk * self.threshold
        return spk, mem


class ConvLIFCell(nn.Module):
    """Standard Conv-LIF with fixed beta."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3, beta=0.9, threshold=1.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.beta = beta
        self.threshold = threshold
        pad = kernel_size // 2
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=pad)

    def forward(self, x, state=None):
        if state is None:
            b, _, h, w = x.shape
            mem = torch.zeros(b, self.hidden_channels, h, w, device=x.device)
        else:
            mem = state
        mem = self.beta * mem + self.conv_in(x)
        spk = spike_fn(mem - self.threshold)
        mem = mem - spk * self.threshold
        return spk, mem


class ConvGRUCell(nn.Module):
    """Minimal ConvGRU for comparison."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        self.conv_gates = nn.Conv2d(in_channels + hidden_channels, 2 * hidden_channels,
                                     kernel_size, padding=pad)
        self.conv_candidate = nn.Conv2d(in_channels + hidden_channels, hidden_channels,
                                         kernel_size, padding=pad)

    def forward(self, x, state=None):
        if state is None:
            b, _, h, w = x.shape
            state = torch.zeros(b, self.hidden_channels, h, w, device=x.device)
        combined = torch.cat([x, state], dim=1)
        gates = torch.sigmoid(self.conv_gates(combined))
        reset_gate, update_gate = gates.chunk(2, dim=1)
        combined_r = torch.cat([x, reset_gate * state], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_r))
        new_state = (1 - update_gate) * state + update_gate * candidate
        return new_state, new_state  # (output, state)


class GridPredictor(nn.Module):
    """Wraps a recurrent 2D cell into a next-position predictor."""
    def __init__(self, cell, hidden_channels, grid_size, is_snn=True):
        super().__init__()
        self.cell = cell
        self.is_snn = is_snn
        self.readout = nn.Linear(hidden_channels * grid_size * grid_size,
                                  grid_size * grid_size)

    def forward(self, x_seq):
        """x_seq: (B, T, C, H, W)"""
        b, t, c, h, w = x_seq.shape
        state = None
        spike_count = 0
        total_neurons = 0

        for step in range(t):
            out, state = self.cell(x_seq[:, step], state)
            if self.is_snn:
                spike_count += out.sum().item()
                total_neurons += out.numel()

        flat = state.reshape(b, -1)
        logits = self.readout(flat)
        sparsity = 1.0 - (spike_count / max(1, total_neurons)) if self.is_snn else 0.0
        return logits, sparsity


def train_and_eval(model, train_loader, test_loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for x_seq, targets in train_loader:
            x_seq, targets = x_seq.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(x_seq)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: train={correct/total*100:.1f}%")

    model.eval()
    correct, total = 0, 0
    total_sparsity, n_batches = 0, 0
    with torch.no_grad():
        for x_seq, targets in test_loader:
            x_seq, targets = x_seq.to(device), targets.to(device)
            logits, sparsity = model(x_seq)
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
            total_sparsity += sparsity
            n_batches += 1

    return correct / total, total_sparsity / max(1, n_batches)


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase78_conv_liquid_lif.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 78: Conv-Liquid-LIF',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 78: Conv-Liquid-LIF - 2D Spatial Liquid Neurons")
    print("  Task: Predict next position of moving pixel on 8x8 grid")
    print(f"  Device: {DEVICE}, Epochs: {EPOCHS}")
    print("=" * 70)

    train_ds = MovingPixelDataset(N_TRAIN, GRID_SIZE, SEQ_LEN, seed=SEED)
    test_ds = MovingPixelDataset(N_TEST, GRID_SIZE, SEQ_LEN, seed=SEED + 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    MODELS = {
        'ConvLIF': lambda: GridPredictor(
            ConvLIFCell(N_CHANNELS, HIDDEN_CHANNELS), HIDDEN_CHANNELS, GRID_SIZE, is_snn=True),
        'ConvGRU': lambda: GridPredictor(
            ConvGRUCell(N_CHANNELS, HIDDEN_CHANNELS), HIDDEN_CHANNELS, GRID_SIZE, is_snn=False),
        'ConvLiquidLIF': lambda: GridPredictor(
            ConvLiquidLIFCell(N_CHANNELS, HIDDEN_CHANNELS), HIDDEN_CHANNELS, GRID_SIZE, is_snn=True),
    }

    all_results = {}
    for name, model_fn in MODELS.items():
        print(f"\n  === {name} ===")
        model = model_fn().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")

        t0 = time.time()
        acc, sparsity = train_and_eval(model, train_loader, test_loader, DEVICE)
        elapsed = time.time() - t0

        all_results[name] = {
            'test_acc': acc, 'sparsity': sparsity,
            'n_params': n_params, 'train_time': elapsed
        }
        print(f"    Test: {acc*100:.1f}%, Sparsity: {sparsity:.3f}, "
              f"Params: {n_params:,}, Time: {elapsed:.0f}s")

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        _save(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Conv-Liquid-LIF")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(f"  {name:20s}: acc={r['test_acc']*100:.1f}%  "
              f"sparsity={r['sparsity']:.3f}  params={r['n_params']:,}")

    _generate_figure(all_results)
    print("\nPhase 78 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        names = list(results.keys())
        colors = {'ConvLIF': '#3B82F6', 'ConvGRU': '#9CA3AF', 'ConvLiquidLIF': '#EC4899'}

        accs = [results[n]['test_acc'] * 100 for n in names]
        bars = ax1.bar(names, accs, color=[colors.get(n, '#333') for n in names],
                      edgecolor='white', linewidth=1.5)
        for b, a in zip(bars, accs):
            ax1.text(b.get_x()+b.get_width()/2, a+0.5, f'{a:.1f}%',
                    ha='center', fontweight='bold')
        ax1.set_ylabel('Test Accuracy (%)'); ax1.set_title('2D Grid Prediction', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        for n in names:
            r = results[n]
            ax2.scatter(r['sparsity']*100, r['test_acc']*100,
                       c=colors.get(n, '#333'), s=150, label=n, edgecolors='white', linewidth=1.5)
        ax2.set_xlabel('Spike Sparsity (%)'); ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Efficiency Frontier', fontweight='bold')
        ax2.legend(); ax2.grid(alpha=0.3)

        fig.suptitle('Phase 78: Conv-Liquid-LIF for 2D Spatial Reasoning',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase78_conv_liquid_lif.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
