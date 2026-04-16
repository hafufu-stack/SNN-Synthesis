"""
Phase 77: 1-Bit Micro-Liquid Resurrection

The ultimate test: Can temporal SR resurrect a 1-bit quantized brain?

Setup:
  1) Train full-precision 16-neuron Liquid-LIF on Sequential MNIST
  2) Quantize ALL weights to 1-bit (torch.sign: +1 or -1)
  3) Measure collapsed accuracy
  4) Inject Temporal SR (tau-noise, NBS K=10) to resurrect performance

This combines:
  - Phase 61: Quantization noise as SR source (LLM)
  - Phase 73: Temporal noise robustness (LSNN)
  - Phase 74: 16-neuron minimum viable model

Expected: 1-bit accuracy collapses, but Temporal SR partially resurrects it.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy
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
EPOCHS = 15
LR = 1e-3

SIGMA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
NBS_K = 10
NBS_SIGMAS = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]


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


class LiquidLIFCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.b_tau = nn.Parameter(torch.ones(hidden_dim) * 1.5)
        self.temporal_sigma = 0.0

    def forward(self, x, state=None):
        if state is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            mem = state
        tau_input = torch.cat([x, mem], dim=-1)
        tau_logit = self.fc_tau(tau_input) + self.b_tau

        if self.temporal_sigma > 0 and not self.training:
            tau_logit = tau_logit + torch.randn_like(tau_logit) * self.temporal_sigma

        beta = torch.sigmoid(tau_logit)
        beta = torch.clamp(beta, 0.01, 0.99)
        mem = beta * mem + self.fc_in(x)
        spk = spike_fn(mem - self.threshold)
        mem = mem - spk * self.threshold
        return spk, mem


class LSNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.cell = LiquidLIFCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        state = None
        for t in range(seq_len):
            _, state = self.cell(x[:, t, :], state)
        return self.readout(state)

    def set_temporal_noise(self, sigma):
        self.cell.temporal_sigma = sigma


def quantize_1bit(model):
    """Force all weights to +1 or -1 using torch.sign."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Scale: preserve approximate magnitude
                scale = param.abs().mean()
                param.copy_(torch.sign(param) * scale)
            # Keep biases as-is (they're tiny)


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
        print(f"      Epoch {epoch+1:2d}: {correct/total*100:.1f}%")


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == labels).sum().item()
            total += x.size(0)
    return correct / total


def evaluate_nbs(model, test_loader, device, sigmas):
    """NBS-style: K runs with diverse temporal noise, majority vote."""
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        x = images.squeeze(1).to(device)
        labels = labels.to(device)
        all_logits = torch.zeros(x.size(0), N_CLASSES, device=device)
        with torch.no_grad():
            for sigma in sigmas:
                model.set_temporal_noise(sigma)
                logits = model(x)
                all_logits += F.softmax(logits, dim=1)
        preds = all_logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += x.size(0)
    model.set_temporal_noise(0.0)
    return correct / total


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase77_1bit_resurrection.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 77: 1-Bit Micro-Liquid Resurrection',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 77: 1-Bit Micro-Liquid Resurrection")
    print("  Can temporal SR resurrect a 1-bit quantized 16-neuron brain?")
    print("=" * 70)

    train_loader, test_loader = get_mnist()

    # 1) Train full-precision model
    print("\n  [1/4] Training full-precision Liquid-LIF (H=16)...")
    model = LSNNClassifier(INPUT_DIM, HIDDEN_DIM, N_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    train_model(model, train_loader, DEVICE)

    fp_acc = evaluate(model, test_loader, DEVICE)
    print(f"\n    Full-precision accuracy: {fp_acc*100:.1f}%")

    # 2) Quantize to 1-bit
    print("\n  [2/4] Quantizing to 1-bit (torch.sign)...")
    model_1bit = copy.deepcopy(model)
    quantize_1bit(model_1bit)

    acc_1bit = evaluate(model_1bit, test_loader, DEVICE)
    print(f"    1-bit accuracy: {acc_1bit*100:.1f}% (delta: {(acc_1bit-fp_acc)*100:+.1f}pp)")

    all_results = {
        'full_precision': fp_acc,
        '1bit_baseline': acc_1bit,
        'n_params': n_params,
    }
    _save(all_results)

    # 3) Single-sigma temporal noise sweep
    print("\n  [3/4] Temporal noise sweep on 1-bit model...")
    noise_results = {}
    for sigma in SIGMA_VALUES:
        model_1bit.set_temporal_noise(sigma)
        acc = evaluate(model_1bit, test_loader, DEVICE)
        model_1bit.set_temporal_noise(0.0)
        delta = (acc - acc_1bit) * 100
        noise_results[f"sigma_{sigma}"] = {'sigma': sigma, 'accuracy': acc}
        print(f"    sigma={sigma:.2f}: {acc*100:.1f}% ({delta:+.1f}pp from 1-bit baseline)")

    all_results['temporal_noise'] = noise_results
    _save(all_results)

    # 4) NBS with sigma-diverse temporal noise
    print(f"\n  [4/4] Temporal NBS (K={NBS_K}, sigma-diverse)...")
    nbs_acc = evaluate_nbs(model_1bit, test_loader, DEVICE, NBS_SIGMAS[:NBS_K])
    delta_nbs = (nbs_acc - acc_1bit) * 100
    delta_fp = (nbs_acc - fp_acc) * 100
    print(f"    NBS accuracy: {nbs_acc*100:.1f}%")
    print(f"    vs 1-bit baseline: {delta_nbs:+.1f}pp")
    print(f"    vs full-precision: {delta_fp:+.1f}pp")

    all_results['nbs_temporal'] = nbs_acc
    _save(all_results)

    # Also test NBS on full-precision model
    print(f"\n  [Bonus] NBS on full-precision model...")
    nbs_fp_acc = evaluate_nbs(model, test_loader, DEVICE, NBS_SIGMAS[:NBS_K])
    print(f"    Full-precision + NBS: {nbs_fp_acc*100:.1f}% ({(nbs_fp_acc-fp_acc)*100:+.1f}pp)")
    all_results['nbs_fullprecision'] = nbs_fp_acc
    _save(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: 1-Bit Resurrection")
    print(f"{'='*70}")
    print(f"  Full-precision (FP32):           {fp_acc*100:.1f}%")
    print(f"  Full-precision + NBS:            {nbs_fp_acc*100:.1f}% ({(nbs_fp_acc-fp_acc)*100:+.1f}pp)")
    print(f"  1-bit (torch.sign):              {acc_1bit*100:.1f}% ({(acc_1bit-fp_acc)*100:+.1f}pp)")
    print(f"  1-bit + Temporal NBS:            {nbs_acc*100:.1f}% ({delta_nbs:+.1f}pp resurrection)")

    recovery_pct = (nbs_acc - acc_1bit) / max(0.001, fp_acc - acc_1bit) * 100
    print(f"\n  Recovery rate: {recovery_pct:.0f}% of lost accuracy recovered by Temporal SR")

    if nbs_acc > acc_1bit + 0.02:
        print(f"\n  >>> TEMPORAL SR RESURRECTS 1-BIT BRAIN! <<<")
    else:
        print(f"\n  >>> 1-bit damage too severe for temporal SR alone <<<")

    del model, model_1bit
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _generate_figure(all_results)
    print("\nPhase 77 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Accuracy levels
        conditions = ['FP32', 'FP32+NBS', '1-bit', '1-bit+NBS']
        accs = [
            results['full_precision'] * 100,
            results['nbs_fullprecision'] * 100,
            results['1bit_baseline'] * 100,
            results['nbs_temporal'] * 100,
        ]
        colors = ['#3B82F6', '#10B981', '#EF4444', '#F59E0B']
        bars = ax1.bar(conditions, accs, color=colors, edgecolor='white', linewidth=1.5)
        for b, a in zip(bars, accs):
            ax1.text(b.get_x() + b.get_width()/2, a + 0.5, f'{a:.1f}%',
                    ha='center', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('1-Bit Resurrection', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

        # Right: Temporal noise sweep on 1-bit
        sigmas = sorted([v['sigma'] for v in results['temporal_noise'].values()])
        noise_accs = [results['temporal_noise'][f'sigma_{s}']['accuracy'] * 100 for s in sigmas]
        ax2.plot(sigmas, noise_accs, '-o', color='#F59E0B', linewidth=2, markersize=6)
        ax2.axhline(y=results['1bit_baseline']*100, color='#EF4444', linestyle='--',
                    alpha=0.7, label=f"1-bit baseline ({results['1bit_baseline']*100:.1f}%)")
        ax2.axhline(y=results['full_precision']*100, color='#3B82F6', linestyle='--',
                    alpha=0.7, label=f"FP32 ({results['full_precision']*100:.1f}%)")
        ax2.set_xlabel('Temporal Noise sigma')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Temporal Noise Sweep (1-bit model)', fontweight='bold')
        ax2.legend(); ax2.grid(alpha=0.3)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

        fig.suptitle('Phase 77: 1-Bit Micro-Liquid Resurrection\n'
                    '16 neurons, 1-bit weights + Temporal SR',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase77_1bit_resurrection.png"),
                   bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
