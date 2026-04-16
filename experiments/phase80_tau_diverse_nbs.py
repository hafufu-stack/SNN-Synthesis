"""
Phase 80: tau-Diverse Beam Search

Instead of diversifying SPATIAL noise (sigma on hidden states),
diversify TEMPORAL parameters (tau-gate bias) across beams.

Each beam has a different "sense of time":
  - Fast beams (low tau-bias -> small beta -> fast decay)
  - Slow beams (high tau-bias -> large beta -> long memory)
  - Normal beams (default tau)

This exploits Phase 73's finding: temporal noise is 5x safer than spatial.

Compare on Sequential MNIST (underfit model):
  A) Baseline (single pass)
  B) sigma-diverse NBS (spatial noise, K=11)
  C) tau-diverse NBS (temporal bias, K=11)
  D) Combined tau+sigma diverse

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

HIDDEN_DIM = 32
N_CLASSES = 10
INPUT_DIM = 28
BATCH_SIZE = 128
TRAIN_EPOCHS = 5   # Deliberate underfit for SR to help
LR = 1e-3
K = 11  # Beam count


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
        self.tau_bias_override = None  # For tau-diverse NBS
        self.spatial_sigma = 0.0
        self.temporal_sigma = 0.0

    def forward(self, x, state=None):
        if state is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            mem = state

        tau_input = torch.cat([x, mem], dim=-1)
        tau_logit = self.fc_tau(tau_input) + self.b_tau

        # Tau-diverse: override the bias
        if self.tau_bias_override is not None:
            tau_logit = self.fc_tau(tau_input) + self.tau_bias_override.to(tau_input.device)

        if self.temporal_sigma > 0:
            tau_logit = tau_logit + torch.randn_like(tau_logit) * self.temporal_sigma

        beta = torch.sigmoid(tau_logit)
        beta = torch.clamp(beta, 0.01, 0.99)

        mem = beta * mem + self.fc_in(x)

        if self.spatial_sigma > 0:
            mem = mem + torch.randn_like(mem) * self.spatial_sigma

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

    def configure_beam(self, tau_bias=None, spatial_sigma=0.0, temporal_sigma=0.0):
        self.cell.tau_bias_override = tau_bias
        self.cell.spatial_sigma = spatial_sigma
        self.cell.temporal_sigma = temporal_sigma

    def reset_beam(self):
        self.cell.tau_bias_override = None
        self.cell.spatial_sigma = 0.0
        self.cell.temporal_sigma = 0.0


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
    for epoch in range(TRAIN_EPOCHS):
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
        print(f"    Epoch {epoch+1}: {correct/total*100:.1f}%")


def evaluate_baseline(model, test_loader, device):
    model.eval()
    model.reset_beam()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == labels).sum().item()
            total += x.size(0)
    return correct / total


def evaluate_nbs(model, test_loader, device, beam_configs):
    """NBS with configurable beams. Each config is a dict with tau_bias/sigma."""
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        x = images.squeeze(1).to(device)
        labels = labels.to(device)
        all_logits = torch.zeros(x.size(0), N_CLASSES, device=device)

        with torch.no_grad():
            for cfg in beam_configs:
                model.configure_beam(
                    tau_bias=cfg.get('tau_bias'),
                    spatial_sigma=cfg.get('spatial_sigma', 0.0),
                    temporal_sigma=cfg.get('temporal_sigma', 0.0),
                )
                logits = model(x)
                all_logits += F.softmax(logits, dim=1)

        preds = all_logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += x.size(0)

    model.reset_beam()
    return correct / total


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase80_tau_diverse_nbs.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 80: tau-Diverse Beam Search',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 80: tau-Diverse Beam Search")
    print("  'Impatient' and 'patient' beams exploring in parallel")
    print(f"  K={K} beams, {TRAIN_EPOCHS} epochs (underfit)")
    print("=" * 70)

    train_loader, test_loader = get_mnist()
    model = LSNNClassifier(INPUT_DIM, HIDDEN_DIM, N_CLASSES).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n  Training (deliberately underfit)...")
    train_model(model, train_loader, DEVICE)

    results = {}

    # A) Baseline
    baseline = evaluate_baseline(model, test_loader, DEVICE)
    print(f"\n  A) Baseline (single pass): {baseline*100:.1f}%")
    results['baseline'] = baseline

    # B) Sigma-diverse NBS (spatial noise, traditional)
    sigma_values = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
    sigma_configs = [{'spatial_sigma': s} for s in sigma_values[:K]]
    sigma_acc = evaluate_nbs(model, test_loader, DEVICE, sigma_configs)
    print(f"  B) sigma-diverse NBS (K={K}): {sigma_acc*100:.1f}% "
          f"({(sigma_acc-baseline)*100:+.1f}pp)")
    results['sigma_diverse'] = sigma_acc

    # C) tau-diverse NBS (temporal bias diversity)
    # Spread tau biases from "fast" (low beta ~0.3) to "slow" (high beta ~0.95)
    tau_biases = torch.linspace(-1.0, 3.0, K)  # sigmoid(-1)=0.27, sigmoid(3)=0.95
    tau_configs = [{'tau_bias': torch.ones(HIDDEN_DIM) * tb} for tb in tau_biases]
    tau_acc = evaluate_nbs(model, test_loader, DEVICE, tau_configs)
    print(f"  C) tau-diverse NBS (K={K}): {tau_acc*100:.1f}% "
          f"({(tau_acc-baseline)*100:+.1f}pp)")
    results['tau_diverse'] = tau_acc

    # D) Combined: tau-diverse + small temporal noise
    combo_configs = [{'tau_bias': torch.ones(HIDDEN_DIM) * tb,
                      'temporal_sigma': 0.1}
                     for tb in tau_biases]
    combo_acc = evaluate_nbs(model, test_loader, DEVICE, combo_configs)
    print(f"  D) tau-diverse + temporal noise (K={K}): {combo_acc*100:.1f}% "
          f"({(combo_acc-baseline)*100:+.1f}pp)")
    results['tau_plus_noise'] = combo_acc

    # E) Temporal noise only (no tau diversity)
    temp_noise_configs = [{'temporal_sigma': s} for s in sigma_values[:K]]
    temp_acc = evaluate_nbs(model, test_loader, DEVICE, temp_noise_configs)
    print(f"  E) temporal-noise NBS (K={K}): {temp_acc*100:.1f}% "
          f"({(temp_acc-baseline)*100:+.1f}pp)")
    results['temporal_noise_diverse'] = temp_acc

    _save(results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: tau-Diverse Beam Search")
    print(f"{'='*70}")
    print(f"  Baseline:                {baseline*100:.1f}%")
    print(f"  sigma-diverse (spatial): {sigma_acc*100:.1f}% ({(sigma_acc-baseline)*100:+.1f}pp)")
    print(f"  tau-diverse (temporal):  {tau_acc*100:.1f}% ({(tau_acc-baseline)*100:+.1f}pp)")
    print(f"  tau + noise (combined):  {combo_acc*100:.1f}% ({(combo_acc-baseline)*100:+.1f}pp)")
    print(f"  temporal-noise-diverse:  {temp_acc*100:.1f}% ({(temp_acc-baseline)*100:+.1f}pp)")

    best_name = max(results, key=lambda k: results[k] if isinstance(results[k], float) else 0)
    best_val = results[best_name]
    if isinstance(best_val, float):
        print(f"\n  >>> BEST: {best_name} at {best_val*100:.1f}% <<<")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    _generate_figure(results)
    print("\nPhase 80 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        names = ['Baseline', 'sigma-diverse\n(spatial)', 'tau-diverse\n(temporal)',
                 'tau+noise\n(combined)', 'temporal\nnoise-diverse']
        keys = ['baseline', 'sigma_diverse', 'tau_diverse', 'tau_plus_noise',
                'temporal_noise_diverse']
        accs = [results[k] * 100 for k in keys]
        colors = ['#9CA3AF', '#3B82F6', '#EC4899', '#F59E0B', '#10B981']

        bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=1.5)
        for b, a in zip(bars, accs):
            ax.text(b.get_x()+b.get_width()/2, a+0.3, f'{a:.1f}%',
                    ha='center', fontweight='bold', fontsize=11)

        ax.axhline(y=results['baseline']*100, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Phase 80: tau-Diverse Beam Search (K=11)\n'
                    'Beams with different "senses of time"',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase80_tau_diverse_nbs.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
