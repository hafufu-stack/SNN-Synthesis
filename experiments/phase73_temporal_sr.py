"""
Phase 73: Temporal Stochastic Resonance

Previous SNN-Synthesis work injected noise into SPATIAL dimensions
(hidden states, weights). Phase 73 injects noise into the TEMPORAL
dimension: the tau-gate of the Liquid-LIF cell.

Conditions:
  A) No noise (baseline)
  B) Spatial noise only (noise on membrane potential)
  C) Temporal noise only (noise on tau-gate logits)
  D) Both spatial + temporal
  E) Sigma-diverse: different sigma per forward pass (NBS-style)

Task: Sequential MNIST with reduced training (5 epochs -> underfitting)
      to create a sub-optimal model where SR can help.

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

HIDDEN_DIM = 64   # Smaller model for underfitting
N_CLASSES = 10
SEQ_LEN = 28
INPUT_DIM = 28
BATCH_SIZE = 128
TRAIN_EPOCHS = 5   # Deliberately underfit
LR = 1e-3
N_EVAL_RUNS = 10   # Multiple noisy forward passes (NBS-style K)


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
    """Liquid-LIF with optional spatial and temporal noise injection."""
    def __init__(self, input_dim, hidden_dim, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.b_tau = nn.Parameter(torch.ones(hidden_dim) * 1.5)

        # Noise parameters (set externally)
        self.spatial_sigma = 0.0
        self.temporal_sigma = 0.0

    def forward(self, x, state=None):
        if state is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            mem = state

        # Liquid tau gate
        tau_input = torch.cat([x, mem], dim=-1)
        tau_logit = self.fc_tau(tau_input) + self.b_tau

        # TEMPORAL NOISE: perturb the time constant
        if self.temporal_sigma > 0 and not self.training:
            tau_logit = tau_logit + torch.randn_like(tau_logit) * self.temporal_sigma

        beta = torch.sigmoid(tau_logit)
        beta = torch.clamp(beta, 0.01, 0.99)

        mem = beta * mem + self.fc_in(x)

        # SPATIAL NOISE: perturb membrane potential
        if self.spatial_sigma > 0 and not self.training:
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
            spk, state = self.cell(x[:, t, :], state)
        return self.readout(state)

    def set_noise(self, spatial_sigma=0.0, temporal_sigma=0.0):
        self.cell.spatial_sigma = spatial_sigma
        self.cell.temporal_sigma = temporal_sigma


def get_sequential_mnist(batch_size):
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_model(model, train_loader, epochs, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(epochs):
        total_loss, total_correct, total_n = 0, 0, 0
        for images, labels in train_loader:
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += x.size(0)
        print(f"    Epoch {epoch+1}: acc={total_correct/total_n*100:.1f}% loss={total_loss/total_n:.4f}")


def evaluate_with_noise(model, test_loader, device, spatial_sigma, temporal_sigma, K=1):
    """Evaluate with NBS-style: run K noisy forward passes, majority vote."""
    model.eval()
    model.set_noise(spatial_sigma, temporal_sigma)

    total_correct, total_n = 0, 0
    for images, labels in test_loader:
        x = images.squeeze(1).to(device)
        labels = labels.to(device)
        batch_size = x.size(0)

        if K == 1:
            with torch.no_grad():
                logits = model(x)
            preds = logits.argmax(1)
        else:
            # K-beam: accumulate logits
            all_logits = torch.zeros(batch_size, N_CLASSES, device=device)
            with torch.no_grad():
                for _ in range(K):
                    logits = model(x)
                    all_logits += F.softmax(logits, dim=1)
            preds = all_logits.argmax(1)

        total_correct += (preds == labels).sum().item()
        total_n += batch_size

    model.set_noise(0.0, 0.0)
    return total_correct / total_n


SIGMA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase73_temporal_sr.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 73: Temporal Stochastic Resonance',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 73: Temporal Stochastic Resonance")
    print("  Does noise on tau-gates (time) help more than noise on states (space)?")
    print(f"  Device: {DEVICE}, Training: {TRAIN_EPOCHS} epochs (deliberate underfit)")
    print("=" * 70)

    train_loader, test_loader = get_sequential_mnist(BATCH_SIZE)

    # Train an underfit model
    print("\n  Training underfit Liquid-LIF model...")
    model = LSNNClassifier(INPUT_DIM, HIDDEN_DIM, N_CLASSES).to(DEVICE)
    train_model(model, train_loader, TRAIN_EPOCHS, DEVICE)

    # Baseline (no noise)
    baseline_acc = evaluate_with_noise(model, test_loader, DEVICE, 0.0, 0.0, K=1)
    print(f"\n  Baseline (no noise): {baseline_acc*100:.1f}%\n")

    all_results = {'baseline': baseline_acc, 'conditions': {}}

    # Test conditions
    CONDITIONS = {
        'spatial_only': {'desc': 'Spatial noise on membrane potential'},
        'temporal_only': {'desc': 'Temporal noise on tau-gate'},
        'both': {'desc': 'Both spatial + temporal'},
    }

    for cond_name, cond_info in CONDITIONS.items():
        print(f"\n  --- {cond_name}: {cond_info['desc']} ---")
        cond_results = {}

        for sigma in SIGMA_VALUES:
            if cond_name == 'spatial_only':
                s_sig, t_sig = sigma, 0.0
            elif cond_name == 'temporal_only':
                s_sig, t_sig = 0.0, sigma
            else:
                s_sig, t_sig = sigma, sigma

            # Single pass
            acc_k1 = evaluate_with_noise(model, test_loader, DEVICE, s_sig, t_sig, K=1)
            # K=10 NBS-style
            acc_k10 = evaluate_with_noise(model, test_loader, DEVICE, s_sig, t_sig, K=N_EVAL_RUNS)

            cond_results[f"sigma_{sigma}"] = {
                'sigma': sigma, 'K1': acc_k1, 'K10': acc_k10
            }
            delta_k1 = (acc_k1 - baseline_acc) * 100
            delta_k10 = (acc_k10 - baseline_acc) * 100
            print(f"    sigma={sigma:.2f}: K=1 {acc_k1*100:.1f}% ({delta_k1:+.1f}pp)  "
                  f"K=10 {acc_k10*100:.1f}% ({delta_k10:+.1f}pp)")

        all_results['conditions'][cond_name] = cond_results
        _save(all_results)

    # Sigma-diverse (different sigma per beam)
    print(f"\n  --- sigma-diverse NBS (K=10, different sigma per beam) ---")
    diverse_sigmas_spatial = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.01, 0.05, 0.1, 0.3]
    diverse_sigmas_temporal = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.01, 0.05, 0.1, 0.3]

    model.eval()
    total_correct, total_n = 0, 0
    for images, labels in test_loader:
        x = images.squeeze(1).to(DEVICE)
        labels = labels.to(DEVICE)
        all_logits = torch.zeros(x.size(0), N_CLASSES, device=DEVICE)
        with torch.no_grad():
            for i in range(N_EVAL_RUNS):
                model.set_noise(diverse_sigmas_spatial[i], diverse_sigmas_temporal[i])
                logits = model(x)
                all_logits += F.softmax(logits, dim=1)
        preds = all_logits.argmax(1)
        total_correct += (preds == labels).sum().item()
        total_n += x.size(0)
    model.set_noise(0.0, 0.0)
    diverse_acc = total_correct / total_n
    print(f"    sigma-diverse: {diverse_acc*100:.1f}% ({(diverse_acc-baseline_acc)*100:+.1f}pp)")
    all_results['sigma_diverse'] = diverse_acc
    _save(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Temporal vs Spatial SR")
    print(f"{'='*70}")
    print(f"  Baseline: {baseline_acc*100:.1f}%")
    for cond_name in CONDITIONS:
        best_k10 = max(
            v['K10'] for v in all_results['conditions'][cond_name].values()
        )
        best_sigma = max(
            all_results['conditions'][cond_name].values(),
            key=lambda v: v['K10']
        )['sigma']
        print(f"  {cond_name:20s}: best K=10 = {best_k10*100:.1f}% (sigma={best_sigma})")
    print(f"  sigma-diverse NBS:       {diverse_acc*100:.1f}%")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _generate_figure(all_results)
    print("\nPhase 73 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = {'spatial_only': '#3B82F6', 'temporal_only': '#EC4899', 'both': '#10B981'}

        baseline = results['baseline']

        # K=1 curves
        for cond_name, cond_data in results['conditions'].items():
            sigmas = sorted([v['sigma'] for v in cond_data.values()])
            accs = [cond_data[f"sigma_{s}"]['K1'] * 100 for s in sigmas]
            ax1.plot(sigmas, accs, '-o', color=colors.get(cond_name), label=cond_name, markersize=5)
        ax1.axhline(y=baseline*100, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax1.set_xlabel('Noise sigma'); ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('K=1 (Single Pass)', fontweight='bold')
        ax1.legend(); ax1.grid(alpha=0.3); ax1.set_xscale('symlog', linthresh=0.01)

        # K=10 curves
        for cond_name, cond_data in results['conditions'].items():
            sigmas = sorted([v['sigma'] for v in cond_data.values()])
            accs = [cond_data[f"sigma_{s}"]['K10'] * 100 for s in sigmas]
            ax2.plot(sigmas, accs, '-o', color=colors.get(cond_name), label=cond_name, markersize=5)
        ax2.axhline(y=baseline*100, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        diverse_acc = results.get('sigma_diverse', 0)
        ax2.axhline(y=diverse_acc*100, color='#F59E0B', linestyle=':', alpha=0.7, label=f'sigma-diverse={diverse_acc*100:.1f}%')
        ax2.set_xlabel('Noise sigma'); ax2.set_ylabel('Accuracy (%)');
        ax2.set_title('K=10 (NBS-style Ensemble)', fontweight='bold')
        ax2.legend(); ax2.grid(alpha=0.3); ax2.set_xscale('symlog', linthresh=0.01)

        fig.suptitle('Phase 73: Temporal vs Spatial Stochastic Resonance\n'
                    'in Liquid-LIF Networks',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase73_temporal_sr.png"),
                   bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
