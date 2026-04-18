"""
Phase 126: Tensor-Decomposed NCA Compiler

Compile a deep CNN into a single-layer NCA that runs for T steps,
WITHOUT any backprop/distillation - pure math only.

Approach:
  1. Train a T-layer CNN teacher
  2. Decompose layer weights W_1...W_T into time-dependent W(t)
     using SVD, Fourier, and polynomial approximations
  3. Run NCA with W(t) for T steps
  4. Compare teacher vs compiled NCA accuracy

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
# Teacher: T-layer CNN with identical conv shapes
# ====================================================================
class DeepCNN(nn.Module):
    """T-layer CNN with same-shape conv layers (compilable target)."""
    def __init__(self, in_ch=1, hidden_ch=16, out_ch=10, n_layers=5):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.layers = nn.ModuleList([
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1)
            for _ in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_ch, out_ch)
        self.n_layers = n_layers

    def forward(self, x):
        h = F.relu(self.stem(x))
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.fc(self.pool(h).view(h.size(0), -1))

    def get_layer_weights(self):
        """Extract weight tensors from each layer."""
        return [layer.weight.data.clone() for layer in self.layers]


# ====================================================================
# Compiled NCA: uses W(t) instead of fixed weights
# ====================================================================
class CompiledNCA(nn.Module):
    """NCA that uses time-dependent weights from decomposition."""
    def __init__(self, stem_weight, stem_bias, fc_weight, fc_bias,
                 hidden_ch=16, out_ch=10):
        super().__init__()
        self.stem = nn.Conv2d(1, hidden_ch, 3, padding=1)
        self.stem.weight.data = stem_weight
        self.stem.bias.data = stem_bias
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_ch, out_ch)
        self.fc.weight.data = fc_weight
        self.fc.bias.data = fc_bias
        self.hidden_ch = hidden_ch

    def forward_with_schedule(self, x, weight_schedule, n_steps):
        """Run NCA with time-dependent weights."""
        h = F.relu(self.stem(x))
        for t in range(n_steps):
            W_t = weight_schedule(t, n_steps)
            h = F.relu(F.conv2d(h, W_t, padding=1))
        return self.fc(self.pool(h).view(h.size(0), -1))


# ====================================================================
# Weight decomposition methods
# ====================================================================
def decompose_mean(weights):
    """Baseline: simple average (Phase 112 approach)."""
    W_mean = torch.stack(weights).mean(dim=0)
    def schedule(t, T):
        return W_mean
    return schedule, "Mean"


def decompose_svd_rank1(weights):
    """SVD rank-1 approximation of weight trajectory."""
    T = len(weights)
    # Stack and reshape: (T, out, in, kh, kw) -> (T, -1)
    shape = weights[0].shape
    W_flat = torch.stack(weights).reshape(T, -1)  # (T, D)

    # SVD
    U, S, Vt = torch.linalg.svd(W_flat, full_matrices=False)

    # Rank-1: W(t) ≈ s[0] * U[:,0] * V[0,:]
    s0 = S[0]
    u0 = U[:, 0]  # (T,)
    v0 = Vt[0, :]  # (D,)
    W_base = (s0 * v0).reshape(shape)

    # Time modulation from U[:,0]
    t_weights = u0  # (T,)

    def schedule(t, T_total):
        idx = min(t, T - 1)
        return W_base * t_weights[idx]
    return schedule, "SVD-Rank1"


def decompose_svd_rank3(weights):
    """SVD rank-3 approximation."""
    T = len(weights)
    shape = weights[0].shape
    W_flat = torch.stack(weights).reshape(T, -1)

    U, S, Vt = torch.linalg.svd(W_flat, full_matrices=False)
    rank = min(3, len(S))

    bases = []
    t_coeffs = []
    for r in range(rank):
        bases.append((S[r] * Vt[r]).reshape(shape))
        t_coeffs.append(U[:, r])

    def schedule(t, T_total):
        idx = min(t, T - 1)
        W_t = torch.zeros_like(bases[0])
        for r in range(rank):
            W_t += bases[r] * t_coeffs[r][idx]
        return W_t
    return schedule, "SVD-Rank3"


def decompose_fourier(weights, n_harmonics=3):
    """Fourier series approximation of weight trajectory."""
    T = len(weights)
    shape = weights[0].shape
    W_flat = torch.stack(weights).reshape(T, -1).cpu().numpy()  # (T, D)

    # Fit Fourier coefficients for each weight dimension
    t_norm = np.linspace(0, 1, T)
    # a0 + sum(an*cos + bn*sin)
    coeffs_a = np.zeros((n_harmonics + 1, W_flat.shape[1]))
    coeffs_b = np.zeros((n_harmonics, W_flat.shape[1]))

    coeffs_a[0] = W_flat.mean(axis=0)
    for n in range(1, n_harmonics + 1):
        cos_basis = np.cos(2 * np.pi * n * t_norm)
        sin_basis = np.sin(2 * np.pi * n * t_norm)
        coeffs_a[n] = 2 * (W_flat * cos_basis[:, None]).mean(axis=0)
        coeffs_b[n-1] = 2 * (W_flat * sin_basis[:, None]).mean(axis=0)

    coeffs_a = torch.tensor(coeffs_a, dtype=torch.float32, device=DEVICE)
    coeffs_b = torch.tensor(coeffs_b, dtype=torch.float32, device=DEVICE)

    def schedule(t, T_total):
        t_n = t / max(T_total - 1, 1)
        W_t = coeffs_a[0].clone()
        for n in range(1, n_harmonics + 1):
            W_t += coeffs_a[n] * np.cos(2 * np.pi * n * t_n)
            W_t += coeffs_b[n-1] * np.sin(2 * np.pi * n * t_n)
        return W_t.reshape(shape)
    return schedule, f"Fourier-{n_harmonics}"


def decompose_polynomial(weights, degree=3):
    """Polynomial approximation of weight trajectory."""
    T = len(weights)
    shape = weights[0].shape
    W_flat = torch.stack(weights).reshape(T, -1).cpu().numpy()

    t_norm = np.linspace(0, 1, T)

    # Fit polynomial for each weight dimension
    coeffs = np.polyfit(t_norm, W_flat, degree)  # (degree+1, D)
    coeffs = torch.tensor(coeffs, dtype=torch.float32, device=DEVICE)

    def schedule(t, T_total):
        t_n = t / max(T_total - 1, 1)
        W_t = torch.zeros(coeffs.shape[1], device=DEVICE)
        for d in range(len(coeffs)):
            W_t += coeffs[d] * (t_n ** (len(coeffs) - 1 - d))
        return W_t.reshape(shape)
    return schedule, f"Poly-{degree}"


def decompose_per_step(weights):
    """Perfect oracle: use exact weight at each step."""
    def schedule(t, T_total):
        idx = min(t, len(weights) - 1)
        return weights[idx]
    return schedule, "Oracle"


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 126: Tensor-Decomposed NCA Compiler")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # MNIST for clean signal
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    results = {}

    for T in [3, 5, 8, 12]:
        print(f"\n{'='*50}")
        print(f"  T = {T} layers/steps")
        print(f"{'='*50}")

        # Train teacher CNN
        teacher = DeepCNN(in_ch=1, hidden_ch=16, out_ch=10, n_layers=T).to(DEVICE)
        opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
        for epoch in range(15):
            teacher.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = F.cross_entropy(teacher(x), y)
                opt.zero_grad(); loss.backward(); opt.step()

        # Evaluate teacher
        teacher.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (teacher(x).argmax(1) == y).sum().item()
                total += y.size(0)
        teacher_acc = correct / total
        print(f"  Teacher accuracy: {teacher_acc*100:.2f}%")

        # Extract weights
        layer_weights = teacher.get_layer_weights()

        # Build compiled NCA shell
        nca = CompiledNCA(
            teacher.stem.weight.data.clone(),
            teacher.stem.bias.data.clone(),
            teacher.fc.weight.data.clone(),
            teacher.fc.bias.data.clone(),
            hidden_ch=16, out_ch=10
        ).to(DEVICE)

        # Test each decomposition
        decompositions = [
            decompose_mean(layer_weights),
            decompose_svd_rank1(layer_weights),
            decompose_svd_rank3(layer_weights),
            decompose_fourier(layer_weights, n_harmonics=2),
            decompose_fourier(layer_weights, n_harmonics=5),
            decompose_polynomial(layer_weights, degree=2),
            decompose_polynomial(layer_weights, degree=min(T-1, 5)),
            decompose_per_step(layer_weights),
        ]

        t_results = {'teacher': teacher_acc}

        for schedule_fn, name in decompositions:
            nca.eval()
            correct = 0; total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = nca.forward_with_schedule(x, schedule_fn, T)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            gap = acc - teacher_acc
            t_results[name] = acc
            marker = " ** LOSSLESS **" if abs(gap) < 0.005 else ""
            print(f"    {name:20s}: {acc*100:.2f}% (gap={gap*100:+.2f}%){marker}")

        results[f'T={T}'] = t_results

    # Summary
    print(f"\n{'='*70}")
    print("  NEURAL COMPILER RESULTS")
    print(f"{'='*70}")
    for t_key, t_res in results.items():
        teacher = t_res['teacher']
        oracle = t_res.get('Oracle', 0)
        best_name = max(
            [(k, v) for k, v in t_res.items() if k not in ('teacher', 'Oracle')],
            key=lambda x: x[1]
        )
        print(f"  {t_key}: teacher={teacher*100:.1f}%, "
              f"oracle={oracle*100:.1f}%, "
              f"best_compiled={best_name[0]}({best_name[1]*100:.1f}%), "
              f"gap={best_name[1]-teacher:+.4f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase126_compiler.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 126: Tensor-Decomposed NCA Compiler',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        t_vals = list(results.keys())
        methods = ['Mean', 'SVD-Rank1', 'SVD-Rank3', 'Fourier-2', 'Fourier-5',
                   'Poly-2', 'Oracle']
        colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#1abc9c',
                  '#3498db', '#9b59b6']

        for mi, method in enumerate(methods):
            accs = []
            for tk in t_vals:
                v = results[tk].get(method, 0)
                accs.append(v * 100)
            if any(a > 0 for a in accs):
                axes[0].plot(range(len(t_vals)), accs, 'o-',
                           label=method, color=colors[mi % len(colors)], markersize=4)

        teacher_accs = [results[tk]['teacher'] * 100 for tk in t_vals]
        axes[0].plot(range(len(t_vals)), teacher_accs, 'k--', label='Teacher', linewidth=2)
        axes[0].set_xticks(range(len(t_vals)))
        axes[0].set_xticklabels(t_vals)
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Compiled NCA vs Teacher CNN')
        axes[0].legend(fontsize=7)

        # Gap chart for T=5
        if 'T=5' in results:
            r5 = results['T=5']
            teacher5 = r5['teacher']
            names = [k for k in r5 if k != 'teacher']
            gaps = [(r5[k] - teacher5) * 100 for k in names]
            c = ['#2ecc71' if g >= -0.5 else '#e74c3c' for g in gaps]
            axes[1].barh(range(len(names)), gaps, color=c)
            axes[1].set_yticks(range(len(names)))
            axes[1].set_yticklabels(names, fontsize=7)
            axes[1].set_xlabel('Gap vs Teacher (%)')
            axes[1].set_title('T=5: Decomposition Quality')
            axes[1].axvline(x=0, color='black', linewidth=0.5)

        plt.suptitle('Phase 126: Neural Compiler (CNN → NCA)', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase126_compiler.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 126 complete!")


if __name__ == '__main__':
    main()
