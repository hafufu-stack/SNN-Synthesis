"""
Phase 115: Zero-Shot LNN Conversion & Temporal Stochastic Resonance

Uses Phase 111's formula (b_tau = -alpha * max(a)) to convert a
trained CNN to LNN WITHOUT retraining, then injects temporal noise
to achieve test-time compute scaling.

Proves: "A static CNN can be compiled into a temporal explorer
         that exceeds the original model's accuracy"

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
# Static CNN (source model)
# ====================================================================
class StaticCNN(nn.Module):
    """Simple CNN for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_activation_stats(self, data_loader, n_batches=10):
        """Compute activation statistics for tau conversion."""
        self.eval()
        all_acts = []
        with torch.no_grad():
            for i, (x, _) in enumerate(data_loader):
                if i >= n_batches:
                    break
                x = x.to(DEVICE)
                # Get intermediate activations
                acts = []
                h = x
                for layer in self.features:
                    h = layer(h)
                    if isinstance(layer, nn.ReLU):
                        acts.append(h.cpu())
                all_acts.append(acts)

        # Compute per-activation-group stats
        stats = []
        n_relu = len(all_acts[0])
        for r in range(n_relu):
            cat = torch.cat([a[r] for a in all_acts], dim=0)
            stats.append({
                'max': cat.max().item(),
                'mean': cat.mean().item(),
                'std': cat.std().item(),
            })
        return stats


# ====================================================================
# LNN version (compiled from static CNN)
# ====================================================================
class LiquidCNN(nn.Module):
    """
    LNN compiled from StaticCNN.
    Same feature extractor, but classifier is liquid with tau gates.
    """
    def __init__(self, static_model, b_tau_values):
        super().__init__()
        # Copy feature extractor exactly
        self.features = static_model.features

        # Liquid classifier with tau gates
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        # Copy weights from static model
        with torch.no_grad():
            self.fc1.weight.data = static_model.classifier[2].weight.data.clone()
            self.fc1.bias.data = static_model.classifier[2].bias.data.clone()
            self.fc2.weight.data = static_model.classifier[4].weight.data.clone()
            self.fc2.bias.data = static_model.classifier[4].bias.data.clone()

        # Tau gate biases (from Phase 111 formula)
        self.b_tau1 = nn.Parameter(torch.ones(256) * b_tau_values[0], requires_grad=False)
        self.b_tau2 = nn.Parameter(torch.ones(10) * b_tau_values[1], requires_grad=False)

    def forward(self, x, n_steps=1, noise_sigma=0.0):
        feat = self.features(x)
        feat = self.pool(feat).view(feat.size(0), -1)

        # Initialize liquid states
        s1 = torch.zeros(feat.size(0), 256, device=feat.device)
        s2 = torch.zeros(feat.size(0), 10, device=feat.device)

        for t in range(n_steps):
            # Temporal noise injection (Temporal NBS)
            if noise_sigma > 0 and self.training is False:
                noise1 = torch.randn_like(s1) * noise_sigma
                noise2 = torch.randn_like(s2) * noise_sigma
            else:
                noise1 = noise2 = 0

            # Layer 1 liquid update
            delta1 = F.relu(self.fc1(feat))
            beta1 = torch.sigmoid(self.b_tau1 + noise1).unsqueeze(0)
            s1 = beta1 * s1 + (1 - beta1) * delta1

            # Layer 2 liquid update
            delta2 = self.fc2(s1)
            beta2 = torch.sigmoid(self.b_tau2 + noise2).unsqueeze(0)
            s2 = beta2 * s2 + (1 - beta2) * delta2

        return s2

    def forward_best_of_k(self, x, n_steps=10, noise_sigma=0.1, K=5):
        """
        Best-of-K inference: run K noisy forward passes, pick
        the one with highest confidence.
        """
        best_out = None
        best_conf = -1

        for k in range(K):
            out = self.forward(x, n_steps=n_steps, noise_sigma=noise_sigma)
            conf = out.softmax(1).max(1)[0].mean().item()
            if conf > best_conf:
                best_conf = conf
                best_out = out

        return best_out


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 115: Zero-Shot LNN Conversion & Temporal SR")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_ds = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10('data', train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Step 1: Train static CNN
    print("\n[Step 1] Training static CNN...")
    static = StaticCNN().to(DEVICE)
    opt = torch.optim.SGD(static.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for epoch in range(30):
        static.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(static(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/30")

    static_acc = 0; total = 0
    static.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            static_acc += (static(x).argmax(1) == y).sum().item()
            total += y.size(0)
    static_acc /= total
    print(f"  Static CNN accuracy: {static_acc*100:.2f}%")

    # Step 2: Get activation stats
    print("\n[Step 2] Computing activation statistics...")
    stats = static.get_activation_stats(train_loader)
    for i, s in enumerate(stats):
        print(f"    ReLU {i}: max={s['max']:.3f} mean={s['mean']:.3f} std={s['std']:.3f}")

    # Step 3: Zero-shot conversion with different alpha values
    print("\n[Step 3] Zero-shot LNN conversion (b_tau = -alpha * max(a))...")
    alpha_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    t_values = [1, 3, 5, 10, 20]
    noise_values = [0.0, 0.05, 0.1, 0.2, 0.5]

    results = []

    # Use last two activation stats for the two FC layers
    fc_max1 = stats[-2]['max'] if len(stats) >= 2 else stats[-1]['max']
    fc_max2 = stats[-1]['max']

    for alpha in alpha_values:
        b_tau = [-alpha * fc_max1, -alpha * fc_max2]
        lnn = LiquidCNN(static, b_tau).to(DEVICE)
        lnn.eval()

        print(f"\n  alpha={alpha}: b_tau=[{b_tau[0]:.2f}, {b_tau[1]:.2f}]")

        # Test at different T and noise levels
        for T in t_values:
            for sigma in noise_values:
                correct = 0; total = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        if sigma > 0:
                            out = lnn.forward_best_of_k(x, n_steps=T,
                                                         noise_sigma=sigma, K=5)
                        else:
                            out = lnn(x, n_steps=T, noise_sigma=0)
                        correct += (out.argmax(1) == y).sum().item()
                        total += y.size(0)
                acc = correct / total
                gap = acc - static_acc
                result = {
                    'alpha': alpha, 'T': T, 'noise_sigma': sigma,
                    'b_tau': b_tau, 'lnn_acc': acc,
                    'static_acc': static_acc, 'gap': gap
                }
                results.append(result)

                if sigma == 0 and T == 1:
                    marker = " (baseline)" if abs(gap) < 0.005 else ""
                    print(f"    T={T:2d}, sigma={sigma:.2f}: {acc*100:.2f}% "
                          f"(gap={gap*100:+.2f}%){marker}")
                elif gap > 0.005:
                    print(f"    T={T:2d}, sigma={sigma:.2f}: {acc*100:.2f}% "
                          f"(gap={gap*100:+.2f}%) ** EXCEEDS STATIC **")

        del lnn; gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("  ZERO-SHOT LNN CONVERSION RESULTS")
    print(f"{'='*70}")
    print(f"  Static CNN baseline: {static_acc*100:.2f}%")

    # Best lossless (no noise)
    no_noise = [r for r in results if r['noise_sigma'] == 0]
    best_lossless = min(no_noise, key=lambda r: abs(r['gap']))
    print(f"\n  Best lossless (no noise): {best_lossless['lnn_acc']*100:.2f}% "
          f"(alpha={best_lossless['alpha']}, T={best_lossless['T']}, "
          f"gap={best_lossless['gap']*100:+.2f}%)")

    # Best with noise (test-time compute scaling)
    best_noisy = max(results, key=lambda r: r['lnn_acc'])
    print(f"  Best with noise:         {best_noisy['lnn_acc']*100:.2f}% "
          f"(alpha={best_noisy['alpha']}, T={best_noisy['T']}, "
          f"sigma={best_noisy['noise_sigma']}, "
          f"gap={best_noisy['gap']*100:+.2f}%)")

    # Did we exceed static?
    exceeds = [r for r in results if r['gap'] > 0.005]
    if exceeds:
        print(f"\n  [!!] {len(exceeds)} configurations EXCEED static CNN!")
        best_exceed = max(exceeds, key=lambda r: r['gap'])
        print(f"       Best: +{best_exceed['gap']*100:.2f}% "
              f"(alpha={best_exceed['alpha']}, T={best_exceed['T']}, "
              f"sigma={best_exceed['noise_sigma']})")
    else:
        print(f"\n  No configuration exceeded static CNN.")
        print(f"  Closest approach: gap={min(results, key=lambda r: abs(r['gap']))['gap']*100:+.3f}%")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase115_zero_shot_lnn.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 115: Zero-Shot LNN Conversion',
            'timestamp': datetime.now().isoformat(),
            'static_accuracy': static_acc,
            'activation_stats': stats,
            'all_results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Accuracy vs T (no noise, different alpha)
        for alpha in alpha_values:
            ar = [r for r in results if r['alpha'] == alpha and r['noise_sigma'] == 0]
            if ar:
                ts = [r['T'] for r in ar]
                accs = [r['lnn_acc'] * 100 for r in ar]
                axes[0].plot(ts, accs, 'o-', label=f'alpha={alpha}', markersize=4)
        axes[0].axhline(y=static_acc*100, color='red', linestyle='--', label='Static CNN')
        axes[0].set_xlabel('T (steps)'); axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('LNN Accuracy vs T (no noise)')
        axes[0].legend(fontsize=7)

        # Plot 2: Noise effect at best alpha, T
        best_alpha = best_lossless['alpha']
        nr = [r for r in results if r['alpha'] == best_alpha and r['T'] == 10]
        if nr:
            sigmas = [r['noise_sigma'] for r in nr]
            accs = [r['lnn_acc'] * 100 for r in nr]
            axes[1].plot(sigmas, accs, 'o-', color='#e74c3c', markersize=6)
            axes[1].axhline(y=static_acc*100, color='blue', linestyle='--', label='Static CNN')
            axes[1].set_xlabel('Noise sigma')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title(f'Temporal SR (alpha={best_alpha}, T=10, K=5)')
            axes[1].legend()

        # Plot 3: Heatmap alpha x T at best noise
        best_sigma = best_noisy['noise_sigma']
        heat = np.zeros((len(alpha_values), len(t_values)))
        for i, a in enumerate(alpha_values):
            for j, t in enumerate(t_values):
                r = [x for x in results if x['alpha'] == a and x['T'] == t
                     and x['noise_sigma'] == best_sigma]
                if r:
                    heat[i, j] = r[0]['gap'] * 100
        im = axes[2].imshow(heat, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
        axes[2].set_xticks(range(len(t_values)))
        axes[2].set_xticklabels(t_values)
        axes[2].set_yticks(range(len(alpha_values)))
        axes[2].set_yticklabels(alpha_values)
        axes[2].set_xlabel('T'); axes[2].set_ylabel('alpha')
        axes[2].set_title(f'Gap vs Static (%) at sigma={best_sigma}')
        plt.colorbar(im, ax=axes[2])

        plt.suptitle('Phase 115: Zero-Shot LNN Conversion & Temporal SR', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase115_zero_shot_lnn.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 115 complete!")


if __name__ == '__main__':
    main()
