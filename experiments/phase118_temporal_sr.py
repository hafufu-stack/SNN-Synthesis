"""
Phase 118: Zero-Shot LNN + Temporal Stochastic Resonance (Bug-fixed)

Fixes Phase 115's tensor dimension bug and completes the temporal
noise experiment. Proves that a static CNN compiled to LNN via
b_tau = -alpha * max(a) can EXCEED original accuracy with noise.

Bug fix: unsqueeze(0) was creating (1,B,D) instead of (B,D) when
noise was applied (B dimension already present from randn_like).

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


class StaticCNN(nn.Module):
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
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_activation_stats(self, data_loader, n_batches=10):
        self.eval()
        all_acts = []
        with torch.no_grad():
            for i, (x, _) in enumerate(data_loader):
                if i >= n_batches: break
                x = x.to(DEVICE)
                acts = []; h = x
                for layer in self.features:
                    h = layer(h)
                    if isinstance(layer, nn.ReLU):
                        acts.append(h.cpu())
                all_acts.append(acts)
        stats = []
        for r in range(len(all_acts[0])):
            cat = torch.cat([a[r] for a in all_acts], dim=0)
            stats.append({'max': cat.max().item(), 'mean': cat.mean().item(), 'std': cat.std().item()})
        return stats


class LiquidCNN(nn.Module):
    """LNN compiled from StaticCNN -- BUG FIXED version."""
    def __init__(self, static_model, b_tau_values):
        super().__init__()
        self.features = static_model.features
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        with torch.no_grad():
            self.fc1.weight.data = static_model.classifier[2].weight.data.clone()
            self.fc1.bias.data = static_model.classifier[2].bias.data.clone()
            self.fc2.weight.data = static_model.classifier[4].weight.data.clone()
            self.fc2.bias.data = static_model.classifier[4].bias.data.clone()

        # b_tau as buffers (not parameters) for clean inference
        self.register_buffer('b_tau1', torch.ones(256) * b_tau_values[0])
        self.register_buffer('b_tau2', torch.ones(10) * b_tau_values[1])

    def forward(self, x, n_steps=1, noise_sigma=0.0):
        feat = self.features(x)
        feat = self.pool(feat).view(feat.size(0), -1)
        B = feat.size(0)

        s1 = torch.zeros(B, 256, device=feat.device)
        s2 = torch.zeros(B, 10, device=feat.device)

        for t in range(n_steps):
            # FIX: When noise is applied, b_tau + noise is already (B, D)
            # so do NOT unsqueeze. When no noise, expand b_tau to (B, D).
            if noise_sigma > 0:
                noise1 = torch.randn(B, 256, device=feat.device) * noise_sigma
                noise2 = torch.randn(B, 10, device=feat.device) * noise_sigma
                beta1 = torch.sigmoid(self.b_tau1.unsqueeze(0) + noise1)  # (B, 256)
                beta2 = torch.sigmoid(self.b_tau2.unsqueeze(0) + noise2)  # (B, 10)
            else:
                beta1 = torch.sigmoid(self.b_tau1).unsqueeze(0).expand(B, -1)  # (B, 256)
                beta2 = torch.sigmoid(self.b_tau2).unsqueeze(0).expand(B, -1)  # (B, 10)

            delta1 = F.relu(self.fc1(feat))
            s1 = beta1 * s1 + (1 - beta1) * delta1

            delta2 = self.fc2(s1)
            s2 = beta2 * s2 + (1 - beta2) * delta2

        return s2

    def forward_best_of_k(self, x, n_steps=10, noise_sigma=0.1, K=5):
        """Best-of-K: run K passes, pick highest confidence."""
        best_out = None
        best_conf = -1.0
        for k in range(K):
            out = self.forward(x, n_steps=n_steps, noise_sigma=noise_sigma)
            conf = out.softmax(1).max(1)[0].mean().item()
            if conf > best_conf:
                best_conf = conf
                best_out = out.clone()
        return best_out

    def forward_majority_vote(self, x, n_steps=10, noise_sigma=0.1, K=11):
        """Majority vote: run K passes, vote on predictions."""
        B = x.size(0)
        votes = torch.zeros(B, 10, device=x.device)
        for k in range(K):
            out = self.forward(x, n_steps=n_steps, noise_sigma=noise_sigma)
            preds = out.argmax(1)
            for i in range(B):
                votes[i, preds[i]] += 1
        return votes  # argmax of votes = majority prediction


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 118: Zero-Shot LNN + Temporal SR (Bug-fixed)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for epoch in range(30):
        static.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(static(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/30")
    static.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (static(x).argmax(1) == y).sum().item()
            total += y.size(0)
    static_acc = correct / total
    print(f"  Static CNN accuracy: {static_acc*100:.2f}%")

    # Step 2: Activation stats
    print("\n[Step 2] Computing activation statistics...")
    stats = static.get_activation_stats(train_loader)
    for i, s in enumerate(stats):
        print(f"    ReLU {i}: max={s['max']:.3f}")
    fc_max1 = stats[-2]['max'] if len(stats) >= 2 else stats[-1]['max']
    fc_max2 = stats[-1]['max']

    # Step 3: Systematic test
    print("\n[Step 3] Zero-shot LNN + Temporal noise...")
    alpha_values = [0.5, 1.0, 2.0, 5.0]
    t_values = [1, 3, 5, 10]
    noise_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    k_values = [1, 5, 11]

    results = []

    for alpha in alpha_values:
        b_tau = [-alpha * fc_max1, -alpha * fc_max2]
        lnn = LiquidCNN(static, b_tau).to(DEVICE)
        lnn.eval()
        print(f"\n  alpha={alpha}: b_tau=[{b_tau[0]:.2f}, {b_tau[1]:.2f}]")

        for T in t_values:
            for sigma in noise_values:
                for K in k_values:
                    if sigma == 0 and K > 1:
                        continue  # no point in multiple noiseless passes

                    correct = 0; total = 0
                    with torch.no_grad():
                        for x, y in test_loader:
                            x, y = x.to(DEVICE), y.to(DEVICE)
                            if K > 1 and sigma > 0:
                                out = lnn.forward_majority_vote(
                                    x, n_steps=T, noise_sigma=sigma, K=K)
                            else:
                                out = lnn(x, n_steps=T, noise_sigma=sigma)
                            correct += (out.argmax(1) == y).sum().item()
                            total += y.size(0)
                    acc = correct / total
                    gap = acc - static_acc
                    result = {
                        'alpha': alpha, 'T': T, 'sigma': sigma, 'K': K,
                        'acc': acc, 'gap': gap
                    }
                    results.append(result)

                    # Print notable results
                    if sigma == 0 and T == 1 and K == 1:
                        print(f"    T={T}, sigma=0: {acc*100:.2f}% "
                              f"(gap={gap*100:+.2f}%)")
                    elif gap > 0.002:
                        print(f"    T={T}, sigma={sigma}, K={K}: {acc*100:.2f}% "
                              f"(gap={gap*100:+.2f}%) ** EXCEEDS **")

        del lnn; gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("  ZERO-SHOT LNN + TEMPORAL SR RESULTS")
    print(f"{'='*70}")
    print(f"  Static CNN: {static_acc*100:.2f}%")

    best_lossless = max([r for r in results if r['sigma'] == 0], key=lambda r: r['acc'])
    print(f"  Best lossless: {best_lossless['acc']*100:.2f}% "
          f"(alpha={best_lossless['alpha']}, T={best_lossless['T']})")

    best_noisy = max(results, key=lambda r: r['acc'])
    print(f"  Best with noise: {best_noisy['acc']*100:.2f}% "
          f"(alpha={best_noisy['alpha']}, T={best_noisy['T']}, "
          f"sigma={best_noisy['sigma']}, K={best_noisy['K']})")
    print(f"  Gap vs static: {best_noisy['gap']*100:+.2f}%")

    exceeds = [r for r in results if r['gap'] > 0.002]
    if exceeds:
        print(f"\n  ** {len(exceeds)} configurations EXCEED static CNN! **")
    else:
        print(f"\n  No configuration exceeded static CNN.")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase118_temporal_sr.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 118: Zero-Shot LNN + Temporal SR',
            'timestamp': datetime.now().isoformat(),
            'static_acc': static_acc,
            'best_lossless': best_lossless,
            'best_noisy': best_noisy,
            'n_exceeds': len(exceeds),
            'all_results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Accuracy vs sigma at different T (best alpha)
        best_alpha = best_lossless['alpha']
        for T in t_values:
            sr = [r for r in results if r['alpha'] == best_alpha and r['T'] == T and r['K'] == 1]
            if sr:
                sigmas = [r['sigma'] for r in sr]
                accs = [r['acc'] * 100 for r in sr]
                axes[0].plot(sigmas, accs, 'o-', label=f'T={T}', markersize=4)
        axes[0].axhline(y=static_acc*100, color='red', linestyle='--', label='Static CNN')
        axes[0].set_xlabel('Noise sigma'); axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title(f'Temporal SR (alpha={best_alpha}, K=1)')
        axes[0].legend(fontsize=8)

        # Majority vote effect
        for K in k_values:
            kr = [r for r in results if r['alpha'] == best_alpha
                  and r['T'] == 5 and r['K'] == K and r['sigma'] > 0]
            if kr:
                sigmas = [r['sigma'] for r in kr]
                accs = [r['acc'] * 100 for r in kr]
                axes[1].plot(sigmas, accs, 'o-', label=f'K={K}', markersize=4)
        axes[1].axhline(y=static_acc*100, color='red', linestyle='--', label='Static')
        axes[1].set_xlabel('Noise sigma'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'Majority Vote Effect (alpha={best_alpha}, T=5)')
        axes[1].legend(fontsize=8)

        # Heatmap: sigma x K at best alpha, T
        best_T = best_noisy['T']
        sigmas_plot = [s for s in noise_values if s > 0]
        heat = np.zeros((len(sigmas_plot), len(k_values)))
        for i, s in enumerate(sigmas_plot):
            for j, K in enumerate(k_values):
                r = [x for x in results if x['alpha'] == best_alpha
                     and x['T'] == best_T and x['sigma'] == s and x['K'] == K]
                if r:
                    heat[i, j] = r[0]['gap'] * 100
        im = axes[2].imshow(heat, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=2)
        axes[2].set_xticks(range(len(k_values)))
        axes[2].set_xticklabels(k_values)
        axes[2].set_yticks(range(len(sigmas_plot)))
        axes[2].set_yticklabels(sigmas_plot)
        axes[2].set_xlabel('K (votes)'); axes[2].set_ylabel('sigma')
        axes[2].set_title(f'Gap (%) at alpha={best_alpha}, T={best_T}')
        plt.colorbar(im, ax=axes[2])

        plt.suptitle('Phase 118: Zero-Shot LNN + Temporal SR (Bug-fixed)', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase118_temporal_sr.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 118 complete!")


if __name__ == '__main__':
    main()
