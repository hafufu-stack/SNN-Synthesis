"""
Phase 112: CNN-to-NCA Dimensional Folding

Demonstrates that spatial depth (L layers) can be folded into
temporal steps (T iterations) of a single NCA update rule.

Core Insight:
  ResNet residual:  x_{l+1} = x_l + F_l(x_l)     # L layers in space
  NCA update:       s_{t+1} = s_t + (1-β) * Δ(s_t)  # T steps in time

These are mathematically identical (Euler discretization of an ODE).

Experiment:
  1. Train a multi-layer CNN on CIFAR-10
  2. Extract and compress weights → single NCA update rule
  3. Run NCA for T=L steps → compare output with CNN
  4. Plot folding ratio vs accuracy

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
# CNN with Residual Blocks (variable depth)
# ====================================================================
class ResBlock(nn.Module):
    """Single residual block: x + F(x) where F = Conv→BN→ReLU→Conv→BN."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ResNet_Variable(nn.Module):
    """ResNet with configurable depth for folding experiments."""
    def __init__(self, n_blocks=4, channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)
        self.n_blocks = n_blocks
        self.channels = channels

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

    def get_block_weights(self):
        """Extract all block weights for folding analysis."""
        weights = []
        for block in self.blocks:
            w = {
                'conv1_w': block.conv1.weight.data.clone(),
                'bn1_w': block.bn1.weight.data.clone(),
                'bn1_b': block.bn1.bias.data.clone(),
                'bn1_rm': block.bn1.running_mean.clone(),
                'bn1_rv': block.bn1.running_var.clone(),
                'conv2_w': block.conv2.weight.data.clone(),
                'bn2_w': block.bn2.weight.data.clone(),
                'bn2_b': block.bn2.bias.data.clone(),
                'bn2_rm': block.bn2.running_mean.clone(),
                'bn2_rv': block.bn2.running_var.clone(),
            }
            weights.append(w)
        return weights


# ====================================================================
# NCA with folded weights (single update rule, T steps)
# ====================================================================
class FoldedNCA(nn.Module):
    """
    NCA that uses a SINGLE update rule derived from CNN block weights.
    The update is applied T times (temporal unrolling = spatial folding).
    """
    def __init__(self, resnet, fold_method='mean'):
        super().__init__()
        self.stem = resnet.stem
        self.pool = resnet.pool
        self.fc = resnet.fc
        self.channels = resnet.channels

        # Fold all block weights into one
        block_weights = resnet.get_block_weights()

        if fold_method == 'mean':
            # Average all block weights
            self.folded_block = self._mean_fold(block_weights)
        elif fold_method == 'first':
            # Use only first block
            self.folded_block = self._single_fold(block_weights, 0)
        elif fold_method == 'last':
            # Use only last block
            self.folded_block = self._single_fold(block_weights, -1)
        elif fold_method == 'svd':
            # SVD-based compression of all blocks
            self.folded_block = self._svd_fold(block_weights)

    def _mean_fold(self, block_weights):
        """Average all block weights into single block."""
        block = ResBlock(self.channels).to(DEVICE)
        n = len(block_weights)
        with torch.no_grad():
            block.conv1.weight.data = sum(w['conv1_w'] for w in block_weights) / n
            block.bn1.weight.data = sum(w['bn1_w'] for w in block_weights) / n
            block.bn1.bias.data = sum(w['bn1_b'] for w in block_weights) / n
            block.bn1.running_mean = sum(w['bn1_rm'] for w in block_weights) / n
            block.bn1.running_var = sum(w['bn1_rv'] for w in block_weights) / n
            block.conv2.weight.data = sum(w['conv2_w'] for w in block_weights) / n
            block.bn2.weight.data = sum(w['bn2_w'] for w in block_weights) / n
            block.bn2.bias.data = sum(w['bn2_b'] for w in block_weights) / n
            block.bn2.running_mean = sum(w['bn2_rm'] for w in block_weights) / n
            block.bn2.running_var = sum(w['bn2_rv'] for w in block_weights) / n
        block.eval()
        return block

    def _single_fold(self, block_weights, idx):
        """Use a single block's weights."""
        block = ResBlock(self.channels).to(DEVICE)
        w = block_weights[idx]
        with torch.no_grad():
            block.conv1.weight.data = w['conv1_w']
            block.bn1.weight.data = w['bn1_w']
            block.bn1.bias.data = w['bn1_b']
            block.bn1.running_mean = w['bn1_rm']
            block.bn1.running_var = w['bn1_rv']
            block.conv2.weight.data = w['conv2_w']
            block.bn2.weight.data = w['bn2_w']
            block.bn2.bias.data = w['bn2_b']
            block.bn2.running_mean = w['bn2_rm']
            block.bn2.running_var = w['bn2_rv']
        block.eval()
        return block

    def _svd_fold(self, block_weights):
        """SVD-based: use rank-1 approximation of weight tensor stack."""
        # Stack conv1 weights: (N_blocks, C_out, C_in, K, K)
        conv1_stack = torch.stack([w['conv1_w'] for w in block_weights])
        conv2_stack = torch.stack([w['conv2_w'] for w in block_weights])

        block = ResBlock(self.channels).to(DEVICE)
        n = len(block_weights)

        with torch.no_grad():
            # SVD along block dimension, take leading component
            c1_flat = conv1_stack.view(n, -1)  # (N, C*C*K*K)
            U1, S1, V1 = torch.linalg.svd(c1_flat, full_matrices=False)
            # Reconstruct using top-1 singular value
            c1_approx = (S1[0] * U1[:, 0:1] @ V1[0:1, :]).mean(0)
            block.conv1.weight.data = c1_approx.view_as(block.conv1.weight)

            c2_flat = conv2_stack.view(n, -1)
            U2, S2, V2 = torch.linalg.svd(c2_flat, full_matrices=False)
            c2_approx = (S2[0] * U2[:, 0:1] @ V2[0:1, :]).mean(0)
            block.conv2.weight.data = c2_approx.view_as(block.conv2.weight)

            # BN: use mean
            block.bn1.weight.data = sum(w['bn1_w'] for w in block_weights) / n
            block.bn1.bias.data = sum(w['bn1_b'] for w in block_weights) / n
            block.bn1.running_mean = sum(w['bn1_rm'] for w in block_weights) / n
            block.bn1.running_var = sum(w['bn1_rv'] for w in block_weights) / n
            block.bn2.weight.data = sum(w['bn2_w'] for w in block_weights) / n
            block.bn2.bias.data = sum(w['bn2_b'] for w in block_weights) / n
            block.bn2.running_mean = sum(w['bn2_rm'] for w in block_weights) / n
            block.bn2.running_var = sum(w['bn2_rv'] for w in block_weights) / n
        block.eval()
        return block

    def forward(self, x, n_steps=None):
        x = self.stem(x)
        self.folded_block.eval()
        for t in range(n_steps):
            x = self.folded_block(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ====================================================================
# Training and Evaluation
# ====================================================================
def train_resnet(model, train_loader, epochs=30, lr=0.01):
    """Train ResNet on CIFAR-10."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * y.size(0); n += y.size(0)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}")

    return model


def evaluate(model, test_loader, **kwargs):
    """Evaluate accuracy."""
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, **kwargs)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 112: CNN→NCA Dimensional Folding")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load CIFAR-10
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

    # Test with different CNN depths
    depths = [2, 4, 6, 8]
    fold_methods = ['mean', 'first', 'last', 'svd']
    all_results = []

    for n_blocks in depths:
        print(f"\n{'='*50}")
        print(f"  ResNet with {n_blocks} residual blocks")
        print(f"{'='*50}")

        # Step 1: Train ResNet
        print(f"  [Training] ResNet-{n_blocks}...")
        resnet = ResNet_Variable(n_blocks=n_blocks, channels=64).to(DEVICE)
        resnet = train_resnet(resnet, train_loader, epochs=30)
        cnn_acc = evaluate(resnet, test_loader)
        print(f"  CNN accuracy: {cnn_acc*100:.2f}%")

        # Step 2: Fold into NCA and test at different T
        t_values = [1, 2, 3, n_blocks//2, n_blocks, n_blocks*2, n_blocks*3]
        t_values = sorted(set([max(1, t) for t in t_values]))

        for method in fold_methods:
            try:
                nca = FoldedNCA(resnet, fold_method=method).to(DEVICE)
            except Exception as e:
                print(f"  [{method}] Error: {e}")
                continue

            for T in t_values:
                nca_acc = evaluate(nca, test_loader, n_steps=T)
                gap = nca_acc - cnn_acc
                marker = " ★" if abs(gap) < 0.02 else ""
                result = {
                    'n_blocks': n_blocks, 'fold_method': method,
                    'T': T, 'cnn_acc': cnn_acc, 'nca_acc': nca_acc,
                    'gap': gap, 'fold_ratio': n_blocks / 1  # L layers → 1 rule
                }
                all_results.append(result)
                print(f"    [{method:5s}] T={T:2d}: {nca_acc*100:.2f}% "
                      f"(gap={gap*100:+.2f}%){marker}")

            del nca; gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        del resnet; gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("  DIMENSIONAL FOLDING RESULTS SUMMARY")
    print(f"{'='*70}")

    # Best result per depth
    for n_blocks in depths:
        depth_results = [r for r in all_results if r['n_blocks'] == n_blocks]
        if not depth_results:
            continue
        best = max(depth_results, key=lambda r: r['nca_acc'])
        cnn = depth_results[0]['cnn_acc']
        print(f"  ResNet-{n_blocks}: CNN={cnn*100:.2f}% → "
              f"Best NCA={best['nca_acc']*100:.2f}% "
              f"({best['fold_method']}, T={best['T']}, "
              f"gap={best['gap']*100:+.2f}%)")

    # Find the sweet spot
    best_overall = max(all_results, key=lambda r: r['nca_acc'])
    print(f"\n  ★ BEST OVERALL: ResNet-{best_overall['n_blocks']} → "
          f"NCA ({best_overall['fold_method']}, T={best_overall['T']})")
    print(f"    NCA accuracy: {best_overall['nca_acc']*100:.2f}% "
          f"(CNN: {best_overall['cnn_acc']*100:.2f}%, "
          f"gap: {best_overall['gap']*100:+.2f}%)")
    print(f"    Folding ratio: {best_overall['n_blocks']}:1 "
          f"(parameter reduction: ~{best_overall['n_blocks']}×)")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase112_cnn_to_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 112: CNN→NCA Dimensional Folding',
            'timestamp': datetime.now().isoformat(),
            'best_overall': best_overall,
            'all_results': all_results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Folding accuracy vs T for each depth
        for n_blocks in depths:
            dr = [r for r in all_results if r['n_blocks'] == n_blocks and r['fold_method'] == 'mean']
            if dr:
                ts = [r['T'] for r in dr]
                accs = [r['nca_acc']*100 for r in dr]
                cnn = dr[0]['cnn_acc']*100
                axes[0, 0].plot(ts, accs, 'o-', label=f'NCA from ResNet-{n_blocks}', markersize=5)
                axes[0, 0].axhline(y=cnn, linestyle='--', alpha=0.3)
        axes[0, 0].set_xlabel('T (NCA steps)')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Folding: NCA Accuracy vs Time Steps (mean fold)')
        axes[0, 0].legend()

        # Plot 2: Fold method comparison at T=n_blocks
        method_accs = {m: [] for m in fold_methods}
        for n_blocks in depths:
            for method in fold_methods:
                r = [x for x in all_results if x['n_blocks'] == n_blocks
                     and x['fold_method'] == method and x['T'] == n_blocks]
                if r:
                    method_accs[method].append(r[0]['nca_acc']*100)
                else:
                    method_accs[method].append(0)
        x = np.arange(len(depths))
        w = 0.2
        for i, (method, accs) in enumerate(method_accs.items()):
            if accs:
                axes[0, 1].bar(x + i*w, accs, w, label=method)
        cnn_accs = [next((r['cnn_acc']*100 for r in all_results if r['n_blocks'] == d), 0) for d in depths]
        axes[0, 1].plot(x + 1.5*w, cnn_accs, 'r*', markersize=12, label='CNN')
        axes[0, 1].set_xticks(x + 1.5*w)
        axes[0, 1].set_xticklabels([f'ResNet-{d}' for d in depths])
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Fold Method Comparison (T=L)')
        axes[0, 1].legend(fontsize=8)

        # Plot 3: Gap heatmap
        gap_matrix = np.full((len(depths), len(fold_methods)), np.nan)
        for i, d in enumerate(depths):
            for j, m in enumerate(fold_methods):
                r = [x for x in all_results if x['n_blocks'] == d
                     and x['fold_method'] == m and x['T'] == d]
                if r:
                    gap_matrix[i, j] = r[0]['gap']*100
        im = axes[1, 0].imshow(gap_matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=5)
        axes[1, 0].set_xticks(range(len(fold_methods)))
        axes[1, 0].set_xticklabels(fold_methods)
        axes[1, 0].set_yticks(range(len(depths)))
        axes[1, 0].set_yticklabels([f'ResNet-{d}' for d in depths])
        axes[1, 0].set_title('Gap (%) at T=L')
        plt.colorbar(im, ax=axes[1, 0])
        for i in range(len(depths)):
            for j in range(len(fold_methods)):
                if not np.isnan(gap_matrix[i, j]):
                    axes[1, 0].text(j, i, f'{gap_matrix[i, j]:.1f}',
                                   ha='center', va='center', fontsize=8)

        # Plot 4: Parameter efficiency
        cnn_params = [d * 2 * 64 * 64 * 9 for d in depths]  # approximate
        nca_params = [2 * 64 * 64 * 9] * len(depths)  # single block
        axes[1, 1].bar(x - 0.2, [p/1000 for p in cnn_params], 0.4, label='CNN', color='#e74c3c')
        axes[1, 1].bar(x + 0.2, [p/1000 for p in nca_params], 0.4, label='NCA (folded)', color='#2ecc71')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'L={d}' for d in depths])
        axes[1, 1].set_ylabel('Parameters (K)')
        axes[1, 1].set_title('Parameter Reduction via Folding')
        axes[1, 1].legend()

        plt.suptitle('Phase 112: CNN→NCA Dimensional Folding', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase112_cnn_to_nca.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 112 complete!")


if __name__ == '__main__':
    main()
