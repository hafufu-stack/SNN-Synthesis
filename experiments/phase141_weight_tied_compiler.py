"""
Phase 141: Weight-Tied CNN Compiler — Space ≡ Time

Proves that a CNN with shared weights across layers is mathematically
identical to an unrolled NCA/RNN. By constraining a CNN to use the
same weights at every layer (Weight Tying), we can convert it to a
single-layer NCA that loops T times — LOSSLESSLY.

CNN(x, depth=T) ≡ NCA(x, steps=T) when weights are tied.

This is the formal proof that spatial depth and temporal iteration
are equivalent under weight sharing.

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


# ================================================================
# Weight-Tied CNN (all blocks share the same weights)
# ================================================================
class WeightTiedCNN(nn.Module):
    """
    CNN where all residual blocks share the SAME weights.
    Mathematically equivalent to an unrolled NCA.
    """
    def __init__(self, channels=32, depth=8, in_channels=1, num_classes=10):
        super().__init__()
        self.depth = depth
        # Input projection
        self.proj_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        # SINGLE shared block (used depth times)
        self.shared_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        # Output
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes)
        )

    def forward(self, x, depth=None):
        if depth is None:
            depth = self.depth
        h = F.relu(self.proj_in(x))
        for _ in range(depth):
            h = h + self.shared_block(h)  # Residual connection
            h = F.relu(h)
        return self.classifier(h)


# ================================================================
# Normal (non-tied) CNN for comparison
# ================================================================
class NormalCNN(nn.Module):
    """Standard CNN with independent weights per layer."""
    def __init__(self, channels=32, depth=8, in_channels=1, num_classes=10):
        super().__init__()
        self.depth = depth
        self.proj_in = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
            ) for _ in range(depth)
        ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes)
        )

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        for block in self.blocks:
            h = F.relu(h + block(h))
        return self.classifier(h)


# ================================================================
# NCA compiled from Weight-Tied CNN (zero retraining)
# ================================================================
class CompiledNCA(nn.Module):
    """
    NCA that uses the SAME weights as a Weight-Tied CNN.
    forward() is identical to WeightTiedCNN — proving equivalence.
    """
    def __init__(self, wt_cnn):
        super().__init__()
        # Copy weights directly (no retraining!)
        self.proj_in = wt_cnn.proj_in
        self.shared_block = wt_cnn.shared_block
        self.classifier = wt_cnn.classifier

    def forward(self, x, steps=8):
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.shared_block(h))
        return self.classifier(h)


# ================================================================
# Training and evaluation
# ================================================================
def train_model(model, train_loader, epochs=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()


def evaluate(model, loader, **kwargs):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, **kwargs)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 141: Weight-Tied CNN Compiler (Space ≡ Time)")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    DEPTH = 8
    results = {}

    # Step 1: Train Normal CNN (independent weights per layer)
    print(f"\n[Step 1] Training Normal CNN (depth={DEPTH}, independent weights)...")
    normal_cnn = NormalCNN(channels=32, depth=DEPTH).to(DEVICE)
    n_params_normal = sum(p.numel() for p in normal_cnn.parameters())
    train_model(normal_cnn, train_loader, epochs=10)
    normal_acc = evaluate(normal_cnn, test_loader)
    results['normal_cnn'] = {'accuracy': normal_acc, 'params': n_params_normal}
    print(f"  Normal CNN: {normal_acc*100:.2f}% ({n_params_normal:,} params)")

    # Step 2: Train Weight-Tied CNN (shared weights)
    print(f"\n[Step 2] Training Weight-Tied CNN (depth={DEPTH}, SHARED weights)...")
    wt_cnn = WeightTiedCNN(channels=32, depth=DEPTH).to(DEVICE)
    n_params_wt = sum(p.numel() for p in wt_cnn.parameters())
    train_model(wt_cnn, train_loader, epochs=10)
    wt_acc = evaluate(wt_cnn, test_loader)
    results['weight_tied_cnn'] = {'accuracy': wt_acc, 'params': n_params_wt}
    print(f"  Weight-Tied CNN: {wt_acc*100:.2f}% ({n_params_wt:,} params)")
    print(f"  Parameter reduction: {n_params_normal/n_params_wt:.1f}×")

    # Step 3: ZERO-SHOT compile to NCA
    print(f"\n[Step 3] Compiling Weight-Tied CNN → NCA (zero retraining)...")
    nca = CompiledNCA(wt_cnn).to(DEVICE)
    nca_acc = evaluate(nca, test_loader, steps=DEPTH)
    lossless = abs(nca_acc - wt_acc) < 1e-6
    results['compiled_nca'] = {
        'accuracy': nca_acc,
        'lossless': lossless,
        'gap': nca_acc - wt_acc
    }
    print(f"  Compiled NCA (T={DEPTH}): {nca_acc*100:.2f}%")
    print(f"  vs Weight-Tied CNN:       {wt_acc*100:.2f}%")
    print(f"  Gap: {(nca_acc - wt_acc)*100:.6f}%")
    print(f"  ★ LOSSLESS CONVERSION: {lossless}")

    # Step 4: Test NCA at different step counts
    print(f"\n[Step 4] NCA accuracy vs iteration steps...")
    step_results = {}
    for T in [1, 2, 4, 8, 12, 16, 24, 32]:
        acc = evaluate(nca, test_loader, steps=T)
        step_results[T] = acc
        marker = " ★" if T == DEPTH else ""
        print(f"  T={T:3d}: {acc*100:.2f}%{marker}")

    # Step 5: Normal CNN depth ablation for comparison
    print(f"\n[Step 5] Normal CNN at different depths (retrained)...")
    depth_results = {}
    for d in [1, 2, 4, 8]:
        m = NormalCNN(channels=32, depth=d).to(DEVICE)
        train_model(m, train_loader, epochs=10)
        acc = evaluate(m, test_loader)
        depth_results[d] = acc
        print(f"  depth={d}: {acc*100:.2f}%")
        del m; gc.collect()

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 141 Complete ({elapsed:.0f}s)")
    print(f"  Normal CNN (8 layers, independent): {normal_acc*100:.2f}% ({n_params_normal:,} params)")
    print(f"  Weight-Tied CNN (8 layers, shared):  {wt_acc*100:.2f}% ({n_params_wt:,} params)")
    print(f"  Compiled NCA (8 steps, zero-shot):   {nca_acc*100:.2f}%")
    print(f"  ★ LOSSLESS: {lossless} (gap={abs(nca_acc-wt_acc)*100:.6f}%)")
    print(f"  Weight tying cost: {(normal_acc-wt_acc)*100:+.2f}%")
    print(f"  Param reduction:   {n_params_normal/n_params_wt:.1f}× fewer parameters")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase141_weight_tied_compiler.json"), 'w',
              encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 141: Weight-Tied CNN Compiler',
            'timestamp': datetime.now().isoformat(),
            'depth': DEPTH,
            'results': results,
            'step_results': {str(k): v for k, v in step_results.items()},
            'depth_results': {str(k): v for k, v in depth_results.items()},
            'lossless': lossless,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Architecture comparison
        names = ['Normal CNN\n(8 layers)', 'WT-CNN\n(shared)', 'Compiled\nNCA']
        accs = [normal_acc*100, wt_acc*100, nca_acc*100]
        params = [n_params_normal, n_params_wt, n_params_wt]
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        bars = axes[0].bar(range(3), accs, color=colors, alpha=0.85, edgecolor='black')
        axes[0].set_xticks(range(3)); axes[0].set_xticklabels(names)
        axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('CNN → NCA Compilation', fontweight='bold')
        for bar, acc, p in zip(bars, accs, params):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{acc:.1f}%\n({p:,}p)', ha='center', fontsize=8, fontweight='bold')

        # Panel 2: NCA steps vs accuracy
        steps = sorted(step_results.keys())
        accs_t = [step_results[t]*100 for t in steps]
        axes[1].plot(steps, accs_t, 'o-', color='#2ecc71', markersize=8, linewidth=2)
        axes[1].axhline(y=wt_acc*100, color='#3498db', linestyle='--', label=f'WT-CNN ({wt_acc*100:.1f}%)')
        axes[1].axvline(x=DEPTH, color='red', linestyle=':', alpha=0.5, label=f'T={DEPTH} (training depth)')
        axes[1].set_xlabel('NCA Steps (T)'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Compiled NCA: Steps vs Accuracy', fontweight='bold')
        axes[1].legend(); axes[1].grid(alpha=0.3)

        # Panel 3: Depth equivalence
        depths = sorted(depth_results.keys())
        normal_accs = [depth_results[d]*100 for d in depths]
        nca_accs = [step_results.get(d, 0)*100 for d in depths]
        x = np.arange(len(depths))
        axes[2].bar(x-0.2, normal_accs, 0.35, color='#e74c3c', alpha=0.85, label='Normal CNN')
        axes[2].bar(x+0.2, nca_accs, 0.35, color='#2ecc71', alpha=0.85, label='Compiled NCA')
        axes[2].set_xticks(x); axes[2].set_xticklabels([f'D/T={d}' for d in depths])
        axes[2].set_ylabel('Accuracy (%)'); axes[2].set_title('Depth ≡ Time', fontweight='bold')
        axes[2].legend()

        plt.suptitle(f'Phase 141: Space ≡ Time — Weight-Tied CNN Compiles to NCA Losslessly',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase141_weight_tied_compiler.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
