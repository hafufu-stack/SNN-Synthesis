"""
Phase 144: Grid Weight-Tied Compiler — Space === Time on Grid Tasks

Game of Life N-step prediction: a pure LOCAL rule task, PERFECT for NCA.
Weight-Tied CNN (shared weights) learns GoL → compile to NCA losslessly.

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
GRID_SIZE = 16
GOL_STEPS = 5


# ================================================================
# Game of Life Dataset (on-the-fly generation)
# ================================================================
def game_of_life_step(grid):
    """One step of Conway's Game of Life (toroidal boundary)."""
    # Count neighbors using conv2d
    kernel = torch.ones(1, 1, 3, 3, device=grid.device)
    kernel[0, 0, 1, 1] = 0  # exclude self
    g = grid.float().unsqueeze(0).unsqueeze(0) if grid.dim() == 2 else grid.float()
    if g.dim() == 3:
        g = g.unsqueeze(0)
    # Pad toroidally
    g_pad = F.pad(g, (1, 1, 1, 1), mode='circular')
    neighbors = F.conv2d(g_pad, kernel).squeeze()
    # Rules: alive if (alive AND 2-3 neighbors) OR (dead AND 3 neighbors)
    survive = (grid > 0.5) & ((neighbors == 2) | (neighbors == 3))
    birth = (grid < 0.5) & (neighbors == 3)
    return (survive | birth).float()


def generate_gol_dataset(n_samples, grid_size=GRID_SIZE, steps=GOL_STEPS):
    """Generate (input, target) pairs for GoL prediction."""
    inputs = []
    targets = []
    for _ in range(n_samples):
        grid = (torch.rand(grid_size, grid_size) > 0.5).float()
        inputs.append(grid.clone())
        for _ in range(steps):
            grid = game_of_life_step(grid)
        targets.append(grid)
    return torch.stack(inputs).unsqueeze(1), torch.stack(targets).unsqueeze(1)


class GoLDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=5000):
        self.X, self.Y = generate_gol_dataset(n_samples)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Weight-Tied CNN (shared block, grid-to-grid)
# ================================================================
class WeightTiedGridCNN(nn.Module):
    def __init__(self, channels=32, depth=GOL_STEPS):
        super().__init__()
        self.depth = depth
        self.proj_in = nn.Conv2d(1, channels, 3, padding=1)
        self.shared_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.proj_out = nn.Conv2d(channels, 1, 1)

    def forward(self, x, depth=None):
        if depth is None: depth = self.depth
        h = F.relu(self.proj_in(x))
        for _ in range(depth):
            h = F.relu(h + self.shared_block(h))
        return self.proj_out(h)


# ================================================================
# Normal CNN (independent weights, grid-to-grid)
# ================================================================
class NormalGridCNN(nn.Module):
    def __init__(self, channels=32, depth=GOL_STEPS):
        super().__init__()
        self.depth = depth
        self.proj_in = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            ) for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        for block in self.blocks:
            h = F.relu(h + block(h))
        return self.proj_out(h)


# ================================================================
# Compiled NCA (zero-shot from Weight-Tied CNN)
# ================================================================
class CompiledGridNCA(nn.Module):
    def __init__(self, wt_cnn):
        super().__init__()
        self.proj_in = wt_cnn.proj_in
        self.shared_block = wt_cnn.shared_block
        self.proj_out = wt_cnn.proj_out

    def forward(self, x, steps=GOL_STEPS):
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.shared_block(h))
        return self.proj_out(h)


# ================================================================
# Training
# ================================================================
def train_grid_model(model, train_loader, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = F.binary_cross_entropy_with_logits(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()


def evaluate_grid(model, loader, **kwargs):
    model.eval()
    total_correct = total_pixels = 0
    total_exact = total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = (torch.sigmoid(model(x, **kwargs)) > 0.5).float()
            total_correct += (pred == y).sum().item()
            total_pixels += y.numel()
            exact = (pred == y).all(dim=-1).all(dim=-1).all(dim=-1)
            total_exact += exact.sum().item()
            total_samples += y.size(0)
    pixel_acc = total_correct / total_pixels
    exact_match = total_exact / total_samples
    return pixel_acc, exact_match


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 144: Grid Weight-Tied Compiler (Space === Time)")
    print(f"  Task: Game of Life {GOL_STEPS}-step prediction on {GRID_SIZE}x{GRID_SIZE}")
    print("=" * 70)

    train_ds = GoLDataset(5000)
    test_ds = GoLDataset(1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    results = {}

    # Step 1: Normal CNN
    print(f"\n[Step 1] Training Normal CNN (depth={GOL_STEPS})...")
    normal = NormalGridCNN().to(DEVICE)
    n_normal = sum(p.numel() for p in normal.parameters())
    train_grid_model(normal, train_loader, epochs=30)
    pa, em = evaluate_grid(normal, test_loader)
    results['normal_cnn'] = {'pixel_acc': pa, 'exact_match': em, 'params': n_normal}
    print(f"  Normal CNN: PA={pa*100:.2f}%, EM={em*100:.2f}% ({n_normal:,} params)")

    # Step 2: Weight-Tied CNN
    print(f"\n[Step 2] Training Weight-Tied CNN (depth={GOL_STEPS})...")
    wt = WeightTiedGridCNN().to(DEVICE)
    n_wt = sum(p.numel() for p in wt.parameters())
    train_grid_model(wt, train_loader, epochs=30)
    pa_wt, em_wt = evaluate_grid(wt, test_loader)
    results['weight_tied'] = {'pixel_acc': pa_wt, 'exact_match': em_wt, 'params': n_wt}
    print(f"  WT-CNN: PA={pa_wt*100:.2f}%, EM={em_wt*100:.2f}% ({n_wt:,} params)")

    # Step 3: Compile to NCA
    print(f"\n[Step 3] Compiling WT-CNN -> NCA (zero retraining)...")
    nca = CompiledGridNCA(wt).to(DEVICE)
    pa_nca, em_nca = evaluate_grid(nca, test_loader, steps=GOL_STEPS)
    gap = abs(pa_nca - pa_wt)
    lossless = gap < 1e-6
    results['compiled_nca'] = {'pixel_acc': pa_nca, 'exact_match': em_nca, 'lossless': lossless}
    print(f"  NCA (T={GOL_STEPS}): PA={pa_nca*100:.2f}%, EM={em_nca*100:.2f}%")
    print(f"  Gap: {gap*100:.6f}%")
    print(f"  LOSSLESS: {lossless}")

    # Step 4: NCA at different T
    print(f"\n[Step 4] NCA at different step counts...")
    step_sweep = {}
    for T in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]:
        pa_t, em_t = evaluate_grid(nca, test_loader, steps=T)
        step_sweep[T] = {'pixel_acc': pa_t, 'exact_match': em_t}
        marker = " <<<" if T == GOL_STEPS else ""
        print(f"  T={T:3d}: PA={pa_t*100:.2f}%, EM={em_t*100:.2f}%{marker}")

    # Step 5: NCA from scratch
    print(f"\n[Step 5] NCA trained from scratch...")
    from_scratch = WeightTiedGridCNN().to(DEVICE)
    train_grid_model(from_scratch, train_loader, epochs=30)
    pa_s, em_s = evaluate_grid(from_scratch, test_loader)
    results['nca_scratch'] = {'pixel_acc': pa_s, 'exact_match': em_s}
    print(f"  NCA scratch: PA={pa_s*100:.2f}%, EM={em_s*100:.2f}%")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 144 Complete ({elapsed:.0f}s)")
    print(f"  Normal CNN:    PA={results['normal_cnn']['pixel_acc']*100:.2f}%")
    print(f"  WT-CNN:        PA={results['weight_tied']['pixel_acc']*100:.2f}%")
    print(f"  Compiled NCA:  PA={results['compiled_nca']['pixel_acc']*100:.2f}% (LOSSLESS={lossless})")
    print(f"  NCA scratch:   PA={results['nca_scratch']['pixel_acc']*100:.2f}%")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase144_grid_weight_tied.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 144: Grid Weight-Tied Compiler',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'step_sweep': {str(k): v for k, v in step_sweep.items()},
            'lossless': lossless, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Accuracy comparison
        names = ['Normal\nCNN', 'WT-CNN', 'Compiled\nNCA', 'NCA\nScratch']
        accs = [results['normal_cnn']['pixel_acc']*100,
                results['weight_tied']['pixel_acc']*100,
                results['compiled_nca']['pixel_acc']*100,
                results['nca_scratch']['pixel_acc']*100]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        bars = axes[0].bar(range(4), accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{acc:.1f}%', ha='center', fontweight='bold')
        axes[0].set_xticks(range(4)); axes[0].set_xticklabels(names)
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('GoL Prediction', fontweight='bold')

        # Panel 2: NCA steps vs accuracy
        steps = sorted(step_sweep.keys())
        pa_vals = [step_sweep[t]['pixel_acc']*100 for t in steps]
        em_vals = [step_sweep[t]['exact_match']*100 for t in steps]
        axes[1].plot(steps, pa_vals, 'o-', color='#2ecc71', label='Pixel Acc', markersize=6)
        axes[1].plot(steps, em_vals, 's--', color='#e74c3c', label='Exact Match', markersize=6)
        axes[1].axvline(x=GOL_STEPS, color='gray', linestyle=':', label=f'T={GOL_STEPS} (train)')
        axes[1].set_xlabel('NCA Steps (T)'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Compiled NCA: Steps vs Performance', fontweight='bold')
        axes[1].legend(); axes[1].grid(alpha=0.3)

        # Panel 3: Visualize example
        test_x, test_y = test_ds[0]
        nca.eval()
        with torch.no_grad():
            pred = torch.sigmoid(nca(test_x.unsqueeze(0).to(DEVICE), steps=GOL_STEPS))
        pred_grid = (pred > 0.5).float().cpu().squeeze()

        axes[2].imshow(np.concatenate([
            test_x.squeeze().numpy(),
            np.ones((GRID_SIZE, 2)),
            test_y.squeeze().numpy(),
            np.ones((GRID_SIZE, 2)),
            pred_grid.numpy()
        ], axis=1), cmap='binary', vmin=0, vmax=1)
        axes[2].set_title('Input | Target | NCA Prediction', fontweight='bold')
        axes[2].axis('off')

        plt.suptitle(f'Phase 144: Game of Life {GOL_STEPS}-Step Prediction (Space === Time)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase144_grid_weight_tied.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
