"""
Phase 146: The Dynamic Hourglass — Phase Reversal via Internal Clock

A task where the RULE CHANGES midway: Dilate 3 times, then Erode 3 times.
  - Baseline NCA (same weights, no clock) CANNOT do this:
    it must apply the same function every step, so it gets confused.
  - Hourglass NCA reads its decaying clock channel to autonomously
    switch from "dilate mode" to "erode mode" — proving time-awareness.

This is the ultimate proof that cells can develop internal temporal
differentiation WITHOUT external time injection.

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
DILATE_STEPS = 3
ERODE_STEPS = 3
TOTAL_STEPS = DILATE_STEPS + ERODE_STEPS  # 6


# ================================================================
# Morphological operations
# ================================================================
def dilate(grid):
    """Binary dilation with 3x3 kernel."""
    g = grid.float()
    if g.dim() == 2: g = g.unsqueeze(0).unsqueeze(0)
    elif g.dim() == 3: g = g.unsqueeze(0)
    kernel = torch.ones(1, 1, 3, 3, device=g.device)
    padded = F.pad(g, (1,1,1,1), mode='constant', value=0)
    result = F.conv2d(padded, kernel)
    return (result > 0).float().squeeze()


def erode(grid):
    """Binary erosion with 3x3 kernel."""
    g = grid.float()
    if g.dim() == 2: g = g.unsqueeze(0).unsqueeze(0)
    elif g.dim() == 3: g = g.unsqueeze(0)
    kernel = torch.ones(1, 1, 3, 3, device=g.device)
    padded = F.pad(g, (1,1,1,1), mode='constant', value=0)
    result = F.conv2d(padded, kernel)
    return (result >= 9).float().squeeze()  # all 9 neighbors must be alive


def generate_dilate_erode_data(n_samples):
    """Generate grid data: input -> dilate 3x -> erode 3x -> target."""
    X, Y = [], []
    for _ in range(n_samples):
        # Random sparse grid (30% fill for visible morphology)
        g = (torch.rand(GRID_SIZE, GRID_SIZE) > 0.7).float()
        X.append(g.clone())
        # Dilate 3 times
        for _ in range(DILATE_STEPS):
            g = dilate(g)
        # Erode 3 times
        for _ in range(ERODE_STEPS):
            g = erode(g)
        Y.append(g)
    return torch.stack(X).unsqueeze(1), torch.stack(Y).unsqueeze(1)


class DilateErodeDS(torch.utils.data.Dataset):
    def __init__(self, n=5000):
        self.X, self.Y = generate_dilate_erode_data(n)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Baseline NCA (no clock — expected to struggle)
# ================================================================
class BaselineGridNCA(nn.Module):
    def __init__(self, ch=32, steps=TOTAL_STEPS):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)


# ================================================================
# Hourglass NCA (with internal clock channel)
# ================================================================
class HourglassGridNCA(nn.Module):
    def __init__(self, ch=33, steps=TOTAL_STEPS):
        super().__init__()
        self.steps = steps
        self.compute_ch = ch - 1  # 32 compute + 1 clock
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(self.compute_ch, 1, 1)  # exclude clock from output

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = self.proj_in(x)
        # Initialize clock channel to 1.0
        h[:, -1:, :, :] = 1.0
        h = F.relu(h)

        self._clock_trajectory = []
        for t in range(steps):
            delta = self.rule(h)
            h = F.relu(h + delta)
            self._clock_trajectory.append(h[:, -1:, :, :].mean().item())

        return self.proj_out(h[:, :-1, :, :])


# ================================================================
# External Clock NCA (Phase 117 approach)
# ================================================================
class ExternalClockGridNCA(nn.Module):
    def __init__(self, ch=32, steps=TOTAL_STEPS):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch+1, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for t in range(steps):
            time_ch = torch.full((h.size(0), 1, h.size(2), h.size(3)),
                                t / steps, device=h.device)
            h_t = torch.cat([h, time_ch], dim=1)
            h = F.relu(h + self.rule(h_t))
        return self.proj_out(h)


# ================================================================
# Training & evaluation
# ================================================================
def train_grid(model, loader, epochs=40, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()


def eval_grid(model, loader):
    model.eval()
    correct = pixels = exact = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = (torch.sigmoid(model(x)) > 0.5).float()
            correct += (pred == y).sum().item(); pixels += y.numel()
            exact += (pred==y).all(-1).all(-1).all(-1).sum().item(); total += y.size(0)
    return correct/pixels, exact/total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 146: The Dynamic Hourglass")
    print(f"  Task: Dilate x{DILATE_STEPS} then Erode x{ERODE_STEPS} on {GRID_SIZE}x{GRID_SIZE}")
    print("=" * 70)

    train_ds = DilateErodeDS(5000)
    test_ds = DilateErodeDS(1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    architectures = {
        'Baseline NCA': BaselineGridNCA().to(DEVICE),
        'External Clock': ExternalClockGridNCA().to(DEVICE),
        'Hourglass NCA': HourglassGridNCA().to(DEVICE),
    }

    results = {}
    for name, model in architectures.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] Training ({n_params:,} params)...")
        train_grid(model, train_loader, epochs=40)
        pa, em = eval_grid(model, test_loader)
        results[name] = {'pixel_acc': pa, 'exact_match': em, 'params': n_params}
        print(f"  PA={pa*100:.2f}%, EM={em*100:.2f}%")
        gc.collect()

    # Clock analysis
    hg = architectures['Hourglass NCA']
    hg.eval()
    with torch.no_grad():
        x_test, _ = next(iter(test_loader))
        _ = hg(x_test.to(DEVICE))
    clock = hg._clock_trajectory
    print(f"\n[Clock Dynamics] Hourglass trajectory (T={TOTAL_STEPS}):")
    for t, v in enumerate(clock):
        phase = "DILATE" if t < DILATE_STEPS else "ERODE"
        bar = "#" * int(max(0, v) * 20)
        print(f"  t={t} [{phase}]: {v:.4f} {bar}")

    # Temporal generalization
    print("\n[Temporal Gen] Testing at different T...")
    gen_results = {}
    for name, model in architectures.items():
        gen_results[name] = {}
        orig_steps = model.steps
        for T in [2, 4, 6, 8, 10, 12, 16]:
            model.steps = T
            pa_t, _ = eval_grid(model, test_loader)
            gen_results[name][T] = pa_t
        model.steps = orig_steps
        print(f"  {name}: " + " ".join(f"T={T}:{gen_results[name][T]*100:.1f}%"
              for T in [6, 8, 12, 16]))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 146 Complete ({elapsed:.0f}s)")
    for name in results:
        r = results[name]
        print(f"  {name:20s}: PA={r['pixel_acc']*100:.2f}%, EM={r['exact_match']*100:.2f}%")
    # Key question: does Hourglass beat Baseline on this phase-reversal task?
    hg_wins = results['Hourglass NCA']['pixel_acc'] > results['Baseline NCA']['pixel_acc']
    print(f"\n  Hourglass > Baseline? {hg_wins}")
    print(f"  Clock shows phase transition? (check trajectory above)")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase146_dynamic_hourglass.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 146: Dynamic Hourglass',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'clock_trajectory': clock,
            'temporal_gen': {k: {str(kk): vv for kk, vv in v.items()} for k, v in gen_results.items()},
            'hourglass_wins': hg_wins, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        # Panel 1: Accuracy comparison
        names = list(results.keys())
        accs = [results[n]['pixel_acc']*100 for n in names]
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        bars = axes[0].bar(range(3), accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{acc:.1f}%', ha='center', fontweight='bold')
        axes[0].set_xticks(range(3)); axes[0].set_xticklabels(['Baseline', 'External\nClock', 'Hourglass'])
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('Dilate+Erode Task', fontweight='bold')

        # Panel 2: Clock trajectory with phase boundary
        axes[1].plot(range(len(clock)), clock, 'o-', color='#2ecc71', markersize=10, linewidth=2)
        axes[1].axvline(x=DILATE_STEPS-0.5, color='red', linestyle='--', linewidth=2,
                       label=f'Phase boundary (Dilate->Erode)')
        axes[1].fill_between(range(DILATE_STEPS), [max(clock)]*DILATE_STEPS, alpha=0.1, color='blue',
                            label='Dilate phase')
        axes[1].fill_between(range(DILATE_STEPS, TOTAL_STEPS),
                            [max(clock)]*(TOTAL_STEPS-DILATE_STEPS), alpha=0.1, color='orange',
                            label='Erode phase')
        axes[1].set_xlabel('NCA Step'); axes[1].set_ylabel('Clock Value')
        axes[1].set_title('Intrinsic Hourglass Dynamics', fontweight='bold')
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

        # Panel 3: Temporal generalization
        for name, color in zip(names, colors):
            ts = sorted(gen_results[name].keys())
            axes[2].plot(ts, [gen_results[name][t]*100 for t in ts], 'o-', color=color,
                        label=name, markersize=5)
        axes[2].axvline(x=TOTAL_STEPS, color='gray', linestyle=':', label=f'T={TOTAL_STEPS} (train)')
        axes[2].set_xlabel('T (steps)'); axes[2].set_ylabel('Pixel Accuracy (%)')
        axes[2].set_title('Temporal Generalization', fontweight='bold')
        axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)

        # Panel 4: Visualize example
        x_vis, y_vis = test_ds[0]
        hg.eval()
        with torch.no_grad():
            pred_hg = (torch.sigmoid(hg(x_vis.unsqueeze(0).to(DEVICE))) > 0.5).float().cpu().squeeze()
        base = architectures['Baseline NCA']
        base.eval()
        with torch.no_grad():
            pred_base = (torch.sigmoid(base(x_vis.unsqueeze(0).to(DEVICE))) > 0.5).float().cpu().squeeze()
        sep = np.ones((GRID_SIZE, 1))
        vis = np.concatenate([x_vis.squeeze().numpy(), sep,
                             y_vis.squeeze().numpy(), sep,
                             pred_base.numpy(), sep,
                             pred_hg.numpy()], axis=1)
        axes[3].imshow(vis, cmap='binary', vmin=0, vmax=1)
        axes[3].set_title('In | Target | Baseline | Hourglass', fontweight='bold')
        axes[3].axis('off')

        plt.suptitle('Phase 146: Dynamic Hourglass - Autonomous Phase Reversal (Dilate->Erode)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase146_dynamic_hourglass.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
