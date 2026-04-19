"""
Phase 148: The Turing NCA — Discontinuous Program Execution

Task: Dilate 2x -> Invert -> Erode 2x (5 steps total)
The Invert operation is a DISCONTINUOUS phase transition that cannot
be approximated by a single continuous rule.

Only NCA with time-awareness can execute multiple "programs" in sequence.

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
TOTAL_STEPS = 5  # Dilate x2 + Invert x1 + Erode x2


# ================================================================
# Operations
# ================================================================
def dilate(grid):
    g = grid.float()
    if g.dim() == 2: g = g.unsqueeze(0).unsqueeze(0)
    elif g.dim() == 3: g = g.unsqueeze(0)
    k = torch.ones(1, 1, 3, 3, device=g.device)
    p = F.pad(g, (1,1,1,1), mode='constant', value=0)
    return (F.conv2d(p, k).squeeze() > 0).float()


def erode(grid):
    g = grid.float()
    if g.dim() == 2: g = g.unsqueeze(0).unsqueeze(0)
    elif g.dim() == 3: g = g.unsqueeze(0)
    k = torch.ones(1, 1, 3, 3, device=g.device)
    p = F.pad(g, (1,1,1,1), mode='constant', value=0)
    return (F.conv2d(p, k).squeeze() >= 9).float()


def invert(grid):
    return 1.0 - grid


def generate_data(n_samples):
    X, Y = [], []
    for _ in range(n_samples):
        g = (torch.rand(GRID_SIZE, GRID_SIZE) > 0.65).float()
        X.append(g.clone())
        # Dilate x2
        g = dilate(g)
        g = dilate(g)
        # Invert
        g = invert(g)
        # Erode x2
        g = erode(g)
        g = erode(g)
        Y.append(g)
    return torch.stack(X).unsqueeze(1), torch.stack(Y).unsqueeze(1)


class TuringDS(torch.utils.data.Dataset):
    def __init__(self, n=5000):
        self.X, self.Y = generate_data(n)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Baseline NCA (no time info)
# ================================================================
class BaselineNCA(nn.Module):
    def __init__(self, ch=32, steps=TOTAL_STEPS):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)


# ================================================================
# External Clock NCA
# ================================================================
class ExternalClockNCA(nn.Module):
    def __init__(self, ch=32, steps=TOTAL_STEPS):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch+1, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
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
# Hourglass NCA
# ================================================================
class HourglassNCA(nn.Module):
    def __init__(self, ch=33, steps=TOTAL_STEPS):
        super().__init__()
        self.steps = steps
        self.compute_ch = ch - 1
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(self.compute_ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = self.proj_in(x)
        h[:, -1:, :, :] = 1.0
        h = F.relu(h)
        self._clock = []
        for t in range(steps):
            h = F.relu(h + self.rule(h))
            self._clock.append(h[:, -1:].mean().item())
        return self.proj_out(h[:, :-1])


# ================================================================
# Larger Baseline (more capacity, still no time)
# ================================================================
class LargeBaselineNCA(nn.Module):
    def __init__(self, ch=64, steps=TOTAL_STEPS):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)


# ================================================================
# Training & evaluation
# ================================================================
def train_grid(model, loader, epochs=50, lr=1e-3):
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
            exact += (pred==y).all(-1).all(-1).all(-1).sum().item()
            total += y.size(0)
    return correct/pixels, exact/total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 148: The Turing NCA")
    print(f"  Task: Dilate x2 -> Invert -> Erode x2 on {GRID_SIZE}x{GRID_SIZE}")
    print("=" * 70)

    train_ds = TuringDS(5000)
    test_ds = TuringDS(1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    architectures = {
        'Baseline (32ch)': BaselineNCA(ch=32).to(DEVICE),
        'Large Baseline (64ch)': LargeBaselineNCA(ch=64).to(DEVICE),
        'External Clock': ExternalClockNCA(ch=32).to(DEVICE),
        'Hourglass NCA': HourglassNCA(ch=33).to(DEVICE),
    }

    results = {}
    for name, model in architectures.items():
        n_p = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] Training ({n_p:,} params)...")
        train_grid(model, train_loader, epochs=50)
        pa, em = eval_grid(model, test_loader)
        results[name] = {'pixel_acc': pa, 'exact_match': em, 'params': n_p}
        print(f"  PA={pa*100:.2f}%, EM={em*100:.2f}%")
        gc.collect()

    # Clock analysis
    hg = architectures['Hourglass NCA']
    hg.eval()
    with torch.no_grad():
        x_t, _ = next(iter(test_loader))
        _ = hg(x_t.to(DEVICE))
    clock = hg._clock
    print(f"\n[Clock] Hourglass trajectory:")
    phases = ['Dilate1', 'Dilate2', 'INVERT', 'Erode1', 'Erode2']
    for t, (v, p) in enumerate(zip(clock, phases)):
        bar = "#" * int(max(0, v) * 20)
        print(f"  t={t} [{p:8s}]: {v:.4f} {bar}")

    # Key comparison
    clock_wins = results['External Clock']['pixel_acc'] > results['Baseline (32ch)']['pixel_acc']
    hg_wins = results['Hourglass NCA']['pixel_acc'] > results['Baseline (32ch)']['pixel_acc']
    best = max(results.keys(), key=lambda k: results[k]['pixel_acc'])

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 148 Complete ({elapsed:.0f}s)")
    for name, r in results.items():
        print(f"  {name:25s}: PA={r['pixel_acc']*100:.2f}%, EM={r['exact_match']*100:.2f}%")
    print(f"\n  Best: {best}")
    print(f"  Clock > Baseline: {clock_wins}")
    print(f"  Hourglass > Baseline: {hg_wins}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase148_turing_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 148: Turing NCA',
            'timestamp': datetime.now().isoformat(),
            'results': results, 'clock': clock,
            'clock_wins': clock_wins, 'hg_wins': hg_wins,
            'best': best, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        names = list(results.keys())
        accs = [results[n]['pixel_acc']*100 for n in names]
        colors = ['#95a5a6', '#7f8c8d', '#e74c3c', '#2ecc71']
        bars = axes[0].bar(range(4), accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{acc:.1f}%', ha='center', fontweight='bold')
        axes[0].set_xticks(range(4))
        axes[0].set_xticklabels(['Baseline\n32ch', 'Baseline\n64ch', 'External\nClock', 'Hourglass'])
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('Turing Task', fontweight='bold')

        axes[1].bar(range(5), clock, color=['#3498db','#3498db','#e74c3c','#2ecc71','#2ecc71'],
                   alpha=0.85, edgecolor='black')
        axes[1].set_xticks(range(5)); axes[1].set_xticklabels(phases, fontsize=8, rotation=30)
        axes[1].set_ylabel('Clock Value'); axes[1].set_title('Hourglass Dynamics', fontweight='bold')

        # Exact match
        ems = [results[n]['exact_match']*100 for n in names]
        bars = axes[2].bar(range(4), ems, color=colors, alpha=0.85, edgecolor='black')
        for bar, e in zip(bars, ems):
            axes[2].text(bar.get_x()+bar.get_width()/2, max(e+0.3, 0.5),
                        f'{e:.1f}%', ha='center', fontweight='bold')
        axes[2].set_xticks(range(4))
        axes[2].set_xticklabels(['Baseline\n32ch', 'Baseline\n64ch', 'External\nClock', 'Hourglass'])
        axes[2].set_ylabel('Exact Match (%)'); axes[2].set_title('Exact Match', fontweight='bold')

        plt.suptitle('Phase 148: Turing NCA (Dilate->Invert->Erode)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase148_turing_nca.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
