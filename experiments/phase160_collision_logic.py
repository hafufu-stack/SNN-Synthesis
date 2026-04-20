"""
Phase 160: Glider Collision Logic - Spatial AND Gate

Phase 159 proved NCA can synthesize gliders (73.6% position correct).
Conway's GoL achieves universal computation via glider collisions.

Task: Signal A from left, Signal B from top. Train NCA so:
  - A AND B collide -> new glider emerges toward bottom-right (output=1)
  - A only or B only -> signal disappears (output=0)

This proves NCA can implement spatial logic gates.

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


def make_signal_pattern():
    """Create a small asymmetric signal pattern (glider-like)."""
    p = torch.zeros(2, 2)
    p[0, 0] = 1; p[0, 1] = 1; p[1, 0] = 1
    return p


def generate_collision_data(n_samples, grid_size=GRID_SIZE):
    """Generate AND-gate collision data.
    
    4 cases: (A=0,B=0), (A=1,B=0), (A=0,B=1), (A=1,B=1)
    Only (1,1) produces output signal at bottom-right.
    """
    X, Y, labels = [], [], []
    pattern = make_signal_pattern()
    ph, pw = pattern.shape

    for _ in range(n_samples):
        a = random.choice([0, 1])
        b = random.choice([0, 1])

        inp = torch.zeros(grid_size, grid_size)
        target = torch.zeros(grid_size, grid_size)

        # Signal A: left edge, middle row
        if a:
            sy = grid_size // 2 - 1
            inp[sy:sy+ph, 0:pw] = pattern

        # Signal B: top edge, middle column
        if b:
            sx = grid_size // 2 - 1
            inp[0:ph, sx:sx+pw] = pattern

        # Output: bottom-right corner (only if AND)
        if a and b:
            oy = grid_size - ph - 1
            ox = grid_size - pw - 1
            target[oy:oy+ph, ox:ox+pw] = pattern

        X.append(inp)
        Y.append(target)
        labels.append((a, b))

    return torch.stack(X).unsqueeze(1), torch.stack(Y).unsqueeze(1), labels


class CollisionDS(torch.utils.data.Dataset):
    def __init__(self, n=5000):
        self.X, self.Y, self.labels = generate_collision_data(n)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


class CollisionNCA(nn.Module):
    def __init__(self, ch=48, steps=15):
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


def eval_logic(model, ds):
    """Evaluate logic gate accuracy per case."""
    model.eval()
    case_correct = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    case_total = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 0}
    total_pa = 0; total_px = 0

    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            x = x.unsqueeze(0).to(DEVICE)
            y = y.unsqueeze(0).to(DEVICE)
            pred = (torch.sigmoid(model(x)) > 0.5).float()
            label = ds.labels[i]

            pa = (pred == y).float().mean().item()
            total_pa += pa; total_px += 1

            # Check if output region is correct
            has_output_pred = pred[0, 0, -4:, -4:].sum() > 0
            has_output_target = y[0, 0, -4:, -4:].sum() > 0
            if has_output_pred == has_output_target:
                case_correct[label] += 1
            case_total[label] += 1

    avg_pa = total_pa / total_px
    case_acc = {}
    for k in case_correct:
        case_acc[k] = case_correct[k] / max(1, case_total[k])
    return avg_pa, case_acc


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 160: Glider Collision Logic")
    print(f"  Spatial AND gate via signal collision")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    train_ds = CollisionDS(8000)
    test_ds = CollisionDS(1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    # Train
    model = CollisionNCA(ch=48, steps=15).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"\n[Training] CollisionNCA ({n_p:,} params)...")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
    for epoch in range(120):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch+1) % 30 == 0:
            pa, ca = eval_logic(model, test_ds)
            print(f"  Epoch {epoch+1}/120: PA={pa*100:.2f}%, "
                  f"AND={ca[(1,1)]*100:.0f}%, A_only={ca[(1,0)]*100:.0f}%, "
                  f"B_only={ca[(0,1)]*100:.0f}%, Neither={ca[(0,0)]*100:.0f}%")

    # Final evaluation
    pa, case_acc = eval_logic(model, test_ds)
    and_acc = case_acc[(1, 1)]
    logic_works = and_acc > 0.6 and case_acc[(1,0)] > 0.6 and case_acc[(0,1)] > 0.6

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 160 Complete ({elapsed:.0f}s)")
    print(f"  PA: {pa*100:.2f}%")
    print(f"  AND (1,1): {case_acc[(1,1)]*100:.1f}%")
    print(f"  A only (1,0): {case_acc[(1,0)]*100:.1f}%")
    print(f"  B only (0,1): {case_acc[(0,1)]*100:.1f}%")
    print(f"  Neither (0,0): {case_acc[(0,0)]*100:.1f}%")
    print(f"  Logic gate works: {logic_works}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase160_collision_logic.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 160: Glider Collision Logic',
            'timestamp': datetime.now().isoformat(),
            'pixel_accuracy': pa,
            'case_accuracy': {str(k): v for k, v in case_acc.items()},
            'logic_works': logic_works,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        cases = ['(0,0)', '(1,0)', '(0,1)', '(1,1)']
        accs = [case_acc[(0,0)], case_acc[(1,0)], case_acc[(0,1)], case_acc[(1,1)]]
        colors = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71']
        bars = axes[0].bar(cases, [a*100 for a in accs], color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f'{acc*100:.0f}%', ha='center', fontweight='bold', fontsize=10)
        axes[0].set_ylabel('Logic Accuracy (%)')
        axes[0].set_title('AND Gate Accuracy', fontweight='bold', fontsize=11)
        axes[0].set_ylim(0, 110)

        # Show example grids
        model.eval()
        with torch.no_grad():
            for idx, (a, b) in enumerate([(1, 1), (1, 0)]):
                for i in range(len(test_ds)):
                    if test_ds.labels[i] == (a, b):
                        x, y = test_ds[i]
                        pred = torch.sigmoid(model(x.unsqueeze(0).to(DEVICE)))[0, 0].cpu()
                        combined = x[0] * 0.5 + pred * 0.5
                        axes[idx+1].imshow(combined.numpy(), cmap='hot', vmin=0, vmax=1)
                        axes[idx+1].set_title(f'A={a}, B={b} -> {"AND=1" if a and b else "AND=0"}',
                                            fontweight='bold', fontsize=10)
                        axes[idx+1].axis('off')
                        break

        fig.suptitle('Phase 160: Glider Collision Logic (Spatial AND Gate)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.85, bottom=0.08, left=0.06, right=0.98, wspace=0.25)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase160_collision_logic.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'pa': pa, 'case_acc': case_acc, 'logic_works': logic_works}


if __name__ == '__main__':
    main()
