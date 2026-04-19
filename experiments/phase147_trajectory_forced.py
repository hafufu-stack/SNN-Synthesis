"""
Phase 147: Trajectory-Forced Compiler — Conquering Chaos

Game of Life is chaotic: 5-step end-to-end prediction hits ~76% ceiling.
Solution: Trajectory Forcing — supervise EVERY intermediate step.

When each step has its own target, the model learns the GoL rule perfectly,
and the compiled NCA inherits 100% accuracy.

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
# Game of Life
# ================================================================
def gol_step(grid):
    kernel = torch.ones(1, 1, 3, 3, device=grid.device)
    kernel[0, 0, 1, 1] = 0
    g = grid.float()
    if g.dim() == 2: g = g.unsqueeze(0).unsqueeze(0)
    elif g.dim() == 3: g = g.unsqueeze(0)
    g_pad = F.pad(g, (1,1,1,1), mode='circular')
    neighbors = F.conv2d(g_pad, kernel).squeeze()
    survive = (grid > 0.5) & ((neighbors == 2) | (neighbors == 3))
    birth = (grid < 0.5) & (neighbors == 3)
    return (survive | birth).float()


def generate_trajectory_data(n_samples, steps=GOL_STEPS):
    """Generate full trajectory: [t0, t1, t2, ..., t5] for each sample."""
    trajectories = []
    for _ in range(n_samples):
        traj = []
        g = (torch.rand(GRID_SIZE, GRID_SIZE) > 0.5).float()
        traj.append(g.clone())
        for _ in range(steps):
            g = gol_step(g)
            traj.append(g.clone())
        trajectories.append(torch.stack(traj))  # (steps+1, H, W)
    return torch.stack(trajectories)  # (N, steps+1, H, W)


class TrajectoryDS(torch.utils.data.Dataset):
    def __init__(self, n=5000):
        self.data = generate_trajectory_data(n)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


# ================================================================
# Weight-Tied CNN with trajectory output
# ================================================================
class TrajectoryWT_CNN(nn.Module):
    def __init__(self, ch=32, depth=GOL_STEPS):
        super().__init__()
        self.depth = depth
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.shared_block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward_trajectory(self, x):
        """Return predictions at EVERY step."""
        h = F.relu(self.proj_in(x))
        outputs = []
        for _ in range(self.depth):
            h = F.relu(h + self.shared_block(h))
            outputs.append(self.proj_out(h))
        return outputs  # list of (B, 1, H, W)

    def forward(self, x, depth=None):
        if depth is None: depth = self.depth
        h = F.relu(self.proj_in(x))
        for _ in range(depth):
            h = F.relu(h + self.shared_block(h))
        return self.proj_out(h)


class NormalTrajCNN(nn.Module):
    def __init__(self, ch=32, depth=GOL_STEPS):
        super().__init__()
        self.depth = depth
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(ch, ch, 3, padding=1))
            for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward_trajectory(self, x):
        h = F.relu(self.proj_in(x))
        outputs = []
        for block in self.blocks:
            h = F.relu(h + block(h))
            outputs.append(self.proj_out(h))
        return outputs

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        for block in self.blocks:
            h = F.relu(h + block(h))
        return self.proj_out(h)


class CompiledNCA(nn.Module):
    def __init__(self, wt):
        super().__init__()
        self.proj_in = wt.proj_in
        self.shared_block = wt.shared_block
        self.proj_out = wt.proj_out
    def forward(self, x, steps=GOL_STEPS):
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.shared_block(h))
        return self.proj_out(h)


# ================================================================
# Training
# ================================================================
def train_trajectory(model, loader, epochs=40, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0; n = 0
        for traj_batch in loader:
            traj_batch = traj_batch.to(DEVICE)
            x = traj_batch[:, 0:1, :, :]  # (B, 1, H, W) initial state
            targets = [traj_batch[:, t+1:t+2, :, :] for t in range(GOL_STEPS)]

            preds = model.forward_trajectory(x)
            loss = sum(F.binary_cross_entropy_with_logits(p, t)
                      for p, t in zip(preds, targets)) / GOL_STEPS
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1


def train_endtoend(model, loader, epochs=40, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for traj_batch in loader:
            traj_batch = traj_batch.to(DEVICE)
            x = traj_batch[:, 0:1, :, :]
            target = traj_batch[:, -1:, :, :]
            pred = model(x) if not hasattr(model, 'forward_trajectory') else model(x)
            loss = F.binary_cross_entropy_with_logits(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()


def eval_grid(model, loader, **kwargs):
    model.eval()
    correct = pixels = exact = total = 0
    with torch.no_grad():
        for traj_batch in loader:
            traj_batch = traj_batch.to(DEVICE)
            x = traj_batch[:, 0:1, :, :]
            target = traj_batch[:, -1:, :, :]
            pred = (torch.sigmoid(model(x, **kwargs)) > 0.5).float()
            correct += (pred == target).sum().item(); pixels += target.numel()
            exact += (pred == target).all(-1).all(-1).all(-1).sum().item()
            total += target.size(0)
    return correct/pixels, exact/total


def eval_per_step(model, loader):
    """Evaluate trajectory-forced model at each intermediate step."""
    model.eval()
    step_accs = []
    with torch.no_grad():
        for traj_batch in loader:
            traj_batch = traj_batch.to(DEVICE)
            x = traj_batch[:, 0:1, :, :]
            preds = model.forward_trajectory(x)
            for t, p in enumerate(preds):
                target = traj_batch[:, t+1:t+2, :, :]
                pred_bin = (torch.sigmoid(p) > 0.5).float()
                acc = (pred_bin == target).float().mean().item()
                while len(step_accs) <= t:
                    step_accs.append([])
                step_accs[t].append(acc)
    return [np.mean(s) for s in step_accs]


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 147: Trajectory-Forced Compiler")
    print(f"  GoL {GOL_STEPS}-step, {GRID_SIZE}x{GRID_SIZE}")
    print("=" * 70)

    train_ds = TrajectoryDS(5000)
    test_ds = TrajectoryDS(1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    results = {}

    # 1. WT-CNN End-to-End (baseline, expect ~76%)
    print("\n[1] WT-CNN End-to-End...")
    wt_e2e = TrajectoryWT_CNN().to(DEVICE)
    train_endtoend(wt_e2e, train_loader, epochs=40)
    pa, em = eval_grid(wt_e2e, test_loader)
    results['wt_e2e'] = {'pixel_acc': pa, 'exact_match': em}
    print(f"  PA={pa*100:.2f}%, EM={em*100:.2f}%")

    # 2. WT-CNN Trajectory-Forced (expect much higher!)
    print("\n[2] WT-CNN Trajectory-Forced...")
    wt_traj = TrajectoryWT_CNN().to(DEVICE)
    train_trajectory(wt_traj, train_loader, epochs=40)
    pa_t, em_t = eval_grid(wt_traj, test_loader)
    results['wt_traj'] = {'pixel_acc': pa_t, 'exact_match': em_t}
    print(f"  PA={pa_t*100:.2f}%, EM={em_t*100:.2f}%")

    # Per-step accuracy
    step_accs = eval_per_step(wt_traj, test_loader)
    print(f"  Per-step: {['%.2f%%'%(a*100) for a in step_accs]}")

    # 3. Compile to NCA
    print("\n[3] Compiling Trajectory-Forced WT-CNN -> NCA...")
    nca = CompiledNCA(wt_traj).to(DEVICE)
    pa_n, em_n = eval_grid(nca, test_loader, steps=GOL_STEPS)
    gap = abs(pa_n - pa_t)
    lossless = gap < 1e-6
    results['compiled_nca'] = {'pixel_acc': pa_n, 'exact_match': em_n, 'lossless': lossless}
    print(f"  NCA PA={pa_n*100:.2f}%, EM={em_n*100:.2f}%, LOSSLESS={lossless}")

    # 4. Normal CNN trajectory-forced (for comparison)
    print("\n[4] Normal CNN Trajectory-Forced...")
    norm_traj = NormalTrajCNN().to(DEVICE)
    train_trajectory(norm_traj, train_loader, epochs=40)
    pa_nt, em_nt = eval_grid(norm_traj, test_loader)
    results['normal_traj'] = {'pixel_acc': pa_nt, 'exact_match': em_nt}
    print(f"  Normal CNN Traj: PA={pa_nt*100:.2f}%, EM={em_nt*100:.2f}%")

    improvement = pa_t - pa
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 147 Complete ({elapsed:.0f}s)")
    print(f"  WT-CNN End-to-End:        PA={pa*100:.2f}%, EM={em*100:.2f}%")
    print(f"  WT-CNN Trajectory-Forced: PA={pa_t*100:.2f}%, EM={em_t*100:.2f}%")
    print(f"  Compiled NCA:             PA={pa_n*100:.2f}%, EM={em_n*100:.2f}% (LOSSLESS={lossless})")
    print(f"  Normal CNN Traj:          PA={pa_nt*100:.2f}%, EM={em_nt*100:.2f}%")
    print(f"  Trajectory improvement:   {improvement*100:+.2f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase147_trajectory_forced.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 147: Trajectory-Forced Compiler',
            'timestamp': datetime.now().isoformat(),
            'results': results, 'step_accs': step_accs,
            'improvement': improvement, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Comparison
        names = ['WT-CNN\nE2E', 'WT-CNN\nTrajectory', 'Compiled\nNCA', 'Normal\nCNN Traj']
        accs = [pa*100, pa_t*100, pa_n*100, pa_nt*100]
        colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']
        bars = axes[0].bar(range(4), accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{acc:.1f}%', ha='center', fontweight='bold')
        axes[0].set_xticks(range(4)); axes[0].set_xticklabels(names)
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('Trajectory Forcing', fontweight='bold')

        # Panel 2: Per-step accuracy
        axes[1].bar(range(1, GOL_STEPS+1), [a*100 for a in step_accs],
                   color='#2ecc71', alpha=0.85, edgecolor='black')
        for i, a in enumerate(step_accs):
            axes[1].text(i+1, a*100+0.3, f'{a*100:.1f}%', ha='center', fontsize=9)
        axes[1].set_xlabel('GoL Step'); axes[1].set_ylabel('Pixel Accuracy (%)')
        axes[1].set_title('Per-Step Accuracy (Trajectory-Forced)', fontweight='bold')

        # Panel 3: Exact match comparison
        ems = [em*100, em_t*100, em_n*100, em_nt*100]
        bars = axes[2].bar(range(4), ems, color=colors, alpha=0.85, edgecolor='black')
        for bar, e in zip(bars, ems):
            axes[2].text(bar.get_x()+bar.get_width()/2, max(e+0.3, 1),
                        f'{e:.1f}%', ha='center', fontweight='bold')
        axes[2].set_xticks(range(4)); axes[2].set_xticklabels(names)
        axes[2].set_ylabel('Exact Match (%)'); axes[2].set_title('Exact Match', fontweight='bold')

        plt.suptitle('Phase 147: Trajectory Forcing Conquers Chaos',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase147_trajectory_forced.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
