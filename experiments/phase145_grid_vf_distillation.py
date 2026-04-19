"""
Phase 145: Grid Vector Field Distillation

Distill a normal (independent-weight) CNN teacher into a single-block NCA
student by matching the VELOCITY FIELD (residuals) at each step.

Teacher: CNN with 5 independent blocks (different weights per layer)
Student: NCA with 1 shared block (same weights, T=5 iterations)
Task: Game of Life 5-step prediction on 16x16 grid

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
CHANNELS = 32


# ================================================================
# Game of Life dataset (reuse from Phase 144)
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

def generate_data(n, steps=GOL_STEPS):
    X, Y = [], []
    for _ in range(n):
        g = (torch.rand(GRID_SIZE, GRID_SIZE) > 0.5).float()
        X.append(g.clone())
        for _ in range(steps): g = gol_step(g)
        Y.append(g)
    return torch.stack(X).unsqueeze(1), torch.stack(Y).unsqueeze(1)

class GoLDS(torch.utils.data.Dataset):
    def __init__(self, n=5000):
        self.X, self.Y = generate_data(n)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Teacher: Normal CNN with independent weights and residual tracking
# ================================================================
class TeacherGridCNN(nn.Module):
    def __init__(self, ch=CHANNELS, depth=GOL_STEPS):
        super().__init__()
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(ch, ch, 3, padding=1))
            for _ in range(depth)
        ])
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        for block in self.blocks:
            h = F.relu(h + block(h))
        return self.proj_out(h)

    def get_residuals(self, x):
        h = F.relu(self.proj_in(x))
        residuals = []; states = [h]
        for block in self.blocks:
            delta = block(h)
            residuals.append(delta)
            h = F.relu(h + delta)
            states.append(h)
        return residuals, states


# ================================================================
# Student: Single-block NCA with update tracking
# ================================================================
class StudentGridNCA(nn.Module):
    def __init__(self, ch=CHANNELS, steps=GOL_STEPS):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.block(h))
        return self.proj_out(h)

    def get_updates(self, x):
        h = F.relu(self.proj_in(x))
        updates = []; states = [h]
        for _ in range(self.steps):
            delta = self.block(h)
            updates.append(delta)
            h = F.relu(h + delta)
            states.append(h)
        return updates, states


# ================================================================
# Distillation methods
# ================================================================
def distill_output_only(student, teacher, loader, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    for epoch in range(epochs):
        student.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_out = teacher(x)
            s_out = student(x)
            loss = 0.5 * F.mse_loss(s_out, t_out) + 0.5 * F.binary_cross_entropy_with_logits(s_out, y)
            opt.zero_grad(); loss.backward(); opt.step()


def distill_vector_field(student, teacher, loader, epochs=30, lr=1e-3):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    for epoch in range(epochs):
        student.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_res, t_states = teacher.get_residuals(x)
            s_upd, s_states = student.get_updates(x)

            # Match velocity fields
            vf_loss = sum(F.mse_loss(s_upd[i], t_res[i])
                         for i in range(min(len(t_res), len(s_upd)))) / GOL_STEPS

            # Match intermediate states
            state_loss = sum(F.mse_loss(s_states[i+1], t_states[i+1])
                            for i in range(min(len(t_states)-1, len(s_states)-1))) / GOL_STEPS

            # Task loss
            s_out = student(x)
            ce_loss = F.binary_cross_entropy_with_logits(s_out, y)

            loss = 0.3 * vf_loss + 0.3 * state_loss + 0.4 * ce_loss
            opt.zero_grad(); loss.backward(); opt.step()


def eval_grid(model, loader, **kwargs):
    model.eval()
    correct = pixels = 0; exact = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = (torch.sigmoid(model(x, **kwargs)) > 0.5).float()
            correct += (pred == y).sum().item()
            pixels += y.numel()
            exact += (pred == y).all(dim=-1).all(dim=-1).all(dim=-1).sum().item()
            total += y.size(0)
    return correct/pixels, exact/total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 145: Grid Vector Field Distillation")
    print(f"  Task: GoL {GOL_STEPS}-step, {GRID_SIZE}x{GRID_SIZE}")
    print("=" * 70)

    train_ds = GoLDS(5000)
    test_ds = GoLDS(1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    results = {}

    # Teacher
    print("\n[Step 1] Training Teacher CNN...")
    teacher = TeacherGridCNN().to(DEVICE)
    n_t = sum(p.numel() for p in teacher.parameters())
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for ep in range(30):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(teacher(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    pa_t, em_t = eval_grid(teacher, test_loader)
    results['teacher'] = {'pixel_acc': pa_t, 'exact_match': em_t, 'params': n_t}
    print(f"  Teacher: PA={pa_t*100:.2f}%, EM={em_t*100:.2f}%")

    # NCA from scratch
    print("\n[Step 2] NCA from scratch...")
    nca_s = StudentGridNCA().to(DEVICE)
    n_s = sum(p.numel() for p in nca_s.parameters())
    opt = torch.optim.Adam(nca_s.parameters(), lr=1e-3)
    for ep in range(30):
        nca_s.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(nca_s(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    pa_s, em_s = eval_grid(nca_s, test_loader)
    results['scratch'] = {'pixel_acc': pa_s, 'exact_match': em_s, 'params': n_s}
    print(f"  Scratch: PA={pa_s*100:.2f}%, EM={em_s*100:.2f}%")

    # Output-only distillation
    print("\n[Step 3] Output-only distillation...")
    nca_out = StudentGridNCA().to(DEVICE)
    distill_output_only(nca_out, teacher, train_loader, epochs=30)
    pa_o, em_o = eval_grid(nca_out, test_loader)
    recovery_o = pa_o / pa_t * 100 if pa_t > 0 else 0
    results['output_distill'] = {'pixel_acc': pa_o, 'exact_match': em_o, 'recovery': recovery_o}
    print(f"  Output distill: PA={pa_o*100:.2f}%, EM={em_o*100:.2f}%, Recovery={recovery_o:.1f}%")

    # Vector Field distillation
    print("\n[Step 4] Vector Field distillation...")
    nca_vf = StudentGridNCA().to(DEVICE)
    distill_vector_field(nca_vf, teacher, train_loader, epochs=30)
    pa_v, em_v = eval_grid(nca_vf, test_loader)
    recovery_v = pa_v / pa_t * 100 if pa_t > 0 else 0
    results['vf_distill'] = {'pixel_acc': pa_v, 'exact_match': em_v, 'recovery': recovery_v}
    print(f"  VF distill: PA={pa_v*100:.2f}%, EM={em_v*100:.2f}%, Recovery={recovery_v:.1f}%")

    improvement = pa_v - pa_o
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 145 Complete ({elapsed:.0f}s)")
    print(f"  Teacher CNN:        PA={pa_t*100:.2f}%")
    print(f"  NCA scratch:        PA={pa_s*100:.2f}%")
    print(f"  Output distill:     PA={pa_o*100:.2f}% (recovery={recovery_o:.1f}%)")
    print(f"  VF distill:         PA={pa_v*100:.2f}% (recovery={recovery_v:.1f}%)")
    print(f"  VF improvement:     {improvement*100:+.2f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase145_grid_vf_distillation.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 145: Grid VF Distillation',
            'timestamp': datetime.now().isoformat(),
            'results': results, 'improvement': improvement, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        names = ['Teacher\nCNN', 'NCA\nScratch', 'Output\nDistill', 'VF\nDistill']
        accs = [pa_t*100, pa_s*100, pa_o*100, pa_v*100]
        colors = ['#e74c3c', '#95a5a6', '#f39c12', '#2ecc71']
        bars = axes[0].bar(range(4), accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{acc:.1f}%', ha='center', fontweight='bold')
        axes[0].set_xticks(range(4)); axes[0].set_xticklabels(names)
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('GoL Distillation', fontweight='bold')

        recs = [recovery_o, recovery_v]
        bars = axes[1].bar([0, 1], recs, color=['#f39c12', '#2ecc71'], alpha=0.85, edgecolor='black')
        axes[1].axhline(y=100, color='red', linestyle='--', label='Teacher')
        for bar, rec in zip(bars, recs):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{rec:.1f}%', ha='center', fontweight='bold')
        axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(['Output\nDistill', 'Vector Field\nDistill'])
        axes[1].set_ylabel('Recovery Rate (%)'); axes[1].set_title('Knowledge Recovery', fontweight='bold')
        axes[1].legend()

        ems = [em_t*100, em_s*100, em_o*100, em_v*100]
        bars = axes[2].bar(range(4), ems, color=colors, alpha=0.85, edgecolor='black')
        for bar, em in zip(bars, ems):
            axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{em:.1f}%', ha='center', fontweight='bold')
        axes[2].set_xticks(range(4)); axes[2].set_xticklabels(names)
        axes[2].set_ylabel('Exact Match (%)'); axes[2].set_title('Exact Match Rate', fontweight='bold')

        plt.suptitle('Phase 145: Vector Field Distillation on Game of Life',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase145_grid_vf_distillation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
