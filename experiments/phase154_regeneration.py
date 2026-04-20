"""
Phase 154: Planarian Regeneration - Self-Repair After Destruction

Phase 153 proved NCA can self-replicate (100% exact match on 2x2+).
Planaria can regenerate from fragments. Can our NCA do the same?

Test: Train replicator, let it tile the grid, then DESTROY portions
(50% random, right-half erase, center hole) and run MORE NCA steps.
Measure how much of the original tiling pattern is autonomously restored.

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


# ================================================================
# Replicator NCA (from Phase 153)
# ================================================================
class ReplicatorNCA(nn.Module):
    def __init__(self, ch=48, steps=20):
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

    def forward_from_state(self, h, steps):
        """Continue from intermediate hidden state."""
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)

    def encode(self, x):
        return F.relu(self.proj_in(x))


def create_tiling_pair(pattern, grid_size=GRID_SIZE):
    ph, pw = pattern.shape
    inp = torch.zeros(grid_size, grid_size)
    inp[:ph, :pw] = pattern
    out = torch.zeros(grid_size, grid_size)
    for y in range(0, grid_size, ph):
        for x in range(0, grid_size, pw):
            h = min(ph, grid_size - y)
            w = min(pw, grid_size - x)
            out[y:y+h, x:x+w] = pattern[:h, :w]
    return inp, out


def generate_data(n_samples, pattern_type='3x3'):
    X, Y = [], []
    for _ in range(n_samples):
        if pattern_type == '3x3':
            pattern = torch.rand(3, 3).round()
            if pattern.sum() == 0: pattern[1, 1] = 1.0
        elif pattern_type == '2x2':
            pattern = torch.rand(2, 2).round()
            if pattern.sum() == 0: pattern[0, 0] = 1.0
        else:
            pattern = torch.rand(3, 3).round()
            if pattern.sum() == 0: pattern[1, 1] = 1.0
        inp, out = create_tiling_pair(pattern, GRID_SIZE)
        X.append(inp); Y.append(out)
    return torch.stack(X).unsqueeze(1), torch.stack(Y).unsqueeze(1)


class TilingDS(torch.utils.data.Dataset):
    def __init__(self, n=3000, pt='3x3'):
        self.X, self.Y = generate_data(n, pt)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Damage functions
# ================================================================
def damage_random_50(grid):
    """Destroy 50% of pixels randomly."""
    mask = torch.rand_like(grid) > 0.5
    return grid * mask.float()

def damage_right_half(grid):
    """Erase the right half."""
    damaged = grid.clone()
    w = grid.shape[-1]
    damaged[..., w//2:] = 0
    return damaged

def damage_center_hole(grid):
    """Erase a large center hole."""
    damaged = grid.clone()
    h, w = grid.shape[-2], grid.shape[-1]
    y1, y2 = h//4, 3*h//4
    x1, x2 = w//4, 3*w//4
    damaged[..., y1:y2, x1:x2] = 0
    return damaged

def damage_quarter(grid):
    """Erase top-right quarter."""
    damaged = grid.clone()
    h, w = grid.shape[-2], grid.shape[-1]
    damaged[..., :h//2, w//2:] = 0
    return damaged


# ================================================================
# Training & Regeneration Test
# ================================================================
def train_model(model, loader, epochs=80, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()


def regeneration_test(model, test_X, test_Y, damage_fn, damage_name, regen_steps=20):
    """Apply damage to the COMPLETED tiling, then run more NCA steps."""
    model.eval()
    pa_before = pa_after = 0
    total_px = 0

    with torch.no_grad():
        for i in range(len(test_X)):
            x = test_X[i:i+1].to(DEVICE)
            y = test_Y[i:i+1].to(DEVICE)

            # Step 1: Get the completed tiling (seed -> full grid)
            completed = torch.sigmoid(model(x))
            completed_binary = (completed > 0.5).float()

            # Step 2: Apply damage
            damaged = damage_fn(completed_binary)

            # Step 3: Run more NCA steps from damaged state
            regenerated = torch.sigmoid(model(damaged, steps=regen_steps))
            regen_binary = (regenerated > 0.5).float()

            # Measure
            pa_before += (completed_binary == y).sum().item()
            pa_after += (regen_binary == y).sum().item()
            total_px += y.numel()

    return pa_before / total_px, pa_after / total_px


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 154: Planarian Regeneration")
    print(f"  Can NCA self-repair after destruction?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Train replicator
    train_ds = TilingDS(3000, '3x3')
    test_ds = TilingDS(200, '3x3')
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    model = ReplicatorNCA(ch=48, steps=20).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"\n[Step 1] Training Replicator ({n_p:,} params)...")
    train_model(model, train_loader, epochs=80)

    # Verify base performance
    model.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(model(test_ds.X.to(DEVICE))) > 0.5).float()
        base_pa = (pred == test_ds.Y.to(DEVICE)).float().mean().item()
    print(f"  Base replication PA: {base_pa*100:.2f}%")

    # Test regeneration with different damage types
    damage_types = {
        'random_50%': damage_random_50,
        'right_half': damage_right_half,
        'center_hole': damage_center_hole,
        'quarter': damage_quarter,
    }

    regen_steps_list = [5, 10, 20, 40]
    results = {}

    print(f"\n[Step 2] Regeneration Tests...")
    for dname, dfn in damage_types.items():
        results[dname] = {}
        for rs in regen_steps_list:
            pa_before, pa_after = regeneration_test(
                model, test_ds.X, test_ds.Y, dfn, dname, regen_steps=rs)
            results[dname][f"regen_{rs}"] = {
                'pa_before_damage': pa_before,
                'pa_after_regen': pa_after,
                'recovery': pa_after - (1.0 - base_pa)  # How much was recovered
            }
        best_rs = max(regen_steps_list, key=lambda rs: results[dname][f"regen_{rs}"]['pa_after_regen'])
        best_pa = results[dname][f"regen_{best_rs}"]['pa_after_regen']
        print(f"  {dname:>15}: best PA after regen = {best_pa*100:.2f}% (at T={best_rs})")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 154 Complete ({elapsed:.0f}s)")
    for dname in damage_types:
        for rs in regen_steps_list:
            r = results[dname][f"regen_{rs}"]
            print(f"  {dname:>15} T={rs:>2}: PA={r['pa_after_regen']*100:.2f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase154_regeneration.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 154: Planarian Regeneration',
            'timestamp': datetime.now().isoformat(),
            'base_pa': base_pa,
            'results': results,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Panel 1: Recovery by damage type (at best regen steps)
        dnames = list(damage_types.keys())
        best_pas = []
        for dn in dnames:
            best_rs = max(regen_steps_list, key=lambda rs: results[dn][f"regen_{rs}"]['pa_after_regen'])
            best_pas.append(results[dn][f"regen_{best_rs}"]['pa_after_regen'] * 100)
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        bars = axes[0].bar(range(len(dnames)), best_pas, color=colors, alpha=0.85, edgecolor='black')
        axes[0].axhline(y=base_pa*100, color='black', linestyle='--', alpha=0.5, label='Undamaged')
        for bar, pa in zip(bars, best_pas):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{pa:.1f}%', ha='center', fontweight='bold', fontsize=8)
        axes[0].set_xticks(range(len(dnames)))
        axes[0].set_xticklabels([d.replace('_', '\n') for d in dnames], fontsize=8)
        axes[0].set_ylabel('PA After Regen (%)')
        axes[0].set_title('Best Recovery by Damage Type', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=8)

        # Panel 2: Recovery vs regen steps for each damage type
        for dn, color in zip(dnames, colors):
            pas = [results[dn][f"regen_{rs}"]['pa_after_regen']*100 for rs in regen_steps_list]
            axes[1].plot(regen_steps_list, pas, 'o-', label=dn, color=color, alpha=0.7)
        axes[1].axhline(y=base_pa*100, color='black', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Regeneration Steps')
        axes[1].set_ylabel('PA After Regen (%)')
        axes[1].set_title('Recovery vs Time', fontweight='bold', fontsize=10)
        axes[1].legend(fontsize=7)

        # Panel 3: Recovery rate (PA_regen / PA_undamaged)
        recovery_rates = [p / (base_pa * 100) * 100 for p in best_pas]
        bars = axes[2].bar(range(len(dnames)), recovery_rates, color=colors, alpha=0.85, edgecolor='black')
        axes[2].axhline(y=100, color='black', linestyle='--', alpha=0.5)
        for bar, rr in zip(bars, recovery_rates):
            axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{rr:.1f}%', ha='center', fontweight='bold', fontsize=8)
        axes[2].set_xticks(range(len(dnames)))
        axes[2].set_xticklabels([d.replace('_', '\n') for d in dnames], fontsize=8)
        axes[2].set_ylabel('Recovery Rate (%)')
        axes[2].set_title('Regeneration Efficiency', fontweight='bold', fontsize=10)

        fig.suptitle('Phase 154: Planarian Regeneration (Self-Repair After Destruction)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.15, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase154_regeneration.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
