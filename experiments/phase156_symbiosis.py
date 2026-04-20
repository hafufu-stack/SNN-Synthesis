"""
Phase 156: Multi-Cellular Symbiosis - Territorial Competition

Two species (Pattern A and Pattern B) placed at opposite corners
of a 32x32 grid, both trying to self-replicate. What happens when
they collide in the middle?

Possible outcomes:
- Domination: one species overwrites the other
- Coexistence: stable boundary (cell membrane)
- Chaos: unstable oscillating frontier
- Fusion: patterns merge into a hybrid

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
GRID_SIZE = 32


# ================================================================
# Multi-Channel Replicator (supports multiple species via colors)
# ================================================================
class MultiSpeciesNCA(nn.Module):
    """NCA that can handle multi-valued grids (multiple species)."""
    def __init__(self, n_colors=4, ch=48, steps=30):
        super().__init__()
        self.steps = steps
        self.n_colors = n_colors
        self.proj_in = nn.Conv2d(n_colors, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, n_colors, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        states = []
        for t in range(steps):
            h = F.relu(h + self.rule(h))
            if t % 5 == 0 or t == steps - 1:
                pred = self.proj_out(h).argmax(dim=1)
                states.append(pred.cpu())
        self._states = states
        return self.proj_out(h)


def create_two_species_data(n_samples, grid_size=GRID_SIZE):
    """Create training data: two species tiling from opposite corners."""
    X, Y = [], []
    for _ in range(n_samples):
        # Pattern A: top-left (values 1,2)
        pa = torch.zeros(2, 2)
        pa[0, 0] = 1; pa[0, 1] = 1; pa[1, 0] = 1; pa[1, 1] = 0
        # Random variation
        if random.random() > 0.5: pa = pa.flip(0)

        # Pattern B: bottom-right (values 2,3)
        pb = torch.zeros(2, 2)
        pb[0, 0] = 2; pb[0, 1] = 2; pb[1, 0] = 0; pb[1, 1] = 2
        if random.random() > 0.5: pb = pb.flip(1)

        # Input: seeds at corners
        inp = torch.zeros(grid_size, grid_size)
        inp[:2, :2] = pa
        inp[-2:, -2:] = pb

        # Output: each species tiles its half
        out = torch.zeros(grid_size, grid_size)
        mid = grid_size // 2
        # Species A tiles left half
        for y in range(0, grid_size, 2):
            for x in range(0, mid, 2):
                h = min(2, grid_size - y)
                w = min(2, mid - x)
                out[y:y+h, x:x+w] = pa[:h, :w]
        # Species B tiles right half
        for y in range(0, grid_size, 2):
            for x in range(mid, grid_size, 2):
                h = min(2, grid_size - y)
                w = min(2, grid_size - x)
                out[y:y+h, x:x+w] = pb[:h, :w]

        X.append(inp)
        Y.append(out)

    # One-hot encode
    X_oh = torch.zeros(n_samples, 4, grid_size, grid_size)
    Y_oh = torch.zeros(n_samples, 4, grid_size, grid_size)
    for i in range(n_samples):
        for c in range(4):
            X_oh[i, c] = (X[i] == c).float()
            Y_oh[i, c] = (Y[i] == c).float()

    return X_oh, Y_oh, X, Y


class SymbiosisDS(torch.utils.data.Dataset):
    def __init__(self, n=2000):
        self.X_oh, self.Y_oh, self.X_raw, self.Y_raw = create_two_species_data(n)
    def __len__(self): return len(self.X_oh)
    def __getitem__(self, i): return self.X_oh[i], self.Y_oh[i]


# ================================================================
# Analysis functions
# ================================================================
def analyze_territory(pred_grid, grid_size=GRID_SIZE):
    """Analyze territorial occupation of each species."""
    pred = pred_grid.numpy() if isinstance(pred_grid, torch.Tensor) else pred_grid
    mid = grid_size // 2

    # Species A territory (values 1)
    a_left = (pred[:, :mid] == 1).sum()
    a_right = (pred[:, mid:] == 1).sum()
    # Species B territory (values 2)  
    b_left = (pred[:, :mid] == 2).sum()
    b_right = (pred[:, mid:] == 2).sum()
    # Empty (value 0)
    empty = (pred == 0).sum()
    # Boundary complexity
    h_boundaries = (pred[1:, :] != pred[:-1, :]).sum()
    v_boundaries = (pred[:, 1:] != pred[:, :-1]).sum()

    total = pred.size
    return {
        'a_total': int((pred == 1).sum()),
        'b_total': int((pred == 2).sum()),
        'a_in_b_territory': int(a_right),
        'b_in_a_territory': int(b_left),
        'empty': int(empty),
        'boundary_complexity': int(h_boundaries + v_boundaries),
        'a_fraction': float((pred == 1).sum() / total),
        'b_fraction': float((pred == 2).sum() / total),
    }


def determine_outcome(territory_info):
    """Classify the interaction outcome."""
    a_frac = territory_info['a_fraction']
    b_frac = territory_info['b_fraction']

    if a_frac < 0.05 and b_frac > 0.3:
        return "B_dominates"
    elif b_frac < 0.05 and a_frac > 0.3:
        return "A_dominates"
    elif abs(a_frac - b_frac) < 0.15 and a_frac > 0.1:
        return "coexistence"
    elif a_frac < 0.05 and b_frac < 0.05:
        return "mutual_extinction"
    else:
        return "asymmetric_coexistence"


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 156: Multi-Cellular Symbiosis")
    print(f"  Two species compete for territory on {GRID_SIZE}x{GRID_SIZE} grid")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Prepare data
    train_ds = SymbiosisDS(2000)
    test_ds = SymbiosisDS(200)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

    # Train multi-species NCA
    model = MultiSpeciesNCA(n_colors=4, ch=48, steps=30).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"\n[Step 1] Training Multi-Species NCA ({n_p:,} params)...")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    for epoch in range(100):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            target = y.argmax(dim=1)  # (B, H, W)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch+1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(test_ds.X_oh[:16].to(DEVICE))
                pred = logits.argmax(dim=1)
                target = test_ds.Y_oh[:16].argmax(dim=1).to(DEVICE)
                pa = (pred == target).float().mean().item()
            print(f"  Epoch {epoch+1}/100: PA={pa*100:.2f}%")

    # Evaluate territorial dynamics
    print(f"\n[Step 2] Analyzing Territorial Dynamics...")
    model.eval()
    territories = []
    outcomes = {}
    step_snapshots = []

    with torch.no_grad():
        for i in range(min(50, len(test_ds))):
            x = test_ds.X_oh[i:i+1].to(DEVICE)
            logits = model(x, steps=30)
            pred = logits[0].argmax(0).cpu().numpy()

            terr = analyze_territory(pred, GRID_SIZE)
            outcome = determine_outcome(terr)
            territories.append(terr)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

            # Save step-by-step evolution for first sample
            if i == 0:
                step_snapshots = [s[0].numpy().tolist() for s in model._states]

    # Aggregate statistics
    avg_a = np.mean([t['a_fraction'] for t in territories])
    avg_b = np.mean([t['b_fraction'] for t in territories])
    avg_boundary = np.mean([t['boundary_complexity'] for t in territories])
    avg_invasion_a = np.mean([t['a_in_b_territory'] for t in territories])
    avg_invasion_b = np.mean([t['b_in_a_territory'] for t in territories])

    # Test at different step counts
    print(f"\n[Step 3] Temporal Evolution...")
    step_results = {}
    for n_steps in [10, 20, 30, 50, 80]:
        model.eval()
        with torch.no_grad():
            x = test_ds.X_oh[:20].to(DEVICE)
            logits = model(x, steps=n_steps)
            pred = logits.argmax(dim=1).cpu()
            terrs = [analyze_territory(pred[b].numpy(), GRID_SIZE) for b in range(pred.shape[0])]
            avg_a_t = np.mean([t['a_fraction'] for t in terrs])
            avg_b_t = np.mean([t['b_fraction'] for t in terrs])
            avg_bound_t = np.mean([t['boundary_complexity'] for t in terrs])
        step_results[n_steps] = {
            'avg_a': avg_a_t, 'avg_b': avg_b_t, 'boundary': avg_bound_t
        }
        print(f"  T={n_steps:>3}: A={avg_a_t*100:.1f}%, B={avg_b_t*100:.1f}%, "
              f"boundary={avg_bound_t:.0f}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 156 Complete ({elapsed:.0f}s)")
    print(f"  Avg species A: {avg_a*100:.1f}%")
    print(f"  Avg species B: {avg_b*100:.1f}%")
    print(f"  Avg boundary complexity: {avg_boundary:.0f}")
    print(f"  Outcomes: {outcomes}")
    print(f"  A invasion into B territory: {avg_invasion_a:.0f} cells")
    print(f"  B invasion into A territory: {avg_invasion_b:.0f} cells")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase156_symbiosis.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 156: Multi-Cellular Symbiosis',
            'timestamp': datetime.now().isoformat(),
            'avg_a_fraction': avg_a, 'avg_b_fraction': avg_b,
            'avg_boundary_complexity': avg_boundary,
            'outcomes': outcomes,
            'step_results': step_results,
            'step_snapshots': step_snapshots,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Panel 1: Territory fractions over time
        steps = sorted(step_results.keys())
        a_fracs = [step_results[s]['avg_a']*100 for s in steps]
        b_fracs = [step_results[s]['avg_b']*100 for s in steps]
        bounds = [step_results[s]['boundary'] for s in steps]
        axes[0].plot(steps, a_fracs, 'o-', color='#e74c3c', label='Species A', linewidth=2)
        axes[0].plot(steps, b_fracs, 's-', color='#3498db', label='Species B', linewidth=2)
        axes[0].set_xlabel('NCA Steps')
        axes[0].set_ylabel('Territory (%)')
        axes[0].set_title('Territorial Expansion', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=9)

        # Panel 2: Boundary complexity over time
        axes[1].plot(steps, bounds, 'o-', color='#2ecc71', linewidth=2)
        axes[1].set_xlabel('NCA Steps')
        axes[1].set_ylabel('Boundary Complexity')
        axes[1].set_title('Border Dynamics', fontweight='bold', fontsize=10)

        # Panel 3: Outcome distribution
        if outcomes:
            outcome_names = list(outcomes.keys())
            outcome_counts = list(outcomes.values())
            colors_pie = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            bars = axes[2].bar(range(len(outcome_names)), outcome_counts,
                              color=colors_pie[:len(outcome_names)], alpha=0.85, edgecolor='black')
            axes[2].set_xticks(range(len(outcome_names)))
            axes[2].set_xticklabels([o.replace('_', '\n') for o in outcome_names], fontsize=8)
            axes[2].set_ylabel('Count')
            axes[2].set_title('Interaction Outcomes', fontweight='bold', fontsize=10)

        fig.suptitle('Phase 156: Multi-Cellular Symbiosis (Two-Species Competition)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.15, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase156_symbiosis.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'outcomes': outcomes, 'step_results': step_results}


if __name__ == '__main__':
    main()
