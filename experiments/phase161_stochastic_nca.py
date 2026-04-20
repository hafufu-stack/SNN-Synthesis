"""
Phase 161: Stochastic NCA - Reviving Chaos via Discrete Noise

Phase 158 showed that NCA learns the mean-field of LV rules,
killing population oscillations. The fix: inject discrete noise
at inference time to break the mean-field approximation.

No new training needed - we retrain the Phase 158 model, then
test with different noise injection strategies:
  1. Deterministic (baseline - should show no oscillation)
  2. Gumbel-softmax sampling
  3. Bernoulli sampling (hardest discretization)
  4. Gaussian noise injection

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
N_SPECIES = 4


# ================================================================
# LV rules + NCA (from Phase 158)
# ================================================================
def lv_step(grid, grass_spread=0.3, sheep_eat=0.5, sheep_die=0.1,
            wolf_eat=0.4, wolf_die=0.2):
    h, w = grid.shape
    new = grid.copy()
    coords = [(y, x) for y in range(h) for x in range(w)]
    random.shuffle(coords)
    for y, x in coords:
        cell = grid[y, x]
        dy, dx = random.choice([(-1,0),(1,0),(0,-1),(0,1)])
        ny, nx = (y+dy)%h, (x+dx)%w
        neighbor = grid[ny, nx]
        if cell == 1:
            if neighbor == 0 and random.random() < grass_spread:
                new[ny, nx] = 1
        elif cell == 2:
            if neighbor == 1 and random.random() < sheep_eat:
                new[ny, nx] = 2
            elif random.random() < sheep_die:
                new[y, x] = 0
        elif cell == 3:
            if neighbor == 2 and random.random() < wolf_eat:
                new[ny, nx] = 3
            elif random.random() < wolf_die:
                new[y, x] = 0
    return new


def generate_lv_data(n_samples, grid_size=GRID_SIZE):
    X, Y = [], []
    for _ in range(n_samples):
        grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        for y in range(grid_size):
            for x in range(grid_size):
                r = random.random()
                if r < 0.3: grid[y, x] = 1
                elif r < 0.4: grid[y, x] = 2
                elif r < 0.45: grid[y, x] = 3
        X.append(grid.copy())
        grid = lv_step(grid)
        Y.append(grid)
    X_oh = np.zeros((n_samples, N_SPECIES, grid_size, grid_size), dtype=np.float32)
    Y_lab = np.zeros((n_samples, grid_size, grid_size), dtype=np.int64)
    for i in range(n_samples):
        for c in range(N_SPECIES):
            X_oh[i, c] = (X[i] == c).astype(np.float32)
        Y_lab[i] = Y[i]
    return torch.from_numpy(X_oh), torch.from_numpy(Y_lab)


class LVDS(torch.utils.data.Dataset):
    def __init__(self, n=3000):
        self.X, self.Y = generate_lv_data(n)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


class LVNCA(nn.Module):
    def __init__(self, n_species=N_SPECIES, ch=48, steps=1):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(n_species, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, n_species, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)


# ================================================================
# Stochastic simulation strategies
# ================================================================
def simulate_deterministic(model, init_oh, total_steps=200):
    """Standard deterministic rollout (baseline - no oscillation expected)."""
    state = init_oh.unsqueeze(0).to(DEVICE)
    history = []
    with torch.no_grad():
        for t in range(total_steps):
            logits = model(state, steps=1)
            pred = logits.argmax(dim=1)
            counts = {s: int((pred[0] == s).sum()) for s in range(N_SPECIES)}
            history.append(counts)
            state = torch.zeros_like(state)
            for c in range(N_SPECIES):
                state[0, c] = (pred[0] == c).float()
    return history


def simulate_gumbel(model, init_oh, total_steps=200, tau=0.5):
    """Gumbel-softmax sampling at each step."""
    state = init_oh.unsqueeze(0).to(DEVICE)
    history = []
    with torch.no_grad():
        for t in range(total_steps):
            logits = model(state, steps=1)
            # Gumbel-softmax sampling
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            noisy_logits = (logits + gumbel) / tau
            pred = noisy_logits.argmax(dim=1)
            counts = {s: int((pred[0] == s).sum()) for s in range(N_SPECIES)}
            history.append(counts)
            state = torch.zeros_like(state)
            for c in range(N_SPECIES):
                state[0, c] = (pred[0] == c).float()
    return history


def simulate_bernoulli(model, init_oh, total_steps=200):
    """Bernoulli sampling: sample each species independently."""
    state = init_oh.unsqueeze(0).to(DEVICE)
    history = []
    with torch.no_grad():
        for t in range(total_steps):
            logits = model(state, steps=1)
            probs = F.softmax(logits, dim=1)
            # Sample from categorical distribution
            B, C, H, W = probs.shape
            flat_probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
            sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
            pred = sampled.reshape(B, H, W)
            counts = {s: int((pred[0] == s).sum()) for s in range(N_SPECIES)}
            history.append(counts)
            state = torch.zeros_like(state)
            for c in range(N_SPECIES):
                state[0, c] = (pred[0] == c).float()
    return history


def simulate_gaussian_noise(model, init_oh, total_steps=200, sigma=0.3):
    """Add Gaussian noise to logits before argmax."""
    state = init_oh.unsqueeze(0).to(DEVICE)
    history = []
    with torch.no_grad():
        for t in range(total_steps):
            logits = model(state, steps=1)
            noisy = logits + torch.randn_like(logits) * sigma
            pred = noisy.argmax(dim=1)
            counts = {s: int((pred[0] == s).sum()) for s in range(N_SPECIES)}
            history.append(counts)
            state = torch.zeros_like(state)
            for c in range(N_SPECIES):
                state[0, c] = (pred[0] == c).float()
    return history


def compute_oscillation_score(history, species=1):
    """Measure oscillation strength via variance of population."""
    pops = [h[species] for h in history]
    if len(pops) < 50:
        return 0
    # Use second half for steady-state analysis
    steady = pops[len(pops)//4:]
    return float(np.var(steady))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 161: Stochastic NCA - Reviving Chaos")
    print(f"  Can discrete noise injection restore population oscillation?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Retrain Phase 158 model
    print("\n[Step 1] Training LV-NCA...")
    train_ds = LVDS(5000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    model = LVNCA(n_species=N_SPECIES, ch=48, steps=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    for epoch in range(80):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch+1) % 40 == 0:
            print(f"  Epoch {epoch+1}/80")

    model.eval()

    # Create initial state
    init_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            r = random.random()
            if r < 0.3: init_grid[y, x] = 1
            elif r < 0.4: init_grid[y, x] = 2
            elif r < 0.45: init_grid[y, x] = 3
    init_oh = torch.zeros(N_SPECIES, GRID_SIZE, GRID_SIZE)
    for c in range(N_SPECIES):
        init_oh[c] = torch.from_numpy((init_grid == c).astype(np.float32))

    # Run all strategies
    strategies = {
        'deterministic': lambda: simulate_deterministic(model, init_oh, 200),
        'gumbel_0.5': lambda: simulate_gumbel(model, init_oh, 200, tau=0.5),
        'gumbel_1.0': lambda: simulate_gumbel(model, init_oh, 200, tau=1.0),
        'bernoulli': lambda: simulate_bernoulli(model, init_oh, 200),
        'gaussian_0.3': lambda: simulate_gaussian_noise(model, init_oh, 200, sigma=0.3),
        'gaussian_1.0': lambda: simulate_gaussian_noise(model, init_oh, 200, sigma=1.0),
    }

    results = {}
    all_histories = {}

    print("\n[Step 2] Testing noise injection strategies...")
    for name, fn in strategies.items():
        torch.manual_seed(SEED)
        history = fn()
        grass_osc = compute_oscillation_score(history, 1)
        sheep_osc = compute_oscillation_score(history, 2)
        wolf_osc = compute_oscillation_score(history, 3)
        total_osc = grass_osc + sheep_osc + wolf_osc

        results[name] = {
            'grass_oscillation': grass_osc,
            'sheep_oscillation': sheep_osc,
            'wolf_oscillation': wolf_osc,
            'total_oscillation': total_osc,
            'final_grass': history[-1][1],
            'final_sheep': history[-1][2],
            'final_wolf': history[-1].get(3, 0),
        }
        all_histories[name] = history
        print(f"  {name:>16}: osc={total_osc:>10.0f}, "
              f"grass={history[-1][1]}, sheep={history[-1][2]}, wolf={history[-1].get(3,0)}")

    # Ground truth LV for comparison
    print("\n[Step 3] Ground truth LV simulation...")
    gt_grid = init_grid.copy()
    gt_history = []
    for t in range(200):
        gt_grid = lv_step(gt_grid)
        gt_history.append({s: int((gt_grid == s).sum()) for s in range(N_SPECIES)})
    gt_osc = compute_oscillation_score(gt_history, 1)
    print(f"  GT oscillation: {gt_osc:.0f}")

    # Check if any stochastic method restored oscillation
    det_osc = results['deterministic']['total_oscillation']
    best_stoch = max(
        [(n, r['total_oscillation']) for n, r in results.items() if n != 'deterministic'],
        key=lambda x: x[1])
    chaos_revived = best_stoch[1] > det_osc * 5 and best_stoch[1] > 100

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 161 Complete ({elapsed:.0f}s)")
    print(f"  Deterministic oscillation: {det_osc:.0f}")
    print(f"  Best stochastic: {best_stoch[0]} = {best_stoch[1]:.0f}")
    print(f"  GT oscillation: {gt_osc:.0f}")
    print(f"  Chaos revived: {chaos_revived}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase161_stochastic_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 161: Stochastic NCA',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'gt_oscillation': gt_osc,
            'chaos_revived': chaos_revived,
            'best_strategy': best_stoch[0],
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Panel 1: Deterministic vs best stochastic (grass population)
        det_grass = [h[1] for h in all_histories['deterministic']]
        best_name = best_stoch[0]
        stoch_grass = [h[1] for h in all_histories[best_name]]
        axes[0].plot(det_grass, color='gray', label='Deterministic', linewidth=1.5)
        axes[0].plot(stoch_grass, color='green', label=f'{best_name}', linewidth=1.5, alpha=0.8)
        gt_grass = [h[1] for h in gt_history]
        axes[0].plot(gt_grass, color='blue', label='Ground Truth', linewidth=1, alpha=0.5)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Grass Population')
        axes[0].set_title('Grass Dynamics', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=7)

        # Panel 2: Oscillation scores comparison
        names = list(results.keys())
        oscs = [results[n]['total_oscillation'] for n in names]
        colors = ['#95a5a6', '#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
        bars = axes[1].bar(range(len(names)), oscs, color=colors[:len(names)],
                          alpha=0.85, edgecolor='black')
        axes[1].axhline(y=gt_osc, color='blue', linestyle='--', alpha=0.5, label='GT')
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
        axes[1].set_ylabel('Oscillation Score')
        axes[1].set_title('Chaos Strength', fontweight='bold', fontsize=10)
        axes[1].legend(fontsize=8)

        # Panel 3: Phase plot (grass vs sheep) for best stochastic
        stoch_sheep = [h[2] for h in all_histories[best_name]]
        axes[2].plot(stoch_grass, stoch_sheep, alpha=0.4, linewidth=0.8, color='#2ecc71')
        axes[2].scatter(stoch_grass[0], stoch_sheep[0], color='green', s=60, zorder=5, label='Start')
        axes[2].scatter(stoch_grass[-1], stoch_sheep[-1], color='red', s=60, zorder=5, label='End')
        axes[2].set_xlabel('Grass')
        axes[2].set_ylabel('Sheep')
        axes[2].set_title(f'Phase Portrait ({best_name})', fontweight='bold', fontsize=10)
        axes[2].legend(fontsize=8)

        fig.suptitle('Phase 161: Stochastic NCA (Reviving Chaos via Discrete Noise)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.15, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase161_stochastic_nca.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
