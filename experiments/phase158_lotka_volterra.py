"""
Phase 158: Lotka-Volterra Automata - Predator-Prey Oscillation

Phase 156 showed coexistence with stable borders. Now we test
dynamic ecosystems with ASYMMETRIC rules (food chain):
  - Grass (1): spreads into empty (0) cells
  - Sheep (2): eats grass (overwrites 1->2), dies without grass
  - Wolf (3): eats sheep (overwrites 2->3), dies without sheep

Train NCA to learn these rules, then observe population oscillations.

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
N_SPECIES = 4  # 0=empty, 1=grass, 2=sheep, 3=wolf


# ================================================================
# Lotka-Volterra Rules (ground truth simulation)
# ================================================================
def lv_step(grid, grass_spread=0.3, sheep_eat=0.5, sheep_die=0.1,
            wolf_eat=0.4, wolf_die=0.2):
    """One step of stochastic Lotka-Volterra CA."""
    h, w = grid.shape
    new = grid.copy()
    # Process in random order
    coords = [(y, x) for y in range(h) for x in range(w)]
    random.shuffle(coords)

    for y, x in coords:
        cell = grid[y, x]
        # Get random neighbor
        dy, dx = random.choice([(-1,0),(1,0),(0,-1),(0,1)])
        ny, nx = (y+dy)%h, (x+dx)%w
        neighbor = grid[ny, nx]

        if cell == 1:  # Grass
            if neighbor == 0 and random.random() < grass_spread:
                new[ny, nx] = 1  # Grass spreads
        elif cell == 2:  # Sheep
            if neighbor == 1 and random.random() < sheep_eat:
                new[ny, nx] = 2  # Sheep eats grass
            elif random.random() < sheep_die:
                new[y, x] = 0  # Sheep dies
        elif cell == 3:  # Wolf
            if neighbor == 2 and random.random() < wolf_eat:
                new[ny, nx] = 3  # Wolf eats sheep
            elif random.random() < wolf_die:
                new[y, x] = 0  # Wolf dies
        elif cell == 0:  # Empty
            pass

    return new


def generate_lv_data(n_samples, n_steps=1, grid_size=GRID_SIZE):
    """Generate Lotka-Volterra input-output pairs."""
    X, Y = [], []
    for _ in range(n_samples):
        # Random initial state
        grid = np.zeros((grid_size, grid_size), dtype=np.int64)
        for y in range(grid_size):
            for x in range(grid_size):
                r = random.random()
                if r < 0.3: grid[y, x] = 1  # grass
                elif r < 0.4: grid[y, x] = 2  # sheep
                elif r < 0.45: grid[y, x] = 3  # wolf

        X.append(grid.copy())
        for _ in range(n_steps):
            grid = lv_step(grid)
        Y.append(grid)

    # One-hot encode
    X_oh = np.zeros((n_samples, N_SPECIES, grid_size, grid_size), dtype=np.float32)
    Y_labels = np.zeros((n_samples, grid_size, grid_size), dtype=np.int64)
    for i in range(n_samples):
        for c in range(N_SPECIES):
            X_oh[i, c] = (X[i] == c).astype(np.float32)
        Y_labels[i] = Y[i]

    return torch.from_numpy(X_oh), torch.from_numpy(Y_labels)


class LVDS(torch.utils.data.Dataset):
    def __init__(self, n=3000):
        self.X, self.Y = generate_lv_data(n)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# LV-NCA
# ================================================================
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


def count_species(grid):
    """Count population of each species."""
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    counts = {}
    for s in range(N_SPECIES):
        counts[s] = int((grid == s).sum())
    return counts


def simulate_ecosystem(model, init_grid_oh, total_steps=100):
    """Run NCA forward many steps, recording population dynamics."""
    model.eval()
    state = init_grid_oh.unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    population_history = []

    with torch.no_grad():
        for t in range(total_steps):
            logits = model(state, steps=1)
            pred = logits.argmax(dim=1)  # (1, H, W)
            counts = count_species(pred[0])
            population_history.append(counts)

            # Convert back to one-hot for next step
            state = torch.zeros_like(state)
            for c in range(N_SPECIES):
                state[0, c] = (pred[0] == c).float()

    return population_history


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 158: Lotka-Volterra Automata")
    print(f"  Predator-prey ecosystem dynamics")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Generate training data
    print("\n[Step 1] Generating LV training data...")
    train_ds = LVDS(5000)
    test_ds = LVDS(500)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    # Train 1-step predictor
    model = LVNCA(n_species=N_SPECIES, ch=48, steps=1).to(DEVICE)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"\n[Step 2] Training LV-NCA ({n_p:,} params)...")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    for epoch in range(80):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 20 == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = model(x).argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            print(f"  Epoch {epoch+1}/80: PA={correct/total*100:.2f}%")

    # Simulate ecosystem
    print(f"\n[Step 3] Simulating ecosystem (200 steps)...")
    # Create balanced initial state
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

    pop_history = simulate_ecosystem(model, init_oh, total_steps=200)

    # Ground truth simulation for comparison
    print(f"  Running ground truth LV simulation...")
    gt_grid = init_grid.copy()
    gt_history = []
    for t in range(200):
        gt_grid = lv_step(gt_grid)
        gt_history.append(count_species(gt_grid))

    # Analyze oscillation
    grass_nca = [h[1] for h in pop_history]
    sheep_nca = [h[2] for h in pop_history]
    wolf_nca = [h.get(3, 0) for h in pop_history]
    grass_gt = [h[1] for h in gt_history]
    sheep_gt = [h[2] for h in gt_history]

    # Check for oscillation (variance in population)
    grass_var = np.var(grass_nca[50:]) if len(grass_nca) > 50 else 0
    sheep_var = np.var(sheep_nca[50:]) if len(sheep_nca) > 50 else 0
    oscillation = grass_var > 100 or sheep_var > 100

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 158 Complete ({elapsed:.0f}s)")
    print(f"  NCA Grass final: {grass_nca[-1]}, Sheep: {sheep_nca[-1]}, Wolf: {wolf_nca[-1]}")
    print(f"  GT  Grass final: {grass_gt[-1]}, Sheep: {sheep_gt[-1]}")
    print(f"  Population variance (NCA): grass={grass_var:.0f}, sheep={sheep_var:.0f}")
    print(f"  Oscillation detected: {oscillation}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase158_lotka_volterra.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 158: Lotka-Volterra Automata',
            'timestamp': datetime.now().isoformat(),
            'nca_population': pop_history,
            'gt_population': gt_history,
            'grass_variance': float(grass_var),
            'sheep_variance': float(sheep_var),
            'oscillation': oscillation,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # NCA population dynamics
        axes[0].plot(grass_nca, color='green', label='Grass', linewidth=1.5)
        axes[0].plot(sheep_nca, color='orange', label='Sheep', linewidth=1.5)
        axes[0].plot(wolf_nca, color='red', label='Wolf', linewidth=1.5)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Population')
        axes[0].set_title('NCA Ecosystem', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=8)

        # Ground truth
        axes[1].plot(grass_gt, color='green', label='Grass', linewidth=1.5)
        axes[1].plot(sheep_gt, color='orange', label='Sheep', linewidth=1.5)
        wolf_gt = [h.get(3, 0) for h in gt_history]
        axes[1].plot(wolf_gt, color='red', label='Wolf', linewidth=1.5)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Population')
        axes[1].set_title('Ground Truth LV', fontweight='bold', fontsize=10)
        axes[1].legend(fontsize=8)

        # Phase plot (Grass vs Sheep)
        axes[2].plot(grass_nca, sheep_nca, alpha=0.5, linewidth=0.8, color='#3498db')
        axes[2].scatter(grass_nca[0], sheep_nca[0], color='green', s=60, zorder=5, label='Start')
        axes[2].scatter(grass_nca[-1], sheep_nca[-1], color='red', s=60, zorder=5, label='End')
        axes[2].set_xlabel('Grass Population')
        axes[2].set_ylabel('Sheep Population')
        axes[2].set_title('Phase Portrait (NCA)', fontweight='bold', fontsize=10)
        axes[2].legend(fontsize=8)

        fig.suptitle('Phase 158: Lotka-Volterra Predator-Prey Dynamics',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase158_lotka_volterra.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'oscillation': oscillation, 'grass_var': grass_var, 'sheep_var': sheep_var}


if __name__ == '__main__':
    main()
