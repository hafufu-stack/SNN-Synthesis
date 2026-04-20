"""
Phase 171: Thermodynamic Autopoiesis

Phase 166 failed (all-or-nothing density) because global Loss is too coarse.
Fix: add a NUTRIENT FIELD that cells consume locally.
  - Nutrients spawn and diffuse each step
  - Cells consume nutrients to stay alive / reproduce
  - No nutrients -> cell dies

This creates LOCAL resource constraints -> emergent density control.

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
N_COLORS = 3  # 0=empty, 1=alive_a, 2=alive_b


class NutrientNCA(nn.Module):
    """NCA that receives cell state + nutrient field."""
    def __init__(self, ch=32, steps=1):
        super().__init__()
        self.steps = steps
        # Input: N_COLORS + 1 nutrient channel
        self.net = nn.Sequential(
            nn.Conv2d(N_COLORS + 1, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, N_COLORS, 1))

    def forward(self, x):
        return self.net(x)


def diffuse(field, rate=0.1):
    """Simple diffusion via average pooling."""
    kernel = torch.ones(1, 1, 3, 3, device=field.device) / 9.0
    padded = F.pad(field.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
    avg = F.conv2d(padded, kernel)[0, 0]
    return field * (1 - rate) + avg * rate


def simulate_thermodynamic(nca, grid_size=GRID_SIZE, total_steps=200,
                            nutrient_spawn_rate=0.05, nutrient_cost=0.3,
                            diffusion_rate=0.15):
    """Simulate cells + nutrient field dynamics."""
    nca.eval()

    # Initialize grid: sparse random cells
    grid = torch.zeros(N_COLORS, grid_size, grid_size, device=DEVICE)
    grid[0] = 1.0  # all empty
    for _ in range(grid_size * grid_size // 8):
        y = random.randint(0, grid_size-1)
        x = random.randint(0, grid_size-1)
        c = random.choice([1, 2])
        grid[0, y, x] = 0; grid[c, y, x] = 1

    # Nutrient field
    nutrients = torch.rand(grid_size, grid_size, device=DEVICE) * 0.5

    pop_history = []
    nutrient_history = []

    with torch.no_grad():
        for t in range(total_steps):
            # Record
            alive = (grid[0] < 0.5).float()
            pop_history.append({
                'alive': alive.sum().item(),
                'color1': grid[1].sum().item(),
                'color2': grid[2].sum().item(),
            })
            nutrient_history.append(nutrients.mean().item())

            # Spawn nutrients (uniform random)
            nutrients = nutrients + torch.rand_like(nutrients) * nutrient_spawn_rate

            # Diffuse nutrients
            nutrients = diffuse(nutrients, diffusion_rate)

            # NCA decides next state
            inp = torch.cat([grid, nutrients.unsqueeze(0)], dim=0).unsqueeze(0)
            logits = nca(inp)
            probs = F.softmax(logits[0], dim=0)

            # Apply thermodynamic rules (vectorized)
            proposed = probs.argmax(dim=0)  # (H, W) cell type each pixel wants

            wants_alive = proposed > 0  # mask: wants to be a living cell
            has_food = nutrients >= nutrient_cost  # mask: enough nutrients

            survives = wants_alive & has_food  # alive AND fed
            starves = wants_alive & ~has_food  # alive BUT starved

            # Build new grid via scatter
            new_grid = torch.zeros_like(grid)
            # Survivors: set new_grid[proposed[y,x], y, x] = 1
            new_grid.scatter_(0, proposed.unsqueeze(0), survives.unsqueeze(0).float())
            # Dead/starved/empty -> background
            dead_mask = ~survives
            new_grid[0][dead_mask] = 1.0

            # Deduct nutrients where cells survived
            nutrients = nutrients - (survives.float() * nutrient_cost)

            grid = new_grid
            nutrients = nutrients.clamp(0, 2.0)

    return pop_history, nutrient_history


def train_nutrient_nca(n_epochs=200):
    """
    Train NCA with thermodynamic fitness:
    maximize alive cells over time (GA since argmax is non-differentiable).
    """
    import copy

    pop_size = 30
    n_elite = 5

    population = [NutrientNCA(ch=16, steps=1).to(DEVICE) for _ in range(pop_size)]

    best_fitness_hist = []

    for gen in range(n_epochs):
        fitnesses = []
        for nca in population:
            pop_hist, _ = simulate_thermodynamic(
                nca, grid_size=GRID_SIZE, total_steps=80,
                nutrient_spawn_rate=0.05, nutrient_cost=0.2)

            # Fitness: average alive cells in second half + survival bonus
            second_half = pop_hist[40:]
            avg_alive = np.mean([h['alive'] for h in second_half])
            # Penalty for full death or full takeover
            final_alive = pop_hist[-1]['alive']
            diversity = min(pop_hist[-1]['color1'], pop_hist[-1]['color2'])

            fitness = avg_alive + diversity * 0.5
            if final_alive < 5:
                fitness -= 20  # Death penalty
            if final_alive > GRID_SIZE * GRID_SIZE * 0.8:
                fitness -= 10  # Cancer penalty

            fitnesses.append(fitness)

        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        best_fitness_hist.append(ranked[0][0])

        if (gen + 1) % 20 == 0 or gen == 0:
            print(f"  Gen {gen+1}/{n_epochs}: best_fit={ranked[0][0]:.1f}, "
                  f"avg_fit={np.mean(fitnesses):.1f}")

        elites = [copy.deepcopy(ranked[i][1]) for i in range(n_elite)]
        new_pop = list(elites)
        while len(new_pop) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            child = copy.deepcopy(p1)
            with torch.no_grad():
                for cp, pp1, pp2 in zip(child.parameters(), p1.parameters(), p2.parameters()):
                    mask = torch.rand_like(cp) > 0.5
                    cp.data = torch.where(mask, pp1.data, pp2.data)
                    cp.add_(torch.randn_like(cp) * 0.05)
            new_pop.append(child)
        population = new_pop

    return ranked[0][1], best_fitness_hist


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 171: Thermodynamic Autopoiesis")
    print(f"  Cells + Nutrient field -> self-organized life")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Train via GA
    print("\n[Step 1] Evolving NCA with nutrient dynamics...")
    best_nca, fitness_hist = train_nutrient_nca(n_epochs=60)

    # Full simulation
    print("\n[Step 2] Full simulation (200 steps)...")
    pop_hist, nut_hist = simulate_thermodynamic(
        best_nca, grid_size=GRID_SIZE, total_steps=200,
        nutrient_spawn_rate=0.05, nutrient_cost=0.2)

    alive_series = [h['alive'] for h in pop_hist]
    c1_series = [h['color1'] for h in pop_hist]
    c2_series = [h['color2'] for h in pop_hist]

    final_alive = pop_hist[-1]['alive']
    avg_alive = np.mean(alive_series[100:])
    alive_var = np.var(alive_series[100:])

    # Autopoiesis = alive, not dead, not cancer, has variance (moves around)
    autopoiesis = (10 < avg_alive < GRID_SIZE * GRID_SIZE * 0.6) and alive_var > 10

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 171 Complete ({elapsed:.0f}s)")
    print(f"  Final alive: {final_alive:.0f}")
    print(f"  Avg alive (t>100): {avg_alive:.1f}")
    print(f"  Alive variance: {alive_var:.1f}")
    print(f"  Autopoiesis achieved: {autopoiesis}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase171_autopoiesis.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 171: Thermodynamic Autopoiesis',
            'timestamp': datetime.now().isoformat(),
            'final_alive': final_alive, 'avg_alive': avg_alive,
            'alive_variance': alive_var, 'autopoiesis': autopoiesis,
            'fitness_history': fitness_hist,
            'population_history': pop_hist,
            'nutrient_history': nut_hist,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(alive_series, color='green', label='Total', linewidth=1.5)
        axes[0].plot(c1_series, color='#e74c3c', label='Species A', linewidth=1, alpha=0.7)
        axes[0].plot(c2_series, color='#3498db', label='Species B', linewidth=1, alpha=0.7)
        axes[0].set_xlabel('Step'); axes[0].set_ylabel('Population')
        axes[0].set_title('Population Dynamics', fontweight='bold'); axes[0].legend(fontsize=7)

        axes[1].plot(nut_hist, color='#f39c12', linewidth=1.5)
        axes[1].set_xlabel('Step'); axes[1].set_ylabel('Avg Nutrients')
        axes[1].set_title('Nutrient Field', fontweight='bold')

        axes[2].plot(fitness_hist, color='#2ecc71', linewidth=2)
        axes[2].set_xlabel('Generation'); axes[2].set_ylabel('Best Fitness')
        axes[2].set_title('GA Evolution', fontweight='bold')

        status = "AUTOPOIESIS!" if autopoiesis else "No autopoiesis"
        fig.suptitle(f'Phase 171: Thermodynamic Autopoiesis ({status})',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase171_autopoiesis.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'autopoiesis': autopoiesis, 'avg_alive': avg_alive}

if __name__ == '__main__':
    main()
