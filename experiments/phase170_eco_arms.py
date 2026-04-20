"""
Phase 170: Ecological Arms Race

Phase 168 failed because: flat grid (no hiding), no predator energy cost.
Fix: add walls (maze terrain) + predator energy depletion.

Prey: survives by hiding behind walls
Predator: must eat to refuel, dies if energy runs out

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy
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
GRID_SIZE = 20
POP_SIZE = 30
N_GENERATIONS = 40
T_STEPS = 40


class EcoNCA(nn.Module):
    """NCA for ecological agents. Input: 4 channels (empty, self, enemy, wall)."""
    def __init__(self, ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 4, 1))

    def forward(self, x):
        return self.net(x)


def make_maze(grid_size, wall_density=0.15):
    """Create a grid with random wall obstacles."""
    walls = torch.zeros(grid_size, grid_size)
    n_walls = int(grid_size * grid_size * wall_density)
    for _ in range(n_walls):
        y, x = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
        walls[y, x] = 1
    # Clear corners for spawning
    walls[:4, :4] = 0; walls[-4:, -4:] = 0
    return walls


def simulate_eco_battle(prey_nca, pred_nca, walls, grid_size=GRID_SIZE, steps=T_STEPS):
    """Simulate with walls and energy constraints."""
    # Channels: 0=empty, 1=prey, 2=predator, 3=wall
    grid = torch.zeros(4, grid_size, grid_size, device=DEVICE)
    grid[0] = 1.0 - walls.to(DEVICE)
    grid[3] = walls.to(DEVICE)

    # Spawn prey in top-left, predator in bottom-right
    for _ in range(grid_size * grid_size // 8):
        y = random.randint(0, grid_size//3)
        x = random.randint(0, grid_size//3)
        if walls[y, x] == 0:
            grid[0, y, x] = 0; grid[1, y, x] = 1

    for _ in range(grid_size * grid_size // 12):
        y = random.randint(2*grid_size//3, grid_size-1)
        x = random.randint(2*grid_size//3, grid_size-1)
        if walls[y, x] == 0:
            grid[0, y, x] = 0; grid[2, y, x] = 1

    # Predator energy
    pred_energy = torch.full((grid_size, grid_size), 10.0, device=DEVICE)

    prey_hist = [grid[1].sum().item()]
    pred_hist = [grid[2].sum().item()]
    captures = 0

    with torch.no_grad():
        for t in range(steps):
            state = grid.unsqueeze(0)

            # Prey NCA output
            prey_logits = prey_nca(state)
            prey_wants = F.softmax(prey_logits, dim=1)[0, 1] > 0.5

            # Predator NCA output
            pred_logits = pred_nca(state)
            pred_wants = F.softmax(pred_logits, dim=1)[0, 2] > 0.5

            new_grid = torch.zeros_like(grid)
            new_grid[3] = walls.to(DEVICE)

            for y in range(grid_size):
                for x in range(grid_size):
                    if walls[y, x] > 0:
                        continue

                    if pred_wants[y, x] and pred_energy[y, x] > 0:
                        if grid[1, y, x] > 0.5:
                            captures += 1
                            pred_energy[y, x] = min(pred_energy[y, x] + 5, 15)
                        new_grid[2, y, x] = 1
                        pred_energy[y, x] -= 1
                        if pred_energy[y, x] <= 0:
                            new_grid[2, y, x] = 0
                            new_grid[0, y, x] = 1
                    elif prey_wants[y, x]:
                        new_grid[1, y, x] = 1
                    else:
                        new_grid[0, y, x] = 1

            grid = new_grid
            prey_hist.append(grid[1].sum().item())
            pred_hist.append(grid[2].sum().item())

    return {
        'final_prey': grid[1].sum().item(),
        'final_pred': grid[2].sum().item(),
        'captures': captures,
        'prey_history': prey_hist,
        'pred_history': pred_hist,
    }


def mutate(model, std=0.06):
    child = copy.deepcopy(model)
    with torch.no_grad():
        for p in child.parameters():
            p.add_(torch.randn_like(p) * std)
    return child


def crossover(p1, p2):
    child = copy.deepcopy(p1)
    with torch.no_grad():
        for cp, pp1, pp2 in zip(child.parameters(), p1.parameters(), p2.parameters()):
            mask = torch.rand_like(cp) > 0.5
            cp.data = torch.where(mask, pp1.data, pp2.data)
    return child


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 170: Ecological Arms Race")
    print(f"  Maze terrain + energy constraints")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Pop: {POP_SIZE}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    walls = make_maze(GRID_SIZE, wall_density=0.15)
    n_walls = int(walls.sum().item())
    print(f"  Walls: {n_walls} cells ({n_walls/(GRID_SIZE**2)*100:.0f}%)")

    prey_pop = [EcoNCA(ch=16).to(DEVICE) for _ in range(POP_SIZE)]
    pred_pop = [EcoNCA(ch=16).to(DEVICE) for _ in range(POP_SIZE)]

    gen_prey_fit = []; gen_pred_fit = []; gen_captures = []
    n_elite = max(2, POP_SIZE // 5)

    for gen in range(N_GENERATIONS):
        eval_pred = random.choice(pred_pop)
        eval_prey = random.choice(prey_pop)

        prey_fits = []
        for prey in prey_pop:
            r = simulate_eco_battle(prey, eval_pred, walls, steps=T_STEPS)
            prey_fits.append(r['final_prey'] - r['captures'] * 0.3)

        pred_fits = []
        for pred in pred_pop:
            r = simulate_eco_battle(eval_prey, pred, walls, steps=T_STEPS)
            pred_fits.append(r['captures'] * 2 + r['final_pred'])

        gen_prey_fit.append(max(prey_fits))
        gen_pred_fit.append(max(pred_fits))

        best_prey = prey_pop[np.argmax(prey_fits)]
        best_pred = pred_pop[np.argmax(pred_fits)]
        champ = simulate_eco_battle(best_prey, best_pred, walls, steps=T_STEPS)
        gen_captures.append(champ['captures'])

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  Gen {gen+1}/{N_GENERATIONS}: "
                  f"prey_fit={max(prey_fits):.1f}, pred_fit={max(pred_fits):.1f}, "
                  f"captures={champ['captures']}, prey={champ['final_prey']:.0f}")

        # Evolve
        for pop, fits in [(prey_pop, prey_fits), (pred_pop, pred_fits)]:
            ranked = sorted(zip(fits, pop), key=lambda x: -x[0])
            elites = [copy.deepcopy(ranked[i][1]) for i in range(n_elite)]
            new_pop = list(elites)
            while len(new_pop) < POP_SIZE:
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
                child = mutate(child, std=0.06)
                new_pop.append(child)
            pop[:] = new_pop

    # Arms race check
    if len(gen_prey_fit) > 10:
        prey_trend = np.mean(gen_prey_fit[-10:]) - np.mean(gen_prey_fit[:10])
        pred_trend = np.mean(gen_pred_fit[-10:]) - np.mean(gen_pred_fit[:10])
        arms_race = prey_trend > 0 or pred_trend > 0
        coexistence = champ['final_prey'] > 5 and champ['final_pred'] > 2
    else:
        arms_race = False; coexistence = False

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 170 Complete ({elapsed:.0f}s)")
    print(f"  Final: prey={champ['final_prey']:.0f}, pred={champ['final_pred']:.0f}")
    print(f"  Captures: {champ['captures']}")
    print(f"  Arms race: {arms_race}, Coexistence: {coexistence}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase170_eco_arms.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 170: Ecological Arms Race',
            'timestamp': datetime.now().isoformat(),
            'final': champ, 'arms_race': arms_race, 'coexistence': coexistence,
            'prey_fitness': gen_prey_fit, 'pred_fitness': gen_pred_fit,
            'captures_hist': gen_captures, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(gen_prey_fit, color='green', label='Prey', linewidth=2)
        axes[0].plot(gen_pred_fit, color='red', label='Predator', linewidth=2)
        axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Fitness')
        axes[0].set_title('Fitness (with Maze)', fontweight='bold'); axes[0].legend()
        axes[1].plot(gen_captures, color='purple', linewidth=2)
        axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Captures')
        axes[1].set_title('Captures Per Battle', fontweight='bold')
        axes[2].plot(champ['prey_history'], color='green', label='Prey', linewidth=1.5)
        axes[2].plot(champ['pred_history'], color='red', label='Predator', linewidth=1.5)
        axes[2].set_xlabel('Step'); axes[2].set_ylabel('Population')
        axes[2].set_title('Final Championship', fontweight='bold'); axes[2].legend()
        fig.suptitle('Phase 170: Ecological Arms Race (Maze + Energy)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase170_eco_arms.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'arms_race': arms_race, 'coexistence': coexistence}

if __name__ == '__main__':
    main()
