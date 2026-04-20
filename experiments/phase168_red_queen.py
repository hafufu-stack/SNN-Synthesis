"""
Phase 168: The Red Queen Co-Evolution

Two NCA populations (Prey and Predator) evolve against each other
via genetic algorithm. No backprop - pure Darwinian evolution.

Prey fitness: survive (more alive cells after T steps)
Predator fitness: consume prey cells (more captures)

Should produce an arms race: defense walls, stealth, speed, etc.

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
GRID_SIZE = 16
POP_SIZE = 30
N_GENERATIONS = 40
T_STEPS = 30


class TinyNCA(nn.Module):
    """Tiny NCA for evolutionary search."""
    def __init__(self, ch=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, 3, 1))

    def forward(self, x):
        return self.net(x)


def simulate_battle(prey_nca, predator_nca, grid_size=GRID_SIZE, steps=T_STEPS):
    """
    Simulate prey vs predator on a shared grid.
    Channels: 0=empty, 1=prey, 2=predator
    """
    # Initialize: prey in left half, predator in right half
    grid = torch.zeros(3, grid_size, grid_size, device=DEVICE)
    # Background
    grid[0] = 1.0

    # Prey: random cells in left half
    for _ in range(grid_size * grid_size // 6):
        y = random.randint(0, grid_size-1)
        x = random.randint(0, grid_size//2 - 1)
        grid[0, y, x] = 0; grid[1, y, x] = 1

    # Predator: random cells in right half
    for _ in range(grid_size * grid_size // 8):
        y = random.randint(0, grid_size-1)
        x = random.randint(grid_size//2, grid_size-1)
        grid[0, y, x] = 0; grid[2, y, x] = 1

    init_prey = grid[1].sum().item()
    init_pred = grid[2].sum().item()

    prey_history = [init_prey]
    pred_history = [init_pred]
    captures = 0

    with torch.no_grad():
        state = grid.unsqueeze(0)  # (1, 3, H, W)

        for t in range(steps):
            # Prey acts
            prey_logits = prey_nca(state)
            prey_probs = F.softmax(prey_logits, dim=1)

            # Predator acts
            pred_logits = predator_nca(state)
            pred_probs = F.softmax(pred_logits, dim=1)

            # Combine: predator dominates prey in conflict
            combined = torch.zeros_like(state)
            # Background by default
            combined[0, 0] = 1.0

            # Prey spreads where predator is absent
            prey_wants = prey_probs[0, 1] > 0.5
            pred_wants = pred_probs[0, 2] > 0.5

            # Predator captures prey
            new_captures = (pred_wants & (state[0, 1] > 0.5)).sum().item()
            captures += new_captures

            # Update grid
            new_grid = torch.zeros(3, grid_size, grid_size, device=DEVICE)
            new_grid[0] = 1.0  # background

            for y in range(grid_size):
                for x in range(grid_size):
                    if pred_wants[y, x]:
                        new_grid[0, y, x] = 0; new_grid[2, y, x] = 1
                    elif prey_wants[y, x]:
                        new_grid[0, y, x] = 0; new_grid[1, y, x] = 1

            state = new_grid.unsqueeze(0)
            prey_history.append(new_grid[1].sum().item())
            pred_history.append(new_grid[2].sum().item())

    final_prey = state[0, 1].sum().item()
    final_pred = state[0, 2].sum().item()

    return {
        'final_prey': final_prey, 'final_pred': final_pred,
        'captures': captures,
        'prey_history': prey_history, 'pred_history': pred_history,
    }


def mutate(model, std=0.05):
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
    print("Phase 168: Red Queen Co-Evolution")
    print(f"  Prey vs Predator evolutionary arms race")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Pop: {POP_SIZE}, Gen: {N_GENERATIONS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Initialize populations
    prey_pop = [TinyNCA(ch=12).to(DEVICE) for _ in range(POP_SIZE)]
    pred_pop = [TinyNCA(ch=12).to(DEVICE) for _ in range(POP_SIZE)]

    # Track evolution
    gen_prey_fitness = []
    gen_pred_fitness = []
    gen_captures = []

    n_elite = max(2, POP_SIZE // 5)

    for gen in range(N_GENERATIONS):
        # Use a random opponent for fitness evaluation
        eval_pred = random.choice(pred_pop)  # fixed opponent for prey
        eval_prey = random.choice(prey_pop)  # fixed opponent for predator

        # Evaluate prey fitness (survive longer = better)
        prey_fits = []
        for prey in prey_pop:
            result = simulate_battle(prey, eval_pred, steps=T_STEPS)
            prey_fits.append(result['final_prey'] - result['captures'] * 0.5)

        # Evaluate predator fitness (capture more = better)
        pred_fits = []
        for pred in pred_pop:
            result = simulate_battle(eval_prey, pred, steps=T_STEPS)
            pred_fits.append(result['captures'] + result['final_pred'] * 0.5)

        gen_prey_fitness.append(max(prey_fits))
        gen_pred_fitness.append(max(pred_fits))

        # Best match result
        best_prey = prey_pop[np.argmax(prey_fits)]
        best_pred = pred_pop[np.argmax(pred_fits)]
        champ_result = simulate_battle(best_prey, best_pred, steps=T_STEPS)
        gen_captures.append(champ_result['captures'])

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  Gen {gen+1}/{N_GENERATIONS}: "
                  f"prey_fit={max(prey_fits):.1f}, pred_fit={max(pred_fits):.1f}, "
                  f"captures={champ_result['captures']:.0f}, "
                  f"prey_alive={champ_result['final_prey']:.0f}")

        # Evolve prey
        ranked = sorted(zip(prey_fits, prey_pop), key=lambda x: -x[0])
        elites = [copy.deepcopy(ranked[i][1]) for i in range(n_elite)]
        new_prey = list(elites)
        while len(new_prey) < POP_SIZE:
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            child = mutate(child, std=0.08)
            new_prey.append(child)
        prey_pop = new_prey

        # Evolve predators
        ranked = sorted(zip(pred_fits, pred_pop), key=lambda x: -x[0])
        elites = [copy.deepcopy(ranked[i][1]) for i in range(n_elite)]
        new_pred = list(elites)
        while len(new_pred) < POP_SIZE:
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            child = mutate(child, std=0.08)
            new_pred.append(child)
        pred_pop = new_pred

    # Final championship
    best_prey = prey_pop[np.argmax([simulate_battle(p, random.choice(pred_pop), steps=T_STEPS)['final_prey'] for p in prey_pop])]
    best_pred = pred_pop[np.argmax([simulate_battle(random.choice(prey_pop), p, steps=T_STEPS)['captures'] for p in pred_pop])]
    final_result = simulate_battle(best_prey, best_pred, steps=T_STEPS)

    # Check for arms race (both fitness increasing)
    if len(gen_prey_fitness) > 5:
        prey_trend = np.mean(gen_prey_fitness[-5:]) - np.mean(gen_prey_fitness[:5])
        pred_trend = np.mean(gen_pred_fitness[-5:]) - np.mean(gen_pred_fitness[:5])
        arms_race = prey_trend > 0 and pred_trend > 0
    else:
        arms_race = False

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 168 Complete ({elapsed:.0f}s)")
    print(f"  Final: prey={final_result['final_prey']:.0f}, "
          f"pred={final_result['final_pred']:.0f}, "
          f"captures={final_result['captures']:.0f}")
    print(f"  Arms race detected: {arms_race}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase168_red_queen.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 168: Red Queen Co-Evolution',
            'timestamp': datetime.now().isoformat(),
            'final_result': final_result,
            'prey_fitness_history': gen_prey_fitness,
            'pred_fitness_history': gen_pred_fitness,
            'captures_history': gen_captures,
            'arms_race': arms_race,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(gen_prey_fitness, color='green', label='Prey', linewidth=2)
        axes[0].plot(gen_pred_fitness, color='red', label='Predator', linewidth=2)
        axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Best Fitness')
        axes[0].set_title('Fitness Evolution', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=8)

        axes[1].plot(gen_captures, color='purple', linewidth=2)
        axes[1].set_xlabel('Generation'); axes[1].set_ylabel('Captures')
        axes[1].set_title('Captures Per Battle', fontweight='bold', fontsize=10)

        axes[2].plot(final_result['prey_history'], color='green', label='Prey', linewidth=1.5)
        axes[2].plot(final_result['pred_history'], color='red', label='Predator', linewidth=1.5)
        axes[2].set_xlabel('Time Step'); axes[2].set_ylabel('Population')
        axes[2].set_title('Final Championship', fontweight='bold', fontsize=10)
        axes[2].legend(fontsize=8)

        fig.suptitle('Phase 168: Red Queen Co-Evolution (Arms Race)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase168_red_queen.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'arms_race': arms_race, 'final': final_result}


if __name__ == '__main__':
    main()
