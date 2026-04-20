"""
Phase 163: Evolutionary NCA - Darwinian Compiler

Phase 160 failed: Backprop fell into "lazy local minima" (output nothing).
Real life doesn't use gradients - it uses natural selection.

Replace Backprop with Genetic Algorithm (GA):
  - Population of 100 NCA individuals (random weights)
  - Fitness: "did collision produce new structure?"
  - Selection + Crossover + Mutation for 50 generations

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
POP_SIZE = 60
N_GENERATIONS = 50
ELITE_FRAC = 0.1
MUTATION_STD = 0.05


class MiniNCA(nn.Module):
    """Small NCA for evolutionary search (fewer params = faster)."""
    def __init__(self, ch=16, steps=15):
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
        return torch.sigmoid(self.proj_out(h))


def make_signal():
    p = torch.zeros(2, 2)
    p[0, 0] = 1; p[0, 1] = 1; p[1, 0] = 1
    return p


def create_test_cases(n=20):
    """Create AND gate test cases."""
    cases = []
    signal = make_signal()
    ph, pw = signal.shape
    gs = GRID_SIZE

    for _ in range(n):
        for a, b in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            inp = torch.zeros(1, 1, gs, gs)
            if a:
                sy = gs // 2 - 1
                inp[0, 0, sy:sy+ph, 0:pw] = signal
            if b:
                sx = gs // 2 - 1
                inp[0, 0, 0:ph, sx:sx+pw] = signal

            # Expected: output exists only for AND
            expected_output = (a == 1 and b == 1)
            cases.append((inp, a, b, expected_output))
    return cases


def fitness(model, cases):
    """Evaluate fitness: reward correct AND behavior."""
    model.eval()
    score = 0
    with torch.no_grad():
        for inp, a, b, expected in cases:
            pred = model(inp.to(DEVICE))
            # Check output region (bottom-right quadrant)
            output_region = pred[0, 0, -5:, -5:]
            has_output = output_region.sum().item() > 1.0

            # Check input region is cleared (signal consumed)
            input_region_sum = pred[0, 0, :4, :4].sum().item() + pred[0, 0, :4, -4:].sum().item()

            if expected:  # AND case: should have output
                if has_output:
                    score += 3.0  # Big reward for creating output
                    score += min(output_region.sum().item(), 3.0)  # Bonus for strong output
                else:
                    score -= 0.5
            else:  # NOT-AND: should NOT have output
                if not has_output:
                    score += 1.0  # Reward for correct suppression
                else:
                    score -= 1.0  # Penalty for false positive

    return score


def mutate(model, std=MUTATION_STD):
    """Apply Gaussian mutation to all parameters."""
    child = copy.deepcopy(model)
    with torch.no_grad():
        for p in child.parameters():
            p.add_(torch.randn_like(p) * std)
    return child


def crossover(parent1, parent2):
    """Uniform crossover: randomly pick params from each parent."""
    child = copy.deepcopy(parent1)
    with torch.no_grad():
        for cp, p1, p2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.rand_like(cp) > 0.5
            cp.data = torch.where(mask, p1.data, p2.data)
    return child


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 163: Evolutionary NCA")
    print(f"  GA search for AND gate (no Backprop)")
    print(f"  Population: {POP_SIZE}, Generations: {N_GENERATIONS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    cases = create_test_cases(n=5)  # 20 cases (5 per combination)
    n_elite = max(2, int(POP_SIZE * ELITE_FRAC))

    # Initialize population
    print(f"\n[Step 1] Initializing population...")
    population = [MiniNCA(ch=16, steps=15).to(DEVICE) for _ in range(POP_SIZE)]

    best_fitness_history = []
    avg_fitness_history = []
    best_and_acc_history = []

    for gen in range(N_GENERATIONS):
        # Evaluate fitness
        fitnesses = [fitness(m, cases) for m in population]

        # Sort by fitness (descending)
        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        best_fit = ranked[0][0]
        avg_fit = np.mean(fitnesses)
        best_fitness_history.append(best_fit)
        avg_fitness_history.append(avg_fit)

        # Evaluate AND accuracy for best individual
        best_model = ranked[0][1]
        best_model.eval()
        and_correct = 0; and_total = 0
        with torch.no_grad():
            for inp, a, b, expected in cases:
                if a == 1 and b == 1:
                    pred = best_model(inp.to(DEVICE))
                    has_output = pred[0, 0, -5:, -5:].sum().item() > 1.0
                    if has_output == expected:
                        and_correct += 1
                    and_total += 1
        and_acc = and_correct / max(1, and_total)
        best_and_acc_history.append(and_acc)

        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"  Gen {gen+1}/{N_GENERATIONS}: best_fit={best_fit:.1f}, "
                  f"avg_fit={avg_fit:.1f}, AND_acc={and_acc*100:.0f}%")

        # Selection + Reproduction
        elites = [copy.deepcopy(ranked[i][1]) for i in range(n_elite)]
        new_pop = list(elites)  # Elite survives

        # Adaptive mutation rate
        mutation_rate = MUTATION_STD * (1.0 + 0.5 * (1 - gen / N_GENERATIONS))

        while len(new_pop) < POP_SIZE:
            if random.random() < 0.7:  # Crossover
                p1 = random.choice(elites)
                p2 = random.choice(elites)
                child = crossover(p1, p2)
                child = mutate(child, std=mutation_rate)
            else:  # Pure mutation
                parent = random.choice(elites)
                child = mutate(parent, std=mutation_rate * 1.5)
            new_pop.append(child)

        population = new_pop

    # Final evaluation
    final_fitnesses = [fitness(m, cases) for m in population]
    best_idx = np.argmax(final_fitnesses)
    best_model = population[best_idx]

    # Detailed AND accuracy
    best_model.eval()
    case_results = {(0,0): [], (1,0): [], (0,1): [], (1,1): []}
    with torch.no_grad():
        for inp, a, b, expected in cases:
            pred = best_model(inp.to(DEVICE))
            has_output = pred[0, 0, -5:, -5:].sum().item() > 1.0
            case_results[(a, b)].append(has_output == expected)

    case_acc = {str(k): np.mean(v) for k, v in case_results.items()}
    and_success = case_acc.get('(1, 1)', 0) > 0.5

    # Also train with Backprop for comparison
    print(f"\n[Step 2] Backprop baseline for comparison...")
    bp_model = MiniNCA(ch=16, steps=15).to(DEVICE)
    bp_cases_X = torch.cat([c[0] for c in cases]).to(DEVICE)
    bp_cases_Y = torch.zeros(len(cases), 1, GRID_SIZE, GRID_SIZE, device=DEVICE)
    signal = make_signal()
    for i, (_, a, b, exp) in enumerate(cases):
        if exp:
            bp_cases_Y[i, 0, -3:-1, -3:-1] = signal

    opt = torch.optim.Adam(bp_model.parameters(), lr=1e-3)
    for epoch in range(200):
        bp_model.train()
        loss = F.binary_cross_entropy(bp_model(bp_cases_X), bp_cases_Y)
        opt.zero_grad(); loss.backward(); opt.step()

    bp_model.eval()
    bp_and_correct = 0; bp_and_total = 0
    with torch.no_grad():
        for inp, a, b, expected in cases:
            if a == 1 and b == 1:
                pred = bp_model(inp.to(DEVICE))
                has_output = pred[0, 0, -5:, -5:].sum().item() > 1.0
                if has_output == expected:
                    bp_and_correct += 1
                bp_and_total += 1
    bp_and_acc = bp_and_correct / max(1, bp_and_total)
    print(f"  Backprop AND accuracy: {bp_and_acc*100:.0f}%")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 163 Complete ({elapsed:.0f}s)")
    print(f"  GA best AND accuracy: {best_and_acc_history[-1]*100:.0f}%")
    print(f"  Backprop AND accuracy: {bp_and_acc*100:.0f}%")
    print(f"  Per-case accuracy: {case_acc}")
    print(f"  Evolution beats Backprop: {best_and_acc_history[-1] > bp_and_acc}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase163_evolutionary.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 163: Evolutionary NCA',
            'timestamp': datetime.now().isoformat(),
            'ga_and_accuracy': best_and_acc_history[-1],
            'bp_and_accuracy': bp_and_acc,
            'case_accuracy': case_acc,
            'fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'and_acc_history': best_and_acc_history,
            'evolution_wins': best_and_acc_history[-1] > bp_and_acc,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        axes[0].plot(best_fitness_history, color='#e74c3c', label='Best', linewidth=2)
        axes[0].plot(avg_fitness_history, color='#3498db', label='Average', linewidth=1.5, alpha=0.7)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title('Fitness Over Generations', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=8)

        axes[1].plot(best_and_acc_history, color='#2ecc71', linewidth=2)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('AND Accuracy')
        axes[1].set_title('AND Gate Accuracy', fontweight='bold', fontsize=10)
        axes[1].set_ylim(-0.05, 1.05)

        methods = ['GA', 'Backprop']
        accs = [best_and_acc_history[-1]*100, bp_and_acc*100]
        colors = ['#2ecc71', '#e74c3c']
        bars = axes[2].bar(methods, accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f'{acc:.0f}%', ha='center', fontweight='bold', fontsize=11)
        axes[2].set_ylabel('AND Accuracy (%)')
        axes[2].set_title('Evolution vs Backprop', fontweight='bold', fontsize=10)
        axes[2].set_ylim(0, 110)

        fig.suptitle('Phase 163: Evolutionary NCA (Darwinian AND Gate)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase163_evolutionary.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'ga_and': best_and_acc_history[-1], 'bp_and': bp_and_acc}


if __name__ == '__main__':
    main()
