"""
Phase 167: Evolutionary TTCT - GA-based Test-Time Context Tuning

Phase 163 proved: GA > Backprop for sparse/discrete objectives.
Apply this to ARC: instead of backprop-optimizing Task Embedding,
evolve a population of embeddings via GA.

This bypasses the VQ Paradox (gradients dying through argmax/VQ).

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
GRID_SIZE = 8
N_COLORS = 6
EMB_DIM = 32
POP_SIZE = 80
N_GENERATIONS = 50


# ================================================================
# ARC-like tasks with varying rules
# ================================================================
def task_fill_color(grid, color):
    """Fill all zeros with the given color."""
    out = grid.clone()
    out[grid == 0] = color
    return out


def task_border(grid, color):
    """Add a border of color around the grid."""
    out = grid.clone()
    out[0, :] = color; out[-1, :] = color
    out[:, 0] = color; out[:, -1] = color
    return out


def task_gravity(grid):
    """Drop all non-zero cells to the bottom."""
    out = torch.zeros_like(grid)
    for x in range(grid.shape[1]):
        col = grid[:, x]
        nonzero = col[col != 0]
        if len(nonzero) > 0:
            out[-len(nonzero):, x] = nonzero
    return out


def generate_task(task_type, n_demos=3, n_test=5, grid_size=GRID_SIZE):
    """Generate demo + test pairs for a task."""
    demos_in, demos_out, tests_in, tests_out = [], [], [], []

    for _ in range(n_demos + n_test):
        grid = torch.zeros(grid_size, grid_size, dtype=torch.long)
        n_cells = random.randint(3, grid_size * grid_size // 3)
        for _ in range(n_cells):
            y, x = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            grid[y, x] = random.randint(1, N_COLORS - 1)

        if task_type == 'fill':
            color = random.choice([1, 2, 3])
            out = task_fill_color(grid, color)
        elif task_type == 'border':
            color = random.choice([1, 2, 3])
            out = task_border(grid, color)
        elif task_type == 'gravity':
            out = task_gravity(grid)
        else:
            raise ValueError(task_type)

        if len(demos_in) < n_demos:
            demos_in.append(grid); demos_out.append(out)
        else:
            tests_in.append(grid); tests_out.append(out)

    return demos_in, demos_out, tests_in, tests_out


# ================================================================
# Context-Conditioned NCA (uses embedding as control)
# ================================================================
class ContextNCA(nn.Module):
    def __init__(self, n_colors=N_COLORS, emb_dim=EMB_DIM, ch=48, steps=10):
        super().__init__()
        self.steps = steps
        self.emb_dim = emb_dim
        # Input: one-hot grid + embedding broadcast
        self.proj_in = nn.Conv2d(n_colors + emb_dim, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, n_colors, 1)

    def forward(self, grid_oh, embedding):
        """
        grid_oh: (B, C, H, W) one-hot encoded
        embedding: (B, emb_dim) context vector
        """
        B, C, H, W = grid_oh.shape
        # Broadcast embedding to spatial dims
        emb_spatial = embedding.unsqueeze(-1).unsqueeze(-1).expand(B, self.emb_dim, H, W)
        x = torch.cat([grid_oh, emb_spatial], dim=1)
        h = F.relu(self.proj_in(x))
        for _ in range(self.steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)


def to_onehot(grids, n_colors=N_COLORS):
    """Convert list of grids to one-hot."""
    gs = grids[0].shape[0]
    oh = torch.zeros(len(grids), n_colors, gs, gs)
    for i, g in enumerate(grids):
        for c in range(n_colors):
            oh[i, c] = (g == c).float()
    return oh


# ================================================================
# TTCT: Backprop vs GA
# ================================================================
def ttct_backprop(model, demos_in, demos_out, n_iters=100, lr=0.1):
    """Traditional TTCT: optimize embedding via backprop on demo pairs."""
    model.eval()
    emb = torch.randn(1, EMB_DIM, device=DEVICE) * 0.1
    emb.requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=lr)

    demo_x = to_onehot(demos_in).to(DEVICE)
    demo_y = torch.stack(demos_out).to(DEVICE)

    for it in range(n_iters):
        emb_batch = emb.expand(len(demos_in), -1)
        logits = model(demo_x, emb_batch)
        loss = F.cross_entropy(logits, demo_y)
        opt.zero_grad(); loss.backward(); opt.step()

    return emb.detach()


def ttct_evolutionary(model, demos_in, demos_out, pop_size=POP_SIZE,
                      n_gens=N_GENERATIONS, mutation_std=0.3):
    """Evolutionary TTCT: optimize embedding via GA."""
    model.eval()
    demo_x = to_onehot(demos_in).to(DEVICE)
    demo_y = torch.stack(demos_out).to(DEVICE)

    # Initialize population
    population = [torch.randn(1, EMB_DIM, device=DEVICE) * 0.1 for _ in range(pop_size)]

    def eval_fitness(emb):
        with torch.no_grad():
            emb_batch = emb.expand(len(demos_in), -1)
            logits = model(demo_x, emb_batch)
            pred = logits.argmax(dim=1)
            pa = (pred == demo_y).float().mean().item()
            em = all((pred[i] == demo_y[i]).all() for i in range(len(demo_y)))
            return pa + (5.0 if em else 0)  # Big bonus for exact match

    for gen in range(n_gens):
        fitnesses = [eval_fitness(e) for e in population]
        ranked = sorted(zip(fitnesses, population), key=lambda x: -x[0])

        n_elite = max(2, pop_size // 10)
        elites = [ranked[i][1].clone() for i in range(n_elite)]

        new_pop = list(elites)
        adaptive_std = mutation_std * (1 - gen / n_gens * 0.5)
        while len(new_pop) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            mask = torch.rand(1, EMB_DIM, device=DEVICE) > 0.5
            child = torch.where(mask, p1, p2)
            child = child + torch.randn_like(child) * adaptive_std
            new_pop.append(child)

        population = new_pop

    # Return best
    fitnesses = [eval_fitness(e) for e in population]
    best_idx = np.argmax(fitnesses)
    return population[best_idx]


def evaluate_with_embedding(model, emb, tests_in, tests_out):
    """Evaluate on test set using the optimized embedding."""
    model.eval()
    test_x = to_onehot(tests_in).to(DEVICE)
    test_y = torch.stack(tests_out).to(DEVICE)

    with torch.no_grad():
        emb_batch = emb.expand(len(tests_in), -1)
        logits = model(test_x, emb_batch)
        pred = logits.argmax(dim=1)
        pa = (pred == test_y).float().mean().item()
        em_count = sum((pred[i] == test_y[i]).all().item() for i in range(len(test_y)))

    return pa, em_count / len(test_y)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 167: Evolutionary TTCT")
    print(f"  GA vs Backprop for Task Embedding optimization")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Emb: {EMB_DIM}D")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    tasks = ['fill', 'border', 'gravity']

    # Pre-train model on all tasks with random embeddings
    print("\n[Step 1] Pre-training ContextNCA...")
    model = ContextNCA(n_colors=N_COLORS, emb_dim=EMB_DIM, ch=48, steps=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(80):
        model.train()
        total_loss = 0; n_batch = 0
        for task in tasks:
            demos_in, demos_out, _, _ = generate_task(task, n_demos=16, n_test=0)
            x = to_onehot(demos_in).to(DEVICE)
            y = torch.stack(demos_out).to(DEVICE)
            emb = torch.randn(len(demos_in), EMB_DIM, device=DEVICE) * 0.1
            logits = model(x, emb)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batch += 1

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/80: loss={total_loss/n_batch:.4f}")

    model.eval()

    # Compare Backprop TTCT vs Evolutionary TTCT
    print("\n[Step 2] Comparing TTCT methods...")
    results = {'backprop': {}, 'evolutionary': {}}
    n_trials = 5

    for task in tasks:
        bp_pas, bp_ems = [], []
        ga_pas, ga_ems = [], []

        for trial in range(n_trials):
            torch.manual_seed(SEED + trial); random.seed(SEED + trial)
            demos_in, demos_out, tests_in, tests_out = generate_task(task, n_demos=3, n_test=5)

            # Backprop TTCT
            emb_bp = ttct_backprop(model, demos_in, demos_out, n_iters=100)
            pa_bp, em_bp = evaluate_with_embedding(model, emb_bp, tests_in, tests_out)
            bp_pas.append(pa_bp); bp_ems.append(em_bp)

            # Evolutionary TTCT
            emb_ga = ttct_evolutionary(model, demos_in, demos_out,
                                       pop_size=POP_SIZE, n_gens=N_GENERATIONS)
            pa_ga, em_ga = evaluate_with_embedding(model, emb_ga, tests_in, tests_out)
            ga_pas.append(pa_ga); ga_ems.append(em_ga)

        results['backprop'][task] = {
            'pa': np.mean(bp_pas), 'em': np.mean(bp_ems),
        }
        results['evolutionary'][task] = {
            'pa': np.mean(ga_pas), 'em': np.mean(ga_ems),
        }
        print(f"  {task:>10}: BP PA={np.mean(bp_pas)*100:.1f}% EM={np.mean(bp_ems)*100:.1f}% | "
              f"GA PA={np.mean(ga_pas)*100:.1f}% EM={np.mean(ga_ems)*100:.1f}%")

    # Summary
    bp_avg_pa = np.mean([results['backprop'][t]['pa'] for t in tasks])
    ga_avg_pa = np.mean([results['evolutionary'][t]['pa'] for t in tasks])
    bp_avg_em = np.mean([results['backprop'][t]['em'] for t in tasks])
    ga_avg_em = np.mean([results['evolutionary'][t]['em'] for t in tasks])
    evolution_wins = ga_avg_pa > bp_avg_pa or ga_avg_em > bp_avg_em

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 167 Complete ({elapsed:.0f}s)")
    print(f"  Backprop TTCT: PA={bp_avg_pa*100:.1f}%, EM={bp_avg_em*100:.1f}%")
    print(f"  Evolutionary TTCT: PA={ga_avg_pa*100:.1f}%, EM={ga_avg_em*100:.1f}%")
    print(f"  Evolution wins: {evolution_wins}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase167_evolutionary_ttct.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 167: Evolutionary TTCT',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'bp_avg_pa': bp_avg_pa, 'ga_avg_pa': ga_avg_pa,
            'bp_avg_em': bp_avg_em, 'ga_avg_em': ga_avg_em,
            'evolution_wins': evolution_wins,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        x = np.arange(len(tasks))
        bp_pas = [results['backprop'][t]['pa']*100 for t in tasks]
        ga_pas = [results['evolutionary'][t]['pa']*100 for t in tasks]
        axes[0].bar(x - 0.15, bp_pas, 0.3, label='Backprop', color='#e74c3c', alpha=0.85)
        axes[0].bar(x + 0.15, ga_pas, 0.3, label='GA', color='#2ecc71', alpha=0.85)
        axes[0].set_xticks(x); axes[0].set_xticklabels(tasks)
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].legend(fontsize=8)
        axes[0].set_title('PA: Backprop vs GA', fontweight='bold', fontsize=10)

        bp_ems = [results['backprop'][t]['em']*100 for t in tasks]
        ga_ems = [results['evolutionary'][t]['em']*100 for t in tasks]
        axes[1].bar(x - 0.15, bp_ems, 0.3, label='Backprop', color='#e74c3c', alpha=0.85)
        axes[1].bar(x + 0.15, ga_ems, 0.3, label='GA', color='#2ecc71', alpha=0.85)
        axes[1].set_xticks(x); axes[1].set_xticklabels(tasks)
        axes[1].set_ylabel('Exact Match (%)'); axes[1].legend(fontsize=8)
        axes[1].set_title('EM: Backprop vs GA', fontweight='bold', fontsize=10)

        methods = ['Backprop', 'Evolutionary']
        avg_pas = [bp_avg_pa*100, ga_avg_pa*100]
        avg_ems = [bp_avg_em*100, ga_avg_em*100]
        x2 = np.arange(2)
        axes[2].bar(x2 - 0.15, avg_pas, 0.3, label='PA', color='#3498db', alpha=0.85)
        axes[2].bar(x2 + 0.15, avg_ems, 0.3, label='EM', color='#f39c12', alpha=0.85)
        axes[2].set_xticks(x2); axes[2].set_xticklabels(methods)
        axes[2].set_ylabel('(%)')
        axes[2].set_title('Overall Average', fontweight='bold', fontsize=10)
        axes[2].legend(fontsize=8)

        fig.suptitle('Phase 167: Evolutionary TTCT (GA vs Backprop for ARC)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.1, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase167_evolutionary_ttct.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
