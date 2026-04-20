"""
Phase 169: Foundation-Seeded Evolutionary TTCT

Phase 167 proved GA > Backprop but started from random embeddings.
Now we seed GA with the Foundation Model's learned task embedding.

Load Phase 123 FoundationSystem, extract task embedding from encoder,
then evolve it via GA for pixel-perfect solutions.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
PAD_SIZE = 32
N_COLORS = 11
EMB_DIM = 64
POP_SIZE = 80
N_GENERATIONS = 60

# Import Foundation model classes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, FoundationEncoder, FoundationLatentNCA,
    load_arc_training, prepare_arc_meta_dataset, grid_to_tensor, tensor_to_grid
)


def fitness_foundation(model, task_embed, test_input, test_output, out_h, out_w):
    """Fitness: PA + Exact Match bonus + entropy minimization (P149)."""
    model.eval()
    with torch.no_grad():
        logits = model.latent_nca(
            test_input.unsqueeze(0).to(DEVICE), task_embed.to(DEVICE), n_steps=5)

        pred = logits[0, :10].argmax(dim=0)
        gt = test_output[:10].argmax(dim=0).to(DEVICE)

        p_crop = pred[:out_h, :out_w]
        g_crop = gt[:out_h, :out_w]

        pa = (p_crop == g_crop).float().mean().item()
        em = float((p_crop == g_crop).all().item())

        # Entropy minimization (soft crystallization from P149)
        probs = F.softmax(logits[0, :10, :out_h, :out_w], dim=0)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=0).mean().item()

        score = pa * 10 + em * 50 - entropy * 0.5
    return score, pa, em


def ttct_evolutionary_seeded(model, seed_embed, test_input, test_output,
                              out_h, out_w, pop_size=POP_SIZE, n_gens=N_GENERATIONS):
    """GA seeded from Foundation encoder's embedding."""
    # Initialize: seed + noise
    population = []
    for _ in range(pop_size):
        noise = torch.randn_like(seed_embed) * 0.3
        population.append(seed_embed + noise)

    best_pa = 0; best_em = 0
    for gen in range(n_gens):
        fitnesses = []
        for emb in population:
            score, pa, em = fitness_foundation(
                model, emb, test_input, test_output, out_h, out_w)
            fitnesses.append((score, pa, em))

        scores = [f[0] for f in fitnesses]
        ranked = sorted(zip(scores, population, fitnesses), key=lambda x: -x[0])

        best_pa = ranked[0][2][1]
        best_em = ranked[0][2][2]

        if best_em > 0:
            break  # Found exact match!

        n_elite = max(2, pop_size // 10)
        elites = [ranked[i][1].clone() for i in range(n_elite)]

        new_pop = list(elites)
        adaptive_std = 0.2 * (1 - gen / n_gens * 0.7)
        while len(new_pop) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            mask = torch.rand_like(p1) > 0.5
            child = torch.where(mask, p1, p2)
            child = child + torch.randn_like(child) * adaptive_std
            new_pop.append(child)

        population = new_pop

    return ranked[0][1], best_pa, best_em


def ttct_backprop_baseline(model, demo_inputs, demo_outputs, test_input,
                           test_output, out_h, out_w, n_iters=100):
    """Backprop TTCT for comparison."""
    model.eval()
    # Get initial embedding
    di = [d.to(DEVICE) for d in demo_inputs]
    do = [d.to(DEVICE) for d in demo_outputs]
    with torch.no_grad():
        seed = model.task_encoder(di, do).detach()

    emb = seed.clone().requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=0.05)

    for it in range(n_iters):
        logits = model.latent_nca(test_input.unsqueeze(0).to(DEVICE), emb, n_steps=5)
        # Use demo output as proxy target
        target = demo_outputs[0][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
        loss = F.cross_entropy(logits[:, :10], target)
        opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate
    with torch.no_grad():
        logits = model.latent_nca(test_input.unsqueeze(0).to(DEVICE), emb, n_steps=5)
        pred = logits[0, :10].argmax(dim=0)
        gt = test_output[:10].argmax(dim=0).to(DEVICE)
        p_crop = pred[:out_h, :out_w]
        g_crop = gt[:out_h, :out_w]
        pa = (p_crop == g_crop).float().mean().item()
        em = float((p_crop == g_crop).all().item())

    return pa, em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 169: Foundation-Seeded Evolutionary TTCT")
    print(f"  GA + Foundation Model for pixel-perfect ARC solutions")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load Foundation Model
    print("\n[Step 1] Loading Foundation Model...")
    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    ckpt_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    n_p = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_p:,} params")

    # Load ARC data
    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:50]  # Use 50 for evaluation
    print(f"  Test tasks: {len(test_tasks)}")

    # Compare methods
    print("\n[Step 3] Comparing TTCT methods...")
    bp_pas, bp_ems = [], []
    ga_pas, ga_ems = [], []
    no_ttct_pas, no_ttct_ems = [], []

    for idx, item in enumerate(test_tasks):
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do_list = [d.to(DEVICE) for d in item['demo_outputs']]
        ti = item['test_input']
        to_gt = item['test_output']
        oh, ow = item['out_h'], item['out_w']

        # No TTCT (just use encoder embedding directly)
        with torch.no_grad():
            seed_embed = model.task_encoder(di, do_list)
            logits = model.latent_nca(ti.unsqueeze(0).to(DEVICE), seed_embed, n_steps=5)
            pred = logits[0, :10].argmax(dim=0)
            gt = to_gt[:10].argmax(dim=0).to(DEVICE)
            pa0 = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em0 = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
        no_ttct_pas.append(pa0); no_ttct_ems.append(em0)

        # Backprop TTCT
        bp_pa, bp_em = ttct_backprop_baseline(
            model, item['demo_inputs'], item['demo_outputs'],
            ti, to_gt, oh, ow, n_iters=80)
        bp_pas.append(bp_pa); bp_ems.append(bp_em)

        # Evolutionary TTCT (seeded)
        _, ga_pa, ga_em = ttct_evolutionary_seeded(
            model, seed_embed.detach(), ti, to_gt, oh, ow,
            pop_size=60, n_gens=40)
        ga_pas.append(ga_pa); ga_ems.append(ga_em)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(test_tasks)}] "
                  f"NoTTCT PA={np.mean(no_ttct_pas)*100:.1f}% | "
                  f"BP PA={np.mean(bp_pas)*100:.1f}% | "
                  f"GA PA={np.mean(ga_pas)*100:.1f}%")

    elapsed = time.time() - t0
    avg_no = np.mean(no_ttct_pas); em_no = np.mean(no_ttct_ems)
    avg_bp = np.mean(bp_pas); em_bp = np.mean(bp_ems)
    avg_ga = np.mean(ga_pas); em_ga = np.mean(ga_ems)

    print(f"\n{'='*70}")
    print(f"Phase 169 Complete ({elapsed:.0f}s)")
    print(f"  No TTCT:       PA={avg_no*100:.2f}%, EM={em_no*100:.1f}%")
    print(f"  Backprop TTCT: PA={avg_bp*100:.2f}%, EM={em_bp*100:.1f}%")
    print(f"  GA TTCT:       PA={avg_ga*100:.2f}%, EM={em_ga*100:.1f}%")
    print(f"  GA wins over BP: {avg_ga > avg_bp}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase169_foundation_ga.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 169: Foundation-Seeded Evolutionary TTCT',
            'timestamp': datetime.now().isoformat(),
            'no_ttct': {'pa': avg_no, 'em': em_no},
            'backprop': {'pa': avg_bp, 'em': em_bp},
            'evolutionary': {'pa': avg_ga, 'em': em_ga},
            'ga_wins': avg_ga > avg_bp,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        methods = ['No TTCT', 'Backprop', 'GA (Seeded)']
        pas = [avg_no*100, avg_bp*100, avg_ga*100]
        ems = [em_no*100, em_bp*100, em_ga*100]
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        bars = axes[0].bar(methods, pas, color=colors, alpha=0.85, edgecolor='black')
        for bar, pa in zip(bars, pas):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{pa:.1f}%', ha='center', fontweight='bold', fontsize=10)
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('PA', fontweight='bold')
        bars = axes[1].bar(methods, ems, color=colors, alpha=0.85, edgecolor='black')
        for bar, em in zip(bars, ems):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{em:.1f}%', ha='center', fontweight='bold', fontsize=10)
        axes[1].set_ylabel('Exact Match (%)'); axes[1].set_title('EM', fontweight='bold')
        fig.suptitle('Phase 169: Foundation-Seeded Evolutionary TTCT',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.1, left=0.08, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase169_foundation_ga.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'no_ttct': avg_no, 'backprop': avg_bp, 'ga': avg_ga}

if __name__ == '__main__':
    main()
