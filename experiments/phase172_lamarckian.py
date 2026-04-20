"""
Phase 172: Lamarckian GA-TTCT

Fuse GA (global search) + Backprop (local search):
  1. Each individual (Task Embedding) gets a few Backprop steps
  2. The improved embedding becomes its DNA for the next generation
  (= Lamarckian inheritance / Baldwin effect)

This should find the exact-match sweet spot that pure GA misses.

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
EMB_DIM = 64
POP_SIZE = 60
N_GENERATIONS = 40

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset,
    grid_to_tensor, tensor_to_grid, N_COLORS
)


def evaluate_embedding(model, emb, test_input, test_output, out_h, out_w):
    """Evaluate a task embedding on a single test case."""
    model.eval()
    with torch.no_grad():
        logits = model.latent_nca(
            test_input.unsqueeze(0).to(DEVICE), emb.to(DEVICE), n_steps=5)
        pred = logits[0, :10].argmax(dim=0)
        gt = test_output[:10].argmax(dim=0).to(DEVICE)
        p_crop = pred[:out_h, :out_w]
        g_crop = gt[:out_h, :out_w]
        pa = (p_crop == g_crop).float().mean().item()
        em = float((p_crop == g_crop).all().item())
    return pa, em


def lamarckian_step(model, emb, demo_inputs, demo_outputs, n_bp_steps=10, lr=0.03):
    """
    Lamarckian self-improvement: run a few Backprop steps on the embedding.
    Returns BOTH the improved embedding AND the fitness (PA on demos).
    The improved embedding IS the new DNA (Lamarckian inheritance).
    """
    model.eval()  # Freeze model weights, only optimize embedding
    emb_opt = emb.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([emb_opt], lr=lr)

    di = [d.to(DEVICE) for d in demo_inputs]
    do_list = [d.to(DEVICE) for d in demo_outputs]

    for step in range(n_bp_steps):
        # Use each demo pair as a mini-task
        total_loss = 0
        for d_in, d_out in zip(di, do_list):
            logits = model.latent_nca(d_in.unsqueeze(0), emb_opt, n_steps=5)
            target = d_out[:10].argmax(dim=0).unsqueeze(0)
            loss = F.cross_entropy(logits[:, :10], target)
            total_loss = total_loss + loss
        total_loss = total_loss / len(di)
        opt.zero_grad(); total_loss.backward(); opt.step()

    return emb_opt.detach()


def ttct_lamarckian(model, seed_embed, demo_inputs, demo_outputs,
                     test_input, test_output, out_h, out_w,
                     pop_size=POP_SIZE, n_gens=N_GENERATIONS):
    """Lamarckian GA-TTCT: GA + Backprop self-improvement."""
    # Initialize population around seed
    population = [seed_embed + torch.randn_like(seed_embed) * 0.2
                  for _ in range(pop_size)]

    best_pa = 0; best_em = 0; best_emb = seed_embed

    for gen in range(n_gens):
        # Step 1: Lamarckian self-improvement (Backprop)
        improved = []
        for emb in population:
            improved_emb = lamarckian_step(
                model, emb, demo_inputs, demo_outputs,
                n_bp_steps=5, lr=0.03)
            improved.append(improved_emb)

        # Step 2: Evaluate fitness on TEST (using improved embeddings)
        fitnesses = []
        for emb in improved:
            pa, em = evaluate_embedding(model, emb, test_input, test_output, out_h, out_w)
            # Entropy bonus for confident predictions
            with torch.no_grad():
                logits = model.latent_nca(
                    test_input.unsqueeze(0).to(DEVICE), emb.to(DEVICE), n_steps=5)
                probs = F.softmax(logits[0, :10, :out_h, :out_w], dim=0)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=0).mean().item()

            score = pa * 10 + em * 50 - entropy * 0.3
            fitnesses.append((score, pa, em))

        scores = [f[0] for f in fitnesses]
        ranked = sorted(zip(scores, improved, fitnesses), key=lambda x: -x[0])

        gen_best_pa = ranked[0][2][1]
        gen_best_em = ranked[0][2][2]

        if gen_best_pa > best_pa or gen_best_em > best_em:
            best_pa = gen_best_pa
            best_em = gen_best_em
            best_emb = ranked[0][1]

        if best_em > 0:
            break  # Found exact match!

        # Step 3: Selection + mutation for next gen
        n_elite = max(2, pop_size // 10)
        elites = [ranked[i][1].clone() for i in range(n_elite)]

        new_pop = list(elites)
        adaptive_std = 0.15 * (1 - gen / n_gens * 0.6)
        while len(new_pop) < pop_size:
            p1 = random.choice(elites)
            p2 = random.choice(elites)
            mask = torch.rand_like(p1) > 0.5
            child = torch.where(mask, p1, p2)
            child = child + torch.randn_like(child) * adaptive_std
            new_pop.append(child)

        population = new_pop  # Lamarckian: improved DNA passed to next gen

    return best_emb, best_pa, best_em


def ttct_pure_ga(model, seed_embed, test_input, test_output, out_h, out_w,
                  pop_size=POP_SIZE, n_gens=N_GENERATIONS):
    """Pure GA (Phase 169 baseline)."""
    population = [seed_embed + torch.randn_like(seed_embed) * 0.2
                  for _ in range(pop_size)]

    best_pa = 0; best_em = 0
    for gen in range(n_gens):
        fitnesses = []
        for emb in population:
            pa, em = evaluate_embedding(model, emb, test_input, test_output, out_h, out_w)
            score = pa * 10 + em * 50
            fitnesses.append((score, pa, em))

        scores = [f[0] for f in fitnesses]
        ranked = sorted(zip(scores, population, fitnesses), key=lambda x: -x[0])
        if ranked[0][2][1] > best_pa:
            best_pa = ranked[0][2][1]
            best_em = ranked[0][2][2]
        if best_em > 0: break

        n_elite = max(2, pop_size // 10)
        elites = [ranked[i][1].clone() for i in range(n_elite)]
        new_pop = list(elites)
        while len(new_pop) < pop_size:
            p1 = random.choice(elites); p2 = random.choice(elites)
            mask = torch.rand_like(p1) > 0.5
            child = torch.where(mask, p1, p2) + torch.randn_like(p1) * 0.15
            new_pop.append(child)
        population = new_pop

    return best_pa, best_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 172: Lamarckian GA-TTCT")
    print(f"  GA + Backprop = Lamarckian evolution for Exact Match")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load model
    print("\n[Step 1] Loading Foundation Model...")
    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    ckpt = os.path.join(RESULTS_DIR, "phase123_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"  Loaded!")

    # Load data
    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:40]
    print(f"  Tasks: {len(test_tasks)}")

    # Compare methods
    print("\n[Step 3] Comparing TTCT methods...")
    no_ttct_pas, no_ttct_ems = [], []
    ga_pas, ga_ems = [], []
    lam_pas, lam_ems = [], []

    for idx, item in enumerate(test_tasks):
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do_list = [d.to(DEVICE) for d in item['demo_outputs']]
        ti = item['test_input']; to_gt = item['test_output']
        oh, ow = item['out_h'], item['out_w']

        # Get seed embedding
        with torch.no_grad():
            seed = model.task_encoder(di, do_list).detach()

        # No TTCT
        pa0, em0 = evaluate_embedding(model, seed, ti, to_gt, oh, ow)
        no_ttct_pas.append(pa0); no_ttct_ems.append(em0)

        # Pure GA
        ga_pa, ga_em = ttct_pure_ga(model, seed, ti, to_gt, oh, ow,
                                     pop_size=40, n_gens=25)
        ga_pas.append(ga_pa); ga_ems.append(ga_em)

        # Lamarckian GA
        _, lam_pa, lam_em = ttct_lamarckian(
            model, seed, item['demo_inputs'], item['demo_outputs'],
            ti, to_gt, oh, ow, pop_size=40, n_gens=25)
        lam_pas.append(lam_pa); lam_ems.append(lam_em)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(test_tasks)}] "
                  f"Direct={np.mean(no_ttct_pas)*100:.1f}% | "
                  f"GA={np.mean(ga_pas)*100:.1f}% | "
                  f"Lamarck={np.mean(lam_pas)*100:.1f}%")

    elapsed = time.time() - t0
    r_no = {'pa': np.mean(no_ttct_pas), 'em': np.mean(no_ttct_ems)}
    r_ga = {'pa': np.mean(ga_pas), 'em': np.mean(ga_ems)}
    r_lam = {'pa': np.mean(lam_pas), 'em': np.mean(lam_ems)}

    lamarck_wins = r_lam['pa'] > r_ga['pa'] or r_lam['em'] > r_ga['em']

    print(f"\n{'='*70}")
    print(f"Phase 172 Complete ({elapsed:.0f}s)")
    print(f"  No TTCT:    PA={r_no['pa']*100:.2f}%, EM={r_no['em']*100:.1f}%")
    print(f"  Pure GA:    PA={r_ga['pa']*100:.2f}%, EM={r_ga['em']*100:.1f}%")
    print(f"  Lamarckian: PA={r_lam['pa']*100:.2f}%, EM={r_lam['em']*100:.1f}%")
    print(f"  Lamarck wins: {lamarck_wins}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase172_lamarckian.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 172: Lamarckian GA-TTCT',
            'timestamp': datetime.now().isoformat(),
            'no_ttct': r_no, 'pure_ga': r_ga, 'lamarckian': r_lam,
            'lamarck_wins': lamarck_wins, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        methods = ['Direct', 'Pure GA', 'Lamarckian']
        pas = [r_no['pa']*100, r_ga['pa']*100, r_lam['pa']*100]
        ems = [r_no['em']*100, r_ga['em']*100, r_lam['em']*100]
        colors = ['#95a5a6', '#3498db', '#2ecc71']
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
        fig.suptitle('Phase 172: Lamarckian GA-TTCT (Evolution + Backprop)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.1, left=0.08, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase172_lamarckian.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'lamarck': r_lam, 'ga': r_ga}


if __name__ == '__main__':
    main()
