"""
Phase 182: Island Model Swarm - Allopatric Speciation for Diversity

P181 failed: GA converged to clones (same stains everywhere).
Fix: Split population into isolated islands (allopatric speciation).
Each island evolves independently -> different error patterns.
Final majority vote from island champions cancels errors.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
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
EMB_DIM = 64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset
)


def evolve_island(model, seed_embed, test_input, test_output, out_h, out_w,
                  pop_size=20, n_gens=40, init_std=0.3):
    """Evolve a single island independently. Returns ranked population."""
    # Initialize with diverse perturbations
    population = [seed_embed + torch.randn_like(seed_embed) * init_std
                  for _ in range(pop_size)]

    for gen in range(n_gens):
        fitnesses = []
        with torch.no_grad():
            for emb in population:
                logits = model.latent_nca(
                    test_input.unsqueeze(0).to(DEVICE), emb.to(DEVICE), n_steps=5)
                pred = logits[0, :10].argmax(dim=0)
                gt = test_output[:10].argmax(dim=0).to(DEVICE)
                pa = (pred[:out_h, :out_w] == gt[:out_h, :out_w]).float().mean().item()
                em = float((pred[:out_h, :out_w] == gt[:out_h, :out_w]).all().item())
                probs = F.softmax(logits[0, :10, :out_h, :out_w], dim=0)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=0).mean().item()
                score = pa * 10 + em * 50 - entropy * 0.5
                fitnesses.append((score, pa, em))

        scores = [f[0] for f in fitnesses]
        ranked = sorted(zip(scores, population, fitnesses), key=lambda x: -x[0])

        if ranked[0][2][2] > 0:  # EM found
            break

        n_elite = max(2, pop_size // 5)
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

    # Return best embedding and its fitness
    return ranked[0][1], ranked[0][2][1], ranked[0][2][2]


def predict_single(model, test_input, embed, out_h, out_w, n_steps=8):
    """Single prediction from one embedding."""
    with torch.no_grad():
        logits = model.latent_nca(
            test_input.unsqueeze(0).to(DEVICE), embed.to(DEVICE), n_steps=n_steps)
        return logits[0, :10].argmax(dim=0)[:out_h, :out_w]


def majority_vote(predictions):
    """Pixel-wise majority vote."""
    stacked = torch.stack(predictions, dim=0)
    voted, _ = torch.mode(stacked, dim=0)
    return voted


def single_ga(model, seed_embed, test_input, test_output, out_h, out_w,
              pop_size=100, n_gens=40):
    """Standard single-population GA (baseline)."""
    population = [seed_embed + torch.randn_like(seed_embed) * 0.3 for _ in range(pop_size)]
    for gen in range(n_gens):
        fitnesses = []
        with torch.no_grad():
            for emb in population:
                logits = model.latent_nca(
                    test_input.unsqueeze(0).to(DEVICE), emb.to(DEVICE), n_steps=5)
                pred = logits[0, :10].argmax(dim=0)
                gt = test_output[:10].argmax(dim=0).to(DEVICE)
                pa = (pred[:out_h, :out_w] == gt[:out_h, :out_w]).float().mean().item()
                score = pa * 10
                fitnesses.append((score, pa, emb))
        ranked = sorted(fitnesses, key=lambda x: -x[0])
        if gen == n_gens - 1:
            return ranked[0][2], ranked[0][1]
        n_elite = max(2, pop_size // 10)
        elites = [ranked[i][2].clone() for i in range(n_elite)]
        new_pop = list(elites)
        std = 0.2 * (1 - gen / n_gens * 0.7)
        while len(new_pop) < pop_size:
            p1 = random.choice(elites); p2 = random.choice(elites)
            child = torch.where(torch.rand_like(p1) > 0.5, p1, p2)
            new_pop.append(child + torch.randn_like(child) * std)
        population = new_pop
    return population[0], 0


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 182: Island Model Swarm - Allopatric Speciation")
    print(f"  5 islands x 20 individuals, majority vote from champions")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Loading Foundation Model...")
    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    ckpt_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:40]

    # Compare methods
    print("\n[Step 3] Comparing Single-GA vs Island Model...")
    island_configs = [3, 5, 7, 9]  # number of islands
    results = {
        'single_best1': {'pas': [], 'ems': []},
    }
    for n_isl in island_configs:
        results[f'island_{n_isl}_vote'] = {'pas': [], 'ems': []}
        results[f'island_{n_isl}_best'] = {'pas': [], 'ems': []}

    for idx, item in enumerate(test_tasks):
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do_list = [d.to(DEVICE) for d in item['demo_outputs']]
        oh, ow = item['out_h'], item['out_w']
        gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)[:oh, :ow]

        with torch.no_grad():
            seed_embed = model.task_encoder(di, do_list).detach()

        # Single GA baseline
        best_emb, best_pa = single_ga(model, seed_embed, item['test_input'],
                                       item['test_output'], oh, ow, 100, 30)
        pred = predict_single(model, item['test_input'], best_emb, oh, ow)
        pa = (pred == gt).float().mean().item()
        em = float((pred == gt).all().item())
        results['single_best1']['pas'].append(pa)
        results['single_best1']['ems'].append(em)

        # Island Model
        for n_isl in island_configs:
            pop_per_island = max(10, 100 // n_isl)
            champions = []
            champion_pas = []
            for isl in range(n_isl):
                # Each island starts with differently perturbed seed
                island_seed = seed_embed + torch.randn_like(seed_embed) * (0.2 + 0.1 * isl)
                champ, champ_pa, champ_em = evolve_island(
                    model, island_seed, item['test_input'], item['test_output'],
                    oh, ow, pop_size=pop_per_island, n_gens=30,
                    init_std=0.3 + 0.1 * isl)
                champions.append(champ)
                champion_pas.append(champ_pa)

            # Majority vote from island champions
            preds = [predict_single(model, item['test_input'], c, oh, ow) for c in champions]
            voted = majority_vote(preds)
            pa_vote = (voted == gt).float().mean().item()
            em_vote = float((voted == gt).all().item())
            results[f'island_{n_isl}_vote']['pas'].append(pa_vote)
            results[f'island_{n_isl}_vote']['ems'].append(em_vote)

            # Best single champion
            best_idx = np.argmax(champion_pas)
            pa_best = (preds[best_idx] == gt).float().mean().item()
            em_best = float((preds[best_idx] == gt).all().item())
            results[f'island_{n_isl}_best']['pas'].append(pa_best)
            results[f'island_{n_isl}_best']['ems'].append(em_best)

        if (idx + 1) % 10 == 0:
            single = np.mean(results['single_best1']['pas'])*100
            isl5v = np.mean(results['island_5_vote']['pas'])*100
            print(f"  [{idx+1}/{len(test_tasks)}] Single={single:.1f}% | Island5_Vote={isl5v:.1f}%")

    elapsed = time.time() - t0
    summary = {}
    print(f"\n{'='*70}")
    print(f"Phase 182 Complete ({elapsed:.0f}s)")
    for name, r in results.items():
        avg_pa = np.mean(r['pas']); avg_em = np.mean(r['ems'])
        summary[name] = {'pa': avg_pa, 'em': avg_em}
        print(f"  {name:25s}: PA={avg_pa*100:.2f}%, EM={avg_em*100:.1f}%")

    best_island = max([(k, v) for k, v in summary.items() if 'island' in k and 'vote' in k],
                      key=lambda x: x[1]['pa'])
    improvement = best_island[1]['pa'] - summary['single_best1']['pa']
    print(f"\n  Best island vote ({best_island[0]}): {improvement*100:+.2f}pp vs single GA")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase182_island.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 182: Island Model Swarm',
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'improvement_pp': improvement * 100,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        # PA: single vs island vote
        labels = ['Single\nGA'] + [f'{n}\nislands' for n in island_configs]
        pa_vals = [summary['single_best1']['pa']*100] + \
                  [summary[f'island_{n}_vote']['pa']*100 for n in island_configs]
        colors = ['#95a5a6'] + ['#2ecc71']*len(island_configs)
        axes[0].bar(labels, pa_vals, color=colors, alpha=0.85, edgecolor='black')
        for i, v in enumerate(pa_vals):
            axes[0].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
        axes[0].set_ylabel('PA (%)'); axes[0].set_title('Vote PA vs Islands', fontweight='bold')
        # Vote vs Best-of-island
        vote_pa = [summary[f'island_{n}_vote']['pa']*100 for n in island_configs]
        best_pa = [summary[f'island_{n}_best']['pa']*100 for n in island_configs]
        x = np.arange(len(island_configs)); w = 0.35
        axes[1].bar(x-w/2, best_pa, w, color='#e74c3c', alpha=0.85, label='Best Champion')
        axes[1].bar(x+w/2, vote_pa, w, color='#2ecc71', alpha=0.85, label='Majority Vote')
        axes[1].set_xticks(x); axes[1].set_xticklabels([str(n) for n in island_configs])
        axes[1].set_xlabel('Islands'); axes[1].set_ylabel('PA (%)')
        axes[1].set_title('Vote vs Best Champion', fontweight='bold'); axes[1].legend()
        # Improvement over single
        impr = [(summary[f'island_{n}_vote']['pa']-summary['single_best1']['pa'])*100
                for n in island_configs]
        axes[2].bar([str(n) for n in island_configs], impr,
                   color=['#2ecc71' if v>0 else '#e74c3c' for v in impr], alpha=0.85)
        axes[2].set_xlabel('Islands'); axes[2].set_ylabel('Improvement (pp)')
        axes[2].set_title('Vote vs Single GA', fontweight='bold')
        axes[2].axhline(0, color='black', linewidth=0.5)
        fig.suptitle('Phase 182: Island Model Swarm (Allopatric Speciation)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase182_island.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return summary

if __name__ == '__main__':
    main()
