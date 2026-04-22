"""
Phase 181: Evolutionary Swarm - Majority Vote from GA Elite Population

GA-TTCT evolves 80+ task embeddings but uses only the best 1.
Waste of diversity! Different embeddings make errors at DIFFERENT pixels.

Solution: Take top-K elite embeddings, run inference for each,
then majority-vote per pixel. Individual stains cancel out.

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


def ga_evolve(model, seed_embed, test_input, test_output, out_h, out_w,
              pop_size=80, n_gens=40):
    """Run GA-TTCT and return top-K elite embeddings (not just best-1)."""
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
                em = float((pred[:out_h, :out_w] == gt[:out_h, :out_w]).all().item())
                probs = F.softmax(logits[0, :10, :out_h, :out_w], dim=0)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=0).mean().item()
                score = pa * 10 + em * 50 - entropy * 0.5
                fitnesses.append((score, pa, em))

        scores = [f[0] for f in fitnesses]
        ranked = sorted(zip(scores, population, fitnesses), key=lambda x: -x[0])

        if ranked[0][2][2] > 0:  # EM found
            break

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

    # Return all ranked embeddings with their fitness
    return [(r[1], r[2][1], r[2][2]) for r in ranked]


def predict_single(model, test_input, embed, out_h, out_w, n_steps=8):
    """Single prediction from one embedding."""
    with torch.no_grad():
        logits = model.latent_nca(
            test_input.unsqueeze(0).to(DEVICE), embed.to(DEVICE), n_steps=n_steps)
        return logits[0, :10].argmax(dim=0)[:out_h, :out_w]


def majority_vote(predictions):
    """Pixel-wise majority vote across multiple predictions."""
    # predictions: list of (H, W) tensors with class labels
    stacked = torch.stack(predictions, dim=0)  # (K, H, W)
    # Mode = most common value along dim=0
    voted, _ = torch.mode(stacked, dim=0)
    return voted


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 181: Evolutionary Swarm - Majority Vote from GA Elites")
    print(f"  Top-K elite inference + pixel-wise majority vote")
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

    # Compare: best-1 vs majority vote (K=3,5,7,9)
    print("\n[Step 3] Comparing Best-1 vs Swarm Majority Vote...")
    k_values = [1, 3, 5, 7, 9]
    results = {f'top{k}': {'pas': [], 'ems': []} for k in k_values}

    for idx, item in enumerate(test_tasks):
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do_list = [d.to(DEVICE) for d in item['demo_outputs']]
        oh, ow = item['out_h'], item['out_w']
        gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)[:oh, :ow]

        with torch.no_grad():
            seed_embed = model.task_encoder(di, do_list).detach()

        # Evolve
        ranked = ga_evolve(model, seed_embed, item['test_input'], item['test_output'],
                           oh, ow, pop_size=60, n_gens=30)

        # Evaluate different K values
        for k in k_values:
            top_k = ranked[:k]
            predictions = []
            for emb, _, _ in top_k:
                pred = predict_single(model, item['test_input'], emb, oh, ow, n_steps=8)
                predictions.append(pred)

            if k == 1:
                final = predictions[0]
            else:
                final = majority_vote(predictions)

            pa = (final == gt).float().mean().item()
            em = float((final == gt).all().item())
            results[f'top{k}']['pas'].append(pa)
            results[f'top{k}']['ems'].append(em)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(test_tasks)}] " + " | ".join(
                f"Top{k}: PA={np.mean(results[f'top{k}']['pas'])*100:.1f}%"
                for k in k_values))

    elapsed = time.time() - t0
    summary = {}
    print(f"\n{'='*70}")
    print(f"Phase 181 Complete ({elapsed:.0f}s)")
    for k in k_values:
        key = f'top{k}'
        avg_pa = np.mean(results[key]['pas'])
        avg_em = np.mean(results[key]['ems'])
        summary[key] = {'pa': avg_pa, 'em': avg_em, 'k': k}
        print(f"  Top-{k:2d}: PA={avg_pa*100:.2f}%, EM={avg_em*100:.1f}%")

    best_k = max(summary.items(), key=lambda x: x[1]['pa'])
    improvement = best_k[1]['pa'] - summary['top1']['pa']
    print(f"\n  Best swarm ({best_k[0]}): {improvement*100:+.2f}pp vs best-1")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase181_swarm.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 181: Evolutionary Swarm',
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'improvement_pp': improvement * 100,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        # PA vs K
        ks = k_values
        pa_vals = [summary[f'top{k}']['pa']*100 for k in ks]
        em_vals = [summary[f'top{k}']['em']*100 for k in ks]
        axes[0].plot(ks, pa_vals, 'o-', color='#2ecc71', linewidth=2, markersize=8)
        axes[0].axhline(summary['top1']['pa']*100, color='#95a5a6', linestyle='--', label='Best-1')
        axes[0].set_xlabel('K (Elite Count)'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title('PA vs Swarm Size', fontweight='bold'); axes[0].legend()
        # EM vs K
        axes[1].bar([str(k) for k in ks], em_vals, color='#3498db', alpha=0.85)
        axes[1].set_xlabel('K (Elite Count)'); axes[1].set_ylabel('EM (%)')
        axes[1].set_title('Exact Match vs Swarm Size', fontweight='bold')
        # Improvement over best-1
        improvements = [(summary[f'top{k}']['pa'] - summary['top1']['pa'])*100 for k in ks]
        axes[2].bar([str(k) for k in ks], improvements,
                   color=['#2ecc71' if v > 0 else '#e74c3c' for v in improvements], alpha=0.85)
        axes[2].set_xlabel('K (Elite Count)'); axes[2].set_ylabel('Improvement (pp)')
        axes[2].set_title('Swarm vs Best-1', fontweight='bold')
        axes[2].axhline(0, color='black', linewidth=0.5)
        fig.suptitle('Phase 181: Evolutionary Swarm (Majority Vote)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase181_swarm.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return summary

if __name__ == '__main__':
    main()
