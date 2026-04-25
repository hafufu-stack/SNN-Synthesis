"""
Phase 214: Image-Level Self-Consistency - Global Majority Vote

Instead of pixel-level voting (P181 chimera failure), cluster
candidates by full-image exact match and pick the most popular.

"Errors scatter randomly, but truth converges."

Uses M=5 diverse models x N=20 trials = 100 total candidates.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import load_arc_training, prepare_arc_meta_dataset
from phase206_causal_volume import ParametricNCA


def train_model_seed(train_tasks, seed, n_epochs=80):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = ParametricNCA(11, 64, 1, 1, 32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        model.train(); random.shuffle(train_tasks)
        for item in train_tasks[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            logits = model(ti, emb)
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def grid_to_hashable(pred_tensor):
    """Convert 2D prediction tensor to hashable tuple."""
    return tuple(pred_tensor.cpu().numpy().flatten().tolist())


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 214: Image-Level Self-Consistency")
    print(f"  Cluster by full-image match, pick most popular")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    M = 5   # diverse models
    N = 20  # trials per model
    noise_scale = 0.3

    # Train M models
    print(f"\n[Training {M} diverse models]")
    models = []
    for i in range(M):
        seed = SEED + i * 100
        print(f"  Model {i+1}/{M} (seed={seed})")
        model = train_model_seed(train, seed, n_epochs=80)
        models.append(model)
    print(f"  Params each: {models[0].count_params():,}")

    # Evaluate
    print(f"\n[Generating {M}x{N}={M*N} candidates per task]")
    for m in models: m.eval()

    greedy_pa, greedy_em = 0, 0
    oracle_pa, oracle_em = 0, 0
    selfcon_pa, selfcon_em = 0, 0
    margin_pa, margin_em = 0, 0
    pixel_vote_pa, pixel_vote_em = 0, 0

    cluster_stats = []  # How many unique images per task, max cluster size

    with torch.no_grad():
        for tidx, item in enumerate(test):
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            ti = item['test_input'].unsqueeze(0).to(DEVICE)

            all_preds = []   # list of (pred_tensor, pa, em, margin)
            all_hashes = []  # hashable grids for clustering

            for model in models:
                do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.encode_task(do_t)

                for trial in range(N):
                    logits = model(ti, emb)
                    if trial > 0:
                        gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                        logits = logits + noise_scale * gumbel

                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    pa = (pred == gt[:oh, :ow]).float().mean().item()
                    em = (pred == gt[:oh, :ow]).all().item()

                    probs = F.softmax(logits[0, :, :oh, :ow], dim=0)
                    top2 = probs.topk(2, dim=0).values
                    margin = (top2[0] - top2[1]).mean().item()

                    h = grid_to_hashable(pred)
                    all_preds.append((pred, pa, em, margin))
                    all_hashes.append(h)

            # Greedy (first model, first trial)
            greedy_pa += all_preds[0][1]; greedy_em += all_preds[0][2]

            # Oracle
            best = max(all_preds, key=lambda c: c[1])
            oracle_pa += best[1]; oracle_em += best[2]

            # Margin
            mar_best = max(all_preds, key=lambda c: c[3])
            margin_pa += mar_best[1]; margin_em += mar_best[2]

            # Self-Consistency: most popular full-image cluster
            hash_counts = Counter(all_hashes)
            most_common_hash = hash_counts.most_common(1)[0][0]
            most_common_count = hash_counts.most_common(1)[0][1]
            n_unique = len(hash_counts)

            # Find the first pred with that hash
            for pred_data, h in zip(all_preds, all_hashes):
                if h == most_common_hash:
                    selfcon_pa += pred_data[1]
                    selfcon_em += pred_data[2]
                    break

            cluster_stats.append({
                'n_unique': n_unique,
                'max_cluster': most_common_count,
                'total': M * N
            })

            # Pixel-level majority vote (for comparison - P181 approach)
            all_pred_tensors = torch.stack([p[0] for p in all_preds])  # (100, oh, ow)
            # Mode along dim 0
            vote_pred = torch.mode(all_pred_tensors, dim=0).values
            pv_pa = (vote_pred == gt[:oh, :ow]).float().mean().item()
            pv_em = (vote_pred == gt[:oh, :ow]).all().item()
            pixel_vote_pa += pv_pa; pixel_vote_em += pv_em

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)}: unique={n_unique}, "
                      f"max_cluster={most_common_count}/{M*N}")

    n_test = len(test)
    results = {
        'greedy': {'pa': greedy_pa/n_test, 'em': greedy_em/n_test},
        'oracle': {'pa': oracle_pa/n_test, 'em': oracle_em/n_test},
        'self_consistency': {'pa': selfcon_pa/n_test, 'em': selfcon_em/n_test},
        'margin': {'pa': margin_pa/n_test, 'em': margin_em/n_test},
        'pixel_vote': {'pa': pixel_vote_pa/n_test, 'em': pixel_vote_em/n_test},
    }

    avg_unique = np.mean([s['n_unique'] for s in cluster_stats])
    avg_max_cluster = np.mean([s['max_cluster'] for s in cluster_stats])

    print(f"\n{'='*70}")
    print(f"  IMAGE-LEVEL SELF-CONSISTENCY ({M}x{N}={M*N}):")
    print(f"  Greedy:        PA={results['greedy']['pa']*100:.1f}%, EM={results['greedy']['em']*100:.1f}%")
    print(f"  Oracle:        PA={results['oracle']['pa']*100:.1f}%, EM={results['oracle']['em']*100:.1f}%")
    print(f"  Self-Consist:  PA={results['self_consistency']['pa']*100:.1f}%, EM={results['self_consistency']['em']*100:.1f}%")
    print(f"  Pixel Vote:    PA={results['pixel_vote']['pa']*100:.1f}%, EM={results['pixel_vote']['em']*100:.1f}%")
    print(f"  Margin:        PA={results['margin']['pa']*100:.1f}%, EM={results['margin']['em']*100:.1f}%")
    print(f"  Avg unique images: {avg_unique:.1f}/{M*N}")
    print(f"  Avg max cluster:   {avg_max_cluster:.1f}/{M*N}")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    for m in models: del m
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase214_self_consistency.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'M': M, 'N': N, 'results': results,
            'avg_unique_images': avg_unique, 'avg_max_cluster': avg_max_cluster,
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        labels = ['Greedy', 'Oracle', 'Self-\nConsist.', 'Pixel\nVote', 'Margin']
        pa_vals = [results[k]['pa']*100 for k in ['greedy','oracle','self_consistency',
                                                   'pixel_vote','margin']]
        em_vals = [results[k]['em']*100 for k in ['greedy','oracle','self_consistency',
                                                   'pixel_vote','margin']]
        colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#f39c12', '#3498db']
        x = np.arange(5); w = 0.35
        axes[0].bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel('%'); axes[0].set_title('Selection Comparison', fontweight='bold')
        axes[0].legend()

        # Cluster stats
        uniques = [s['n_unique'] for s in cluster_stats]
        max_clusters = [s['max_cluster'] for s in cluster_stats]
        axes[1].hist(uniques, bins=20, color='#3498db', alpha=0.7, label='Unique')
        axes[1].set_xlabel('# Unique Images'); axes[1].set_ylabel('Count')
        axes[1].set_title(f'Avg={avg_unique:.0f} unique/{M*N}', fontweight='bold')
        axes[1].legend()

        axes[2].hist(max_clusters, bins=20, color='#e74c3c', alpha=0.7, label='Max cluster')
        axes[2].set_xlabel('Max Cluster Size'); axes[2].set_ylabel('Count')
        axes[2].set_title(f'Avg max={avg_max_cluster:.0f}/{M*N}', fontweight='bold')
        axes[2].legend()

        fig.suptitle('Phase 214: Image-Level Self-Consistency', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.06, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase214_self_consistency.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
