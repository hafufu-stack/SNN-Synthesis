"""
Phase 186: Unsupervised Champion Selection - Best-of-N via Margin

P182: Best single island champion (+1.84pp) >> majority vote (+0.49pp).
Problem: At test time we don't know which candidate is best (no ground truth).

Solution: Use average Margin (top1-top2) across all pixels as an
unsupervised quality metric. The candidate with highest mean margin
is the "most confident crystal" = likely the best answer.

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


def generate_candidates(model, seed_embed, test_input, n_candidates=10,
                         noise_std=0.3, n_steps=8):
    """Generate N candidate outputs by perturbing task embedding."""
    candidates = []
    nca = model.latent_nca
    ti = test_input.unsqueeze(0).to(DEVICE)

    for i in range(n_candidates):
        if i == 0:
            emb = seed_embed  # First candidate = unperturbed
        else:
            emb = seed_embed + torch.randn_like(seed_embed) * noise_std

        with torch.no_grad():
            logits = nca(ti, emb.to(DEVICE), n_steps=n_steps)
        candidates.append(logits[0])  # (C, H, W)

    return candidates


def select_by_margin(candidates, out_h, out_w):
    """Select candidate with highest mean margin (unsupervised)."""
    scores = []
    for logits in candidates:
        raw = logits[:10, :out_h, :out_w]
        sorted_l, _ = raw.sort(dim=0, descending=True)
        margin = (sorted_l[0] - sorted_l[1]).mean().item()
        scores.append(margin)
    best_idx = np.argmax(scores)
    return best_idx, scores


def select_by_entropy(candidates, out_h, out_w):
    """Select candidate with lowest mean entropy (unsupervised)."""
    scores = []
    for logits in candidates:
        probs = F.softmax(logits[:10, :out_h, :out_w], dim=0)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=0).mean().item()
        scores.append(-entropy)  # negate so higher = better
    best_idx = np.argmax(scores)
    return best_idx, scores


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 186: Unsupervised Champion Selection")
    print(f"  Best-of-N via Margin/Entropy (no ground truth needed)")
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
    test_tasks = all_tasks[:50]

    # Compare methods
    print("\n[Step 3] Comparing selection strategies...")
    n_candidates_list = [1, 3, 5, 9, 15]
    results = {}

    for n_cand in n_candidates_list:
        r = {'single_pas': [], 'single_ems': [],
             'margin_pas': [], 'margin_ems': [],
             'entropy_pas': [], 'entropy_ems': [],
             'oracle_pas': [], 'oracle_ems': [],
             'margin_rank': [], 'entropy_rank': []}

        with torch.no_grad():
            for item in test_tasks:
                di = [d.to(DEVICE) for d in item['demo_inputs']]
                do = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.task_encoder(di, do)
                oh, ow = item['out_h'], item['out_w']
                gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)[:oh, :ow]

                candidates = generate_candidates(model, emb, item['test_input'],
                                                  n_candidates=n_cand, noise_std=0.3)

                # Evaluate all candidates
                all_pas = []
                for logits in candidates:
                    pred = logits[:10].argmax(dim=0)[:oh, :ow]
                    pa = (pred == gt).float().mean().item()
                    all_pas.append(pa)

                # Single (first = unperturbed)
                r['single_pas'].append(all_pas[0])
                r['single_ems'].append(float(all_pas[0] == 1.0))

                if n_cand == 1:
                    for key in ['margin', 'entropy', 'oracle']:
                        r[f'{key}_pas'].append(all_pas[0])
                        r[f'{key}_ems'].append(float(all_pas[0] == 1.0))
                    continue

                # Margin selection
                m_idx, _ = select_by_margin(candidates, oh, ow)
                r['margin_pas'].append(all_pas[m_idx])
                r['margin_ems'].append(float(all_pas[m_idx] == 1.0))
                # Rank: where does margin's pick fall in PA ranking?
                sorted_pas = sorted(all_pas, reverse=True)
                r['margin_rank'].append(sorted_pas.index(all_pas[m_idx]))

                # Entropy selection
                e_idx, _ = select_by_entropy(candidates, oh, ow)
                r['entropy_pas'].append(all_pas[e_idx])
                r['entropy_ems'].append(float(all_pas[e_idx] == 1.0))
                r['entropy_rank'].append(sorted_pas.index(all_pas[e_idx]))

                # Oracle (best possible)
                best_idx = np.argmax(all_pas)
                r['oracle_pas'].append(all_pas[best_idx])
                r['oracle_ems'].append(float(all_pas[best_idx] == 1.0))

        results[n_cand] = {k: np.mean(v) if v else 0 for k, v in r.items()}

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 186 Complete ({elapsed:.0f}s)")
    for n_cand in n_candidates_list:
        r = results[n_cand]
        print(f"  N={n_cand:2d}: Single={r['single_pas']*100:.1f}% | "
              f"Margin={r['margin_pas']*100:.1f}% | "
              f"Entropy={r['entropy_pas']*100:.1f}% | "
              f"Oracle={r['oracle_pas']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase186_champion.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 186: Unsupervised Champion Selection',
            'timestamp': datetime.now().isoformat(),
            'results': {str(k): v for k, v in results.items()},
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ns = n_candidates_list
        single = [results[n]['single_pas']*100 for n in ns]
        margin = [results[n]['margin_pas']*100 for n in ns]
        entropy = [results[n]['entropy_pas']*100 for n in ns]
        oracle = [results[n]['oracle_pas']*100 for n in ns]
        axes[0].plot(ns, single, 'o--', color='#95a5a6', label='Single', linewidth=2)
        axes[0].plot(ns, margin, 's-', color='#2ecc71', label='Margin Select', linewidth=2)
        axes[0].plot(ns, entropy, '^-', color='#e74c3c', label='Entropy Select', linewidth=2)
        axes[0].plot(ns, oracle, 'D-', color='#3498db', label='Oracle (Best)', linewidth=2)
        axes[0].set_xlabel('N Candidates'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title('PA vs N Candidates', fontweight='bold'); axes[0].legend()
        # Margin rank
        ranks_m = [results[n].get('margin_rank', 0) for n in ns]
        ranks_e = [results[n].get('entropy_rank', 0) for n in ns]
        x = np.arange(len(ns)); w = 0.35
        axes[1].bar(x-w/2, ranks_m, w, color='#2ecc71', alpha=0.85, label='Margin')
        axes[1].bar(x+w/2, ranks_e, w, color='#e74c3c', alpha=0.85, label='Entropy')
        axes[1].set_xticks(x); axes[1].set_xticklabels([str(n) for n in ns])
        axes[1].set_xlabel('N'); axes[1].set_ylabel('Avg Rank (0=best)')
        axes[1].set_title('Selection Quality', fontweight='bold'); axes[1].legend()
        fig.suptitle('Phase 186: Unsupervised Champion Selection', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase186_champion.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    main()
