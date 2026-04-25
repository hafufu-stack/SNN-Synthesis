"""
Phase 219: Critic-Guided Hill Climbing - Discrete Zero-Order Optimization

Instead of gradient-based logit repair (P215 failed catastrophically),
use discrete color replacement on low-confidence pixels.

"Try each color on uncertain pixels, keep whichever looks best."

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
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
from phase199_gated import GatedHybridNCA
from phase205_critic import ArcAutoEncoder, train_autoencoder


def hill_climb(pred, logits, ae, n_colors=11, max_pixels=10, max_iters=5):
    """Discrete hill climbing on low-confidence pixels.

    1. Find pixels with lowest margin (most uncertain)
    2. For each, try all 11 colors, pick the one that minimizes AE error
    3. Repeat for max_iters rounds

    Returns: improved prediction (H, W) tensor
    """
    oh, ow = pred.shape
    current = pred.clone()

    for iteration in range(max_iters):
        # Find uncertain pixels via margin
        probs = F.softmax(logits[:, :oh, :ow], dim=0)  # (C, H, W)
        top2 = probs.topk(2, dim=0).values
        margin = (top2[0] - top2[1])  # (H, W)

        # Get pixels sorted by uncertainty (lowest margin first)
        flat_margin = margin.reshape(-1)
        _, indices = flat_margin.sort()
        n_try = min(max_pixels, len(indices))

        improved = False
        for pidx in range(n_try):
            flat_idx = indices[pidx].item()
            py = flat_idx // ow
            px = flat_idx % ow
            current_color = current[py, px].item()

            # Current AE error
            current_oh = F.one_hot(current.long(), n_colors).permute(2, 0, 1).float()
            current_err = ae.reconstruction_error(current_oh.unsqueeze(0)).item()

            best_color = current_color
            best_err = current_err

            # Try each alternative color
            for c in range(n_colors):
                if c == current_color:
                    continue
                trial = current.clone()
                trial[py, px] = c
                trial_oh = F.one_hot(trial.long(), n_colors).permute(2, 0, 1).float()
                trial_err = ae.reconstruction_error(trial_oh.unsqueeze(0)).item()
                if trial_err < best_err:
                    best_err = trial_err
                    best_color = c

            if best_color != current_color:
                current[py, px] = best_color
                improved = True

        if not improved:
            break  # No improvement found, stop early

    return current


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 219: Critic-Guided Hill Climbing")
    print(f"  Discrete zero-order optimization on uncertain pixels")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train GatedHybrid
    print(f"\n[Training GatedHybridNCA]")
    torch.manual_seed(SEED)
    model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        model.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  Params: {model.count_params():,}")

    # Train AE
    print(f"\n[Training AE Critic]")
    output_grids = []
    for item in train:
        output_grids.append(item['test_output'][:11])
        for do in item['demo_outputs']:
            output_grids.append(do)
    ae = ArcAutoEncoder(11, 32).to(DEVICE)
    train_autoencoder(ae, output_grids, n_epochs=200, lr=1e-3)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # Evaluate
    configs = [(5, 3), (10, 5), (15, 5)]  # (max_pixels, max_iters)
    model.eval()

    greedy_pa, greedy_em = 0, 0
    hc_results = {}

    print(f"\n[Evaluating Hill Climbing]")
    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            logits_crop = logits[0, :, :oh, :ow]
            pred = logits_crop.argmax(dim=0)

            greedy_pa += (pred == gt[:oh, :ow]).float().mean().item()
            greedy_em += float((pred == gt[:oh, :ow]).all().item())

            for (max_px, max_it) in configs:
                key = f"px{max_px}_it{max_it}"
                if key not in hc_results:
                    hc_results[key] = {'pa': 0, 'em': 0}

                repaired = hill_climb(pred, logits_crop, ae,
                                     max_pixels=max_px, max_iters=max_it)
                hc_results[key]['pa'] += (repaired == gt[:oh, :ow]).float().mean().item()
                hc_results[key]['em'] += float((repaired == gt[:oh, :ow]).all().item())

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    greedy_pa /= n_test; greedy_em /= n_test
    for k in hc_results:
        hc_results[k]['pa'] /= n_test
        hc_results[k]['em'] /= n_test

    print(f"\n{'='*70}")
    print(f"  CRITIC-GUIDED HILL CLIMBING:")
    print(f"  Greedy:       PA={greedy_pa*100:.1f}%, EM={greedy_em*100:.1f}%")
    for k, r in hc_results.items():
        print(f"  HC({k:12s}): PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model, ae; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase219_hill_climb.json"), 'w', encoding='utf-8') as f:
        json.dump({'greedy': {'pa': greedy_pa, 'em': greedy_em},
                   'hill_climb': hc_results,
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = ['Greedy'] + [k for k in hc_results]
        pa_vals = [greedy_pa*100] + [r['pa']*100 for r in hc_results.values()]
        em_vals = [greedy_em*100] + [r['em']*100 for r in hc_results.values()]
        colors = ['#95a5a6', '#e74c3c', '#e67e22', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors[:len(labels)], alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors[:len(labels)], alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 219: Critic-Guided Hill Climbing', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase219_hill_climb.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'greedy_pa': greedy_pa, 'hill_climb': hc_results}


if __name__ == '__main__':
    main()
