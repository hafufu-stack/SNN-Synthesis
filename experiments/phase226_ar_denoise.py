"""
Phase 226: Autoregressive Denoising NCA

Instead of updating all pixels simultaneously (which causes "stains"
from conflicting cell opinions), freeze high-confidence pixels first
and let uncertainty propagate from known to unknown regions.

"Build from certainty outward, like a crystal growing."

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
from phase199_gated import GatedHybridNCA, train_and_eval


def autoregressive_inference(model, x, task_emb, oh, ow, n_rounds=5, threshold=0.9):
    """Autoregressive denoising: freeze high-confidence pixels progressively.

    1. Run full NCA forward
    2. Find pixels where max_prob > threshold
    3. Freeze those pixels (replace their logits with hard one-hot)
    4. Run NCA again with frozen context
    5. Repeat for n_rounds
    """
    B, C_in, H, W = x.shape
    frozen_mask = torch.zeros(B, 1, H, W, device=x.device, dtype=torch.bool)
    frozen_logits = torch.zeros(B, 11, H, W, device=x.device)

    for round_idx in range(n_rounds):
        out = model(x, task_emb)
        logits = out[0] if isinstance(out, tuple) else out
        logits_crop = logits[:, :, :oh, :ow]

        probs = F.softmax(logits_crop, dim=1)  # (B, 11, oh, ow)
        max_probs, _ = probs.max(dim=1, keepdim=True)  # (B, 1, oh, ow)

        # Find newly confident pixels
        new_frozen = (max_probs > threshold) & (~frozen_mask[:, :, :oh, :ow])
        frozen_mask[:, :, :oh, :ow] = frozen_mask[:, :, :oh, :ow] | new_frozen

        # Store frozen logits (hard one-hot * large value)
        pred = logits_crop.argmax(dim=1)  # (B, oh, ow)
        hard_logits = F.one_hot(pred, 11).permute(0, 3, 1, 2).float() * 10.0
        frozen_logits[:, :, :oh, :ow] = torch.where(
            frozen_mask[:, :, :oh, :ow].expand_as(frozen_logits[:, :, :oh, :ow]),
            hard_logits,
            frozen_logits[:, :, :oh, :ow]
        )

        # Inject frozen pixels into input for next round
        # Replace input channels with frozen one-hot encoding
        frozen_input = F.one_hot(pred, 11).permute(0, 3, 1, 2).float()
        x_new = x.clone()
        expand_mask = frozen_mask[:, :, :oh, :ow].expand(B, min(C_in, 11), oh, ow)
        x_new[:, :11, :oh, :ow] = torch.where(
            expand_mask,
            frozen_input,
            x[:, :11, :oh, :ow]
        )
        x = x_new

        n_frozen = frozen_mask[:, :, :oh, :ow].float().mean().item()
        if n_frozen > 0.99:
            break  # Almost all frozen

    # Final output: combine frozen logits with last NCA output
    final_logits = logits.clone()
    expand_frozen = frozen_mask[:, :, :oh, :ow].expand_as(final_logits[:, :, :oh, :ow])
    final_logits[:, :, :oh, :ow] = torch.where(
        expand_frozen,
        frozen_logits[:, :, :oh, :ow],
        logits[:, :, :oh, :ow]
    )
    return final_logits


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 226: Autoregressive Denoising NCA")
    print(f"  Progressive pixel freezing for stain elimination")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train model
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

    # Evaluate: greedy vs autoregressive
    model.eval()
    greedy_pa, greedy_em = 0, 0
    ar_results = {}
    configs = [(3, 0.9), (5, 0.9), (5, 0.8), (10, 0.7)]

    print(f"\n[Evaluating Autoregressive Denoising]")
    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            # Greedy
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits[0, :, :oh, :ow].argmax(dim=0)
            greedy_pa += (pred == gt[:oh, :ow]).float().mean().item()
            greedy_em += float((pred == gt[:oh, :ow]).all().item())

            # Autoregressive
            for (n_rounds, threshold) in configs:
                key = f"AR(r{n_rounds},t{threshold})"
                if key not in ar_results:
                    ar_results[key] = {'pa': 0, 'em': 0}

                ar_logits = autoregressive_inference(
                    model, ti, emb, oh, ow,
                    n_rounds=n_rounds, threshold=threshold
                )
                ar_pred = ar_logits[0, :, :oh, :ow].argmax(dim=0)
                ar_results[key]['pa'] += (ar_pred == gt[:oh, :ow]).float().mean().item()
                ar_results[key]['em'] += float((ar_pred == gt[:oh, :ow]).all().item())

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    greedy_pa /= n_test; greedy_em /= n_test
    for k in ar_results:
        ar_results[k]['pa'] /= n_test; ar_results[k]['em'] /= n_test

    print(f"\n{'='*70}")
    print(f"  AUTOREGRESSIVE DENOISING NCA:")
    print(f"  Greedy:      PA={greedy_pa*100:.1f}%, EM={greedy_em*100:.1f}%")
    for k, r in ar_results.items():
        d = (r['pa'] - greedy_pa) * 100
        print(f"  {k:20s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}% (Δ={d:+.1f}pp)")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase226_ar_denoise.json"), 'w', encoding='utf-8') as f:
        json.dump({'greedy': {'pa': greedy_pa, 'em': greedy_em},
                   'autoregressive': ar_results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        labels = ['Greedy'] + list(ar_results.keys())
        pa_vals = [greedy_pa*100] + [r['pa']*100 for r in ar_results.values()]
        em_vals = [greedy_em*100] + [r['em']*100 for r in ar_results.values()]
        colors = ['#95a5a6', '#e74c3c', '#e67e22', '#3498db', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors[:len(labels)], alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors[:len(labels)], alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=15)
        ax.set_ylabel('%'); ax.set_title('Phase 226: Autoregressive Denoising NCA', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase226_ar_denoise.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'greedy_pa': greedy_pa, 'ar': ar_results}


if __name__ == '__main__':
    main()
