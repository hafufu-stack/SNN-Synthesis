"""
Phase 173: Thermodynamic Apoptosis

Use P171's nutrient mechanism to kill uncertain pixels:
  - High-confidence pixels get nutrients -> survive
  - Low-confidence (shimmering) pixels starve -> die (go to 0)
  = Programmed cell death for pixel-perfect output

No VQ needed. Life's own rules clean up the noise.

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
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset,
    grid_to_tensor, tensor_to_grid, N_COLORS
)


def apoptosis_inference(model, test_input, task_embed, out_h, out_w,
                         base_steps=5, apoptosis_steps=3,
                         confidence_threshold=0.6):
    """
    Run NCA inference with thermodynamic apoptosis at the end:
      1. Normal NCA for base_steps
      2. Then apoptosis_steps where uncertain pixels die

    Returns cleaned-up predictions.
    """
    model.eval()
    with torch.no_grad():
        x = test_input.unsqueeze(0).to(DEVICE)
        te = task_embed.to(DEVICE)

        # Step 1: Normal inference
        logits = model.latent_nca(x, te, n_steps=base_steps)
        probs = F.softmax(logits[0, :10], dim=0)  # (10, H, W)

        # Step 2: Apoptosis - kill uncertain pixels
        for step in range(apoptosis_steps):
            max_prob, pred = probs.max(dim=0)  # (H, W)

            # Nutrient = confidence
            nutrients = max_prob.clone()

            # Kill cells with low confidence
            alive = nutrients >= confidence_threshold
            dead = ~alive

            # Dead pixels -> color 0 (background)
            pred_cleaned = pred.clone()
            pred_cleaned[dead] = 0

            # Re-run through NCA with cleaned input
            # Convert pred_cleaned back to one-hot
            cleaned_oh = torch.zeros(1, N_COLORS, x.shape[2], x.shape[3], device=DEVICE)
            for c in range(10):
                cleaned_oh[0, c] = (pred_cleaned == c).float()
            cleaned_oh[0, 10] = x[0, 10]  # Preserve mask

            logits = model.latent_nca(cleaned_oh, te, n_steps=2)
            probs = F.softmax(logits[0, :10], dim=0)

            # Increase threshold each step (progressive apoptosis)
            confidence_threshold = min(confidence_threshold + 0.05, 0.9)

        # Final prediction
        final_pred = probs.argmax(dim=0)
        final_confidence = probs.max(dim=0)[0]

    return final_pred, final_confidence


def standard_inference(model, test_input, task_embed, out_h, out_w, n_steps=5):
    """Standard inference without apoptosis."""
    model.eval()
    with torch.no_grad():
        logits = model.latent_nca(
            test_input.unsqueeze(0).to(DEVICE), task_embed.to(DEVICE), n_steps=n_steps)
        pred = logits[0, :10].argmax(dim=0)
        confidence = F.softmax(logits[0, :10], dim=0).max(dim=0)[0]
    return pred, confidence


def evaluate(pred, test_output, out_h, out_w):
    """Compute PA and EM."""
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
    print("Phase 173: Thermodynamic Apoptosis")
    print(f"  Programmed cell death for pixel-perfect output")
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
    test_tasks = all_tasks[:50]
    print(f"  Tasks: {len(test_tasks)}")

    # Compare: standard vs apoptosis with different thresholds
    print("\n[Step 3] Comparing inference methods...")

    configs = [
        ('Standard', None),
        ('Apoptosis_0.5', 0.5),
        ('Apoptosis_0.6', 0.6),
        ('Apoptosis_0.7', 0.7),
        ('Apoptosis_0.8', 0.8),
    ]

    results = {}
    for name, threshold in configs:
        pas, ems = [], []
        conf_means = []

        for item in test_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do_list = [d.to(DEVICE) for d in item['demo_outputs']]
            oh, ow = item['out_h'], item['out_w']

            with torch.no_grad():
                task_embed = model.task_encoder(di, do_list)

            if threshold is None:
                pred, conf = standard_inference(
                    model, item['test_input'], task_embed, oh, ow)
            else:
                pred, conf = apoptosis_inference(
                    model, item['test_input'], task_embed, oh, ow,
                    confidence_threshold=threshold)

            pa, em = evaluate(pred, item['test_output'], oh, ow)
            pas.append(pa); ems.append(em)
            conf_means.append(conf[:oh, :ow].mean().item())

        results[name] = {
            'pa': np.mean(pas), 'em': np.mean(ems),
            'avg_confidence': np.mean(conf_means),
        }
        print(f"  {name:>15}: PA={np.mean(pas)*100:.2f}%, "
              f"EM={np.mean(ems)*100:.1f}%, "
              f"conf={np.mean(conf_means):.3f}")

    # Best apoptosis config
    best = max([k for k in results if k != 'Standard'],
               key=lambda k: results[k]['pa'])
    improvement = results[best]['pa'] - results['Standard']['pa']

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 173 Complete ({elapsed:.0f}s)")
    print(f"  Standard:      PA={results['Standard']['pa']*100:.2f}%, EM={results['Standard']['em']*100:.1f}%")
    print(f"  Best apoptosis: {best}")
    print(f"    PA={results[best]['pa']*100:.2f}%, EM={results[best]['em']*100:.1f}%")
    print(f"  PA improvement: {improvement*100:+.2f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase173_apoptosis.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 173: Thermodynamic Apoptosis',
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'best_config': best,
            'pa_improvement': improvement,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        names = list(results.keys())
        pas = [results[n]['pa']*100 for n in names]
        ems = [results[n]['em']*100 for n in names]
        confs = [results[n]['avg_confidence'] for n in names]

        colors_pa = ['#e74c3c'] + ['#3498db'] * (len(names)-1)
        axes[0].bar(range(len(names)), pas, color=colors_pa, alpha=0.85, edgecolor='black')
        axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=30, fontsize=7)
        axes[0].set_ylabel('PA (%)')
        axes[0].set_title('Pixel Accuracy', fontweight='bold', fontsize=10)

        axes[1].bar(range(len(names)), ems, color=colors_pa, alpha=0.85, edgecolor='black')
        axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=30, fontsize=7)
        axes[1].set_ylabel('EM (%)')
        axes[1].set_title('Exact Match', fontweight='bold', fontsize=10)

        axes[2].bar(range(len(names)), confs, color='#f39c12', alpha=0.85, edgecolor='black')
        axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names, rotation=30, fontsize=7)
        axes[2].set_ylabel('Confidence')
        axes[2].set_title('Avg Prediction Confidence', fontweight='bold', fontsize=10)

        fig.suptitle('Phase 173: Thermodynamic Apoptosis (Programmed Cell Death)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.18, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase173_apoptosis.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
