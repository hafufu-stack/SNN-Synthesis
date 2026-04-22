"""
Phase 183: Surface Tension TTCT - TV Loss for Stain Removal

Isolated 1-2px stains resist local repair (P178) and global hormone
(P180). Physical approach: add Total Variation (TV) Loss to TTCT.
TV Loss penalizes adjacent pixel differences = surface tension.
Isolated stain pixels become energetically unstable and collapse
into surrounding majority color regions.

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


def tv_loss(logits):
    """Total Variation loss on logits (surface tension)."""
    diff_h = (logits[:, :, 1:, :] - logits[:, :, :-1, :]).abs().mean()
    diff_w = (logits[:, :, :, 1:] - logits[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def ttct_standard(model, demo_inputs, demo_outputs, test_input, test_output,
                  out_h, out_w, n_iters=80):
    """Standard TTCT (backprop on task embedding, no TV)."""
    model.eval()
    di = [d.to(DEVICE) for d in demo_inputs]
    do = [d.to(DEVICE) for d in demo_outputs]
    with torch.no_grad():
        seed = model.task_encoder(di, do).detach()
    emb = seed.clone().requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=0.05)
    for it in range(n_iters):
        logits = model.latent_nca(test_input.unsqueeze(0).to(DEVICE), emb, n_steps=5)
        target = demo_outputs[0][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
        loss = F.cross_entropy(logits[:, :10], target)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = model.latent_nca(test_input.unsqueeze(0).to(DEVICE), emb, n_steps=8)
        pred = logits[0, :10].argmax(dim=0)
        gt = test_output[:10].argmax(dim=0).to(DEVICE)
        pa = (pred[:out_h, :out_w] == gt[:out_h, :out_w]).float().mean().item()
        em = float((pred[:out_h, :out_w] == gt[:out_h, :out_w]).all().item())
    return pa, em


def ttct_surface_tension(model, demo_inputs, demo_outputs, test_input, test_output,
                          out_h, out_w, n_iters=80, tv_weight=0.1):
    """TTCT with Total Variation loss (surface tension)."""
    model.eval()
    di = [d.to(DEVICE) for d in demo_inputs]
    do = [d.to(DEVICE) for d in demo_outputs]
    with torch.no_grad():
        seed = model.task_encoder(di, do).detach()
    emb = seed.clone().requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=0.05)
    for it in range(n_iters):
        logits = model.latent_nca(test_input.unsqueeze(0).to(DEVICE), emb, n_steps=5)
        target = demo_outputs[0][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
        ce_loss = F.cross_entropy(logits[:, :10], target)
        tv = tv_loss(logits[:, :10, :out_h, :out_w])
        loss = ce_loss + tv_weight * tv
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = model.latent_nca(test_input.unsqueeze(0).to(DEVICE), emb, n_steps=8)
        pred = logits[0, :10].argmax(dim=0)
        gt = test_output[:10].argmax(dim=0).to(DEVICE)
        pa = (pred[:out_h, :out_w] == gt[:out_h, :out_w]).float().mean().item()
        em = float((pred[:out_h, :out_w] == gt[:out_h, :out_w]).all().item())
    return pa, em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 183: Surface Tension TTCT - TV Loss for Stain Removal")
    print(f"  Physical surface tension to crush isolated stains")
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

    # Compare standard TTCT vs surface tension TTCT
    print("\n[Step 3] Comparing Standard TTCT vs Surface Tension TTCT...")
    tv_weights = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = {'standard': {'pas': [], 'ems': []}}
    for w in tv_weights:
        results[f'tv_{w}'] = {'pas': [], 'ems': []}

    for idx, item in enumerate(test_tasks):
        oh, ow = item['out_h'], item['out_w']

        # Standard
        pa, em = ttct_standard(model, item['demo_inputs'], item['demo_outputs'],
                                item['test_input'], item['test_output'], oh, ow)
        results['standard']['pas'].append(pa)
        results['standard']['ems'].append(em)

        # Surface tension variants
        for w in tv_weights:
            pa, em = ttct_surface_tension(model, item['demo_inputs'], item['demo_outputs'],
                                           item['test_input'], item['test_output'],
                                           oh, ow, tv_weight=w)
            results[f'tv_{w}']['pas'].append(pa)
            results[f'tv_{w}']['ems'].append(em)

        if (idx + 1) % 10 == 0:
            std_pa = np.mean(results['standard']['pas'])*100
            best_tv = max([(w, np.mean(results[f'tv_{w}']['pas'])*100) for w in tv_weights],
                         key=lambda x: x[1])
            print(f"  [{idx+1}/{len(test_tasks)}] Standard={std_pa:.1f}% | "
                  f"Best TV(w={best_tv[0]})={best_tv[1]:.1f}%")

    elapsed = time.time() - t0
    summary = {}
    print(f"\n{'='*70}")
    print(f"Phase 183 Complete ({elapsed:.0f}s)")
    for name, r in results.items():
        avg_pa = np.mean(r['pas']); avg_em = np.mean(r['ems'])
        summary[name] = {'pa': avg_pa, 'em': avg_em}
        print(f"  {name:15s}: PA={avg_pa*100:.2f}%, EM={avg_em*100:.1f}%")

    best_tv = max([(k, v) for k, v in summary.items() if k.startswith('tv_')],
                  key=lambda x: x[1]['pa'])
    improvement = best_tv[1]['pa'] - summary['standard']['pa']
    print(f"\n  Best TV ({best_tv[0]}): {improvement*100:+.2f}pp vs standard TTCT")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase183_surface_tension.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 183: Surface Tension TTCT',
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'improvement_pp': improvement * 100,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        # PA vs TV weight
        std_pa = summary['standard']['pa']*100
        tv_pa = [summary[f'tv_{w}']['pa']*100 for w in tv_weights]
        axes[0].plot(tv_weights, tv_pa, 'o-', color='#2ecc71', linewidth=2, label='TV TTCT')
        axes[0].axhline(std_pa, color='#95a5a6', linestyle='--', label='Standard')
        axes[0].set_xlabel('TV Weight'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title('PA vs Surface Tension', fontweight='bold')
        axes[0].set_xscale('log'); axes[0].legend()
        # EM comparison
        labels = ['Std'] + [f'TV\n{w}' for w in tv_weights]
        em_vals = [summary['standard']['em']*100] + [summary[f'tv_{w}']['em']*100 for w in tv_weights]
        colors = ['#95a5a6'] + ['#3498db']*len(tv_weights)
        axes[1].bar(labels, em_vals, color=colors, alpha=0.85)
        axes[1].set_ylabel('EM (%)'); axes[1].set_title('Exact Match', fontweight='bold')
        # Improvement
        impr = [(summary[f'tv_{w}']['pa'] - summary['standard']['pa'])*100 for w in tv_weights]
        axes[2].bar([str(w) for w in tv_weights], impr,
                   color=['#2ecc71' if v>0 else '#e74c3c' for v in impr], alpha=0.85)
        axes[2].set_xlabel('TV Weight'); axes[2].set_ylabel('Improvement (pp)')
        axes[2].set_title('TV vs Standard', fontweight='bold')
        axes[2].axhline(0, color='black', linewidth=0.5)
        fig.suptitle('Phase 183: Surface Tension TTCT (TV Loss)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.16, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase183_surface_tension.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return summary

if __name__ == '__main__':
    main()
