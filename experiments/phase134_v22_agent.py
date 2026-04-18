"""
Phase 134: v22 The Quantum Agent (Kaggle Final Dry Run)

Full pipeline:
  1. Context Encoder → Zero-Shot Task Embedding
  2. VQ-TTCT → Optimize embedding on demos
  3. Gumbel-Temporal NBS (G=2.0, K=21) → Discrete beam search
  4. Discrete Auto-T → Crystallization early-stop
  5. Size Cropper → Output correct dimensions

The moment of truth: Can VQ cells achieve Exact Match on Real ARC?

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase132_foundation_vq import (
    ContextVQNCA, load_arc_training, prepare_arc_meta_dataset,
    DEVICE, SEED, PAD_SIZE, N_COLORS, IN_CH
)
from phase133_vq_ttct import vq_ttct_optimize, predict_with_embed

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def gumbel_nbs_with_auto_t(model, test_input, task_embed, K=21, max_steps=15,
                            gumbel_scale=2.0):
    """
    Gumbel Temporal NBS + Auto-T:
    Each beam runs G-noisy VQ-NCA until crystallized, then votes.
    """
    B = 1
    H, W = PAD_SIZE, PAD_SIZE

    # Collect K predictions
    all_preds = []

    for k in range(K):
        x = test_input.unsqueeze(0)
        te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

        state = model.stem(x)
        prev_indices = None

        for t in range(max_steps):
            ctx = torch.cat([state, te], dim=1)
            delta = model.update(ctx)
            beta = model.tau(ctx)
            state = beta * state + (1 - beta) * delta
            state, _, _, indices = model.vq(state, gumbel_scale=gumbel_scale)

            # Auto-T: stop if crystallized
            if prev_indices is not None:
                n_changed = (indices != prev_indices).sum().item()
                if n_changed == 0:
                    break
            prev_indices = indices

        logits = model.decoder(state)
        pred = logits[0, :10].argmax(dim=0)  # (H, W)
        all_preds.append(pred)

    # Majority vote
    stacked = torch.stack(all_preds)  # (K, H, W)
    votes = torch.zeros(10, H, W, device=DEVICE)
    for c in range(10):
        votes[c] = (stacked == c).float().sum(dim=0)

    return votes.argmax(dim=0)  # (H, W)


def v22_solve(model, item, time_budget=30.0):
    """Full v22 pipeline for one task."""
    di = [d.to(DEVICE) for d in item['demo_inputs']]
    do = [d.to(DEVICE) for d in item['demo_outputs']]
    ti = item['test_input'].to(DEVICE)
    oh, ow = item['out_h'], item['out_w']

    t0 = time.time()

    # Step 1: VQ-TTCT
    ttct_steps = min(150, max(50, int(time_budget * 5)))
    best_embed, best_loss = vq_ttct_optimize(
        model, di, do, n_steps_nca=5, ttct_steps=ttct_steps, ttct_lr=0.01)

    # Step 2: Gumbel NBS + Auto-T
    with torch.no_grad():
        pred = gumbel_nbs_with_auto_t(
            model, ti, best_embed, K=21, max_steps=10, gumbel_scale=2.0)

    # Crop
    pred_crop = pred[:oh, :ow].cpu()
    elapsed = time.time() - t0

    return pred_crop, elapsed, best_loss


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 134: v22 The Quantum Agent (Kaggle Final Dry Run)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load model
    model_path = os.path.join(RESULTS_DIR, "phase132_model.pt")
    if not os.path.exists(model_path):
        print("  ERROR: Phase 132 model not found!")
        return

    model = ContextVQNCA(embed_dim=64, hidden_ch=32, n_codes=64).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    # Load ALL ARC tasks
    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=400)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:50]
    print(f"  Test tasks: {len(test_tasks)}")

    # Run v22 vs Zero-Shot
    print("\n[Step 2] v22 Quantum Agent vs Zero-Shot...")
    zs_px = 0; v22_px = 0; total_px = 0
    zs_exact = 0; v22_exact = 0
    times = []; task_results = []

    for i, item in enumerate(test_tasks):
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do = [d.to(DEVICE) for d in item['demo_outputs']]
        ti = item['test_input'].to(DEVICE)
        to_gt = item['test_output'].to(DEVICE)
        oh, ow = item['out_h'], item['out_w']
        gt = to_gt[:10].argmax(dim=0)[:oh, :ow]

        # Zero-shot
        with torch.no_grad():
            logits = model(di, do, ti, n_steps=5)
            pred_zs = logits[0, :10].argmax(dim=0)[:oh, :ow]
            zs_px += (pred_zs == gt).sum().item()
            zs_ex = (pred_zs == gt).all().item()
            zs_exact += zs_ex

        # v22
        pred_v22, elapsed, ttct_loss = v22_solve(model, item, time_budget=30.0)
        v22_match = (pred_v22 == gt.cpu())
        v22_px += v22_match.sum().item()
        v22_ex = v22_match.all().item()
        v22_exact += v22_ex

        total_px += oh * ow
        times.append(elapsed)

        task_results.append({
            'task_id': item['task_id'],
            'zs_exact': zs_ex, 'v22_exact': v22_ex,
            'zs_pixel': (pred_zs == gt).float().mean().item(),
            'v22_pixel': v22_match.float().mean().item(),
            'time': elapsed,
        })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_tasks)}: "
                  f"ZS_ex={zs_exact}, v22_ex={v22_exact}, "
                  f"avg_time={np.mean(times):.1f}s")

    # Final
    zs_px_acc = zs_px / max(total_px, 1)
    v22_px_acc = v22_px / max(total_px, 1)

    print(f"\n{'='*70}")
    print(f"  v22 QUANTUM AGENT - FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Tasks tested:   {len(test_tasks)}")
    print(f"  Parameters:     {n_params:,}")
    print(f"")
    print(f"  Zero-Shot:")
    print(f"    Pixel acc:    {zs_px_acc*100:.2f}%")
    print(f"    Exact match:  {zs_exact}/{len(test_tasks)}")
    print(f"")
    print(f"  v22 Quantum Agent (VQ-TTCT + Gumbel NBS K=21 + Auto-T):")
    print(f"    Pixel acc:    {v22_px_acc*100:.2f}%")
    print(f"    Exact match:  {v22_exact}/{len(test_tasks)}")
    print(f"    Avg time:     {np.mean(times):.2f}s (budget=432s)")
    print(f"")
    print(f"  Gap vs ZS:      pixel={((v22_px_acc-zs_px_acc)*100):+.2f}%, "
          f"exact={v22_exact-zs_exact:+d}")

    if v22_exact > 0:
        print(f"\n  🎉🎉🎉 v22 QUANTUM AGENT SOLVED {v22_exact} REAL ARC TASKS! 🎉🎉🎉")
        solved = [r for r in task_results if r['v22_exact']]
        for s in solved:
            print(f"    ✅ Task {s['task_id']}: pixel={s['v22_pixel']*100:.1f}%")

    # Compare v21 vs v22
    print(f"\n  === v21 (Season 7) vs v22 (Season 10) ===")
    print(f"  v21: pixel=72.60%, exact=0/50  (continuous)")
    print(f"  v22: pixel={v22_px_acc*100:.2f}%, exact={v22_exact}/50  (quantized)")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase134_v22_agent.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 134: v22 Quantum Agent',
                   'timestamp': datetime.now().isoformat(),
                   'summary': {
                       'zs_pixel': zs_px_acc, 'v22_pixel': v22_px_acc,
                       'zs_exact': zs_exact, 'v22_exact': v22_exact,
                       'avg_time': float(np.mean(times)),
                       'n_params': n_params,
                   },
                   'per_task': task_results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Scatter: ZS vs v22 per task
        zs_pxs = [r['zs_pixel'] for r in task_results]
        v22_pxs = [r['v22_pixel'] for r in task_results]
        colors_scatter = ['#2ecc71' if r['v22_exact'] else '#3498db' for r in task_results]
        axes[0].scatter(zs_pxs, v22_pxs, c=colors_scatter, alpha=0.6, s=30)
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[0].set_xlabel('Zero-Shot pixel acc')
        axes[0].set_ylabel('v22 pixel acc')
        axes[0].set_title('Per-Task: ZS vs v22')

        # Time distribution
        axes[1].hist(times, bins=20, color='#3498db', alpha=0.7)
        axes[1].axvline(x=432, color='red', linestyle='--', label='Kaggle budget')
        axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Count')
        axes[1].set_title('v22 Runtime'); axes[1].legend()

        # v21 vs v22 comparison
        labels = ['v21\n(Continuous)', 'v22\n(Quantum)']
        pixel_vals = [0.726, v22_px_acc]
        exact_vals = [0, v22_exact / max(len(test_tasks), 1)]

        x = np.arange(2)
        w = 0.35
        axes[2].bar(x - w/2, [p * 100 for p in pixel_vals], w,
                   label='Pixel %', color='#3498db')
        axes[2].bar(x + w/2, [e * 100 for e in exact_vals], w,
                   label='Exact %', color='#2ecc71')
        axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
        axes[2].set_ylabel('Rate (%)'); axes[2].set_title('v21 vs v22 Agent')
        axes[2].legend()

        plt.suptitle('Phase 134: v22 Quantum Agent (Real ARC)', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase134_v22_agent.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 134 complete!")


if __name__ == '__main__':
    main()
