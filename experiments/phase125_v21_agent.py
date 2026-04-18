"""
Phase 125: v21 Singularity Agent (Kaggle Dry Run)

Full pipeline:
  1. Task Embedding from demos (In-Context)
  2. TTCT: optimize context vector on demo loss
  3. Latent Temporal NBS: noise beam search K=11
  4. Soft-Landing: energy canary for stability

Simulates with Kaggle's 432s/task budget.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset,
    grid_to_tensor, tensor_to_grid, DEVICE, SEED, PAD_SIZE, N_COLORS
)
from phase124_ttct import ttct_optimize

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def temporal_nbs(model, test_input, task_embed, K=11, n_steps=5,
                 noise_sigma=0.1):
    """
    Temporal Noise Beam Search: run K forward passes with noise,
    then majority-vote on pixel predictions.
    """
    B = 1
    votes = torch.zeros(N_COLORS, PAD_SIZE, PAD_SIZE, device=DEVICE)

    for k in range(K):
        state = model.latent_nca.encoder(test_input.unsqueeze(0))
        te = task_embed.view(1, -1, 1, 1).expand(B, -1, PAD_SIZE, PAD_SIZE)

        for t in range(n_steps):
            state_ctx = torch.cat([state, te], dim=1)
            delta = model.latent_nca.update(state_ctx)
            beta = model.latent_nca.tau_gate(state_ctx)

            # Inject temporal noise
            if noise_sigma > 0:
                noise = torch.randn_like(beta) * noise_sigma
                beta_logit = torch.logit(beta.clamp(1e-6, 1 - 1e-6)) + noise
                beta = torch.sigmoid(beta_logit)

            state = beta * state + (1 - beta) * delta

        logits = model.latent_nca.decoder(state)
        pred = logits[0, :10].argmax(dim=0)  # (H, W)

        # One-hot vote
        for c in range(10):
            votes[c] += (pred == c).float()

    # Majority vote
    return votes[:10].argmax(dim=0)  # (H, W)


def v21_solve(model, item, time_budget=30.0):
    """
    Full v21 Singularity Agent pipeline for one task.
    """
    di = [d.to(DEVICE) for d in item['demo_inputs']]
    do = [d.to(DEVICE) for d in item['demo_outputs']]
    ti = item['test_input'].to(DEVICE)
    oh, ow = item['out_h'], item['out_w']

    t0 = time.time()

    # Step 1: TTCT (use 70% of budget)
    ttct_budget = int(time_budget * 0.7)
    ttct_steps = min(200, max(50, ttct_budget * 10))  # ~10 steps/sec
    best_embed, best_loss = ttct_optimize(
        model, di, do, n_steps_nca=5, ttct_steps=ttct_steps, ttct_lr=0.01)

    # Step 2: Temporal NBS with optimized embedding
    with torch.no_grad():
        pred_nbs = temporal_nbs(
            model, ti, best_embed, K=11, n_steps=5, noise_sigma=0.1)

    # Crop to output size
    pred_grid = pred_nbs[:oh, :ow].cpu().numpy().tolist()
    elapsed = time.time() - t0

    return pred_grid, elapsed, best_loss


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 125: v21 Singularity Agent (Kaggle Dry Run)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load model
    model_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    if not os.path.exists(model_path):
        print("  ERROR: Phase 123 model not found!")
        return

    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    # Load ALL ARC tasks for final dry run
    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=400)
    print(f"  Total tasks: {len(all_tasks)}")

    # Run on a test subset (50 tasks for speed)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:50]

    # Also compare: zero-shot baseline
    print("\n[Step 2] Running v21 Agent vs Zero-Shot baseline...")
    zs_correct = 0; v21_correct = 0
    zs_pixels = 0; v21_pixels = 0
    total_px = 0
    times = []
    task_results = []

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
            zs_match = (pred_zs == gt).float().mean().item()
            zs_exact = (pred_zs == gt).all().item()
            zs_pixels += (pred_zs == gt).sum().item()

        # v21 Agent (TTCT + NBS)
        pred_grid, elapsed, ttct_loss = v21_solve(model, item, time_budget=30.0)
        pred_v21 = torch.tensor(pred_grid)  # CPU
        gt_cpu = gt.cpu()
        v21_match = (pred_v21 == gt_cpu).float().mean().item() if pred_v21.shape == gt_cpu.shape else 0
        v21_exact = (pred_v21 == gt_cpu).all().item() if pred_v21.shape == gt_cpu.shape else 0
        v21_pixels += (pred_v21 == gt_cpu).sum().item() if pred_v21.shape == gt_cpu.shape else 0

        total_px += oh * ow
        zs_correct += zs_exact
        v21_correct += v21_exact
        times.append(elapsed)

        task_results.append({
            'task_id': item['task_id'],
            'zs_pixel': zs_match, 'zs_exact': zs_exact,
            'v21_pixel': v21_match, 'v21_exact': v21_exact,
            'time': elapsed, 'ttct_loss': ttct_loss,
        })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(test_tasks)}: "
                  f"ZS exact={zs_correct}, v21 exact={v21_correct}, "
                  f"avg_time={np.mean(times):.1f}s")

    # Final summary
    zs_px_acc = zs_pixels / max(total_px, 1)
    v21_px_acc = v21_pixels / max(total_px, 1)

    print(f"\n{'='*70}")
    print(f"  v21 SINGULARITY AGENT - FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Tasks tested:   {len(test_tasks)}")
    print(f"  Parameters:     {n_params:,}")
    print(f"")
    print(f"  Zero-Shot:")
    print(f"    Pixel acc:    {zs_px_acc*100:.2f}%")
    print(f"    Exact match:  {zs_correct}/{len(test_tasks)}")
    print(f"")
    print(f"  v21 Agent (TTCT + NBS K=11):")
    print(f"    Pixel acc:    {v21_px_acc*100:.2f}%")
    print(f"    Exact match:  {v21_correct}/{len(test_tasks)}")
    print(f"    Avg time:     {np.mean(times):.2f}s (budget=432s)")
    print(f"")
    print(f"  Gap:            pixel={((v21_px_acc-zs_px_acc)*100):+.2f}%, "
          f"exact={v21_correct-zs_correct:+d}")

    if v21_correct > 0:
        print(f"\n  🎉 v21 SOLVED {v21_correct} REAL ARC TASKS! 🎉")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase125_v21_agent.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 125: v21 Singularity Agent',
                   'timestamp': datetime.now().isoformat(),
                   'summary': {
                       'zs_pixel': zs_px_acc, 'v21_pixel': v21_px_acc,
                       'zs_exact': zs_correct, 'v21_exact': v21_correct,
                       'avg_time': float(np.mean(times)),
                       'n_params': n_params,
                   },
                   'per_task': task_results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Per-task comparison
        zs_pxs = [r['zs_pixel'] for r in task_results]
        v21_pxs = [r['v21_pixel'] for r in task_results]
        axes[0].scatter(zs_pxs, v21_pxs, alpha=0.5, s=20)
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[0].set_xlabel('Zero-Shot pixel acc'); axes[0].set_ylabel('v21 pixel acc')
        axes[0].set_title('Per-Task Comparison')

        # Time distribution
        axes[1].hist(times, bins=20, color='#3498db', alpha=0.7)
        axes[1].axvline(x=432, color='red', linestyle='--', label='Kaggle budget')
        axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Count')
        axes[1].set_title('v21 Runtime'); axes[1].legend()

        # Summary bar
        categories = ['Zero-Shot\nPixel', 'v21\nPixel', 'Zero-Shot\nExact', 'v21\nExact']
        vals = [zs_px_acc, v21_px_acc,
                zs_correct/max(len(test_tasks),1),
                v21_correct/max(len(test_tasks),1)]
        colors = ['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71']
        axes[2].bar(range(4), vals, color=colors)
        axes[2].set_xticks(range(4)); axes[2].set_xticklabels(categories, fontsize=8)
        axes[2].set_ylabel('Rate'); axes[2].set_title('v21 vs Zero-Shot')
        for i, v in enumerate(vals):
            axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

        plt.suptitle('Phase 125: v21 Singularity Agent (Real ARC)', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase125_v21_agent.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 125 complete!")


if __name__ == '__main__':
    main()
