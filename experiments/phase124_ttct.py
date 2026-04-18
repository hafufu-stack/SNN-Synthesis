"""
Phase 124: Test-Time Context Tuning (TTCT)

Instead of updating all weights at test time (TTT = unstable, slow),
only optimize the Task Embedding vector using gradient descent on
demo pair loss. This is "Prompt Tuning" for NCA.

Budget: 432 seconds per task on Kaggle.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

# Reuse Foundation model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset,
    grid_to_tensor, tensor_to_grid, DEVICE, SEED, PAD_SIZE, N_COLORS
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def ttct_optimize(model, demo_inputs, demo_outputs, n_steps_nca=5,
                  ttct_steps=200, ttct_lr=0.01):
    """
    Optimize ONLY the task embedding vector on demo loss.
    Returns: optimized task_embed (detached)
    """
    model.eval()
    # Freeze all model weights
    for p in model.parameters():
        p.requires_grad = False

    # Get initial task embedding
    with torch.no_grad():
        task_embed_init = model.task_encoder(demo_inputs, demo_outputs)

    # Make it a learnable parameter
    task_embed = task_embed_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([task_embed], lr=ttct_lr)

    best_loss = float('inf')
    best_embed = task_embed.clone().detach()

    for step in range(ttct_steps):
        opt.zero_grad()
        total_loss = 0

        # Compute loss on each demo pair (self-consistency)
        for di, do_gt in zip(demo_inputs, demo_outputs):
            logits = model.latent_nca(
                di.unsqueeze(0), task_embed, n_steps=n_steps_nca)
            target = do_gt[:10].argmax(dim=0).unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            total_loss += loss

        total_loss /= len(demo_inputs)
        total_loss.backward()
        opt.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_embed = task_embed.clone().detach()

    # Restore requires_grad
    for p in model.parameters():
        p.requires_grad = True

    return best_embed, best_loss


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 124: Test-Time Context Tuning (TTCT)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load model
    model_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    if not os.path.exists(model_path):
        print("  ERROR: Phase 123 model not found! Run Phase 123 first.")
        return

    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print("  Model loaded!")

    # Load ARC test data
    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)

    # Use last 30 as test
    test_tasks = all_tasks[-30:]
    print(f"  Test tasks: {len(test_tasks)}")

    # Evaluate: Zero-shot vs TTCT
    print("\n[Step 2] Comparing Zero-Shot vs TTCT...")
    results = {'zero_shot': [], 'ttct': []}

    model.eval()
    for i, item in enumerate(test_tasks):
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do = [d.to(DEVICE) for d in item['demo_outputs']]
        ti = item['test_input'].to(DEVICE)
        to_gt = item['test_output'].to(DEVICE)
        oh, ow = item['out_h'], item['out_w']

        # Zero-shot
        with torch.no_grad():
            logits_zs = model(di, do, ti, n_steps=5)
            pred_zs = logits_zs[0, :10].argmax(dim=0)
            gt = to_gt[:10].argmax(dim=0)
            p_zs = pred_zs[:oh, :ow]
            g = gt[:oh, :ow]
            px_acc_zs = (p_zs == g).float().mean().item()
            exact_zs = (p_zs == g).all().item()

        # TTCT
        t0 = time.time()
        best_embed, best_loss = ttct_optimize(
            model, di, do, n_steps_nca=5, ttct_steps=100, ttct_lr=0.01)
        ttct_time = time.time() - t0

        with torch.no_grad():
            logits_ttct = model.latent_nca(
                ti.unsqueeze(0), best_embed, n_steps=5)
            pred_ttct = logits_ttct[0, :10].argmax(dim=0)
            p_ttct = pred_ttct[:oh, :ow]
            px_acc_ttct = (p_ttct == g).float().mean().item()
            exact_ttct = (p_ttct == g).all().item()

        results['zero_shot'].append({
            'task_id': item['task_id'],
            'pixel_acc': px_acc_zs,
            'exact': exact_zs,
        })
        results['ttct'].append({
            'task_id': item['task_id'],
            'pixel_acc': px_acc_ttct,
            'exact': exact_ttct,
            'time': ttct_time,
            'loss': best_loss,
        })

        if (i + 1) % 10 == 0:
            zs_mean = np.mean([r['pixel_acc'] for r in results['zero_shot']])
            tt_mean = np.mean([r['pixel_acc'] for r in results['ttct']])
            print(f"    {i+1}/{len(test_tasks)}: "
                  f"ZS={zs_mean:.3f}, TTCT={tt_mean:.3f}, "
                  f"gap={tt_mean-zs_mean:+.3f}")

    # Summary
    zs_px = np.mean([r['pixel_acc'] for r in results['zero_shot']])
    tt_px = np.mean([r['pixel_acc'] for r in results['ttct']])
    zs_ex = sum(r['exact'] for r in results['zero_shot'])
    tt_ex = sum(r['exact'] for r in results['ttct'])
    tt_time_avg = np.mean([r['time'] for r in results['ttct']])

    print(f"\n{'='*70}")
    print(f"  TTCT RESULTS")
    print(f"{'='*70}")
    print(f"  Zero-Shot:  pixel={zs_px*100:.2f}%, exact={zs_ex}/{len(test_tasks)}")
    print(f"  TTCT:       pixel={tt_px*100:.2f}%, exact={tt_ex}/{len(test_tasks)}")
    print(f"  Gap:        pixel={((tt_px-zs_px)*100):+.2f}%, exact={tt_ex-zs_ex:+d}")
    print(f"  Avg time:   {tt_time_avg:.2f}s per task (budget=432s)")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase124_ttct.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 124: TTCT',
                   'timestamp': datetime.now().isoformat(),
                   'results': {
                       'zs_pixel_acc': zs_px, 'ttct_pixel_acc': tt_px,
                       'zs_exact': zs_ex, 'ttct_exact': tt_ex,
                       'avg_time': tt_time_avg,
                   }}, f, indent=2, default=str)

    print("\nPhase 124 complete!")


if __name__ == '__main__':
    main()
