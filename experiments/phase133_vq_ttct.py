"""
Phase 133: VQ-TTCT (Discrete Context Tuning)

Freeze all model weights + VQ codebook.
Optimize ONLY the Task Embedding via STE-compatible backprop.

"Tilt the landscape with context wind, don't reshape the terrain"

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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def vq_ttct_optimize(model, demo_inputs, demo_outputs, n_steps_nca=5,
                     ttct_steps=150, ttct_lr=0.01):
    """
    Optimize ONLY the task embedding on demo loss.
    VQ uses STE so gradients flow through to embedding.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Get initial embedding
    with torch.no_grad():
        task_embed_init = model.encoder(demo_inputs, demo_outputs)

    task_embed = task_embed_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([task_embed], lr=ttct_lr)

    best_loss = float('inf')
    best_embed = task_embed.clone().detach()

    for step in range(ttct_steps):
        opt.zero_grad()
        total_loss = 0

        for di, do_gt in zip(demo_inputs, demo_outputs):
            # Manual forward with custom embedding
            B = 1
            x = di.unsqueeze(0)
            _, _, H, W = x.shape
            te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

            state = model.stem(x)
            for t in range(n_steps_nca):
                ctx = torch.cat([state, te], dim=1)
                delta = model.update(ctx)
                beta = model.tau(ctx)
                state = beta * state + (1 - beta) * delta
                # VQ with STE (gradients pass through)
                state, _, _, _ = model.vq(state)

            logits = model.decoder(state)
            target = do_gt[:10].argmax(dim=0).unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            total_loss += loss

        total_loss /= len(demo_inputs)
        total_loss.backward()
        opt.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_embed = task_embed.clone().detach()

    for p in model.parameters():
        p.requires_grad = True

    return best_embed, best_loss


def predict_with_embed(model, test_input, task_embed, n_steps=5, gumbel_scale=0.0):
    """Forward pass using custom task embedding."""
    B = 1
    x = test_input.unsqueeze(0)
    _, _, H, W = x.shape
    te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

    state = model.stem(x)
    for t in range(n_steps):
        ctx = torch.cat([state, te], dim=1)
        delta = model.update(ctx)
        beta = model.tau(ctx)
        state = beta * state + (1 - beta) * delta
        state, _, _, _ = model.vq(state, gumbel_scale=gumbel_scale)

    return model.decoder(state)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 133: VQ-TTCT (Discrete Context Tuning)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load model
    model_path = os.path.join(RESULTS_DIR, "phase132_model.pt")
    if not os.path.exists(model_path):
        print("  ERROR: Phase 132 model not found!")
        return

    model = ContextVQNCA(embed_dim=64, hidden_ch=32, n_codes=64).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    print("  Model loaded!")

    # Load data
    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=350)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[-40:]
    print(f"  Test tasks: {len(test_tasks)}")

    # Compare: Zero-shot vs TTCT
    print("\n[Step 2] Zero-Shot vs VQ-TTCT...")
    zs_px = 0; zs_exact = 0
    tt_px = 0; tt_exact = 0
    total_px = 0; n = 0
    times = []

    model.eval()
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
            zs_exact += (pred_zs == gt).all().item()

        # TTCT
        t0 = time.time()
        best_embed, best_loss = vq_ttct_optimize(
            model, di, do, n_steps_nca=5, ttct_steps=100, ttct_lr=0.01)
        ttct_time = time.time() - t0
        times.append(ttct_time)

        with torch.no_grad():
            logits_tt = predict_with_embed(model, ti, best_embed, n_steps=5)
            pred_tt = logits_tt[0, :10].argmax(dim=0)[:oh, :ow]
            tt_px += (pred_tt == gt).sum().item()
            tt_exact += (pred_tt == gt).all().item()

        total_px += oh * ow
        n += 1

        if (i + 1) % 10 == 0:
            zs_a = zs_px / total_px * 100
            tt_a = tt_px / total_px * 100
            print(f"    {i+1}/{len(test_tasks)}: ZS={zs_a:.1f}%, "
                  f"TTCT={tt_a:.1f}%, gap={tt_a-zs_a:+.1f}%, "
                  f"ZS_ex={zs_exact}, TT_ex={tt_exact}")

    zs_px_acc = zs_px / max(total_px, 1)
    tt_px_acc = tt_px / max(total_px, 1)

    print(f"\n{'='*70}")
    print(f"  VQ-TTCT RESULTS")
    print(f"{'='*70}")
    print(f"  Zero-Shot:  pixel={zs_px_acc*100:.2f}%, exact={zs_exact}/{n}")
    print(f"  VQ-TTCT:    pixel={tt_px_acc*100:.2f}%, exact={tt_exact}/{n}")
    print(f"  Gap:        pixel={((tt_px_acc-zs_px_acc)*100):+.2f}%, exact={tt_exact-zs_exact:+d}")
    print(f"  Avg time:   {np.mean(times):.2f}s per task")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase133_vq_ttct.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 133: VQ-TTCT',
                   'timestamp': datetime.now().isoformat(),
                   'results': {
                       'zs_pixel': zs_px_acc, 'tt_pixel': tt_px_acc,
                       'zs_exact': zs_exact, 'tt_exact': tt_exact,
                       'total': n, 'avg_time': float(np.mean(times)),
                   }}, f, indent=2, default=str)

    print("\nPhase 133 complete!")


if __name__ == '__main__':
    main()
