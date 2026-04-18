"""
Phase 137: v23 The Chimera Agent

Integrates the best of both worlds:
  - Continuous NCA loop (v21 expressiveness, 72%+ pixel)
  - Readout-VQ OR TTA for crystallization
  - TTCT (works because no STE in loop)
  - NBS for diversity

The final battle against Exact Match = 0.

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
    load_arc_training, prepare_arc_meta_dataset,
    DEVICE, SEED, PAD_SIZE, N_COLORS, IN_CH
)
from phase135_readout_vq import ReadoutVQNCA, ttct_readout_vq
from phase136_tta import ContinuousContextNCA, tta_predict, d8_transforms, d8_inverse

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def nbs_predict(model, test_input, task_embed, oh, ow, K=11, n_steps=5, noise=0.05):
    """NBS with latent noise + majority vote."""
    all_preds = []
    for k in range(K):
        x = test_input.unsqueeze(0)
        _, _, H, W = x.shape
        te = task_embed.view(1, -1, 1, 1).expand(1, -1, H, W)

        state = model.stem(x)
        for t in range(n_steps):
            ctx = torch.cat([state, te], dim=1)
            delta = model.update(ctx)
            beta = model.tau(ctx)
            if noise > 0:
                beta = beta + torch.randn_like(beta) * noise
                beta = beta.clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta

        # Readout VQ if available
        if hasattr(model, 'vq') and model.use_vq:
            state, _ = model.vq(state)

        logits = model.decoder(state)
        pred = logits[0, :10].argmax(dim=0)[:oh, :ow]
        all_preds.append(pred)

    stacked = torch.stack(all_preds)
    votes = torch.zeros(10, oh, ow, device=DEVICE)
    for c in range(10):
        votes[c] = (stacked == c).float().sum(dim=0)
    return votes.argmax(dim=0)


def v23_solve_readout_vq(model, item):
    """v23 with Readout-VQ + TTCT + NBS."""
    di = [d.to(DEVICE) for d in item['demo_inputs']]
    do = [d.to(DEVICE) for d in item['demo_outputs']]
    ti = item['test_input'].to(DEVICE)
    oh, ow = item['out_h'], item['out_w']

    t0 = time.time()
    best_embed, _ = ttct_readout_vq(model, di, do, n_steps=5, ttct_steps=80)
    with torch.no_grad():
        pred = nbs_predict(model, ti, best_embed, oh, ow, K=11, n_steps=5, noise=0.05)
    return pred.cpu(), time.time() - t0


def v23_solve_tta(model, item):
    """v23 with TTA (D8) + NBS."""
    di = [d.to(DEVICE) for d in item['demo_inputs']]
    do = [d.to(DEVICE) for d in item['demo_outputs']]
    ti = item['test_input'].to(DEVICE)
    oh, ow = item['out_h'], item['out_w']

    t0 = time.time()
    with torch.no_grad():
        task_embed = model.encode_demos(di, do)
        pred = tta_predict(model, ti, task_embed, oh, ow, n_steps=5)
    return pred.cpu(), time.time() - t0


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 137: v23 The Chimera Agent")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load ARC
    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=400)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:50]
    print(f"  Test tasks: {len(test_tasks)}")

    # Try to load pre-trained models
    agents = {}

    # Agent A: Readout-VQ (Phase 135)
    p135_path = os.path.join(RESULTS_DIR, "phase135_model_256.pt")
    if os.path.exists(p135_path):
        print("\n  Loading Readout-VQ model...")
        model_rvq = ReadoutVQNCA(embed_dim=64, hidden_ch=32, n_codes=256).to(DEVICE)
        model_rvq.load_state_dict(torch.load(p135_path, map_location=DEVICE, weights_only=True))
        model_rvq.eval()
        agents['v23-ReadoutVQ'] = ('rvq', model_rvq)
    else:
        print("  Readout-VQ model not found, skipping")

    # Agent B: TTA (Phase 136)
    p136_path = os.path.join(RESULTS_DIR, "phase136_model.pt")
    if os.path.exists(p136_path):
        print("  Loading Continuous+TTA model...")
        model_tta = ContinuousContextNCA(embed_dim=64, hidden_ch=32).to(DEVICE)
        model_tta.load_state_dict(torch.load(p136_path, map_location=DEVICE, weights_only=True))
        model_tta.eval()
        agents['v23-TTA'] = ('tta', model_tta)
    else:
        print("  TTA model not found, skipping")

    if not agents:
        print("  ERROR: No pre-trained models found! Run Phase 135/136 first.")
        return

    # Also run zero-shot baseline on first available model
    _, first_model = list(agents.values())[0]
    n_params = sum(p.numel() for p in first_model.parameters())

    # Run all agents
    print("\n[Step 2] Running v23 agents...")
    all_results = {}

    for agent_name, (agent_type, model) in agents.items():
        print(f"\n  === {agent_name} ===")
        zs_px = 0; ag_px = 0; total_px = 0
        zs_exact = 0; ag_exact = 0
        times = []; n = 0

        for i, item in enumerate(test_tasks):
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            ti = item['test_input'].to(DEVICE)
            to_gt = item['test_output'].to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            gt = to_gt[:10].argmax(dim=0)[:oh, :ow]

            # Zero-shot
            with torch.no_grad():
                if agent_type == 'rvq':
                    logits = model(di, do, ti, n_steps=5)
                else:
                    logits = model(di, do, ti, n_steps=5)
                pred_zs = logits[0, :10].argmax(dim=0)[:oh, :ow]
                zs_px += (pred_zs == gt).sum().item()
                zs_exact += (pred_zs == gt).all().item()

            # Agent
            if agent_type == 'rvq':
                pred, elapsed = v23_solve_readout_vq(model, item)
            else:
                pred, elapsed = v23_solve_tta(model, item)

            ag_match = (pred == gt.cpu())
            ag_px += ag_match.sum().item()
            ag_exact += ag_match.all().item()
            total_px += oh * ow
            times.append(elapsed)
            n += 1

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(test_tasks)}: "
                      f"ZS_ex={zs_exact}, v23_ex={ag_exact}, "
                      f"time={np.mean(times):.1f}s")

        zs_acc = zs_px / max(total_px, 1)
        ag_acc = ag_px / max(total_px, 1)

        all_results[agent_name] = {
            'zs_pixel': zs_acc, 'zs_exact': zs_exact,
            'ag_pixel': ag_acc, 'ag_exact': ag_exact,
            'total': n, 'avg_time': float(np.mean(times)),
        }

    # Summary
    print(f"\n{'='*70}")
    print("  v23 CHIMERA AGENT - FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Previous results:")
    print(f"    v21 (S7, continuous):    pixel=72.60%, exact=0/50")
    print(f"    v22 (S10, full-VQ):      pixel=60.07%, exact=0/50")
    print(f"")
    for name, res in all_results.items():
        print(f"  {name}:")
        print(f"    ZS:  pixel={res['zs_pixel']*100:.2f}%, exact={res['zs_exact']}/{res['total']}")
        print(f"    v23: pixel={res['ag_pixel']*100:.2f}%, exact={res['ag_exact']}/{res['total']}")
        print(f"    Gap: pixel={((res['ag_pixel']-res['zs_pixel'])*100):+.2f}%, "
              f"exact={res['ag_exact']-res['zs_exact']:+d}")
        print(f"    Time: {res['avg_time']:.2f}s/task (budget=432s)")

        if res['ag_exact'] > 0:
            print(f"    🎉 EXACT MATCHES ACHIEVED! 🎉")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase137_v23_agent.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 137: v23 Chimera Agent',
                   'timestamp': datetime.now().isoformat(),
                   'results': all_results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        versions = ['v21\n(S7)', 'v22\n(S10)']
        px_vals = [72.60, 60.07]
        ex_vals = [0, 0]

        for name, res in all_results.items():
            versions.append(f'{name}\n(S11)')
            px_vals.append(res['ag_pixel'] * 100)
            ex_vals.append(res['ag_exact'])

        x = np.arange(len(versions))
        w = 0.35
        ax.bar(x - w/2, px_vals, w, label='Pixel %', color='#3498db')
        ax.bar(x + w/2, ex_vals, w, label='Exact Matches', color='#2ecc71')
        ax.set_xticks(x); ax.set_xticklabels(versions, fontsize=8)
        ax.set_ylabel('Score')
        ax.set_title('Agent Evolution: v21 → v22 → v23')
        ax.legend()

        for i, (p, e) in enumerate(zip(px_vals, ex_vals)):
            ax.text(i - w/2, p + 1, f'{p:.1f}%', ha='center', fontsize=7)
            ax.text(i + w/2, e + 0.3, str(int(e)), ha='center', fontsize=7)

        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase137_v23_agent.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 137 complete!")


if __name__ == '__main__':
    main()
