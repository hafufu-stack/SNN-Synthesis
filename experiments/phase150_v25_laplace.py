"""
Phase 150: v25 The Laplace Demon — Ultimate Soft-Crystallization Agent

Combines ALL discoveries into the final ARC agent:
  1. Phase 123 pre-trained Foundation Model (148K params, 400 tasks)
  2. TTCT + Entropy Minimization (Phase 149: +3.51% gain)
  3. Fixed D8 TTA (Phase 136 approach, non-square padding fixed)
  4. Temporal NBS (Phase 125: majority vote across noise samples)

Goal: Break v23's record (83.5% PA, 1 Exact Match).

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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


# ================================================================
# D8 Symmetry Transforms (fixed for non-square grids)
# ================================================================
def d8_transform(x, idx):
    """Apply D8 transform idx to tensor (B,C,H,W). H=W=PAD_SIZE."""
    if idx == 0: return x
    elif idx == 1: return torch.rot90(x, 1, [2, 3])
    elif idx == 2: return torch.rot90(x, 2, [2, 3])
    elif idx == 3: return torch.rot90(x, 3, [2, 3])
    elif idx == 4: return torch.flip(x, [3])
    elif idx == 5: return torch.flip(torch.rot90(x, 1, [2, 3]), [3])
    elif idx == 6: return torch.flip(x, [2])
    elif idx == 7: return torch.flip(torch.rot90(x, 1, [2, 3]), [2])
    return x


def d8_inverse(pred, idx):
    """Inverse D8 transform on 2D tensor (H, W) in FULL padded space."""
    if idx == 0: return pred
    elif idx == 1: return torch.rot90(pred, -1, [0, 1])
    elif idx == 2: return torch.rot90(pred, -2, [0, 1])
    elif idx == 3: return torch.rot90(pred, -3, [0, 1])
    elif idx == 4: return torch.flip(pred, [1])
    elif idx == 5: return torch.flip(torch.rot90(pred, -1, [0, 1]), [1])
    elif idx == 6: return torch.flip(pred, [0])
    elif idx == 7: return torch.flip(torch.rot90(pred, -1, [0, 1]), [0])
    return pred


# ================================================================
# Entropy Minimization Loss
# ================================================================
def entropy_loss(logits):
    """Minimize entropy of softmax to sharpen predictions."""
    probs = F.softmax(logits[:, :10], dim=1)
    ent = -(probs * (probs + 1e-8).log()).sum(dim=1)
    return ent.mean()


# ================================================================
# TTCT with Soft Crystallization
# ================================================================
def ttct_crystallize(model, di, do, n_steps=5, ttct_steps=200,
                     lr=0.01, ent_weight=0.1):
    """TTCT with entropy minimization for soft crystallization."""
    task_embed = model.task_encoder(di, do).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([task_embed], lr=lr)
    best_loss = float('inf')
    best_te = task_embed.data.clone()

    for step in range(ttct_steps):
        opt.zero_grad()
        total = 0
        for inp, out in zip(di, do):
            logits = model.latent_nca(inp.unsqueeze(0), task_embed, n_steps)
            target = out[:10].argmax(0).unsqueeze(0)
            demo_loss = F.cross_entropy(logits, target)
            ent = entropy_loss(logits)
            total += demo_loss + ent_weight * ent
        total /= len(di)
        total.backward()
        opt.step()

        if total.item() < best_loss:
            best_loss = total.item()
            best_te = task_embed.data.clone()

    return best_te, best_loss


# ================================================================
# D8 TTA + NBS Majority Vote
# ================================================================
def d8_tta_vote(model, test_input, task_embed, oh, ow,
                n_steps=5, use_nbs=True, K=7, noise_sigma=0.05):
    """
    Full D8 TTA with optional NBS per view.
    All transforms done in FULL padded space (PAD_SIZE x PAD_SIZE),
    inverse transform BEFORE cropping to handle non-square grids.
    """
    votes = torch.zeros(10, PAD_SIZE, PAD_SIZE, device=DEVICE)

    for d8_idx in range(8):
        # Transform input in padded space
        x_view = d8_transform(test_input.unsqueeze(0), d8_idx)

        if use_nbs:
            for k in range(K):
                state = model.latent_nca.encoder(x_view)
                B, _, H, W = state.shape
                te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

                for t in range(n_steps):
                    state_ctx = torch.cat([state, te], dim=1)
                    delta = model.latent_nca.update(state_ctx)
                    beta = model.latent_nca.tau_gate(state_ctx)

                    if noise_sigma > 0 and k > 0:
                        noise = torch.randn_like(beta) * noise_sigma
                        beta = torch.sigmoid(
                            torch.logit(beta.clamp(1e-6, 1-1e-6)) + noise)

                    state = beta * state + (1 - beta) * delta

                logits = model.latent_nca.decoder(state)
                pred = logits[0, :10].argmax(dim=0)  # (PAD_SIZE, PAD_SIZE)

                # Inverse transform in FULL padded space
                pred_inv = d8_inverse(pred, d8_idx)
                for c in range(10):
                    votes[c] += (pred_inv == c).float()
        else:
            logits = model.latent_nca(x_view, task_embed, n_steps)
            pred = logits[0, :10].argmax(dim=0)
            pred_inv = d8_inverse(pred, d8_idx)
            for c in range(10):
                votes[c] += (pred_inv == c).float()

    # Crop AFTER voting
    return votes.argmax(dim=0)[:oh, :ow]


# ================================================================
# v25 Solve Pipeline
# ================================================================
def v25_solve(model, item, time_budget=30.0):
    """Complete v25 pipeline for one task."""
    di = [d.to(DEVICE) for d in item['demo_inputs']]
    do = [d.to(DEVICE) for d in item['demo_outputs']]
    ti = item['test_input'].to(DEVICE)
    oh, ow = item['out_h'], item['out_w']

    t0 = time.time()

    # Step 1: TTCT with Entropy Minimization (use 60% of budget)
    ttct_budget = max(50, int(time_budget * 0.6 * 10))
    best_te, best_loss = ttct_crystallize(
        model, di, do, n_steps=5, ttct_steps=min(ttct_budget, 200),
        ent_weight=0.1)

    # Step 2: D8 TTA + NBS majority vote
    with torch.no_grad():
        pred = d8_tta_vote(model, ti, best_te, oh, ow,
                          n_steps=5, use_nbs=True, K=7, noise_sigma=0.05)

    elapsed = time.time() - t0
    return pred.cpu().numpy().tolist(), elapsed, best_loss


# ================================================================
# Main
# ================================================================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 150: v25 The Laplace Demon")
    print("  Foundation + TTCT + Entropy Min + D8 TTA + NBS")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Step 1: Load pre-trained Foundation Model
    print("\n[Step 1] Loading Phase 123 Foundation Model...")
    model_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    if not os.path.exists(model_path):
        print("  ERROR: phase123_model.pt not found! Run Phase 123 first.")
        return

    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters")

    # Step 2: Load test tasks
    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=400)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:50]  # Dry run on 50 tasks
    print(f"  Test tasks: {len(test_tasks)}")

    # Step 3: Compare methods
    print("\n[Step 3] Running v25 Laplace Demon Evaluation...")

    # Metrics
    metrics = {
        'zero_shot': {'px': 0, 'ex': 0},
        'ttct_only': {'px': 0, 'ex': 0},
        'v25_full': {'px': 0, 'ex': 0},
    }
    total_px = 0
    times = []
    task_details = []

    for i, item in enumerate(test_tasks):
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do_ = [d.to(DEVICE) for d in item['demo_outputs']]
        ti = item['test_input'].to(DEVICE)
        to_ = item['test_output'].to(DEVICE)
        oh, ow = item['out_h'], item['out_w']
        gt = to_[:10].argmax(0)[:oh, :ow]

        # (a) Zero-shot baseline
        with torch.no_grad():
            logits = model(di, do_, ti, n_steps=5)
            pred_zs = logits[0, :10].argmax(0)[:oh, :ow]
            zs_match = (pred_zs == gt).sum().item()
            zs_exact = (pred_zs == gt).all().item()
            metrics['zero_shot']['px'] += zs_match
            metrics['zero_shot']['ex'] += zs_exact

        # (b) TTCT only (no TTA)
        best_te, _ = ttct_crystallize(model, di, do_, ttct_steps=100, ent_weight=0.1)
        with torch.no_grad():
            logits_t = model.latent_nca(ti.unsqueeze(0), best_te, n_steps=5)
            pred_t = logits_t[0, :10].argmax(0)[:oh, :ow]
            t_match = (pred_t == gt).sum().item()
            t_exact = (pred_t == gt).all().item()
            metrics['ttct_only']['px'] += t_match
            metrics['ttct_only']['ex'] += t_exact

        # (c) v25 Full: TTCT + D8 TTA + NBS
        t_start = time.time()
        with torch.no_grad():
            pred_v25 = d8_tta_vote(model, ti, best_te, oh, ow,
                                   n_steps=5, use_nbs=True, K=7, noise_sigma=0.05)
        v25_match = (pred_v25 == gt).sum().item()
        v25_exact = (pred_v25 == gt).all().item()
        metrics['v25_full']['px'] += v25_match
        metrics['v25_full']['ex'] += v25_exact
        elapsed_task = time.time() - t_start
        times.append(elapsed_task)

        total_px += oh * ow

        detail = {
            'task_id': item['task_id'],
            'zs_pa': zs_match / (oh*ow), 'zs_ex': zs_exact,
            'ttct_pa': t_match / (oh*ow), 'ttct_ex': t_exact,
            'v25_pa': v25_match / (oh*ow), 'v25_ex': v25_exact,
        }
        task_details.append(detail)

        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(test_tasks)}: "
                  f"ZS_ex={metrics['zero_shot']['ex']}, "
                  f"TTCT_ex={metrics['ttct_only']['ex']}, "
                  f"v25_ex={metrics['v25_full']['ex']}")

    # Summary
    elapsed = time.time() - t0
    n = len(test_tasks)

    print(f"\n{'='*70}")
    print(f"Phase 150: v25 The Laplace Demon ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Tasks: {n}")
    print(f"  Parameters: {n_params:,}")
    print(f"")

    for method in ['zero_shot', 'ttct_only', 'v25_full']:
        pa = metrics[method]['px'] / max(total_px, 1) * 100
        ex = metrics[method]['ex']
        label = {'zero_shot': 'Zero-Shot', 'ttct_only': 'TTCT+Entropy',
                 'v25_full': 'v25 Full (TTCT+D8+NBS)'}[method]
        print(f"  {label:30s}: PA={pa:.2f}%, Exact={ex}/{n}")

    v25_pa = metrics['v25_full']['px'] / max(total_px, 1) * 100
    zs_pa = metrics['zero_shot']['px'] / max(total_px, 1) * 100
    print(f"\n  Total improvement: PA={v25_pa-zs_pa:+.2f}%, "
          f"Exact={metrics['v25_full']['ex']-metrics['zero_shot']['ex']:+d}")
    print(f"  Avg D8+NBS time/task: {np.mean(times):.2f}s")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase150_v25_laplace.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 150: v25 The Laplace Demon',
            'timestamp': datetime.now().isoformat(),
            'n_params': n_params,
            'metrics': {k: {'pixel_acc': v['px']/max(total_px,1), 'exact': v['ex']}
                       for k, v in metrics.items()},
            'total_tasks': n, 'total_pixels': total_px,
            'avg_tta_time': float(np.mean(times)),
            'per_task': task_details,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        # Panel 1: PA comparison
        methods = ['Zero-Shot', 'TTCT+\nEntropy', 'v25 Full\n(TTCT+D8+NBS)']
        pas = [metrics[k]['px']/max(total_px,1)*100 for k in ['zero_shot','ttct_only','v25_full']]
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        bars = axes[0].bar(range(3), pas, color=colors, alpha=0.85, edgecolor='black')
        for bar, pa in zip(bars, pas):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{pa:.1f}%', ha='center', fontweight='bold')
        axes[0].set_xticks(range(3)); axes[0].set_xticklabels(methods)
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('v25 Laplace Demon', fontweight='bold')

        # Panel 2: Exact match
        exs = [metrics[k]['ex'] for k in ['zero_shot','ttct_only','v25_full']]
        bars = axes[1].bar(range(3), exs, color=colors, alpha=0.85, edgecolor='black')
        for bar, ex in zip(bars, exs):
            axes[1].text(bar.get_x()+bar.get_width()/2, max(ex+0.2, 0.3),
                        str(ex), ha='center', fontweight='bold')
        axes[1].set_xticks(range(3)); axes[1].set_xticklabels(methods)
        axes[1].set_ylabel('Exact Matches'); axes[1].set_title('Exact Match Count', fontweight='bold')

        # Panel 3: Per-task PA distribution
        zs_pas = [d['zs_pa'] for d in task_details]
        v25_pas = [d['v25_pa'] for d in task_details]
        axes[2].scatter(zs_pas, v25_pas, alpha=0.5, s=30, c='#2ecc71')
        axes[2].plot([0,1], [0,1], 'k--', alpha=0.3)
        axes[2].set_xlabel('Zero-Shot PA'); axes[2].set_ylabel('v25 PA')
        axes[2].set_title('Per-Task Comparison', fontweight='bold')

        # Panel 4: Improvement histogram
        improvements = [d['v25_pa'] - d['zs_pa'] for d in task_details]
        axes[3].hist(improvements, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        axes[3].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[3].set_xlabel('PA Improvement (v25 - ZS)'); axes[3].set_ylabel('Count')
        mean_imp = np.mean(improvements)
        axes[3].set_title(f'Improvement Distribution (mean={mean_imp*100:+.1f}%)', fontweight='bold')

        plt.suptitle('Phase 150: v25 The Laplace Demon (Foundation + TTCT + Entropy + D8 + NBS)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase150_v25_laplace.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print(f"\nPhase 150 complete! ({elapsed:.0f}s)")
    return metrics


if __name__ == '__main__':
    main()
