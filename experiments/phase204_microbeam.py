"""
Phase 204: Cellular Micro-Beam Search - Branch on Disagreement Pixels

Instead of noising (P200) or inpainting (P203), enumerate candidate
assignments at dissonance pixels and select the configuration with
the highest per-pixel confidence margin (from P201's Time-Travel insight).

Strategy:
  1. Run Gated Hybrid -> get fused output
  2. Find dissonance pixels (S1 argmax != S2 argmax)
  3. For each dissonance pixel, branch: assign Top-1 vs Top-2 color
  4. Re-run NCA t*=3 (P201 optimal) with each branch
  5. Select branch with highest average margin across entire grid

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime
from itertools import product

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import load_arc_training, prepare_arc_meta_dataset
from phase199_gated import GatedHybridNCA
from phase191_generalization import ScalableNCA


def compute_margin(logits):
    """Average confidence margin: top1_prob - top2_prob across all pixels."""
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    top2 = probs.topk(2, dim=1).values  # (B, 2, H, W)
    margin = (top2[:, 0] - top2[:, 1]).mean()
    return margin.item()


def micro_beam_search(model, x, task_emb, oh, ow, max_branch_pixels=8, n_rerun_steps=3):
    """Cellular Micro-Beam Search on dissonance pixels.

    1. Forward pass to get S1, S2, fused output
    2. Identify dissonance pixels (up to max_branch_pixels)
    3. For small pixel counts: enumerate all 2^N assignments
       For large: use greedy sequential assignment
    4. Re-run NCA for each branch, pick highest margin
    """
    B, _, H, W = x.shape
    with torch.no_grad():
        output, s1_out, s2_out, gate = model(x, task_emb)

        s1_pred = s1_out[0, :, :oh, :ow].argmax(dim=0)  # (oh, ow)
        s2_pred = s2_out[0, :, :oh, :ow].argmax(dim=0)

        # Dissonance pixels
        dis_mask = (s1_pred != s2_pred)
        dis_coords = torch.nonzero(dis_mask, as_tuple=False)  # (N, 2)
        n_dis = dis_coords.shape[0]

        if n_dis == 0:
            # No dissonance - return as-is
            return output, 0, compute_margin(output[:, :, :oh, :ow])

        # Get fused prediction
        fused_pred = output[0, :, :oh, :ow].argmax(dim=0)

        # For each dissonance pixel, get top-1 and top-2 candidates
        fused_probs = F.softmax(output[0, :, :oh, :ow], dim=0)
        candidates = []
        for idx in range(min(n_dis, max_branch_pixels)):
            r, c = dis_coords[idx]
            pixel_probs = fused_probs[:, r, c]
            top2_vals, top2_idx = pixel_probs.topk(2)
            candidates.append((r.item(), c.item(), top2_idx[0].item(), top2_idx[1].item()))

        n_cands = len(candidates)

        # Branch enumeration (limit to 2^8 = 256 max)
        if n_cands <= max_branch_pixels:
            best_margin = -1
            best_pred = fused_pred.clone()
            n_branches = 2 ** n_cands

            for branch_id in range(n_branches):
                trial = fused_pred.clone()
                for bit, (r, c, c1, c2) in enumerate(candidates):
                    trial[r, c] = c1 if (branch_id >> bit) & 1 else c2

                # Convert back to one-hot, run brief NCA pass, measure margin
                trial_oh = F.one_hot(trial.long(), 11).permute(2, 0, 1).float().unsqueeze(0)
                trial_oh_padded = F.pad(trial_oh, (0, W - ow, 0, H - oh))
                # Quick NCA pass with n_rerun_steps
                te = task_emb.view(1, -1, 1, 1).expand(1, -1, H, W)
                inp = torch.cat([trial_oh_padded, te], dim=1)
                s1_rerun = model.s1(inp)
                margin = compute_margin(s1_rerun[:, :, :oh, :ow])

                if margin > best_margin:
                    best_margin = margin
                    best_pred = trial.clone()

            # Convert best_pred back to logits
            best_logits = F.one_hot(best_pred.long(), 11).permute(2, 0, 1).float().unsqueeze(0)
            result = F.pad(best_logits, (0, W - ow, 0, H - oh)) * 10  # scale to logit-like
            return result, n_branches, best_margin
        else:
            return output, 0, compute_margin(output[:, :, :oh, :ow])


def train_and_eval(model, train_tasks, test_tasks, n_epochs, label):
    """Standard training loop for Gated Hybrid."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist_pa, hist_em = [], []
    for epoch in range(n_epochs):
        model.train()
        random.shuffle(train_tasks)
        for item in train_tasks[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            tpa, tem = 0, 0
            with torch.no_grad():
                for item in test_tasks:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    out = model(ti, emb)
                    logits = out[0] if isinstance(out, tuple) else out
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
                    tem += float((pred == gt[:oh, :ow]).all().item())
            avg_pa = tpa / len(test_tasks)
            avg_em = tem / len(test_tasks)
            hist_pa.append(avg_pa); hist_em.append(avg_em)
            print(f"    {label} Ep{epoch+1}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%")
    return hist_pa, hist_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 204: Cellular Micro-Beam Search")
    print(f"  Branch on S1/S2 dissonance pixels, pick best-margin branch")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    C, ep = 64, 100

    # Train Gated Hybrid
    print(f"\n[Training Gated Hybrid]")
    model = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {model.count_params():,}")
    h_pa, h_em = train_and_eval(model, train, test, ep, "Gated")

    # Evaluate with Micro-Beam Search
    print(f"\n[Micro-Beam Search Evaluation]")
    model.eval()
    beam_pa, beam_em = 0, 0
    total_branches, total_dissonance = 0, 0
    margins_baseline, margins_beam = [], []

    with torch.no_grad():
        for item in test:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            # Baseline (no beam)
            out_base = model(ti, emb)
            base_logits = out_base[0]
            base_pred = base_logits[0, :, :oh, :ow].argmax(dim=0)
            base_margin = compute_margin(base_logits[:, :, :oh, :ow])
            margins_baseline.append(base_margin)

            # Micro-Beam Search
            beam_logits, n_br, beam_margin = micro_beam_search(
                model, ti, emb, oh, ow, max_branch_pixels=6, n_rerun_steps=3
            )
            beam_pred = beam_logits[0, :, :oh, :ow].argmax(dim=0)
            total_branches += n_br
            margins_beam.append(beam_margin)

            # S1/S2 dissonance count
            s1_pred = out_base[1][0, :, :oh, :ow].argmax(dim=0)
            s2_pred = out_base[2][0, :, :oh, :ow].argmax(dim=0)
            n_dis = (s1_pred != s2_pred).sum().item()
            total_dissonance += n_dis

            beam_pa += (beam_pred == gt[:oh, :ow]).float().mean().item()
            beam_em += float((beam_pred == gt[:oh, :ow]).all().item())

    n_test = len(test)
    base_pa_final = h_pa[-1]
    base_em_final = h_em[-1]
    beam_pa_final = beam_pa / n_test
    beam_em_final = beam_em / n_test
    avg_dis = total_dissonance / n_test
    avg_br = total_branches / n_test

    print(f"\n  Results:")
    print(f"    Baseline:    PA={base_pa_final*100:.1f}%, EM={base_em_final*100:.1f}%")
    print(f"    Micro-Beam:  PA={beam_pa_final*100:.1f}%, EM={beam_em_final*100:.1f}%")
    print(f"    Avg dissonance pixels: {avg_dis:.1f}")
    print(f"    Avg branches explored: {avg_br:.1f}")
    print(f"    Margin: base={np.mean(margins_baseline):.3f}, beam={np.mean(margins_beam):.3f}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 204 Complete ({elapsed:.0f}s)")
    print(f"{'='*70}")

    del model; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase204_microbeam.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': {'pa': base_pa_final, 'em': base_em_final},
            'microbeam': {'pa': beam_pa_final, 'em': beam_em_final},
            'avg_dissonance_pixels': avg_dis,
            'avg_branches': avg_br,
            'margin_baseline': np.mean(margins_baseline),
            'margin_beam': np.mean(margins_beam),
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        labels = ['Baseline', 'Micro-Beam']
        pa_vals = [base_pa_final*100, beam_pa_final*100]
        em_vals = [base_em_final*100, beam_em_final*100]
        x = np.arange(2); w = 0.35
        axes[0].bar(x-w/2, pa_vals, w, color=['#3498db','#e74c3c'], alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=['#3498db','#e74c3c'], alpha=0.4,
                   hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
        axes[0].set_ylabel('%'); axes[0].set_title('PA & EM: Baseline vs Micro-Beam', fontweight='bold')
        axes[0].legend()

        m_base = np.mean(margins_baseline)
        m_beam = np.mean(margins_beam)
        axes[1].bar(['Baseline', 'Micro-Beam'], [m_base, m_beam],
                    color=['#3498db','#e74c3c'], alpha=0.85)
        axes[1].set_ylabel('Avg Margin')
        axes[1].set_title('Confidence Margin', fontweight='bold')

        # Training curve
        epochs = [20*(i+1) for i in range(len(h_pa))]
        axes[2].plot(epochs, [v*100 for v in h_pa], 'o-', color='#3498db', lw=2)
        axes[2].axhline(y=beam_pa_final*100, color='#e74c3c', ls='--', lw=2, label='Micro-Beam')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Test PA (%)')
        axes[2].set_title('Training + Beam PA', fontweight='bold')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 204: Cellular Micro-Beam Search', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase204_microbeam.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'baseline_pa': base_pa_final, 'beam_pa': beam_pa_final}


if __name__ == '__main__':
    main()
