"""
Phase 213: Backward Cycle-Consistency - Inverse NCA Verification

Train an Inverse NCA (Output -> Input) and use reconstruction
error to verify forward NCA candidates.

"If the candidate is correct, the inverse model should perfectly
reconstruct the original input."

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
from phase206_causal_volume import ParametricNCA


class InverseNCA(nn.Module):
    """NCA that maps Output -> Input (backward direction)."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32):
        super().__init__()
        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )
        self.net = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, hidden_ch, 1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1), nn.ReLU(),
            nn.Conv2d(hidden_ch, n_colors, 1),
        )

    def encode_task(self, demo_inputs):
        """Encode from demo INPUT images (reverse direction)."""
        embeddings = []
        for di in demo_inputs:
            emb = self.demo_encoder(di.unsqueeze(0))
            embeddings.append(emb)
        return torch.stack(embeddings).mean(dim=0)

    def forward(self, output_grid, task_emb):
        B, _, H, W = output_grid.shape
        te = task_emb.view(B, -1, 1, 1).expand(B, -1, H, W)
        inp = torch.cat([output_grid, te], dim=1)
        return self.net(inp)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 213: Backward Cycle-Consistency")
    print(f"  Inverse NCA (Output->Input) for verification")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    has_di = 'demo_inputs' in train[0] if train else False
    if not has_di:
        print("  WARNING: No demo_inputs found. Using output-based encoding.")

    # Train forward NCA (K=1, C=64, T=1)
    print(f"\n[Training Forward NCA]")
    torch.manual_seed(SEED)
    fwd_model = ParametricNCA(11, 64, 1, 1, 32).to(DEVICE)
    opt_f = torch.optim.Adam(fwd_model.parameters(), lr=1e-3)
    for epoch in range(100):
        fwd_model.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = fwd_model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            logits = fwd_model(ti, emb)
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt_f.zero_grad(); loss.backward(); opt_f.step()
    print(f"  Forward NCA Params: {fwd_model.count_params():,}")

    # Train Inverse NCA (Output -> Input)
    print(f"\n[Training Inverse NCA]")
    inv_model = InverseNCA(11, 64, 32).to(DEVICE)
    opt_i = torch.optim.Adam(inv_model.parameters(), lr=1e-3)
    for epoch in range(100):
        inv_model.train(); random.shuffle(train)
        for item in train[:50]:
            # Inverse: predict INPUT from OUTPUT
            # Use demo_outputs as encoder source (task ID)
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            if has_di:
                di_t = [d.to(DEVICE) for d in item['demo_inputs']]
                inv_emb = inv_model.encode_task(di_t)
            else:
                inv_emb = inv_model.encode_task(do_t)

            # Feed output, predict input
            test_out = item['test_output'][:11].unsqueeze(0).to(DEVICE)
            test_in_gt = item['test_input'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            ih, iw = item.get('in_h', test_in_gt.shape[1]), item.get('in_w', test_in_gt.shape[2])
            oh, ow = item['out_h'], item['out_w']

            # Pad output to match input spatial size if needed
            maxH = max(ih, oh); maxW = max(iw, ow)
            test_out_pad = F.pad(test_out, (0, maxW - ow, 0, maxH - oh))
            pred_input = inv_model(test_out_pad, inv_emb)
            loss = F.cross_entropy(pred_input[:, :, :ih, :iw], test_in_gt[:, :ih, :iw])
            opt_i.zero_grad(); loss.backward(); opt_i.step()
        if (epoch + 1) % 50 == 0:
            print(f"    Inverse NCA Ep{epoch+1}")
    print(f"  Inverse NCA Params: {inv_model.count_params():,}")

    # Generate N=100 candidates & evaluate selectors
    N = 100
    noise_scale = 0.3
    print(f"\n[Generating {N} candidates & cycle-consistency check]")

    fwd_model.eval(); inv_model.eval()
    greedy_pa, greedy_em = 0, 0
    oracle_pa, oracle_em = 0, 0
    cycle_pa, cycle_em = 0, 0
    margin_pa, margin_em = 0, 0

    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            fwd_emb = fwd_model.encode_task(do_t)
            if has_di:
                di_t = [d.to(DEVICE) for d in item['demo_inputs']]
                inv_emb = inv_model.encode_task(di_t)
            else:
                inv_emb = inv_model.encode_task(do_t)

            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            test_in_gt = item['test_input'][:11].to(DEVICE)  # one-hot input
            ih, iw = test_in_gt.shape[1], test_in_gt.shape[2]

            candidates = []
            for trial in range(N):
                logits = fwd_model(ti, fwd_emb)
                if trial > 0:
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                    logits = logits + noise_scale * gumbel

                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                pa = (pred == gt[:oh, :ow]).float().mean().item()
                em = (pred == gt[:oh, :ow]).all().item()

                # Cycle-consistency: Output -> Inverse NCA -> predicted Input
                cand_oh = F.one_hot(pred.long(), 11).permute(2, 0, 1).float().unsqueeze(0)
                maxH = max(ih, oh); maxW = max(iw, ow)
                cand_pad = F.pad(cand_oh, (0, maxW - ow, 0, maxH - oh))
                recon_input = inv_model(cand_pad, inv_emb)
                recon_pred = recon_input[0, :, :ih, :iw]
                # MSE between reconstructed input and actual input
                cycle_err = ((recon_pred - test_in_gt[:, :ih, :iw]) ** 2).mean().item()

                # Margin
                probs = F.softmax(logits[0, :, :oh, :ow], dim=0)
                top2 = probs.topk(2, dim=0).values
                margin = (top2[0] - top2[1]).mean().item()

                candidates.append((pa, em, cycle_err, margin))

            greedy_pa += candidates[0][0]; greedy_em += candidates[0][1]
            best = max(candidates, key=lambda c: c[0])
            oracle_pa += best[0]; oracle_em += best[1]
            # Cycle: LOWEST reconstruction error
            cyc_best = min(candidates, key=lambda c: c[2])
            cycle_pa += cyc_best[0]; cycle_em += cyc_best[1]
            mar_best = max(candidates, key=lambda c: c[3])
            margin_pa += mar_best[0]; margin_em += mar_best[1]

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    results = {
        'greedy': {'pa': greedy_pa/n_test, 'em': greedy_em/n_test},
        'oracle': {'pa': oracle_pa/n_test, 'em': oracle_em/n_test},
        'cycle': {'pa': cycle_pa/n_test, 'em': cycle_em/n_test},
        'margin': {'pa': margin_pa/n_test, 'em': margin_em/n_test},
    }
    vgap_cyc = (oracle_pa - cycle_pa) / n_test * 100
    vgap_mar = (oracle_pa - margin_pa) / n_test * 100

    print(f"\n{'='*70}")
    print(f"  BACKWARD CYCLE-CONSISTENCY (N={N}):")
    print(f"  Greedy: PA={results['greedy']['pa']*100:.1f}%, EM={results['greedy']['em']*100:.1f}%")
    print(f"  Oracle: PA={results['oracle']['pa']*100:.1f}%, EM={results['oracle']['em']*100:.1f}%")
    print(f"  Cycle:  PA={results['cycle']['pa']*100:.1f}%, EM={results['cycle']['em']*100:.1f}%")
    print(f"  Margin: PA={results['margin']['pa']*100:.1f}%, EM={results['margin']['em']*100:.1f}%")
    print(f"  V-Gap Cycle: {vgap_cyc:.1f}pp  |  V-Gap Margin: {vgap_mar:.1f}pp")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del fwd_model, inv_model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase213_cycle.json"), 'w', encoding='utf-8') as f:
        json.dump({'N': N, 'results': results,
                   'vgap_cycle': vgap_cyc, 'vgap_margin': vgap_mar,
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ['Greedy', 'Oracle', 'Cycle\nConsist.', 'Margin']
        pa_vals = [results[k]['pa']*100 for k in ['greedy','oracle','cycle','margin']]
        em_vals = [results[k]['em']*100 for k in ['greedy','oracle','cycle','margin']]
        colors = ['#95a5a6', '#2ecc71', '#9b59b6', '#3498db']
        x = np.arange(4); w = 0.35
        axes[0].bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
        axes[0].set_ylabel('%'); axes[0].set_title('Selection Comparison', fontweight='bold')
        axes[0].legend()

        gaps = [0, 0, vgap_cyc, vgap_mar]
        axes[1].bar(labels, gaps, color=colors, alpha=0.85)
        axes[1].set_ylabel('Gap from Oracle (pp)')
        axes[1].set_title('Verification Gap', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        fig.suptitle('Phase 213: Backward Cycle-Consistency', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase213_cycle.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
