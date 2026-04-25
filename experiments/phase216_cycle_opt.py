"""
Phase 216: Cycle-Consistency Optimization - Inverse NCA as Optimizer

Use Backward NCA's reconstruction error as gradient signal to
directly optimize output candidates until they are "reversible."

"If the answer is correct, the inverse model can perfectly
reconstruct the question. Optimize the answer until this is true."

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
from phase213_cycle import InverseNCA


def cycle_optimize(logits_crop, inv_model, inv_emb, test_input_oh, ih, iw, oh, ow,
                   n_steps=50, lr=0.05):
    """Optimize output logits to minimize cycle-consistency error."""
    opt_logits = logits_crop.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_logits], lr=lr)

    maxH = max(ih, oh); maxW = max(iw, ow)

    for step in range(n_steps):
        soft = F.softmax(opt_logits, dim=1)
        # Pad for inverse model
        soft_pad = F.pad(soft, (0, maxW - ow, 0, maxH - oh))
        recon = inv_model(soft_pad, inv_emb)
        # Reconstruction error against true input
        cycle_loss = ((recon[:, :, :ih, :iw] - test_input_oh[:, :, :ih, :iw]) ** 2).mean()
        optimizer.zero_grad()
        cycle_loss.backward()
        optimizer.step()

    return opt_logits.detach()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 216: Cycle-Consistency Optimization")
    print(f"  Optimize output via inverse NCA reconstruction error")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    has_di = 'demo_inputs' in train[0] if train else False

    # Train Forward NCA
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

    # Train Inverse NCA
    print(f"\n[Training Inverse NCA]")
    inv_model = InverseNCA(11, 64, 32).to(DEVICE)
    opt_i = torch.optim.Adam(inv_model.parameters(), lr=1e-3)
    for epoch in range(100):
        inv_model.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            if has_di:
                di_t = [d.to(DEVICE) for d in item['demo_inputs']]
                inv_emb = inv_model.encode_task(di_t)
            else:
                inv_emb = inv_model.encode_task(do_t)
            test_out = item['test_output'][:11].unsqueeze(0).to(DEVICE)
            test_in_gt = item['test_input'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            ih, iw = test_in_gt.shape[1], test_in_gt.shape[2]
            oh, ow = item['out_h'], item['out_w']
            maxH = max(ih, oh); maxW = max(iw, ow)
            test_out_pad = F.pad(test_out, (0, maxW - ow, 0, maxH - oh))
            pred_input = inv_model(test_out_pad, inv_emb)
            loss = F.cross_entropy(pred_input[:, :, :ih, :iw], test_in_gt[:, :ih, :iw])
            opt_i.zero_grad(); loss.backward(); opt_i.step()
    print(f"  Inverse NCA Params: {inv_model.count_params():,}")
    # Freeze inverse model
    inv_model.eval()
    for p in inv_model.parameters():
        p.requires_grad = False

    # Evaluate
    repair_configs = [10, 30, 50]
    print(f"\n[Evaluating Cycle-Consistency Optimization]")
    fwd_model.eval()

    greedy_pa, greedy_em = 0, 0
    repair_results = {n: {'pa': 0, 'em': 0} for n in repair_configs}

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
        test_in_oh = item['test_input'][:11].unsqueeze(0).to(DEVICE)
        ih, iw = test_in_oh.shape[2], test_in_oh.shape[3]

        with torch.no_grad():
            logits = fwd_model(ti, fwd_emb)
            pred = logits[0, :, :oh, :ow].argmax(dim=0)
            greedy_pa += (pred == gt[:oh, :ow]).float().mean().item()
            greedy_em += float((pred == gt[:oh, :ow]).all().item())

        for n_steps in repair_configs:
            repaired = cycle_optimize(
                logits[:, :, :oh, :ow], inv_model, inv_emb,
                test_in_oh, ih, iw, oh, ow,
                n_steps=n_steps, lr=0.05
            )
            pred_r = repaired[0].argmax(dim=0)
            repair_results[n_steps]['pa'] += (pred_r == gt[:oh, :ow]).float().mean().item()
            repair_results[n_steps]['em'] += float((pred_r == gt[:oh, :ow]).all().item())

        if (tidx + 1) % 10 == 0:
            print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    greedy_pa /= n_test; greedy_em /= n_test
    for n in repair_configs:
        repair_results[n]['pa'] /= n_test
        repair_results[n]['em'] /= n_test

    print(f"\n{'='*70}")
    print(f"  CYCLE-CONSISTENCY OPTIMIZATION:")
    print(f"  Greedy:      PA={greedy_pa*100:.1f}%, EM={greedy_em*100:.1f}%")
    for n in repair_configs:
        r = repair_results[n]
        print(f"  CycleOpt({n:3d}): PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del fwd_model, inv_model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase216_cycle_opt.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'greedy': {'pa': greedy_pa, 'em': greedy_em},
            'repair': {str(k): v for k, v in repair_results.items()},
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = ['Greedy'] + [f'Cycle({n})' for n in repair_configs]
        pa_vals = [greedy_pa*100] + [repair_results[n]['pa']*100 for n in repair_configs]
        em_vals = [greedy_em*100] + [repair_results[n]['em']*100 for n in repair_configs]
        colors = ['#95a5a6','#9b59b6','#8e44ad','#6c3483']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('%'); ax.set_title('Phase 216: Cycle-Consistency Optimization', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase216_cycle_opt.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'greedy_pa': greedy_pa, 'repair': repair_results}


if __name__ == '__main__':
    main()
