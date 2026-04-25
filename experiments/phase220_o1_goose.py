"""
Phase 220: The O1-Goose - Ultimate Test-Time Compute Pipeline

Combines ALL successful techniques into one meta-pipeline:
1. TTT-LoRA (P218): Adapt synapses to the task
2. Sampling (P209): Generate N diverse candidates
3. Hill Climbing (P219): Repair uncertain pixels discretely
4. Self-Consistency (P214): Pick the most converged answer

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
from collections import Counter
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
from phase199_gated import GatedHybridNCA
from phase205_critic import ArcAutoEncoder, train_autoencoder
from phase218_gated_ttt import add_lora_to_gated, get_lora_params, reset_lora, ttt_lora
from phase219_hill_climb import hill_climb


def grid_to_hashable(pred_tensor):
    return tuple(pred_tensor.cpu().numpy().flatten().tolist())


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 220: The O1-Goose")
    print(f"  TTT-LoRA + Sampling + Hill Climbing + Self-Consistency")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    has_di = 'demo_inputs' in train[0] if train else False

    # Train GatedHybrid
    print(f"\n[Training GatedHybridNCA]")
    torch.manual_seed(SEED)
    model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        model.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  Params: {model.count_params():,}")

    # Add LoRA
    model = add_lora_to_gated(model, rank=8)
    model = model.to(DEVICE)

    # Train AE
    print(f"\n[Training AE Critic]")
    output_grids = []
    for item in train:
        output_grids.append(item['test_output'][:11])
        for do in item['demo_outputs']:
            output_grids.append(do)
    ae = ArcAutoEncoder(11, 32).to(DEVICE)
    train_autoencoder(ae, output_grids, n_epochs=200, lr=1e-3)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # Evaluate: compare pipelines
    N = 20  # candidates per task
    noise_scale = 0.3

    # Pipeline configs to test
    pipelines = {
        'greedy': False,         # No TTT, no HC, no SC
        'ttt_only': True,        # TTT only
        'ttt+sample+sc': True,   # TTT + N samples + Self-Consistency
        'full_o1': True,         # TTT + N samples + HC + Self-Consistency
    }

    results = {k: {'pa': 0, 'em': 0} for k in pipelines}

    print(f"\n[Evaluating O1-Goose Pipeline (N={N})]")

    for tidx, item in enumerate(test):
        do_t = [d.to(DEVICE) for d in item['demo_outputs']]
        gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
        oh, ow = item['out_h'], item['out_w']
        ti = item['test_input'].unsqueeze(0).to(DEVICE)

        # --- Greedy baseline (no TTT) ---
        reset_lora(model)
        model.eval()
        with torch.no_grad():
            emb = model.encode_task(do_t)
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits[0, :, :oh, :ow].argmax(dim=0)
            pa = (pred == gt[:oh, :ow]).float().mean().item()
            em = float((pred == gt[:oh, :ow]).all().item())
            results['greedy']['pa'] += pa
            results['greedy']['em'] += em

        # --- TTT step ---
        reset_lora(model)
        if has_di:
            model.train()
            ttt_lora(model, item, n_steps=30, lr=0.01)
        model.eval()

        with torch.no_grad():
            emb = model.encode_task(do_t)
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            ttt_pred = logits[0, :, :oh, :ow].argmax(dim=0)
            pa = (ttt_pred == gt[:oh, :ow]).float().mean().item()
            em = float((ttt_pred == gt[:oh, :ow]).all().item())
            results['ttt_only']['pa'] += pa
            results['ttt_only']['em'] += em

        # --- TTT + N samples + Self-Consistency ---
        candidates_raw = []
        candidates_hc = []
        with torch.no_grad():
            for trial in range(N):
                out = model(ti, emb)
                trial_logits = out[0] if isinstance(out, tuple) else out
                if trial > 0:
                    gumbel = -torch.log(-torch.log(torch.rand_like(trial_logits)+1e-8)+1e-8)
                    trial_logits = trial_logits + noise_scale * gumbel
                trial_logits_crop = trial_logits[0, :, :oh, :ow]
                trial_pred = trial_logits_crop.argmax(dim=0)
                candidates_raw.append(trial_pred)

                # Hill Climb this candidate
                hc_pred = hill_climb(trial_pred, trial_logits_crop, ae,
                                    max_pixels=5, max_iters=3)
                candidates_hc.append(hc_pred)

        # Self-Consistency on raw candidates
        raw_hashes = [grid_to_hashable(c) for c in candidates_raw]
        raw_counts = Counter(raw_hashes)
        best_hash = raw_counts.most_common(1)[0][0]
        for c, h in zip(candidates_raw, raw_hashes):
            if h == best_hash:
                pa = (c == gt[:oh, :ow]).float().mean().item()
                em = float((c == gt[:oh, :ow]).all().item())
                results['ttt+sample+sc']['pa'] += pa
                results['ttt+sample+sc']['em'] += em
                break

        # Self-Consistency on HC candidates (full O1 pipeline)
        hc_hashes = [grid_to_hashable(c) for c in candidates_hc]
        hc_counts = Counter(hc_hashes)
        best_hc_hash = hc_counts.most_common(1)[0][0]
        for c, h in zip(candidates_hc, hc_hashes):
            if h == best_hc_hash:
                pa = (c == gt[:oh, :ow]).float().mean().item()
                em = float((c == gt[:oh, :ow]).all().item())
                results['full_o1']['pa'] += pa
                results['full_o1']['em'] += em
                break

        if (tidx + 1) % 10 == 0:
            print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    for k in results:
        results[k]['pa'] /= n_test
        results[k]['em'] /= n_test

    print(f"\n{'='*70}")
    print(f"  THE O1-GOOSE PIPELINE (N={N}):")
    for k, r in results.items():
        print(f"  {k:20s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model, ae; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase220_o1_goose.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'N': N,
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        labels = list(results.keys())
        pa_vals = [results[k]['pa']*100 for k in labels]
        em_vals = [results[k]['em']*100 for k in labels]
        colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('%'); ax.set_title('Phase 220: The O1-Goose Pipeline', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase220_o1_goose.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
