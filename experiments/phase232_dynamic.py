"""
Phase 232: Equation-Driven Dynamic Compute Allocation

Use the scaling law to predict per-task difficulty, then allocate
compute (N trials) dynamically: easy tasks get few trials,
hard tasks get many.

"Spend your budget where it matters most."

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
from phase199_gated import GatedHybridNCA


def estimate_difficulty(logits, oh, ow):
    """Estimate task difficulty from initial inference margin.
    Low margin = hard (model is uncertain).
    High margin = easy (model is confident).
    Returns difficulty score in [0, 1] (higher = harder).
    """
    probs = F.softmax(logits[:, :, :oh, :ow], dim=1)
    top2 = probs.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).mean().item()
    # Invert: low margin -> high difficulty
    difficulty = 1.0 - min(1.0, margin)
    return difficulty


def dynamic_allocate(difficulties, total_budget, min_N=5, max_N=500):
    """Allocate N trials per task proportional to difficulty.
    Total trials across all tasks <= total_budget.
    """
    n_tasks = len(difficulties)
    # Normalize difficulties
    d_arr = np.array(difficulties)
    d_arr = np.clip(d_arr, 0.01, 1.0)
    # Allocate proportional to difficulty
    raw_alloc = d_arr / d_arr.sum() * total_budget
    allocations = np.clip(raw_alloc, min_N, max_N).astype(int)
    # Adjust to fit budget
    while allocations.sum() > total_budget:
        idx = allocations.argmax()
        allocations[idx] = max(min_N, allocations[idx] - 10)
    return allocations


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 232: Equation-Driven Dynamic Compute Allocation")
    print(f"  Allocate trials per task based on difficulty")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train model
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
    model.eval()

    # Step 1: Estimate difficulty for each test task
    print(f"\n[Estimating task difficulties]")
    difficulties = []
    noise_scale = 0.3

    with torch.no_grad():
        for item in test:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            diff = estimate_difficulty(logits[0:1], oh, ow)
            difficulties.append(diff)

    diff_arr = np.array(difficulties)
    print(f"  Difficulty: mean={diff_arr.mean():.3f}, min={diff_arr.min():.3f}, max={diff_arr.max():.3f}")

    # Step 2: Compare fixed vs dynamic allocation
    total_budget = len(test) * 100  # Same total trials
    fixed_N = 100
    dynamic_allocs = dynamic_allocate(difficulties, total_budget, min_N=5, max_N=500)
    print(f"  Total budget: {total_budget} trials")
    print(f"  Fixed: {fixed_N}/task")
    print(f"  Dynamic: min={dynamic_allocs.min()}, max={dynamic_allocs.max()}, mean={dynamic_allocs.mean():.0f}")

    # Step 3: Evaluate both
    fixed_pa, fixed_em = 0, 0
    dynamic_pa, dynamic_em = 0, 0

    print(f"\n[Evaluating Fixed vs Dynamic allocation]")
    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            # Fixed N=100
            best_pa_f, best_em_f = 0, 0
            for trial in range(fixed_N):
                out = model(ti, emb)
                logits = out[0] if isinstance(out, tuple) else out
                if trial > 0:
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                    logits = logits + noise_scale * gumbel
                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                pa = (pred == gt[:oh, :ow]).float().mean().item()
                em = float((pred == gt[:oh, :ow]).all().item())
                if pa > best_pa_f:
                    best_pa_f = pa
                    best_em_f = em
            fixed_pa += best_pa_f; fixed_em += best_em_f

            # Dynamic N
            N_dyn = int(dynamic_allocs[tidx])
            best_pa_d, best_em_d = 0, 0
            for trial in range(N_dyn):
                out = model(ti, emb)
                logits = out[0] if isinstance(out, tuple) else out
                if trial > 0:
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                    logits = logits + noise_scale * gumbel
                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                pa = (pred == gt[:oh, :ow]).float().mean().item()
                em = float((pred == gt[:oh, :ow]).all().item())
                if pa > best_pa_d:
                    best_pa_d = pa
                    best_em_d = em
            dynamic_pa += best_pa_d; dynamic_em += best_em_d

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)} (N_dyn={N_dyn}, diff={difficulties[tidx]:.2f})")

    n_test = len(test)
    fixed_pa /= n_test; fixed_em /= n_test
    dynamic_pa /= n_test; dynamic_em /= n_test

    print(f"\n{'='*70}")
    print(f"  DYNAMIC COMPUTE ALLOCATION (budget={total_budget} trials):")
    print(f"  Fixed  (N=100): PA={fixed_pa*100:.1f}%, EM={fixed_em*100:.1f}%")
    print(f"  Dynamic       : PA={dynamic_pa*100:.1f}%, EM={dynamic_em*100:.1f}%")
    delta = (dynamic_pa - fixed_pa) * 100
    print(f"  Δ PA = {delta:+.1f}pp")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase232_dynamic.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'fixed': {'pa': fixed_pa, 'em': fixed_em, 'N': fixed_N},
            'dynamic': {'pa': dynamic_pa, 'em': dynamic_em,
                       'N_min': int(dynamic_allocs.min()), 'N_max': int(dynamic_allocs.max())},
            'difficulties': difficulties,
            'total_budget': total_budget, 'delta_pa': delta,
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Fixed vs Dynamic
        axes[0].bar([0, 1], [fixed_pa*100, dynamic_pa*100], color=['#95a5a6', '#2ecc71'], alpha=0.85)
        axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Fixed N=100', 'Dynamic'])
        axes[0].set_ylabel('PA (%)'); axes[0].set_title('PA Comparison', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Plot 2: Difficulty distribution
        axes[1].hist(difficulties, bins=20, color='#3498db', alpha=0.7)
        axes[1].set_xlabel('Difficulty'); axes[1].set_ylabel('Count')
        axes[1].set_title('Task Difficulty Distribution', fontweight='bold')

        # Plot 3: N allocation vs difficulty
        axes[2].scatter(difficulties, dynamic_allocs, c='#e74c3c', alpha=0.6, s=30)
        axes[2].set_xlabel('Difficulty'); axes[2].set_ylabel('Allocated N')
        axes[2].set_title('Dynamic N Allocation', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 232: Dynamic Compute Allocation', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.85, wspace=0.35)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase232_dynamic.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")

    gc.collect()
    return {'fixed_pa': fixed_pa, 'dynamic_pa': dynamic_pa, 'delta': delta}

if __name__ == '__main__':
    main()
