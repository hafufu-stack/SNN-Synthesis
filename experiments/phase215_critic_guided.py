"""
Phase 215: Critic-Guided Generation - AE Gradient Repair

Use AE Critic's gradient to directly optimize output logits,
removing "stains" (non-ARC-like artifacts) from generated images.

Like Classifier-Guided Diffusion / DeepDream for ARC grids.

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
from phase205_critic import ArcAutoEncoder, train_autoencoder


def critic_guided_repair(logits, ae, n_steps=50, lr=0.1):
    """Optimize output logits to minimize AE reconstruction error.
    
    Args:
        logits: (1, C, H, W) raw output logits from NCA
        ae: trained AE critic (frozen)
        n_steps: gradient descent steps
        lr: learning rate for logit optimization
    Returns:
        optimized logits
    """
    # Detach and make optimizable
    opt_logits = logits.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_logits], lr=lr)

    for step in range(n_steps):
        soft = F.softmax(opt_logits, dim=1)
        recon_err = ae.reconstruction_error(soft)
        loss = recon_err.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return opt_logits.detach()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 215: Critic-Guided Generation")
    print(f"  AE gradient repair of output logits")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train NCA
    print(f"\n[Training NCA (K=1, C=64, T=1)]")
    torch.manual_seed(SEED)
    model = ParametricNCA(11, 64, 1, 1, 32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        model.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            logits = model(ti, emb)
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  NCA Params: {model.count_params():,}")

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

    # Evaluate: Greedy vs Critic-Guided Repair
    repair_configs = [10, 30, 50, 100]
    print(f"\n[Evaluating Critic-Guided Repair]")
    model.eval()

    greedy_pa, greedy_em = 0, 0
    repair_results = {n: {'pa': 0, 'em': 0} for n in repair_configs}

    for tidx, item in enumerate(test):
        do_t = [d.to(DEVICE) for d in item['demo_outputs']]
        emb = model.encode_task(do_t)
        ti = item['test_input'].unsqueeze(0).to(DEVICE)
        gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
        oh, ow = item['out_h'], item['out_w']

        with torch.no_grad():
            logits = model(ti, emb)
            pred = logits[0, :, :oh, :ow].argmax(dim=0)
            greedy_pa += (pred == gt[:oh, :ow]).float().mean().item()
            greedy_em += float((pred == gt[:oh, :ow]).all().item())

        # Critic-guided repair with different step counts
        for n_steps in repair_configs:
            repaired = critic_guided_repair(
                logits[:, :, :oh, :ow], ae, n_steps=n_steps, lr=0.05
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
    print(f"  CRITIC-GUIDED GENERATION:")
    print(f"  Greedy:      PA={greedy_pa*100:.1f}%, EM={greedy_em*100:.1f}%")
    for n in repair_configs:
        r = repair_results[n]
        print(f"  Repair({n:3d}): PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model, ae; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase215_critic_guided.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'greedy': {'pa': greedy_pa, 'em': greedy_em},
            'repair': {str(k): v for k, v in repair_results.items()},
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ['Greedy'] + [f'Repair\n({n})' for n in repair_configs]
        pa_vals = [greedy_pa*100] + [repair_results[n]['pa']*100 for n in repair_configs]
        em_vals = [greedy_em*100] + [repair_results[n]['em']*100 for n in repair_configs]
        colors = ['#95a5a6'] + ['#e74c3c','#e67e22','#f1c40f','#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        axes[0].bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel('%'); axes[0].set_title('Critic-Guided Repair', fontweight='bold')
        axes[0].legend()

        steps = repair_configs
        pa_curve = [repair_results[n]['pa']*100 for n in steps]
        axes[1].plot(steps, pa_curve, 'o-', color='#e74c3c', lw=2, ms=8)
        axes[1].axhline(y=greedy_pa*100, color='#95a5a6', ls='--', label='Greedy')
        axes[1].set_xlabel('Repair Steps'); axes[1].set_ylabel('PA (%)')
        axes[1].set_title('PA vs Repair Steps', fontweight='bold')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        fig.suptitle('Phase 215: Critic-Guided Generation', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase215_critic_guided.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'greedy_pa': greedy_pa, 'repair': repair_results}


if __name__ == '__main__':
    main()
