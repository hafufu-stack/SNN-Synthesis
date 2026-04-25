"""
Phase 207: Unsupervised AE-TTCT - Test-Time Training with Aesthetic Critic

Integrate P205's AE Critic into TTCT (Test-Time Compute Training).
At test time, optimize task embedding using:
  Loss = demo_loss + lambda * ae_reconstruction_loss(test_output)

The AE loss provides gradient signal even on unseen test tasks,
allowing self-correction toward "beautiful" ARC outputs.

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
from phase205_critic import ArcAutoEncoder, train_autoencoder


def ttct_standard(model, item, n_steps=30, lr=0.01):
    """Standard TTCT: optimize task embedding on demo examples only."""
    model.eval()
    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
    emb = model.encode_task(do_t).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=lr)

    # Demo inputs/outputs for TTCT
    di_t = [d.to(DEVICE) for d in item['demo_inputs']] if 'demo_inputs' in item else []
    do_gt = [d[:11].argmax(dim=0).to(DEVICE) for d in item['demo_outputs']]
    do_oh = [item.get('demo_out_h', [item['out_h']])] if 'demo_out_h' in item else None
    do_ow = [item.get('demo_out_w', [item['out_w']])] if 'demo_out_w' in item else None

    for step in range(n_steps):
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for di, dgt in zip(di_t, do_gt):
            doh, dow = dgt.shape
            logits = model(di.unsqueeze(0), emb)
            logits = logits[0] if isinstance(logits, tuple) else logits
            total_loss = total_loss + F.cross_entropy(logits[:, :, :doh, :dow], dgt.unsqueeze(0))
        if total_loss.item() > 0:
            opt.zero_grad(); total_loss.backward(); opt.step()

    return emb.detach()


def ttct_with_ae(model, ae, item, n_steps=30, lr=0.01, ae_weight=0.1):
    """AE-TTCT: optimize task embedding using demo loss + AE critic on test output."""
    model.eval(); ae.eval()
    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
    emb = model.encode_task(do_t).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([emb], lr=lr)

    di_t = [d.to(DEVICE) for d in item['demo_inputs']] if 'demo_inputs' in item else []
    do_gt = [d[:11].argmax(dim=0).to(DEVICE) for d in item['demo_outputs']]
    ti = item['test_input'].unsqueeze(0).to(DEVICE)
    oh, ow = item['out_h'], item['out_w']

    for step in range(n_steps):
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        # Demo loss
        for di, dgt in zip(di_t, do_gt):
            doh, dow = dgt.shape
            logits = model(di.unsqueeze(0), emb)
            logits = logits[0] if isinstance(logits, tuple) else logits
            total_loss = total_loss + F.cross_entropy(logits[:, :, :doh, :dow], dgt.unsqueeze(0))

        # AE critic loss on test output (unsupervised signal!)
        test_out = model(ti, emb)
        test_logits = test_out[0] if isinstance(test_out, tuple) else test_out
        test_soft = F.softmax(test_logits[:, :, :oh, :ow], dim=1)
        ae_err = ae.reconstruction_error(test_soft)
        total_loss = total_loss + ae_weight * ae_err.mean()

        if total_loss.item() > 0:
            opt.zero_grad(); total_loss.backward(); opt.step()

    return emb.detach()


def train_foundation(model, train_tasks, n_epochs=100):
    """Train foundation model."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
        if (epoch + 1) % 50 == 0:
            print(f"    Foundation Ep{epoch+1}")
    return model


def evaluate(model, test_tasks, label, emb_fn=None):
    """Evaluate with optional TTCT embedding function.
    Note: emb_fn (TTCT) needs gradients, so we can't wrap in no_grad."""
    model.eval()
    tpa, tem = 0, 0
    for item in test_tasks:
        do_t = [d.to(DEVICE) for d in item['demo_outputs']]
        if emb_fn:
            # TTCT needs gradients internally, call outside no_grad
            emb = emb_fn(item)
        else:
            with torch.no_grad():
                emb = model.encode_task(do_t)
        with torch.no_grad():
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits[0, :, :oh, :ow].argmax(dim=0)
            tpa += (pred == gt[:oh, :ow]).float().mean().item()
            tem += float((pred == gt[:oh, :ow]).all().item())
    pa = tpa / len(test_tasks)
    em = tem / len(test_tasks)
    print(f"    {label}: PA={pa*100:.1f}%, EM={em*100:.1f}%")
    return pa, em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 207: Unsupervised AE-TTCT")
    print(f"  AE Critic as test-time learning signal")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Step 1: Train AE on output grids
    print(f"\n[Step 1: Train AE Critic]")
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
    print(f"  AE trained ({sum(p.numel() for p in ae.parameters()):,} params)")

    # Step 2: Train foundation Gated Hybrid
    print(f"\n[Step 2: Train Foundation Model]")
    model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    print(f"  Params: {model.count_params():,}")
    model = train_foundation(model, train, n_epochs=100)

    # Step 3: Evaluate without TTCT
    print(f"\n[Step 3: Evaluation]")
    pa_base, em_base = evaluate(model, test, "No TTCT")

    # Step 4: Evaluate with standard TTCT
    print(f"\n[Step 4: Standard TTCT]")
    # Need demo_inputs in items - check if available
    has_di = 'demo_inputs' in test[0] if test else False
    if has_di:
        def std_ttct_fn(item):
            return ttct_standard(model, item, n_steps=20)
        pa_ttct, em_ttct = evaluate(model, test, "Standard TTCT", std_ttct_fn)
    else:
        print("    No demo_inputs available, skipping standard TTCT")
        pa_ttct, em_ttct = pa_base, em_base

    # Step 5: Evaluate with AE-TTCT (different weights)
    results_ae = {}
    for w in [0.05, 0.1, 0.3]:
        print(f"\n[Step 5: AE-TTCT (w={w})]")
        if has_di:
            def ae_ttct_fn(item, weight=w):
                return ttct_with_ae(model, ae, item, n_steps=20, ae_weight=weight)
            pa_ae, em_ae = evaluate(model, test, f"AE-TTCT(w={w})", ae_ttct_fn)
        else:
            # Simplified: use AE loss on test output to refine embedding
            def ae_only_fn(item, weight=w):
                return ttct_with_ae(model, ae, item, n_steps=20, ae_weight=weight)
            pa_ae, em_ae = evaluate(model, test, f"AE-TTCT(w={w})", ae_only_fn)
        results_ae[f'w{w}'] = {'pa': pa_ae, 'em': em_ae}

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 207 Complete ({elapsed:.0f}s)")
    print(f"  Baseline:     PA={pa_base*100:.1f}%, EM={em_base*100:.1f}%")
    print(f"  Standard TTCT: PA={pa_ttct*100:.1f}%, EM={em_ttct*100:.1f}%")
    for k, v in results_ae.items():
        print(f"  AE-TTCT({k}): PA={v['pa']*100:.1f}%, EM={v['em']*100:.1f}%")
    print(f"{'='*70}")

    del model, ae; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase207_ae_ttct.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': {'pa': pa_base, 'em': em_base},
            'standard_ttct': {'pa': pa_ttct, 'em': em_ttct},
            'ae_ttct': results_ae,
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = ['Base', 'TTCT'] + [f'AE({k})' for k in results_ae]
        pa_vals = [pa_base*100, pa_ttct*100] + [v['pa']*100 for v in results_ae.values()]
        em_vals = [em_base*100, em_ttct*100] + [v['em']*100 for v in results_ae.values()]
        x = np.arange(len(labels)); w = 0.35
        colors = ['#95a5a6', '#3498db'] + ['#e67e22', '#e74c3c', '#2ecc71'][:len(results_ae)]
        axes[0].bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel('%'); axes[0].set_title('PA & EM Comparison', fontweight='bold')
        axes[0].legend()

        # PA improvement bar
        improvements = [0, (pa_ttct-pa_base)*100] + [(v['pa']-pa_base)*100 for v in results_ae.values()]
        axes[1].bar(labels, improvements, color=colors, alpha=0.85)
        axes[1].axhline(y=0, color='black', lw=0.5)
        axes[1].set_ylabel('PA Change (pp)'); axes[1].set_title('PA Improvement vs Baseline', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        fig.suptitle('Phase 207: Unsupervised AE-TTCT', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase207_ae_ttct.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'baseline_pa': pa_base, 'ae_ttct': results_ae}


if __name__ == '__main__':
    main()
