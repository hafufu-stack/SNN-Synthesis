"""
Phase 184: Proxy Confidence Sleep - Zero-Cost Margin-Based Sleep

P177/P179: Entropy sleep boosts PA by +2.26pp (anti-drift), but
Softmax+Log is computationally expensive (2x slowdown).
P179: State-delta doesn't work (Liquid NCA never stops moving).

New approach: Use logit MARGIN (top1 - top2) as confidence proxy.
No Softmax, no Log, just a subtraction on raw logits.
Nearly zero computational cost while preserving meaning-based sleep.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
EMB_DIM = 64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset
)


def inference_standard(model, x, task_embed, n_steps=8):
    """Standard inference."""
    return model.latent_nca(x, task_embed, n_steps=n_steps)


def inference_entropy_sleep(model, x, task_embed, n_steps=8, entropy_thresh=0.3):
    """Entropy-based sleep (P177 style, expensive)."""
    nca = model.latent_nca
    B, _, H, W = x.shape
    state = nca.encoder(x)
    te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)
    for step in range(n_steps):
        state_ctx = torch.cat([state, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        new_state = beta * state + (1 - beta) * delta
        logits = nca.decoder(new_state)
        probs = F.softmax(logits[:, :10], dim=1)
        pixel_entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
        awake = (pixel_entropy > entropy_thresh).float().unsqueeze(1).expand_as(state)
        state = awake * new_state + (1 - awake) * state
    return nca.decoder(state)


def inference_margin_sleep(model, x, task_embed, n_steps=8, margin_thresh=2.0):
    """Phase 184: Margin-based sleep (near-zero cost).
    
    Confidence = top1_logit - top2_logit (no softmax needed).
    If margin > threshold -> pixel is confident -> sleep.
    """
    nca = model.latent_nca
    B, _, H, W = x.shape
    state = nca.encoder(x)
    te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)
    sleep_ratio_hist = []

    for step in range(n_steps):
        state_ctx = torch.cat([state, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        new_state = beta * state + (1 - beta) * delta

        # Decode to get logits (cheap - just 2 conv layers)
        logits = nca.decoder(new_state)
        raw = logits[:, :10]  # (B, 10, H, W)

        # Margin: top1 - top2 (NO softmax, NO log!)
        sorted_logits, _ = raw.sort(dim=1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]  # (B, H, W)

        # Sleep: confident pixels (high margin) keep old state
        confident = (margin > margin_thresh).float().unsqueeze(1).expand_as(state)
        sleep_ratio_hist.append(confident[:, 0].mean().item())

        state = confident * state + (1 - confident) * new_state

    return nca.decoder(state), sleep_ratio_hist


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 184: Proxy Confidence Sleep - Margin-Based Zero-Cost Sleep")
    print(f"  Margin (top1-top2) as confidence proxy, no Softmax/Log")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Loading Foundation Model...")
    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    ckpt_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:50]

    # Benchmark
    print("\n[Step 3] Benchmarking Standard vs Entropy Sleep vs Margin Sleep...")
    margin_thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    methods = {
        'standard': {'pas': [], 'ems': [], 'time': 0},
        'entropy_sleep': {'pas': [], 'ems': [], 'time': 0},
    }
    for mt in margin_thresholds:
        methods[f'margin_{mt}'] = {'pas': [], 'ems': [], 'time': 0, 'sleep_ratios': []}

    with torch.no_grad():
        for item in test_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.task_encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            # Standard
            t1 = time.time()
            logits = inference_standard(model, ti, emb, n_steps=8)
            methods['standard']['time'] += time.time() - t1
            pred = logits[0, :10].argmax(dim=0)
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            methods['standard']['pas'].append(pa); methods['standard']['ems'].append(em)

            # Entropy sleep
            t1 = time.time()
            logits = inference_entropy_sleep(model, ti, emb, n_steps=8, entropy_thresh=0.3)
            methods['entropy_sleep']['time'] += time.time() - t1
            pred = logits[0, :10].argmax(dim=0)
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            methods['entropy_sleep']['pas'].append(pa); methods['entropy_sleep']['ems'].append(em)

            # Margin sleep variants
            for mt in margin_thresholds:
                key = f'margin_{mt}'
                t1 = time.time()
                logits, slp_hist = inference_margin_sleep(model, ti, emb, n_steps=8,
                                                          margin_thresh=mt)
                methods[key]['time'] += time.time() - t1
                pred = logits[0, :10].argmax(dim=0)
                pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
                em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
                methods[key]['pas'].append(pa); methods[key]['ems'].append(em)
                methods[key]['sleep_ratios'].append(np.mean(slp_hist))

    elapsed = time.time() - t0
    summary = {}
    print(f"\n{'='*70}")
    print(f"Phase 184 Complete ({elapsed:.0f}s)")
    std_time = methods['standard']['time']
    for name, m in methods.items():
        avg_pa = np.mean(m['pas']); avg_em = np.mean(m['ems'])
        speedup = std_time / max(0.001, m['time'])
        extra = ""
        if 'sleep_ratios' in m and m['sleep_ratios']:
            extra = f", Sleep={np.mean(m['sleep_ratios'])*100:.0f}%"
        summary[name] = {'pa': avg_pa, 'em': avg_em, 'time': m['time'], 'speedup': speedup}
        if 'sleep_ratios' in m:
            summary[name]['sleep_ratio'] = np.mean(m['sleep_ratios'])
        print(f"  {name:20s}: PA={avg_pa*100:.2f}%, Speed={speedup:.2f}x{extra}")

    # Key comparison
    ent_pa = summary['entropy_sleep']['pa']
    best_margin = max([(k, v) for k, v in summary.items() if k.startswith('margin_')],
                      key=lambda x: x[1]['pa'])
    print(f"\n  Entropy Sleep:  PA={ent_pa*100:.2f}%, Speed={summary['entropy_sleep']['speedup']:.2f}x")
    print(f"  Best Margin ({best_margin[0]}): PA={best_margin[1]['pa']*100:.2f}%, "
          f"Speed={best_margin[1]['speedup']:.2f}x")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase184_margin_sleep.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 184: Proxy Confidence Sleep',
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        # PA comparison
        keys = ['standard', 'entropy_sleep'] + [f'margin_{mt}' for mt in margin_thresholds]
        labels = ['Std', 'Entropy'] + [f'M={mt}' for mt in margin_thresholds]
        pa_vals = [summary[k]['pa']*100 for k in keys]
        colors = ['#95a5a6', '#e74c3c'] + ['#2ecc71']*len(margin_thresholds)
        axes[0].bar(range(len(labels)), pa_vals, color=colors, alpha=0.85, edgecolor='black')
        axes[0].set_xticks(range(len(labels))); axes[0].set_xticklabels(labels, fontsize=7, rotation=30)
        axes[0].set_ylabel('PA (%)'); axes[0].set_title('Pixel Accuracy', fontweight='bold')
        # Speed comparison
        speeds = [summary[k]['speedup'] for k in keys]
        axes[1].bar(range(len(labels)), speeds, color=colors, alpha=0.85, edgecolor='black')
        axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels, fontsize=7, rotation=30)
        axes[1].set_ylabel('Speedup (vs Std)'); axes[1].set_title('Speed', fontweight='bold')
        axes[1].axhline(1.0, color='black', linewidth=0.5)
        # Sleep ratio
        mrg_keys = [f'margin_{mt}' for mt in margin_thresholds]
        slp_ratios = [summary[k].get('sleep_ratio', 0)*100 for k in mrg_keys]
        axes[2].bar([f'{mt}' for mt in margin_thresholds], slp_ratios, color='#3498db', alpha=0.85)
        axes[2].set_xlabel('Margin Threshold'); axes[2].set_ylabel('Sleep Ratio (%)')
        axes[2].set_title('Pixels Sleeping', fontweight='bold')
        fig.suptitle('Phase 184: Margin-Based Sleep (Zero-Cost Confidence Proxy)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.18, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase184_margin_sleep.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return summary

if __name__ == '__main__':
    main()
