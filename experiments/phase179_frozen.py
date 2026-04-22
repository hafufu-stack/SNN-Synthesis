"""
Phase 179: Frozen Metabolism - Zero-Overhead Sleep via State Delta

P177 proved Sleep boosts PA by +6.2pp (anti-drift protection), but
entropy computation (Softmax+Log) halved speed (0.5x).

Fix: Replace entropy with hidden state delta ||state_t - state_{t-1}||.
If delta < threshold, pixel has reached fixed point -> skip update.
Cost: one subtraction per step (essentially free).

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
    """Standard NCA inference (no sleep)."""
    return model.latent_nca(x, task_embed, n_steps=n_steps)


def inference_entropy_sleep(model, x, task_embed, n_steps=8, entropy_thresh=0.3):
    """P177-style: entropy-based sleep (expensive)."""
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


def inference_frozen(model, x, task_embed, n_steps=8, delta_thresh=0.01):
    """Phase 179: Frozen Metabolism - state-delta based sleep (near-zero cost).
    
    Sleep criterion: max channel change < threshold -> pixel is frozen.
    """
    nca = model.latent_nca
    B, _, H, W = x.shape
    state = nca.encoder(x)
    te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)
    frozen_ratio_hist = []

    for step in range(n_steps):
        prev_state = state.clone()
        state_ctx = torch.cat([state, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        new_state = beta * state + (1 - beta) * delta

        # Frozen criterion: max absolute change across channels
        state_delta = torch.abs(new_state - prev_state).max(dim=1)[0]  # (B, H, W)
        frozen = (state_delta < delta_thresh).float().unsqueeze(1).expand_as(state)
        frozen_ratio_hist.append(frozen[:, 0].mean().item())

        # Frozen pixels keep old state, active pixels update
        state = frozen * prev_state + (1 - frozen) * new_state

    return nca.decoder(state), frozen_ratio_hist


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 179: Frozen Metabolism - Zero-Overhead Sleep")
    print(f"  State-delta sleep vs entropy sleep vs standard")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load
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

    # Benchmark 3 methods
    print("\n[Step 3] Benchmarking 3 inference methods...")
    methods = {
        'standard': {'fn': 'std', 'pas': [], 'ems': [], 'time': 0},
        'entropy_sleep': {'fn': 'ent', 'pas': [], 'ems': [], 'time': 0},
    }
    # Test multiple delta thresholds for frozen
    delta_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    for dt in delta_thresholds:
        methods[f'frozen_{dt}'] = {'fn': 'frz', 'dt': dt, 'pas': [], 'ems': [],
                                    'time': 0, 'frozen_ratios': []}

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
            methods['standard']['pas'].append(pa)
            methods['standard']['ems'].append(em)

            # Entropy sleep
            t1 = time.time()
            logits = inference_entropy_sleep(model, ti, emb, n_steps=8, entropy_thresh=0.3)
            methods['entropy_sleep']['time'] += time.time() - t1
            pred = logits[0, :10].argmax(dim=0)
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            methods['entropy_sleep']['pas'].append(pa)
            methods['entropy_sleep']['ems'].append(em)

            # Frozen (multiple thresholds)
            for dt in delta_thresholds:
                key = f'frozen_{dt}'
                t1 = time.time()
                logits, frz_hist = inference_frozen(model, ti, emb, n_steps=8, delta_thresh=dt)
                methods[key]['time'] += time.time() - t1
                pred = logits[0, :10].argmax(dim=0)
                pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
                em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
                methods[key]['pas'].append(pa)
                methods[key]['ems'].append(em)
                methods[key]['frozen_ratios'].append(np.mean(frz_hist))

    # Results
    elapsed = time.time() - t0
    summary = {}
    print(f"\n{'='*70}")
    print(f"Phase 179 Complete ({elapsed:.0f}s)")
    for name, m in methods.items():
        avg_pa = np.mean(m['pas']); avg_em = np.mean(m['ems'])
        total_time = m['time']
        extra = ""
        if 'frozen_ratios' in m and m['frozen_ratios']:
            extra = f", FrozenRatio={np.mean(m['frozen_ratios'])*100:.0f}%"
        summary[name] = {'pa': avg_pa, 'em': avg_em, 'total_time': total_time}
        if 'frozen_ratios' in m:
            summary[name]['frozen_ratio'] = np.mean(m['frozen_ratios'])
        print(f"  {name:20s}: PA={avg_pa*100:.2f}%, Time={total_time:.2f}s{extra}")

    # Key comparison
    std_time = methods['standard']['time']
    ent_time = methods['entropy_sleep']['time']
    best_frozen = max(
        [(k, v) for k, v in methods.items() if k.startswith('frozen_')],
        key=lambda x: np.mean(x[1]['pas']))
    frz_time = best_frozen[1]['time']
    print(f"\n  Standard time:       {std_time:.2f}s")
    print(f"  Entropy sleep time:  {ent_time:.2f}s ({ent_time/std_time:.1f}x)")
    print(f"  Best frozen time:    {frz_time:.2f}s ({frz_time/std_time:.1f}x)")
    print(f"  Best frozen PA:      {np.mean(best_frozen[1]['pas'])*100:.2f}% ({best_frozen[0]})")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase179_frozen.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 179: Frozen Metabolism',
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        # PA comparison
        keys = ['standard', 'entropy_sleep'] + [f'frozen_{dt}' for dt in delta_thresholds]
        labels = ['Std', 'Entropy\nSleep'] + [f'Frozen\n{dt}' for dt in delta_thresholds]
        pa_vals = [np.mean(methods[k]['pas'])*100 for k in keys]
        colors = ['#95a5a6', '#e74c3c'] + ['#2ecc71']*len(delta_thresholds)
        axes[0].bar(range(len(labels)), pa_vals, color=colors, alpha=0.85, edgecolor='black')
        axes[0].set_xticks(range(len(labels))); axes[0].set_xticklabels(labels, fontsize=7)
        axes[0].set_ylabel('PA (%)'); axes[0].set_title('Pixel Accuracy', fontweight='bold')
        # Time comparison
        times = [methods[k]['time'] for k in keys]
        axes[1].bar(range(len(labels)), times, color=colors, alpha=0.85, edgecolor='black')
        axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels, fontsize=7)
        axes[1].set_ylabel('Total Time (s)'); axes[1].set_title('Speed', fontweight='bold')
        # Frozen ratio
        frz_keys = [f'frozen_{dt}' for dt in delta_thresholds]
        frz_ratios = [np.mean(methods[k]['frozen_ratios'])*100 for k in frz_keys]
        axes[2].bar([f'{dt}' for dt in delta_thresholds], frz_ratios, color='#3498db', alpha=0.85)
        axes[2].set_xlabel('Delta Threshold'); axes[2].set_ylabel('Frozen Ratio (%)')
        axes[2].set_title('Pixels Frozen', fontweight='bold')
        fig.suptitle('Phase 179: Frozen Metabolism (Zero-Cost Sleep)', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.16, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase179_frozen.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return summary

if __name__ == '__main__':
    main()
