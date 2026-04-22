"""
Phase 180: Hormonal Regeneration - Global Context for Self-Repair

P178: local autophagy improves +1.56pp but can't fix "semantic stains"
because NCA's 3x3 view lacks global context (symmetry, color ratios).
P174: Hormone broadcast provides global info but is redundant at high steps.

Fusion: Use autophagy (destroy stains) + hormone broadcast (ONLY during
repair steps) so destroyed pixels can read global context to reconstruct.

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
N_COLORS = 11

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset
)


def inference_standard(model, x, task_embed, n_steps=8):
    """Standard inference."""
    return model.latent_nca(x, task_embed, n_steps=n_steps)


def inference_autophagy_only(model, x, task_embed, n_steps_initial=5,
                              n_steps_repair=3, entropy_percentile=90):
    """P178-style: autophagy without hormone (baseline)."""
    nca = model.latent_nca
    B, _, H, W = x.shape
    state = nca.encoder(x)
    te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)
    for step in range(n_steps_initial):
        state_ctx = torch.cat([state, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        state = beta * state + (1 - beta) * delta
    logits_mid = nca.decoder(state)
    probs = F.softmax(logits_mid[:, :10], dim=1)
    pixel_entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
    threshold = torch.quantile(pixel_entropy.flatten(), entropy_percentile / 100.0)
    stain_mask = pixel_entropy > threshold
    stain_3d = stain_mask.unsqueeze(1).expand_as(state)
    state[stain_3d] = 0.0
    for step in range(n_steps_repair):
        state_ctx = torch.cat([state, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        new_state = beta * state + (1 - beta) * delta
        state = stain_3d.float() * new_state + (~stain_3d).float() * state
    return nca.decoder(state)


def inference_hormonal_regen(model, x, task_embed, n_steps_initial=5,
                              n_steps_repair=3, entropy_percentile=90,
                              n_hormone_ch=4):
    """Phase 180: Autophagy + Hormonal Broadcast during repair.
    
    1. Normal inference for n_steps_initial
    2. Identify stains (high entropy pixels)
    3. Zero-reset stain hidden states (autophagy)
    4. During repair steps: inject global hormone (GAP of latent state)
       into ALL pixels as additional context for inpainting
    """
    nca = model.latent_nca
    B, _, H, W = x.shape
    state = nca.encoder(x)
    te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

    # Phase 1: Normal inference
    for step in range(n_steps_initial):
        state_ctx = torch.cat([state, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        state = beta * state + (1 - beta) * delta

    # Identify stains
    logits_mid = nca.decoder(state)
    probs = F.softmax(logits_mid[:, :10], dim=1)
    pixel_entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
    threshold = torch.quantile(pixel_entropy.flatten(), entropy_percentile / 100.0)
    stain_mask = pixel_entropy > threshold
    stain_3d = stain_mask.unsqueeze(1).expand_as(state)

    # Autophagy: zero stain hidden states
    state[stain_3d] = 0.0

    # Phase 2: Repair with hormone broadcast
    for step in range(n_steps_repair):
        # Hormone: Global Average Pool of healthy (non-stain) state
        # This gives the repair step global context
        healthy_state = state * (~stain_3d).float()
        n_healthy = (~stain_3d[:, :1]).float().sum(dim=(-2, -1), keepdim=True).clamp(min=1)
        hormone = healthy_state.sum(dim=(-2, -1), keepdim=True) / n_healthy  # (B, C, 1, 1)
        hormone_broadcast = hormone.expand_as(state)  # broadcast to all pixels

        # Inject hormone into state before update (additive injection)
        state_augmented = state + hormone_broadcast * 0.1

        state_ctx = torch.cat([state_augmented, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        new_state = beta * state_augmented + (1 - beta) * delta

        # Only update stained pixels; healthy ones stay frozen
        state = stain_3d.float() * new_state + (~stain_3d).float() * state

    return nca.decoder(state)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 180: Hormonal Regeneration - Global Self-Repair")
    print(f"  Autophagy + Hormone Broadcast for semantic inpainting")
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

    # Baseline
    print("\n[Step 3] Standard baseline...")
    std_pas, std_ems = [], []
    with torch.no_grad():
        for item in test_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.task_encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            logits = inference_standard(model, ti, emb, n_steps=8)
            pred = logits[0, :10].argmax(dim=0)
            gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            std_pas.append(pa); std_ems.append(em)
    std_pa = np.mean(std_pas); std_em = np.mean(std_ems)
    print(f"  Standard: PA={std_pa*100:.2f}%, EM={std_em*100:.1f}%")

    # Compare autophagy-only vs hormonal regen at different percentiles
    print("\n[Step 4] Comparing Autophagy-Only vs Hormonal Regeneration...")
    percentiles = [80, 85, 90, 95]
    repair_steps = [2, 3, 5]
    results = {}

    for pct in percentiles:
        for n_rep in repair_steps:
            # Autophagy only
            key_a = f"auto_p{pct}_r{n_rep}"
            a_pas, a_ems = [], []
            with torch.no_grad():
                for item in test_tasks:
                    di = [d.to(DEVICE) for d in item['demo_inputs']]
                    do = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.task_encoder(di, do)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    logits = inference_autophagy_only(model, ti, emb, 5, n_rep, pct)
                    pred = logits[0, :10].argmax(dim=0)
                    gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
                    em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
                    a_pas.append(pa); a_ems.append(em)
            results[key_a] = {'pa': np.mean(a_pas), 'em': np.mean(a_ems), 'type': 'autophagy'}

            # Hormonal regen
            key_h = f"horm_p{pct}_r{n_rep}"
            h_pas, h_ems = [], []
            with torch.no_grad():
                for item in test_tasks:
                    di = [d.to(DEVICE) for d in item['demo_inputs']]
                    do = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.task_encoder(di, do)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    logits = inference_hormonal_regen(model, ti, emb, 5, n_rep, pct)
                    pred = logits[0, :10].argmax(dim=0)
                    gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
                    em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
                    h_pas.append(pa); h_ems.append(em)
            results[key_h] = {'pa': np.mean(h_pas), 'em': np.mean(h_ems), 'type': 'hormonal'}

            if n_rep == 3:
                print(f"  pct={pct}: Auto PA={np.mean(a_pas)*100:.2f}%, "
                      f"Horm PA={np.mean(h_pas)*100:.2f}% "
                      f"(D={((np.mean(h_pas)-np.mean(a_pas))*100):+.2f}pp)")

    best_config = max(results.items(), key=lambda x: x[1]['pa'])
    improvement = best_config[1]['pa'] - std_pa

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 180 Complete ({elapsed:.0f}s)")
    print(f"  Standard:  PA={std_pa*100:.2f}%")
    print(f"  Best ({best_config[0]}): PA={best_config[1]['pa']*100:.2f}%")
    print(f"  Improvement vs standard: {improvement*100:+.2f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase180_hormonal_regen.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 180: Hormonal Regeneration',
            'timestamp': datetime.now().isoformat(),
            'standard': {'pa': std_pa, 'em': std_em},
            'configs': results,
            'best_config': best_config[0],
            'improvement_pp': improvement * 100,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        # Auto vs Horm at repair=3
        pcts = percentiles
        auto_pa = [results[f'auto_p{p}_r3']['pa']*100 for p in pcts]
        horm_pa = [results[f'horm_p{p}_r3']['pa']*100 for p in pcts]
        axes[0].plot(pcts, auto_pa, 'o-', color='#e74c3c', linewidth=2, label='Autophagy Only')
        axes[0].plot(pcts, horm_pa, 's-', color='#2ecc71', linewidth=2, label='+ Hormone')
        axes[0].axhline(std_pa*100, color='#95a5a6', linestyle='--', label='Standard')
        axes[0].set_xlabel('Percentile'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title('PA: Autophagy vs Hormonal', fontweight='bold'); axes[0].legend(fontsize=8)
        # Hormone advantage
        advantage = [h - a for h, a in zip(horm_pa, auto_pa)]
        axes[1].bar([str(p) for p in pcts], advantage,
                   color=['#2ecc71' if v > 0 else '#e74c3c' for v in advantage], alpha=0.85)
        axes[1].set_xlabel('Percentile'); axes[1].set_ylabel('Hormone Advantage (pp)')
        axes[1].set_title('Hormone vs Autophagy', fontweight='bold')
        axes[1].axhline(0, color='black', linewidth=0.5)
        # Repair steps comparison (pct=90)
        reps = repair_steps
        auto_r = [results[f'auto_p90_r{r}']['pa']*100 for r in reps]
        horm_r = [results[f'horm_p90_r{r}']['pa']*100 for r in reps]
        x = np.arange(len(reps)); w = 0.35
        axes[2].bar(x - w/2, auto_r, w, color='#e74c3c', alpha=0.85, label='Auto')
        axes[2].bar(x + w/2, horm_r, w, color='#2ecc71', alpha=0.85, label='Horm')
        axes[2].set_xticks(x); axes[2].set_xticklabels([str(r) for r in reps])
        axes[2].set_xlabel('Repair Steps'); axes[2].set_ylabel('PA (%)')
        axes[2].set_title('Repair Steps (pct=90)', fontweight='bold'); axes[2].legend()
        fig.suptitle('Phase 180: Hormonal Regeneration (Global Self-Repair)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase180_hormonal_regen.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'standard_pa': std_pa, 'best_pa': best_config[1]['pa']}

if __name__ == '__main__':
    main()
