"""
Phase 178: Autophagic Regeneration - Self-Repair for Exact Match

P176 (global annealing) failed: uniform cooling freezes errors in place.
P154 proved NCA has self-repair ability. P175 showed confident pixels
can be identified via entropy.

Strategy: identify uncertain pixels (high entropy "stains"), destroy
their hidden state (autophagy), then let confident frozen neighbors
inpaint the gaps via additional NCA steps.

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
PAD_SIZE = 32
N_COLORS = 11
EMB_DIM = 64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset
)


def inference_standard(model, x, task_embed, n_steps=8):
    """Standard inference."""
    return model.latent_nca(x, task_embed, n_steps=n_steps)


def inference_autophagic(model, x, task_embed, n_steps_initial=5,
                          n_steps_repair=3, entropy_percentile=80):
    """Autophagic regeneration inference.
    
    Uses FoundationLatentNCA's actual structure:
    - self.encoder, self.update, self.tau_gate, self.decoder
    
    1. Run NCA for n_steps_initial (normal thinking)
    2. Identify uncertain pixels (high entropy > percentile threshold)
    3. Zero-reset their hidden state (autophagy/self-eating)
    4. Freeze confident pixels (metabolic sleep)
    5. Run n_steps_repair more steps for inpainting from neighbors
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

    # Checkpoint: measure entropy to find "stains"
    logits_mid = nca.decoder(state)
    probs = F.softmax(logits_mid[:, :10], dim=1)
    pixel_entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)  # (B, H, W)

    # Autophagy: destroy uncertain pixels' hidden state
    threshold = torch.quantile(pixel_entropy.flatten(), entropy_percentile / 100.0)
    stain_mask = pixel_entropy > threshold  # (B, H, W)
    stain_3d = stain_mask.unsqueeze(1).expand_as(state)  # (B, C, H, W)

    # Zero-reset stained pixels (autophagy)
    state_repaired = state.clone()
    state_repaired[stain_3d] = 0.0

    n_stains = stain_mask.float().sum().item()
    n_total = stain_mask.numel()

    # Phase 2: Repair - frozen confident pixels help reconstruct stains
    for step in range(n_steps_repair):
        state_ctx = torch.cat([state_repaired, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        new_state = beta * state_repaired + (1 - beta) * delta

        # Only update stained (destroyed) pixels - confident ones stay frozen
        state_repaired = stain_3d.float() * new_state + (~stain_3d).float() * state_repaired

    final_logits = nca.decoder(state_repaired)
    return final_logits, {
        'n_stains': n_stains,
        'n_total': n_total,
        'stain_ratio': n_stains / max(n_total, 1)
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 178: Autophagic Regeneration - Self-Repair for EM")
    print(f"  Destroy uncertain pixels, let neighbors inpaint")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load model
    print("\n[Step 1] Loading Foundation Model...")
    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    ckpt_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Load ARC
    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:50]

    # Evaluate standard
    print("\n[Step 3] Standard inference baseline...")
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

    # Test different autophagy percentiles
    print("\n[Step 4] Autophagic regeneration at different aggressiveness levels...")
    percentiles = [50, 60, 70, 80, 90, 95]
    repair_steps_list = [2, 3, 5]
    config_results = {}

    for pct in percentiles:
        for n_repair in repair_steps_list:
            key = f"p{pct}_r{n_repair}"
            a_pas, a_ems, stain_ratios = [], [], []
            with torch.no_grad():
                for item in test_tasks:
                    di = [d.to(DEVICE) for d in item['demo_inputs']]
                    do = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.task_encoder(di, do)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    logits, stats = inference_autophagic(
                        model, ti, emb, n_steps_initial=5,
                        n_steps_repair=n_repair, entropy_percentile=pct)
                    pred = logits[0, :10].argmax(dim=0)
                    gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
                    em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
                    a_pas.append(pa); a_ems.append(em)
                    stain_ratios.append(stats['stain_ratio'])

            avg_pa = np.mean(a_pas); avg_em = np.mean(a_ems)
            config_results[key] = {
                'pa': avg_pa, 'em': avg_em,
                'percentile': pct, 'repair_steps': n_repair,
                'avg_stain_ratio': np.mean(stain_ratios)
            }
            if n_repair == 3:  # only print main config
                print(f"  Autophagy(pct={pct}, repair={n_repair}): "
                      f"PA={avg_pa*100:.2f}%, EM={avg_em*100:.1f}%, "
                      f"Stains={np.mean(stain_ratios)*100:.0f}%")

    best_config = max(config_results.items(), key=lambda x: x[1]['pa'])
    improvement = best_config[1]['pa'] - std_pa

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 178 Complete ({elapsed:.0f}s)")
    print(f"  Standard:  PA={std_pa*100:.2f}%, EM={std_em*100:.1f}%")
    print(f"  Best ({best_config[0]}): PA={best_config[1]['pa']*100:.2f}%, "
          f"EM={best_config[1]['em']*100:.1f}%")
    print(f"  Improvement: {improvement*100:+.2f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase178_autophagic.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 178: Autophagic Regeneration',
            'timestamp': datetime.now().isoformat(),
            'standard': {'pa': std_pa, 'em': std_em},
            'configs': config_results,
            'best_config': best_config[0],
            'improvement_pp': improvement * 100,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # PA vs percentile (repair=3)
        pcts = percentiles
        pa_r3 = [config_results[f'p{p}_r3']['pa']*100 for p in pcts]
        axes[0].plot(pcts, pa_r3, 'o-', color='#e74c3c', linewidth=2, label='Autophagic')
        axes[0].axhline(std_pa*100, color='#95a5a6', linestyle='--', label='Standard')
        axes[0].set_xlabel('Entropy Percentile (% destroyed)')
        axes[0].set_ylabel('PA (%)'); axes[0].set_title('PA vs Autophagy Level', fontweight='bold')
        axes[0].legend()

        # PA vs repair steps (pct=80)
        repairs = repair_steps_list
        pa_p80 = [config_results[f'p80_r{r}']['pa']*100 for r in repairs]
        axes[1].bar([str(r) for r in repairs], pa_p80, color='#2ecc71', alpha=0.85)
        axes[1].axhline(std_pa*100, color='#95a5a6', linestyle='--')
        axes[1].set_xlabel('Repair Steps'); axes[1].set_ylabel('PA (%)')
        axes[1].set_title('PA vs Repair Steps (pct=80)', fontweight='bold')

        # Stain ratio vs percentile
        stain_r3 = [config_results[f'p{p}_r3']['avg_stain_ratio']*100 for p in pcts]
        axes[2].bar([str(p) for p in pcts], stain_r3, color='#f39c12', alpha=0.85)
        axes[2].set_xlabel('Percentile'); axes[2].set_ylabel('Pixels Destroyed (%)')
        axes[2].set_title('Autophagy Intensity', fontweight='bold')

        fig.suptitle('Phase 178: Autophagic Regeneration (Self-Repair)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase178_autophagic.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'standard_pa': std_pa, 'best_pa': best_config[1]['pa']}

if __name__ == '__main__':
    main()
