"""
Phase 200: Cognitive Dissonance - Self-Correction via S1/S2 Disagreement
 MILESTONE: Phase 200!

Use |O1 - O2| (System 1 vs System 2 disagreement) to detect "stains"
(uncertain pixels), then inject Gumbel noise ONLY at those pixels
to force self-correction.

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
from phase191_generalization import ScalableNCA


class CognitiveDissonanceNCA(GatedHybridNCA):
    """GatedHybrid + self-correction at dissonance pixels."""
    def __init__(self, *args, correction_steps=3, noise_scale=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.correction_steps = correction_steps
        self.noise_scale = noise_scale

    def forward(self, x, task_emb):
        B, _, H, W = x.shape
        te = task_emb.view(B, -1, 1, 1).expand(B, -1, H, W)

        # Stage 1: System 1 sketches
        inp1 = torch.cat([x, te], dim=1)
        s1_out = self.s1(inp1)

        # Stage 2: System 2 refines
        inp2 = torch.cat([s1_out, te], dim=1)
        state = self.s2_encoder(inp2)
        for t in range(self.s2_steps):
            delta = self.s2_update(state)
            beta = self.s2_tau(state)
            state = beta * state + (1 - beta) * delta
        s2_out = self.s2_decoder(state)

        # Detect cognitive dissonance: where S1 and S2 disagree
        dissonance = (s1_out - s2_out).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
        dissonance_threshold = dissonance.mean() + dissonance.std()
        dissonance_mask = (dissonance > dissonance_threshold).float()

        # Self-correction: inject Gumbel noise at dissonance pixels, re-run S2
        if dissonance_mask.sum() > 0 and self.training:
            for _ in range(self.correction_steps):
                gumbel = -torch.log(-torch.log(torch.rand_like(state) + 1e-8) + 1e-8)
                noise = gumbel * self.noise_scale * dissonance_mask
                state_corrected = state + noise
                delta = self.s2_update(state_corrected)
                beta = self.s2_tau(state_corrected)
                state = beta * state_corrected + (1 - beta) * delta
            s2_out = self.s2_decoder(state)

        # Per-pixel gate fusion
        gate_input = torch.cat([x, s1_out, s2_out], dim=1)
        gate = self.pixel_gate(gate_input)
        output = gate * s1_out + (1 - gate) * s2_out

        return output, s1_out, s2_out, gate, dissonance_mask


def train_and_eval(model, train_tasks, test_tasks, n_epochs, label):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist_pa, hist_em = [], []
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
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            tpa, tem, dissonance_pcts = 0, 0, []
            with torch.no_grad():
                for item in test_tasks:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    out = model(ti, emb)
                    logits = out[0] if isinstance(out, tuple) else out
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
                    tem += float((pred == gt[:oh, :ow]).all().item())
                    if isinstance(out, tuple) and len(out) >= 5:
                        dm = out[4][0, 0, :oh, :ow]
                        dissonance_pcts.append(dm.mean().item() * 100)
            avg_pa = tpa / len(test_tasks)
            avg_em = tem / len(test_tasks)
            hist_pa.append(avg_pa); hist_em.append(avg_em)
            dp = np.mean(dissonance_pcts) if dissonance_pcts else 0
            print(f"    {label} Ep{epoch+1}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%, "
                  f"Dissonance={dp:.1f}%")
    return hist_pa, hist_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print(" Phase 200: Cognitive Dissonance (Self-Correction)")
    print(f"  S1 vs S2 disagreement -> Gumbel noise -> self-correct")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    C, ep = 64, 100

    print(f"\n[Gated Hybrid (P199 baseline)]")
    m1 = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1_pa, h1_em = train_and_eval(m1, train, test, ep, "Gated")
    del m1; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    print(f"\n[Cognitive Dissonance NCA]")
    m2 = CognitiveDissonanceNCA(11, C, 32, 10, correction_steps=3).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2_pa, h2_em = train_and_eval(m2, train, test, ep, "CogDis")
    del m2; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f" Phase 200 Complete ({elapsed:.0f}s)")
    print(f"  Gated:    PA={h1_pa[-1]*100:.1f}%, EM={h1_em[-1]*100:.1f}%")
    print(f"  CogDis:   PA={h2_pa[-1]*100:.1f}%, EM={h2_em[-1]*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase200_cogdis.json"), 'w', encoding='utf-8') as f:
        json.dump({'gated': {'pa': h1_pa[-1], 'em': h1_em[-1]},
                   'cogdis': {'pa': h2_pa[-1], 'em': h2_em[-1]},
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        epochs = [20*(i+1) for i in range(len(h1_pa))]
        ax.plot(epochs, [h*100 for h in h1_pa], 'o-', color='#3498db', lw=2, label='Gated Hybrid')
        ax.plot(epochs, [h*100 for h in h2_pa], 'D-', color='#e74c3c', lw=2, label='Cognitive Dissonance')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test PA (%)')
        ax.set_title(' Phase 200: Cognitive Dissonance', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase200_cogdis.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")
    gc.collect()
    return {'gated': h1_pa[-1], 'cogdis': h2_pa[-1]}

if __name__ == '__main__':
    main()
