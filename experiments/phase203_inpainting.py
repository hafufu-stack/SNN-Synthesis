"""
Phase 203: Dissonance Inpainting - Logical Infill of Uncertain Pixels

P200's noise injection failed. Instead of noise, we MASK disagreement pixels
and let the NCA "inpaint" them from frozen, agreed-upon context.

Strategy:
  1. Run Gated Hybrid -> get S1 and S2 outputs
  2. Identify dissonance pixels (S1 != S2 argmax)
  3. Freeze agreed pixels (S1 == S2), mask disagreement ones
  4. Re-run NCA with frozen context to inpaint masked pixels
  5. Compare PA/EM with vanilla Gated Hybrid

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


class InpaintingNCA(GatedHybridNCA):
    """Gated Hybrid + post-hoc dissonance inpainting.

    After the main forward pass, pixels where S1 and S2 disagree
    are masked. The NCA is re-run with agreed pixels FROZEN to
    propagate geometric context into masked positions.
    """
    def __init__(self, *args, inpaint_steps=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.inpaint_steps = inpaint_steps
        # Inpainting refiner: takes masked logits + context -> fills holes
        C = kwargs.get('hidden_ch', 64)
        n_colors = args[0] if args else 11
        self.inpaint_refiner = nn.Sequential(
            nn.Conv2d(n_colors + 1, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

    def forward(self, x, task_emb, do_inpaint=True):
        B, _, H, W = x.shape
        te = task_emb.view(B, -1, 1, 1).expand(B, -1, H, W)

        # Stage 1: System 1 sketches
        inp1 = torch.cat([x, te], dim=1)
        s1_out = self.s1(inp1)

        # Stage 2: System 2 refines S1's sketch
        inp2 = torch.cat([s1_out, te], dim=1)
        state = self.s2_encoder(inp2)
        for t in range(self.s2_steps):
            delta = self.s2_update(state)
            beta = self.s2_tau(state)
            state = beta * state + (1 - beta) * delta
        s2_out = self.s2_decoder(state)

        # Per-pixel gate fusion (standard Gated Hybrid)
        gate_input = torch.cat([x, s1_out, s2_out], dim=1)
        gate = self.pixel_gate(gate_input)
        fused = gate * s1_out + (1 - gate) * s2_out

        if not do_inpaint or not self.training:
            # During eval, apply inpainting post-hoc
            pass

        # Detect dissonance: where S1 and S2 predict different classes
        s1_pred = s1_out.argmax(dim=1)  # (B, H, W)
        s2_pred = s2_out.argmax(dim=1)  # (B, H, W)
        agree_mask = (s1_pred == s2_pred).float().unsqueeze(1)  # (B,1,H,W), 1=agree
        dissonance_mask = 1.0 - agree_mask  # 1=disagree

        # Inpainting: iteratively refine masked pixels from frozen context
        current = fused.clone()
        for step in range(self.inpaint_steps):
            # Feed current state + mask to refiner
            refiner_input = torch.cat([current, dissonance_mask], dim=1)
            refined = self.inpaint_refiner(refiner_input)
            # Only update disagreement pixels; freeze agreed ones
            current = agree_mask * fused + dissonance_mask * refined

        return current, s1_out, s2_out, gate, dissonance_mask

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_and_eval(model, train_tasks, test_tasks, n_epochs, label, is_inpaint=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist_pa, hist_em, hist_dis = [], [], []
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
            tpa, tem, dis_pcts = 0, 0, []
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
                        dis_pcts.append(dm.mean().item() * 100)
            avg_pa = tpa / len(test_tasks)
            avg_em = tem / len(test_tasks)
            hist_pa.append(avg_pa); hist_em.append(avg_em)
            dp = np.mean(dis_pcts) if dis_pcts else 0
            hist_dis.append(dp)
            print(f"    {label} Ep{epoch+1}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%, "
                  f"Dissonance={dp:.1f}%")
    return hist_pa, hist_em, hist_dis


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 203: Dissonance Inpainting")
    print(f"  Mask S1/S2 disagreement, freeze agreed context, inpaint holes")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    C, ep = 64, 100

    # Baseline: Gated Hybrid (P199)
    print(f"\n[Gated Hybrid Baseline (P199)]")
    m1 = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1_pa, h1_em, _ = train_and_eval(m1, train, test, ep, "Gated")
    del m1; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Inpainting NCA with 3 steps
    print(f"\n[Inpainting NCA (3 steps)]")
    m2 = InpaintingNCA(11, hidden_ch=C, embed_dim=32, s2_steps=10, inpaint_steps=3).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2_pa, h2_em, h2_dis = train_and_eval(m2, train, test, ep, "Inpaint3", True)
    del m2; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Inpainting NCA with 5 steps
    print(f"\n[Inpainting NCA (5 steps)]")
    m3 = InpaintingNCA(11, hidden_ch=C, embed_dim=32, s2_steps=10, inpaint_steps=5).to(DEVICE)
    print(f"  Params: {m3.count_params():,}")
    h3_pa, h3_em, h3_dis = train_and_eval(m3, train, test, ep, "Inpaint5", True)
    del m3; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 203 Complete ({elapsed:.0f}s)")
    print(f"  Gated Hybrid: PA={h1_pa[-1]*100:.1f}%, EM={h1_em[-1]*100:.1f}%")
    print(f"  Inpaint(3):   PA={h2_pa[-1]*100:.1f}%, EM={h2_em[-1]*100:.1f}%")
    print(f"  Inpaint(5):   PA={h3_pa[-1]*100:.1f}%, EM={h3_em[-1]*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase203_inpainting.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'gated': {'pa': h1_pa[-1], 'em': h1_em[-1]},
            'inpaint3': {'pa': h2_pa[-1], 'em': h2_em[-1],
                         'avg_dissonance_pct': h2_dis[-1] if h2_dis else 0},
            'inpaint5': {'pa': h3_pa[-1], 'em': h3_em[-1],
                         'avg_dissonance_pct': h3_dis[-1] if h3_dis else 0},
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        epochs = [20*(i+1) for i in range(len(h1_pa))]

        for h, c, l in [(h1_pa,'#3498db','Gated'), (h2_pa,'#e67e22','Inpaint(3)'),
                         (h3_pa,'#2ecc71','Inpaint(5)')]:
            axes[0].plot(epochs, [v*100 for v in h], 'o-', color=c, lw=2, label=l)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test PA (%)')
        axes[0].set_title('PA: Inpainting vs Gated', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        labels = ['Gated', 'Inp(3)', 'Inp(5)']
        pa_vals = [h1_pa[-1]*100, h2_pa[-1]*100, h3_pa[-1]*100]
        em_vals = [h1_em[-1]*100, h2_em[-1]*100, h3_em[-1]*100]
        x = np.arange(3); w = 0.35
        axes[1].bar(x-w/2, pa_vals, w, color=['#3498db','#e67e22','#2ecc71'], alpha=0.85, label='PA')
        axes[1].bar(x+w/2, em_vals, w, color=['#3498db','#e67e22','#2ecc71'], alpha=0.4,
                   hatch='//', label='EM')
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel('%'); axes[1].set_title('PA & EM Comparison', fontweight='bold')
        axes[1].legend()

        if h2_dis:
            axes[2].plot(epochs, h2_dis, 's-', color='#e67e22', lw=2, label='Inp(3)')
        if h3_dis:
            axes[2].plot(epochs, h3_dis, 'D-', color='#2ecc71', lw=2, label='Inp(5)')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Dissonance (%)')
        axes[2].set_title('Dissonance % Over Training', fontweight='bold')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 203: Dissonance Inpainting', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase203_inpainting.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'gated_pa': h1_pa[-1], 'inpaint3_pa': h2_pa[-1], 'inpaint5_pa': h3_pa[-1]}


if __name__ == '__main__':
    main()
