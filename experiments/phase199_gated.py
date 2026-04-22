"""
Phase 199: Gated Hybrid Dual-Process - Per-Pixel Intuition/Logic Fusion

P194: Parallel wins PA (59.7%), Sequential wins EM (4.0%).
Solution: Sequential pipeline + pixel-wise gate that learns WHERE
to trust System 1 (intuition) vs System 2 (logic).

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
from phase191_generalization import ScalableNCA
from phase192_dual_process import DualProcessNCA
from phase194_sequential import SequentialDualNCA


class GatedHybridNCA(nn.Module):
    """Sequential S1->S2 with per-pixel learned gate for fusion."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32, s2_steps=10):
        super().__init__()
        self.embed_dim = embed_dim
        C = hidden_ch

        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

        # System 1: Fast sketch (1x1, T=1)
        self.s1 = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

        # System 2: Refine (3x3 NCA, T steps, receives S1 output)
        self.s2_encoder = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )
        self.s2_update = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 1),
        )
        self.s2_tau = nn.Sequential(nn.Conv2d(C, C, 1), nn.Sigmoid())
        self.s2_decoder = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )
        self.s2_steps = s2_steps

        # Per-pixel fusion gate: sees both S1 output, S2 output, AND original input
        self.pixel_gate = nn.Sequential(
            nn.Conv2d(n_colors * 3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

    def encode_task(self, demo_outputs):
        embeddings = []
        for do in demo_outputs:
            emb = self.demo_encoder(do.unsqueeze(0))
            embeddings.append(emb)
        return torch.stack(embeddings).mean(dim=0)

    def forward(self, x, task_emb):
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

        # Per-pixel gate: input + S1 + S2 -> gate
        gate_input = torch.cat([x, s1_out, s2_out], dim=1)
        gate = self.pixel_gate(gate_input)  # (B, 1, H, W)

        # Fusion: gate=1 -> trust S1, gate=0 -> trust S2
        output = gate * s1_out + (1 - gate) * s2_out

        return output, s1_out, s2_out, gate

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


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
            tpa, tem = 0, 0
            gate_means = []
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
                    if isinstance(out, tuple) and len(out) >= 4:
                        gate_means.append(out[3][0, 0, :oh, :ow].mean().item())
            avg_pa = tpa / len(test_tasks)
            avg_em = tem / len(test_tasks)
            hist_pa.append(avg_pa); hist_em.append(avg_em)
            g = np.mean(gate_means) if gate_means else -1
            gs = f", Gate(S1)={g:.2f}" if g >= 0 else ""
            print(f"    {label} Ep{epoch+1}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%{gs}")
    return hist_pa, hist_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 199: Gated Hybrid Dual-Process")
    print(f"  Per-pixel S1/S2 fusion for max PA AND max EM")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    C, ep = 64, 100

    # Standard baseline
    print(f"\n[Standard NCA]")
    m0 = ScalableNCA(11, C, 5, 32).to(DEVICE)
    print(f"  Params: {m0.count_params():,}")
    h0_pa, h0_em = train_and_eval(m0, train, test, ep, "Std")
    del m0; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Parallel Dual (P192)
    print(f"\n[Parallel Dual (P192)]")
    m1 = DualProcessNCA(11, C, 32).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1_pa, h1_em = train_and_eval(m1, train, test, ep, "Para")
    del m1; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Sequential Dual (P194)
    print(f"\n[Sequential Dual (P194)]")
    m2 = SequentialDualNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2_pa, h2_em = train_and_eval(m2, train, test, ep, "Seq")
    del m2; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Gated Hybrid (NEW)
    print(f"\n[Gated Hybrid Dual-Process]")
    m3 = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {m3.count_params():,}")
    h3_pa, h3_em = train_and_eval(m3, train, test, ep, "Gated")
    del m3; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 199 Complete ({elapsed:.0f}s)")
    print(f"  Standard:   PA={h0_pa[-1]*100:.1f}%, EM={h0_em[-1]*100:.1f}%")
    print(f"  Parallel:   PA={h1_pa[-1]*100:.1f}%, EM={h1_em[-1]*100:.1f}%")
    print(f"  Sequential: PA={h2_pa[-1]*100:.1f}%, EM={h2_em[-1]*100:.1f}%")
    print(f"  Gated:      PA={h3_pa[-1]*100:.1f}%, EM={h3_em[-1]*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase199_gated.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'standard': {'pa': h0_pa[-1], 'em': h0_em[-1]},
            'parallel': {'pa': h1_pa[-1], 'em': h1_em[-1]},
            'sequential': {'pa': h2_pa[-1], 'em': h2_em[-1]},
            'gated': {'pa': h3_pa[-1], 'em': h3_em[-1]},
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        epochs = [20*(i+1) for i in range(len(h0_pa))]
        for h, c, l in [(h0_pa,'#95a5a6','Std'), (h1_pa,'#3498db','Para'),
                         (h2_pa,'#e67e22','Seq'), (h3_pa,'#2ecc71','Gated')]:
            axes[0].plot(epochs, [v*100 for v in h], 'o-', color=c, lw=2, label=l)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test PA (%)')
        axes[0].set_title('PA Curves', fontweight='bold'); axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        labels = ['Std', 'Para', 'Seq', 'Gated']
        pa_vals = [h0_pa[-1]*100, h1_pa[-1]*100, h2_pa[-1]*100, h3_pa[-1]*100]
        em_vals = [h0_em[-1]*100, h1_em[-1]*100, h2_em[-1]*100, h3_em[-1]*100]
        x = np.arange(4); w = 0.35
        axes[1].bar(x-w/2, pa_vals, w, color=['#95a5a6','#3498db','#e67e22','#2ecc71'],
                   alpha=0.85, label='PA')
        axes[1].bar(x+w/2, em_vals, w, color=['#95a5a6','#3498db','#e67e22','#2ecc71'],
                   alpha=0.4, hatch='//', label='EM')
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel('%'); axes[1].set_title('PA & EM', fontweight='bold')
        axes[1].legend()

        # PA vs EM scatter
        axes[2].scatter(pa_vals, em_vals, s=120,
                       c=['#95a5a6','#3498db','#e67e22','#2ecc71'],
                       edgecolor='black', zorder=5)
        for i, l in enumerate(labels):
            axes[2].annotate(l, (pa_vals[i], em_vals[i]), textcoords="offset points",
                           xytext=(8, 5), fontweight='bold')
        axes[2].set_xlabel('PA (%)'); axes[2].set_ylabel('EM (%)')
        axes[2].set_title('PA vs EM Tradeoff', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 199: Gated Hybrid Dual-Process', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase199_gated.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'gated_pa': h3_pa[-1], 'gated_em': h3_em[-1]}

if __name__ == '__main__':
    main()
