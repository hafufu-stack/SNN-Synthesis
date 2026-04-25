"""
Phase 233: Parameter-Free Expansion via Depthwise-Separable Conv

The Grand Unified Equation reveals:
  C (channels) has the LARGEST positive coefficient (+0.052)
  P (params) has a NEGATIVE coefficient (-0.010)

Standard 3x3 Conv: P grows as C^2 (disaster for the equation)
Depthwise-Separable: P grows as C (best of both worlds!)

Goal: Implement C=192 NCA with minimal parameter growth.

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
from phase199_gated import GatedHybridNCA, train_and_eval
from phase227_complexity import generate_complex_batch


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-Separable Conv2d: P ~ C instead of C^2."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DepthwiseGatedNCA(nn.Module):
    """GatedHybrid with Depthwise-Separable Convs for C scaling."""
    def __init__(self, n_colors=11, hidden_ch=192, embed_dim=32, s2_steps=10):
        super().__init__()
        self.embed_dim = embed_dim
        C = hidden_ch

        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

        # System 1: Fast sketch (1x1 only, no spatial conv)
        self.s1 = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

        # System 2: Depthwise-Separable NCA
        self.s2_encoder = nn.Sequential(
            DepthwiseSeparableConv(n_colors + embed_dim, C, 3, 1), nn.ReLU(),
            DepthwiseSeparableConv(C, C, 3, 1), nn.ReLU(),
        )
        self.s2_update = nn.Sequential(
            DepthwiseSeparableConv(C, C, 3, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1),
        )
        self.s2_tau = nn.Sequential(nn.Conv2d(C, C, 1), nn.Sigmoid())
        self.s2_decoder = nn.Sequential(
            DepthwiseSeparableConv(C, C, 3, 1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )
        self.s2_steps = s2_steps

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
        inp1 = torch.cat([x, te], dim=1)
        s1_out = self.s1(inp1)
        inp2 = torch.cat([s1_out, te], dim=1)
        state = self.s2_encoder(inp2)
        for t in range(self.s2_steps):
            delta = self.s2_update(state)
            beta = self.s2_tau(state)
            state = beta * state + (1 - beta) * delta
        s2_out = self.s2_decoder(state)
        gate_input = torch.cat([x, s1_out, s2_out], dim=1)
        gate = self.pixel_gate(gate_input)
        output = gate * s1_out + (1 - gate) * s2_out
        return output, s1_out, s2_out, gate

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def cotrain(model, train_tasks, test_tasks, n_epochs, label, syn_ratio=5):
    """Co-Training with synthetic data (P229 golden ratio 1:5)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng_syn = np.random.RandomState(SEED + 777)
    hist_pa, hist_em = [], []
    for epoch in range(n_epochs):
        model.train(); random.shuffle(train_tasks)
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
            # Interleave synthetic
            if syn_ratio > 0:
                sbatch = generate_complex_batch(rng_syn, batch_size=syn_ratio)
                for inp_oh, out_oh, soh, sow in sbatch:
                    inp_t = inp_oh.unsqueeze(0).to(DEVICE)
                    out_gt = out_oh[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
                    semb = model.encode_task([out_oh.to(DEVICE)])
                    sout = model(inp_t, semb)
                    slogits = sout[0] if isinstance(sout, tuple) else sout
                    sloss = F.cross_entropy(slogits[:, :, :soh, :sow], out_gt[:, :soh, :sow])
                    opt.zero_grad(); sloss.backward(); opt.step()
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            tpa, tem = 0, 0
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
            avg_pa = tpa / len(test_tasks); avg_em = tem / len(test_tasks)
            hist_pa.append(avg_pa); hist_em.append(avg_em)
            print(f"    {label} Ep{epoch+1}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%")
    return hist_pa, hist_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 233: Parameter-Free Expansion (Depthwise-Separable)")
    print(f"  C=192 with P~C instead of P~C^2")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    ep = 100

    # Baseline: GatedHybrid C=64 (standard conv)
    print(f"\n[Baseline: GatedHybrid C=64]")
    torch.manual_seed(SEED)
    m0 = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    print(f"  Params: {m0.count_params():,}")
    h0_pa, h0_em = train_and_eval(m0, train, test, ep, "Base64")
    del m0; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # DW-Separable C=192 (no co-training)
    print(f"\n[Depthwise C=192 (no co-training)]")
    torch.manual_seed(SEED)
    m1 = DepthwiseGatedNCA(11, 192, 32, 10).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1_pa, h1_em = train_and_eval(m1, train, test, ep, "DW192")
    del m1; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # DW-Separable C=192 + Co-Training 1:5
    print(f"\n[Depthwise C=192 + Co-Training 1:5]")
    torch.manual_seed(SEED)
    m2 = DepthwiseGatedNCA(11, 192, 32, 10).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2_pa, h2_em = cotrain(m2, train, test, ep, "DW192+CT", syn_ratio=5)
    del m2; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # Also try C=128 for comparison
    print(f"\n[Depthwise C=128 + Co-Training 1:5]")
    torch.manual_seed(SEED)
    m3 = DepthwiseGatedNCA(11, 128, 32, 10).to(DEVICE)
    print(f"  Params: {m3.count_params():,}")
    h3_pa, h3_em = cotrain(m3, train, test, ep, "DW128+CT", syn_ratio=5)
    del m3; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'base64': {'pa': h0_pa[-1], 'em': h0_em[-1], 'params': 166679},
        'dw192': {'pa': h1_pa[-1], 'em': h1_em[-1]},
        'dw192_ct': {'pa': h2_pa[-1], 'em': h2_em[-1]},
        'dw128_ct': {'pa': h3_pa[-1], 'em': h3_em[-1]},
    }

    print(f"\n{'='*70}")
    print(f"  DEPTHWISE-SEPARABLE EXPANSION:")
    for k, r in results.items():
        print(f"  {k:15s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase233_depthwise.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(results.keys())
        pa_vals = [r['pa']*100 for r in results.values()]
        em_vals = [r['em']*100 for r in results.values()]
        colors = ['#95a5a6', '#3498db', '#2ecc71', '#e67e22']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 233: Depthwise-Separable (C scaling)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase233_depthwise.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
