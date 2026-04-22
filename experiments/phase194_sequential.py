"""
Phase 194: Sequential Dual-Process - Intuition then Logic

P192 proved: Parallel Dual-Process (+6.3pp, first EM!)
Now test: Sequential pipeline where System 1's output
becomes System 2's input (human-like: sketch then refine).

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


class SequentialDualNCA(nn.Module):
    """Sequential: S1 sketches -> S2 refines on top of sketch."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32,
                 s2_steps=10):
        super().__init__()
        self.embed_dim = embed_dim
        C = hidden_ch

        # Shared task encoder
        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

        # System 1: Fast sketch (1x1 only, T=1)
        self.s1 = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

        # System 2: Refine (3x3 NCA, T steps, takes S1 output as input)
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

        # Residual gate: how much S2 modifies S1's output
        self.residual_gate = nn.Sequential(
            nn.Conv2d(n_colors * 2, 16, 1), nn.ReLU(),
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

        # Stage 1: System 1 sketches output from input
        inp1 = torch.cat([x, te], dim=1)
        s1_out = self.s1(inp1)  # (B, n_colors, H, W)

        # Stage 2: System 2 refines S1's output
        inp2 = torch.cat([s1_out, te], dim=1)  # S1 output as S2 input!
        state = self.s2_encoder(inp2)
        for t in range(self.s2_steps):
            delta = self.s2_update(state)
            beta = self.s2_tau(state)
            state = beta * state + (1 - beta) * delta
        s2_out = self.s2_decoder(state)

        # Residual: output = S1 + gate * (S2 - S1)
        gate_input = torch.cat([s1_out, s2_out], dim=1)
        gate = self.residual_gate(gate_input)
        output = s1_out + gate * (s2_out - s1_out)

        return output, s1_out, s2_out, gate

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_and_eval(model, train_tasks, test_tasks, n_epochs, label):
    """Train and evaluate, return test PA history."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    test_pa_hist = []

    for epoch in range(n_epochs):
        model.train()
        random.shuffle(train_tasks)
        for item in train_tasks[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            output = model(ti, emb)
            logits = output[0] if isinstance(output, tuple) else output
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            test_pa, test_em = 0, 0
            gate_means = []
            with torch.no_grad():
                for item in test_tasks:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    output = model(ti, emb)
                    logits = output[0] if isinstance(output, tuple) else output
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    pa = (pred == gt[:oh, :ow]).float().mean().item()
                    em = float((pred == gt[:oh, :ow]).all().item())
                    test_pa += pa; test_em += em
                    if isinstance(output, tuple) and len(output) >= 4:
                        gate_means.append(output[3][0, 0, :oh, :ow].mean().item())
            avg_pa = test_pa / len(test_tasks)
            avg_em = test_em / len(test_tasks)
            test_pa_hist.append(avg_pa)
            gate_str = f", Gate={np.mean(gate_means):.2f}" if gate_means else ""
            print(f"    {label} Ep{epoch+1}: TestPA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%{gate_str}")
    return test_pa_hist


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 194: Sequential Dual-Process (Intuition -> Logic)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train_tasks = all_tasks[:200]
    test_tasks = all_tasks[200:250]
    C, n_epochs = 64, 100

    # Parallel Dual (P192 baseline)
    print(f"\n[Parallel Dual-Process (P192 baseline)]")
    m1 = DualProcessNCA(11, C, embed_dim=32).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1 = train_and_eval(m1, train_tasks, test_tasks, n_epochs, "Parallel")
    del m1; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Sequential Dual
    print(f"\n[Sequential Dual-Process]")
    m2 = SequentialDualNCA(11, C, embed_dim=32, s2_steps=10).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2 = train_and_eval(m2, train_tasks, test_tasks, n_epochs, "Sequential")
    del m2; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Standard single NCA
    print(f"\n[Standard NCA (baseline)]")
    m3 = ScalableNCA(11, C, n_steps=5, embed_dim=32).to(DEVICE)
    print(f"  Params: {m3.count_params():,}")
    h3 = train_and_eval(m3, train_tasks, test_tasks, n_epochs, "Standard")
    del m3; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 194 Complete ({elapsed:.0f}s)")
    print(f"  Standard:   TestPA={h3[-1]*100:.1f}%")
    print(f"  Parallel:   TestPA={h1[-1]*100:.1f}%")
    print(f"  Sequential: TestPA={h2[-1]*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase194_sequential.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 194: Sequential Dual-Process',
            'timestamp': datetime.now().isoformat(),
            'standard': h3[-1], 'parallel': h1[-1], 'sequential': h2[-1],
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = [20*(i+1) for i in range(len(h1))]
        axes[0].plot(epochs, [h*100 for h in h3], 'o-', color='#95a5a6', lw=2, label='Standard')
        axes[0].plot(epochs, [h*100 for h in h1], 's-', color='#3498db', lw=2, label='Parallel Dual')
        axes[0].plot(epochs, [h*100 for h in h2], 'D-', color='#2ecc71', lw=2, label='Sequential Dual')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test PA (%)')
        axes[0].set_title('Learning Curves', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        labels = ['Standard', 'Parallel\nDual', 'Sequential\nDual']
        vals = [h3[-1]*100, h1[-1]*100, h2[-1]*100]
        colors = ['#95a5a6', '#3498db', '#2ecc71']
        axes[1].bar(labels, vals, color=colors, alpha=0.85, edgecolor='black')
        for i, v in enumerate(vals):
            axes[1].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold')
        axes[1].set_ylabel('Test PA (%)'); axes[1].set_title('Final Accuracy', fontweight='bold')
        fig.suptitle('Phase 194: Sequential vs Parallel Dual-Process', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase194_sequential.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'standard': h3[-1], 'parallel': h1[-1], 'sequential': h2[-1]}

if __name__ == '__main__':
    main()
