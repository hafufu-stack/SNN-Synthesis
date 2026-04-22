"""
Phase 195: Heterogeneous NCA with Self-Attention

Break NCA's locality wall: combine 3x3 Conv (local) with
Self-Attention (global) for instant long-range communication.

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


class SpatialSelfAttention(nn.Module):
    """Lightweight self-attention over spatial positions."""
    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = max(1, ch // n_heads)
        self.qkv = nn.Conv2d(ch, 3 * n_heads * self.head_dim, 1)
        self.proj = nn.Conv2d(n_heads * self.head_dim, ch, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)  # (B, 3*nh*hd, H, W)
        qkv = qkv.view(B, 3, self.n_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # q,k,v: (B, nh, hd, N)
        attn = (q.transpose(-1, -2) @ k) * self.scale  # (B, nh, N, N)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-1, -2))  # (B, nh, hd, N)
        out = out.reshape(B, self.n_heads * self.head_dim, H, W)
        return self.proj(out)


class HeterogeneousNCA(nn.Module):
    """NCA with both local Conv and global Self-Attention."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32,
                 n_steps=5, attn_heads=4):
        super().__init__()
        self.n_steps = n_steps
        self.embed_dim = embed_dim
        C = hidden_ch

        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )

        # Local path: 3x3 Conv
        self.local_update = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 1),
        )
        # Global path: Self-Attention
        self.global_attn = SpatialSelfAttention(C, n_heads=attn_heads)
        self.attn_norm = nn.InstanceNorm2d(C)

        # Mix gate: learned balance between local and global
        self.mix_gate = nn.Sequential(
            nn.Conv2d(C * 2, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

        self.tau_gate = nn.Sequential(nn.Conv2d(C, C, 1), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
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
        inp = torch.cat([x, te], dim=1)
        state = self.encoder(inp)

        gate_history = []
        for t in range(self.n_steps):
            local_delta = self.local_update(state)
            global_delta = self.global_attn(self.attn_norm(state))

            # Mix local and global
            mix_input = torch.cat([local_delta, global_delta], dim=1)
            alpha = self.mix_gate(mix_input)  # (B, 1, H, W)
            gate_history.append(alpha.mean().item())
            delta = alpha * local_delta + (1 - alpha) * global_delta

            beta = self.tau_gate(state)
            state = beta * state + (1 - beta) * delta

        return self.decoder(state), gate_history

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_and_eval(model, train_tasks, test_tasks, n_epochs, label):
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
            avg_pa = test_pa / len(test_tasks)
            avg_em = test_em / len(test_tasks)
            test_pa_hist.append(avg_pa)
            print(f"    {label} Ep{epoch+1}: TestPA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%")
    return test_pa_hist


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 195: Heterogeneous NCA with Self-Attention")
    print(f"  Local Conv + Global Attention = break locality wall")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train_tasks = all_tasks[:200]
    test_tasks = all_tasks[200:250]
    C, n_epochs = 64, 100

    # Standard NCA baseline
    print(f"\n[Standard NCA]")
    m1 = ScalableNCA(11, C, n_steps=5, embed_dim=32).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1 = train_and_eval(m1, train_tasks, test_tasks, n_epochs, "Standard")
    del m1; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Heterogeneous NCA
    print(f"\n[Heterogeneous NCA (Conv + Attention)]")
    m2 = HeterogeneousNCA(11, C, embed_dim=32, n_steps=5, attn_heads=4).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2 = train_and_eval(m2, train_tasks, test_tasks, n_epochs, "Hetero")
    del m2; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 195 Complete ({elapsed:.0f}s)")
    print(f"  Standard:      TestPA={h1[-1]*100:.1f}%")
    print(f"  Heterogeneous: TestPA={h2[-1]*100:.1f}%")
    delta = (h2[-1] - h1[-1]) * 100
    print(f"  Delta: {'+' if delta >= 0 else ''}{delta:.1f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase195_hetero.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 195: Heterogeneous NCA',
            'timestamp': datetime.now().isoformat(),
            'standard': h1[-1], 'heterogeneous': h2[-1],
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = [20*(i+1) for i in range(len(h1))]
        axes[0].plot(epochs, [h*100 for h in h1], 'o-', color='#95a5a6', lw=2, label='Standard NCA')
        axes[0].plot(epochs, [h*100 for h in h2], 'D-', color='#e74c3c', lw=2, label='Heterogeneous')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test PA (%)')
        axes[0].set_title('Learning Curves', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        labels = ['Standard\nNCA', 'Heterogeneous\n(Conv+Attn)']
        vals = [h1[-1]*100, h2[-1]*100]
        axes[1].bar(labels, vals, color=['#95a5a6', '#e74c3c'], alpha=0.85, edgecolor='black')
        for i, v in enumerate(vals):
            axes[1].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold')
        axes[1].set_ylabel('Test PA (%)'); axes[1].set_title('Final Accuracy', fontweight='bold')
        fig.suptitle('Phase 195: Heterogeneous NCA (Conv + Self-Attention)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase195_hetero.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'standard': h1[-1], 'heterogeneous': h2[-1]}

if __name__ == '__main__':
    main()
