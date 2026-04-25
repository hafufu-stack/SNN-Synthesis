"""
Phase 240: U-NCA (Hierarchical Neural Cellular Automata)

Past Locality Wall (P195-198): global communication broke geometry.
Solution: U-Net-style multi-scale NCA that preserves topology.

Each update step:
- 1x scale: 3x3 Conv (local, standard NCA)
- 1/2 scale: AvgPool -> 3x3 Conv -> Upsample (medium range)
- 1/4 scale: AvgPool -> 3x3 Conv -> Upsample (global view)
Concatenate all scales -> fusion Conv -> update

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


class UNCAUpdateBlock(nn.Module):
    """Multi-scale update: local + medium + global vision."""
    def __init__(self, C):
        super().__init__()
        # Local (1x)
        self.local_conv = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )
        # Medium (1/2x)
        self.med_conv = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )
        # Global (1/4x)
        self.glob_conv = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )
        # Fusion (3C -> C)
        self.fusion = nn.Sequential(
            nn.Conv2d(C * 3, C, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1),
        )

    def forward(self, state):
        _, _, H, W = state.shape
        # Local
        local_out = self.local_conv(state)
        # Medium: downsample -> conv -> upsample
        # Ensure even dimensions for pooling
        pad_h = H % 2; pad_w = W % 2
        if pad_h or pad_w:
            state_padded = F.pad(state, (0, pad_w, 0, pad_h))
        else:
            state_padded = state
        med_down = F.avg_pool2d(state_padded, 2)
        med_out = self.med_conv(med_down)
        med_up = F.interpolate(med_out, size=(H, W), mode='bilinear', align_corners=False)
        # Global: downsample 4x -> conv -> upsample
        Hp, Wp = state_padded.shape[2], state_padded.shape[3]
        pad_h4 = (4 - Hp % 4) % 4; pad_w4 = (4 - Wp % 4) % 4
        state_padded4 = F.pad(state_padded, (0, pad_w4, 0, pad_h4))
        glob_down = F.avg_pool2d(state_padded4, 4)
        glob_out = self.glob_conv(glob_down)
        glob_up = F.interpolate(glob_out, size=(H, W), mode='bilinear', align_corners=False)
        # Concatenate and fuse
        combined = torch.cat([local_out, med_up, glob_up], dim=1)
        return self.fusion(combined)


class UGatedNCA(nn.Module):
    """GatedHybrid with U-Net-style hierarchical update."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32, s2_steps=10):
        super().__init__()
        self.embed_dim = embed_dim
        C = hidden_ch

        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

        # System 1: Fast sketch
        self.s1 = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

        # System 2: U-NCA (hierarchical update)
        self.s2_encoder = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )
        self.s2_update = UNCAUpdateBlock(C)
        self.s2_tau = nn.Sequential(nn.Conv2d(C, C, 1), nn.Sigmoid())
        self.s2_decoder = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
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


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 240: U-NCA (Hierarchical Neural Cellular Automata)")
    print(f"  Multi-scale update: local + medium + global")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    ep = 100

    # Baseline: GatedHybrid
    print(f"\n[Baseline: GatedHybrid C=64]")
    torch.manual_seed(SEED)
    m0 = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    print(f"  Params: {m0.count_params():,}")
    from phase199_gated import train_and_eval
    h0_pa, h0_em = train_and_eval(m0, train, test, ep, "Base")
    del m0; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # U-NCA C=64
    print(f"\n[U-NCA C=64]")
    torch.manual_seed(SEED)
    m1 = UGatedNCA(11, 64, 32, 10).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1_pa, h1_em = train_and_eval(m1, train, test, ep, "UNCA64")
    del m1; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # U-NCA C=96 (bigger brain)
    print(f"\n[U-NCA C=96]")
    torch.manual_seed(SEED)
    m2 = UGatedNCA(11, 96, 32, 10).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2_pa, h2_em = train_and_eval(m2, train, test, ep, "UNCA96")
    del m2; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    results = {
        'baseline': {'pa': h0_pa[-1], 'em': h0_em[-1]},
        'unca64': {'pa': h1_pa[-1], 'em': h1_em[-1]},
        'unca96': {'pa': h2_pa[-1], 'em': h2_em[-1]},
    }

    print(f"\n{'='*70}")
    print(f"  U-NCA RESULTS:")
    for k, r in results.items():
        print(f"  {k:10s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase240_unca.json"), 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(results.keys())
        pa_vals = [r['pa']*100 for r in results.values()]
        em_vals = [r['em']*100 for r in results.values()]
        colors = ['#95a5a6', '#3498db', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('%'); ax.set_title('Phase 240: U-NCA Hierarchical', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase240_unca.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
