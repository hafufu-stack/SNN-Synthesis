"""
Phase 225: Pixel-level Sparse Mixture of Experts (SMoE)

Replace GatedHybrid's System 2 with Sparse MoE:
- N independent expert Conv2d modules
- Per-pixel Top-K routing via lightweight gating network
- Massively more parameters, same FLOPs per pixel

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


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts: pixel-level Top-K routing."""
    def __init__(self, channels, n_experts=32, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1) for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.Conv2d(channels, n_experts, 1),
        )
        # Load balancing loss coefficient
        self.aux_loss = 0.0

    def forward(self, x):
        B, C, H, W = x.shape
        gate_logits = self.gate(x)  # (B, n_experts, H, W)

        # Top-K routing per pixel
        gate_logits_flat = gate_logits.permute(0, 2, 3, 1).reshape(-1, self.n_experts)
        topk_vals, topk_idx = gate_logits_flat.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B*H*W, top_k)

        # Compute all expert outputs (efficient for small n_experts)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_exp, C, H, W)
        expert_outs_flat = expert_outs.permute(0, 3, 4, 1, 2).reshape(-1, self.n_experts, C)

        # Gather top-k expert outputs
        bhw = topk_idx.shape[0]
        selected = torch.zeros(bhw, C, device=x.device)
        for k in range(self.top_k):
            idx = topk_idx[:, k]  # (bhw,)
            expert_out_k = expert_outs_flat[torch.arange(bhw, device=x.device), idx]  # (bhw, C)
            selected += topk_weights[:, k:k+1] * expert_out_k

        output = selected.view(B, H, W, C).permute(0, 3, 1, 2)

        # Load balancing aux loss
        gate_probs = F.softmax(gate_logits_flat, dim=-1).mean(dim=0)
        self.aux_loss = (gate_probs * torch.log(gate_probs + 1e-8)).sum() * -1.0

        return F.relu(output)


class MoEGatedHybridNCA(nn.Module):
    """GatedHybrid with Sparse MoE in System 2."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32, s2_steps=10,
                 n_experts=32, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        C = hidden_ch

        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

        # System 1 (same as GatedHybrid)
        self.s1 = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

        # System 2 with MoE
        self.s2_encoder = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 3, padding=1), nn.ReLU(),
        )
        self.s2_moe = SparseMoELayer(C, n_experts=n_experts, top_k=top_k)
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
        state = self.s2_moe(state)  # MoE layer
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
    print("Phase 225: Pixel-level Sparse MoE")
    print(f"  Replace S2 with Sparse Mixture of Experts")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    C, ep = 64, 100

    # Baseline
    print(f"\n[Baseline: Standard GatedHybrid]")
    torch.manual_seed(SEED)
    m0 = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {m0.count_params():,}")
    h0_pa, h0_em = train_and_eval(m0, train, test, ep, "Base")
    del m0; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # MoE configs
    configs = [(16, 2), (32, 2)]
    moe_results = {}

    for n_exp, top_k in configs:
        label = f"MoE({n_exp},k{top_k})"
        print(f"\n[{label}]")
        torch.manual_seed(SEED)
        m = MoEGatedHybridNCA(11, C, 32, 10, n_experts=n_exp, top_k=top_k).to(DEVICE)
        print(f"  Params: {m.count_params():,}")
        h_pa, h_em = train_and_eval(m, train, test, ep, label)
        moe_results[label] = {'pa': h_pa[-1], 'em': h_em[-1], 'params': m.count_params()}
        del m; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  SPARSE MoE RESULTS:")
    print(f"  Baseline:   PA={h0_pa[-1]*100:.1f}%, EM={h0_em[-1]*100:.1f}%")
    for k, r in moe_results.items():
        print(f"  {k:15s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}% ({r['params']:,} params)")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase225_moe.json"), 'w', encoding='utf-8') as f:
        json.dump({'baseline': {'pa': h0_pa[-1], 'em': h0_em[-1]},
                   'moe': moe_results, 'elapsed': elapsed,
                   'timestamp': datetime.now().isoformat()}, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = ['Baseline'] + list(moe_results.keys())
        pa_vals = [h0_pa[-1]*100] + [r['pa']*100 for r in moe_results.values()]
        em_vals = [h0_em[-1]*100] + [r['em']*100 for r in moe_results.values()]
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors[:len(labels)], alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors[:len(labels)], alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 225: Sparse MoE', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase225_moe.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'baseline_pa': h0_pa[-1], 'moe': moe_results}


if __name__ == '__main__':
    main()
