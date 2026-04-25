"""
Phase 208: The 20B Swap Leviathan - CPU-only mega-NCA

CPU-only experiment using swap/virtual memory for massive NCA.
Start small to validate, then scale up.

Strategy:
  - device='cpu' (no VRAM dependency)
  - Scale channels: 1000, 5000, 20000 (incrementally)
  - Very few tasks, very few epochs (due to extreme slowness)
  - Checkpoint after each epoch

NOTE: This is a background/romance experiment. The extreme channel
counts will thrash swap and take hours per epoch. Run alongside
GPU experiments.

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
DEVICE = "cpu"  # FORCED CPU for swap experiment
SEED = 2026

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import load_arc_training, prepare_arc_meta_dataset


class LeviathanNCA(nn.Module):
    """Massive NCA with configurable channel count, CPU only."""
    def __init__(self, n_colors=11, hidden_ch=1000, steps=3, embed_dim=32):
        super().__init__()
        self.steps = steps
        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )
        # Use 1x1 convs to reduce parameter count while keeping massive hidden
        # Full 3x3 at C=45000 -> 3*3*45000^2 = 18.2B params (too much)
        # Strategy: 1x1 bottleneck with 3x3 depthwise (groups=hidden_ch)
        self.encoder = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, hidden_ch, 1), nn.ReLU(),
        )
        # Depthwise 3x3 (groups=hidden_ch) + pointwise 1x1
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, groups=hidden_ch), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_ch, n_colors, 1),
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
        state = self.encoder(torch.cat([x, te], dim=1))
        for t in range(self.steps):
            state = state + self.update(state)
        return self.decoder(state)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def estimate_memory_gb(params):
    """Estimate Adam optimizer memory in GB (param + grad + 2 moments)."""
    return params * 4 * 4 / (1024**3)  # 4 copies * 4 bytes each


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 208: The Swap Leviathan (CPU-only)")
    print(f"  Massive NCA on CPU with virtual memory")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=50)
    random.shuffle(all_tasks)
    train = all_tasks[:10]  # Very few tasks due to extreme slowness
    test = all_tasks[10:20]

    # Scale progression
    channel_configs = [
        (1000, 3, 5),     # ~4M params, ~0.06 GB RAM
        (5000, 3, 5),     # ~100M params, ~1.5 GB RAM
        (20000, 3, 3),    # ~1.6B params, ~24 GB RAM
    ]

    results = []
    for C, T, n_epochs in channel_configs:
        print(f"\n{'='*50}")
        print(f"  Leviathan: C={C}, T={T}")
        torch.manual_seed(SEED)
        model = LeviathanNCA(11, C, T, 32)
        params = model.count_params()
        mem_gb = estimate_memory_gb(params)
        print(f"  Params: {params:,}")
        print(f"  Estimated RAM (Adam): {mem_gb:.1f} GB")

        if mem_gb > 300:
            print(f"  SKIPPING: {mem_gb:.0f} GB exceeds safe limit")
            results.append({'C': C, 'T': T, 'params': params, 'mem_gb': mem_gb,
                           'pa': 0, 'em': 0, 'status': 'skipped'})
            continue

        opt = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD for less memory than Adam
        best_pa = 0

        for epoch in range(n_epochs):
            model.train()
            ep_t0 = time.time()
            random.shuffle(train)
            for item in train[:5]:
                do_t = [d for d in item['demo_outputs']]
                emb = model.encode_task(do_t)
                ti = item['test_input'].unsqueeze(0)
                gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0)
                oh, ow = item['out_h'], item['out_w']
                logits = model(ti, emb)
                loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
                opt.zero_grad(); loss.backward(); opt.step()
            ep_time = time.time() - ep_t0

            # Eval
            model.eval()
            tpa, tem = 0, 0
            with torch.no_grad():
                for item in test:
                    do_t = [d for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0)
                    gt = item['test_output'][:11].argmax(dim=0)
                    oh, ow = item['out_h'], item['out_w']
                    logits = model(ti, emb)
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
                    tem += float((pred == gt[:oh, :ow]).all().item())
            pa = tpa / len(test)
            em = tem / len(test)
            if pa > best_pa: best_pa = pa
            print(f"    Ep{epoch+1}: PA={pa*100:.1f}%, EM={em*100:.1f}%, "
                  f"Loss={loss.item():.3f}, Time={ep_time:.1f}s")

        results.append({'C': C, 'T': T, 'params': params, 'mem_gb': mem_gb,
                       'pa': best_pa, 'em': em, 'status': 'done'})
        del model, opt; gc.collect()
        print(f"  Best PA: {best_pa*100:.1f}%")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 208 Complete ({elapsed:.0f}s)")
    for r in results:
        print(f"  C={r['C']}: {r['params']:,} params, {r['mem_gb']:.1f}GB, "
              f"PA={r['pa']*100:.1f}%, Status={r['status']}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase208_leviathan.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        done = [r for r in results if r['status'] == 'done']
        if done:
            params_list = [r['params']/1e6 for r in done]
            pa_list = [r['pa']*100 for r in done]
            axes[0].plot(params_list, pa_list, 'o-', color='#e74c3c', lw=2, ms=10)
            axes[0].set_xlabel('Parameters (M)'); axes[0].set_ylabel('Best PA (%)')
            axes[0].set_title('Leviathan Scaling', fontweight='bold')
            axes[0].set_xscale('log'); axes[0].grid(True, alpha=0.3)
            for r in done:
                axes[0].annotate(f"C={r['C']}", (r['params']/1e6, r['pa']*100),
                               textcoords="offset points", xytext=(8, 5), fontsize=9)

            mem_list = [r['mem_gb'] for r in done]
            axes[1].bar(range(len(done)), mem_list, color='#9b59b6', alpha=0.8)
            axes[1].set_xticks(range(len(done)))
            axes[1].set_xticklabels([f"C={r['C']}" for r in done])
            axes[1].set_ylabel('Estimated RAM (GB)')
            axes[1].set_title('Memory Usage', fontweight='bold')

        fig.suptitle('Phase 208: The Swap Leviathan', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.08, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase208_leviathan.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
