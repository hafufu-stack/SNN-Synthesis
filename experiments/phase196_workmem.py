"""
Phase 196: Working Memory NCA - Grid + Global Variable Space

Give NCA a "notepad": a global vector that can store abstract
concepts (counts, colors, rules) without spatial binding.

Architecture:
  Grid Space: (B, C, H, W) - spatial processing (drawing)
  Global Vector: (B, G) - abstract variable storage (counting, matching)

Each step:
  1. Grid -> Global: Pool grid info into global vector
  2. Global update: MLP processes global vector
  3. Global -> Grid: Broadcast global vector back to all cells
  4. Grid update: Normal NCA step with global context

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


class WorkingMemoryNCA(nn.Module):
    """NCA with global working memory vector."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32,
                 global_dim=32, n_steps=5):
        super().__init__()
        self.n_steps = n_steps
        self.embed_dim = embed_dim
        self.global_dim = global_dim
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

        # Grid -> Global (read from grid)
        self.grid_to_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(C, global_dim), nn.ReLU(),
        )

        # Global update (MLP working memory)
        self.global_update = nn.Sequential(
            nn.Linear(global_dim + embed_dim, global_dim * 2), nn.ReLU(),
            nn.Linear(global_dim * 2, global_dim),
        )
        self.global_gate = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim), nn.Sigmoid()
        )

        # Global -> Grid (write to grid)
        self.global_to_grid = nn.Sequential(
            nn.Linear(global_dim, C), nn.ReLU(),
        )

        # Grid update (NCA with global context)
        self.update = nn.Sequential(
            nn.Conv2d(C + C, C, 3, padding=1), nn.ReLU(),  # C grid + C global_broadcast
            nn.Conv2d(C, C, 1),
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

        # Initialize global working memory
        global_mem = torch.zeros(B, self.global_dim, device=x.device)
        te_flat = task_emb.view(B, -1)

        for t in range(self.n_steps):
            # Step 1: Grid -> Global (read)
            grid_summary = self.grid_to_global(state)  # (B, G)

            # Step 2: Global update (think)
            global_input = torch.cat([grid_summary, te_flat], dim=1)
            new_global = self.global_update(global_input)
            gate = self.global_gate(torch.cat([global_mem, new_global], dim=1))
            global_mem = gate * global_mem + (1 - gate) * new_global

            # Step 3: Global -> Grid (write/broadcast)
            global_broadcast = self.global_to_grid(global_mem)  # (B, C)
            global_spatial = global_broadcast.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)

            # Step 4: Grid update with global context
            combined = torch.cat([state, global_spatial], dim=1)
            delta = self.update(combined)
            beta = self.tau_gate(state)
            state = beta * state + (1 - beta) * delta

        return self.decoder(state), global_mem

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
    print("Phase 196: Working Memory NCA (Grid + Global Vector)")
    print(f"  Give NCA a notepad for abstract variables")
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

    # Working Memory NCA
    print(f"\n[Working Memory NCA]")
    m2 = WorkingMemoryNCA(11, C, embed_dim=32, global_dim=32, n_steps=5).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2 = train_and_eval(m2, train_tasks, test_tasks, n_epochs, "WorkMem")
    del m2; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 196 Complete ({elapsed:.0f}s)")
    print(f"  Standard:       TestPA={h1[-1]*100:.1f}%")
    print(f"  Working Memory: TestPA={h2[-1]*100:.1f}%")
    delta = (h2[-1] - h1[-1]) * 100
    print(f"  Delta: {'+' if delta >= 0 else ''}{delta:.1f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase196_workmem.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 196: Working Memory NCA',
            'timestamp': datetime.now().isoformat(),
            'standard': h1[-1], 'working_memory': h2[-1],
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = [20*(i+1) for i in range(len(h1))]
        axes[0].plot(epochs, [h*100 for h in h1], 'o-', color='#95a5a6', lw=2, label='Standard NCA')
        axes[0].plot(epochs, [h*100 for h in h2], 'D-', color='#9b59b6', lw=2, label='Working Memory')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test PA (%)')
        axes[0].set_title('Learning Curves', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        labels = ['Standard\nNCA', 'Working\nMemory']
        vals = [h1[-1]*100, h2[-1]*100]
        axes[1].bar(labels, vals, color=['#95a5a6', '#9b59b6'], alpha=0.85, edgecolor='black')
        for i, v in enumerate(vals):
            axes[1].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold')
        axes[1].set_ylabel('Test PA (%)'); axes[1].set_title('Final Accuracy', fontweight='bold')
        fig.suptitle('Phase 196: Working Memory NCA (Grid + Global Vector)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase196_workmem.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'standard': h1[-1], 'working_memory': h2[-1]}

if __name__ == '__main__':
    main()
