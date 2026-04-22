"""
Phase 192: Dual-Process NCA - System 1 & System 2

P189 proved: Time DESTROYS memory (T=1 is best for memorization).
But time is needed for reasoning (P174-176 showed T>1 helps ARC).

Solution: Split NCA into two parallel pathways:
  System 1 (Memory/Hippocampus): T=1, 1x1 Conv, preserves input exactly
  System 2 (Reasoning/Cortex): T=10, 3x3 NCA, iterative rule application
  Final: Learnable gate fuses both outputs

This is Kahneman's Dual Process Theory in silicon!

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


class DualProcessNCA(nn.Module):
    """Dual-Process NCA: System 1 (Memory) + System 2 (Reasoning)."""
    def __init__(self, n_colors=11, hidden_ch=32, embed_dim=32,
                 s1_steps=1, s2_steps=10):
        super().__init__()
        self.embed_dim = embed_dim
        C = hidden_ch

        # Shared task encoder
        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

        # ===== System 1: Memory (T=1, 1x1 Conv only) =====
        # No spatial mixing -> preserves exact pixel positions
        self.s1_encoder = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 1), nn.ReLU(),
            nn.Conv2d(C, C, 1), nn.ReLU(),
        )
        self.s1_decoder = nn.Sequential(
            nn.Conv2d(C, C, 1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )
        self.s1_steps = s1_steps  # typically 1

        # ===== System 2: Reasoning (T=10, 3x3 NCA) =====
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
        self.s2_steps = s2_steps  # typically 10

        # ===== Fusion Gate =====
        # Learnable per-pixel gate: how much to trust memory vs reasoning
        self.fusion_gate = nn.Sequential(
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
        inp = torch.cat([x, te], dim=1)

        # System 1: Fast memory (T=1, no spatial mixing)
        s1_state = self.s1_encoder(inp)
        s1_out = self.s1_decoder(s1_state)

        # System 2: Slow reasoning (T=10, full NCA dynamics)
        s2_state = self.s2_encoder(inp)
        for t in range(self.s2_steps):
            delta = self.s2_update(s2_state)
            beta = self.s2_tau(s2_state)
            s2_state = beta * s2_state + (1 - beta) * delta
        s2_out = self.s2_decoder(s2_state)

        # Fusion: gate between memory and reasoning
        gate_input = torch.cat([s1_out, s2_out], dim=1)
        gate = self.fusion_gate(gate_input)  # (B, 1, H, W)
        output = gate * s1_out + (1 - gate) * s2_out

        return output, s1_out, s2_out, gate

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_and_eval_model(model, train_tasks, test_tasks, n_epochs=100, label="model"):
    """Train model and track metrics."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    test_pa_hist = []

    for epoch in range(n_epochs):
        model.train()
        total_pa = 0
        random.shuffle(train_tasks)
        for item in train_tasks[:50]:
            do_tensors = [d.to(DEVICE) for d in item['demo_outputs']]
            task_emb = model.encode_task(do_tensors)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            output = model(ti, task_emb)
            logits = output[0] if isinstance(output, tuple) else output
            loss = F.cross_entropy(logits[:, :10, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()

            with torch.no_grad():
                pred = logits[0, :10, :oh, :ow].argmax(dim=0)
                total_pa += (pred == gt[0, :oh, :ow]).float().mean().item()

        # Test every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            test_pa, test_em = 0, 0
            gate_means = []
            with torch.no_grad():
                for item in test_tasks:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    output = model(ti, emb)
                    logits = output[0] if isinstance(output, tuple) else output
                    pred = logits[0, :10, :oh, :ow].argmax(dim=0)
                    pa = (pred == gt[:oh, :ow]).float().mean().item()
                    em = float((pred == gt[:oh, :ow]).all().item())
                    test_pa += pa; test_em += em
                    if isinstance(output, tuple) and len(output) >= 4:
                        gate_means.append(output[3][0, 0, :oh, :ow].mean().item())

            avg_test_pa = test_pa / len(test_tasks)
            avg_test_em = test_em / len(test_tasks)
            avg_gate = np.mean(gate_means) if gate_means else -1
            test_pa_hist.append(avg_test_pa)
            gate_str = f", Gate(S1)={avg_gate:.2f}" if avg_gate >= 0 else ""
            print(f"    {label} Epoch {epoch+1}: TestPA={avg_test_pa*100:.1f}%, "
                  f"TestEM={avg_test_em*100:.1f}%{gate_str}")

    return test_pa_hist


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 192: Dual-Process NCA - System 1 & System 2")
    print(f"  Memory (T=1, 1x1) + Reasoning (T=10, 3x3) + Fusion Gate")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train_tasks = all_tasks[:200]
    test_tasks = all_tasks[200:250]

    # Test at C=64 (good balance of speed and capacity)
    C = 64
    n_epochs = 100

    # Baseline 1: Standard NCA (T=5)
    print(f"\n[Step 2a] Baseline: Standard NCA (T=5, C={C})...")
    model_std = ScalableNCA(11, C, n_steps=5, embed_dim=32).to(DEVICE)
    print(f"  Params: {model_std.count_params():,}")
    hist_std = train_and_eval_model(model_std, train_tasks, test_tasks, n_epochs, "Standard")
    del model_std; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # Baseline 2: Memory-only NCA (T=1)
    print(f"\n[Step 2b] Baseline: Memory NCA (T=1, C={C})...")
    model_mem = ScalableNCA(11, C, n_steps=1, embed_dim=32).to(DEVICE)
    print(f"  Params: {model_mem.count_params():,}")
    hist_mem = train_and_eval_model(model_mem, train_tasks, test_tasks, n_epochs, "Memory(T=1)")
    del model_mem; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # Dual-Process NCA
    print(f"\n[Step 2c] Dual-Process NCA (S1=T1+1x1, S2=T10+3x3, C={C})...")
    model_dual = DualProcessNCA(11, C, embed_dim=32, s1_steps=1, s2_steps=10).to(DEVICE)
    print(f"  Params: {model_dual.count_params():,}")
    hist_dual = train_and_eval_model(model_dual, train_tasks, test_tasks, n_epochs, "DualProcess")
    del model_dual; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 192 Complete ({elapsed:.0f}s)")
    print(f"  Standard (T=5): Final TestPA = {hist_std[-1]*100:.1f}%")
    print(f"  Memory (T=1):   Final TestPA = {hist_mem[-1]*100:.1f}%")
    print(f"  Dual-Process:   Final TestPA = {hist_dual[-1]*100:.1f}%")
    best = max(hist_std[-1], hist_mem[-1], hist_dual[-1])
    if hist_dual[-1] == best:
        print(f"  -> Dual-Process WINS!")
    print(f"{'='*70}")

    summary = {
        'standard': {'final_test_pa': hist_std[-1], 'history': hist_std},
        'memory_t1': {'final_test_pa': hist_mem[-1], 'history': hist_mem},
        'dual_process': {'final_test_pa': hist_dual[-1], 'history': hist_dual},
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase192_dual_process.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 192: Dual-Process NCA',
            'timestamp': datetime.now().isoformat(),
            'summary': {k: v['final_test_pa'] for k, v in summary.items()},
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = [20*(i+1) for i in range(len(hist_std))]
        axes[0].plot(epochs, [h*100 for h in hist_std], 'o-', color='#95a5a6',
                    linewidth=2, label='Standard (T=5)')
        axes[0].plot(epochs, [h*100 for h in hist_mem], 's-', color='#e74c3c',
                    linewidth=2, label='Memory (T=1)')
        axes[0].plot(epochs, [h*100 for h in hist_dual], 'D-', color='#2ecc71',
                    linewidth=2, label='Dual-Process')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test PA (%)')
        axes[0].set_title('Learning Curves', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        labels = ['Standard\n(T=5)', 'Memory\n(T=1)', 'Dual\nProcess']
        final_pas = [hist_std[-1]*100, hist_mem[-1]*100, hist_dual[-1]*100]
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        axes[1].bar(labels, final_pas, color=colors, alpha=0.85, edgecolor='black')
        for i, v in enumerate(final_pas):
            axes[1].text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold')
        axes[1].set_ylabel('Test PA (%)'); axes[1].set_title('Final Accuracy', fontweight='bold')

        fig.suptitle('Phase 192: Dual-Process NCA (System 1 + System 2)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase192_dual_process.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return summary

if __name__ == '__main__':
    main()
