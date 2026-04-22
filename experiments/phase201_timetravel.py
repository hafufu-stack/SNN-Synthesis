"""
Phase 201: Time-Traveling NCA - Optimal Step Selection

P189 proved: too many steps = Drift destroys memory.
Solution: run T=20 steps, record margin at each step,
then "time travel" back to the step with highest confidence.

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


class TimeTravelNCA(nn.Module):
    """NCA that records state at each step and returns to best one."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32, max_steps=20):
        super().__init__()
        self.max_steps = max_steps
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
        self.update = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
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

    def forward(self, x, task_emb, return_all=False):
        B, _, H, W = x.shape
        te = task_emb.view(B, -1, 1, 1).expand(B, -1, H, W)
        inp = torch.cat([x, te], dim=1)
        state = self.encoder(inp)

        # Record snapshots
        snapshots = []
        margins = []

        for t in range(self.max_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)
            state = beta * state + (1 - beta) * delta

            logits = self.decoder(state)
            snapshots.append(logits)

            # Compute margin: top1 - top2 confidence (higher = more certain)
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                top2 = probs.topk(2, dim=1).values
                margin = (top2[:, 0] - top2[:, 1]).mean().item()
                margins.append(margin)

        if return_all:
            return snapshots, margins

        # Time travel: pick step with highest margin
        best_t = int(np.argmax(margins))
        return snapshots[best_t], best_t, margins

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_and_eval(model, train_tasks, test_tasks, n_epochs, label, time_travel=False):
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

            if time_travel:
                snaps, margins = model(ti, emb, return_all=True)
                # Train on ALL snapshots (multi-step supervision)
                loss = 0
                for snap in snaps:
                    loss += F.cross_entropy(snap[:, :, :oh, :ow], gt[:, :oh, :ow])
                loss /= len(snaps)
            else:
                out = model(ti, emb)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            tpa, tem, best_ts = 0, 0, []
            with torch.no_grad():
                for item in test_tasks:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']

                    if time_travel:
                        out = model(ti, emb)
                        logits, best_t, margins = out
                        best_ts.append(best_t)
                    else:
                        out = model(ti, emb)
                        logits = out[0] if isinstance(out, tuple) else out

                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
                    tem += float((pred == gt[:oh, :ow]).all().item())
            avg_pa = tpa / len(test_tasks)
            avg_em = tem / len(test_tasks)
            hist_pa.append(avg_pa); hist_em.append(avg_em)
            bt = f", BestT={np.mean(best_ts):.1f}" if best_ts else ""
            print(f"    {label} Ep{epoch+1}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%{bt}")
    return hist_pa, hist_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 201: Time-Traveling NCA")
    print(f"  Run T=20, rewind to highest-confidence step")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    C, ep = 64, 100

    # Standard NCA (fixed T=5)
    print(f"\n[Standard NCA (T=5)]")
    m1 = ScalableNCA(11, C, 5, 32).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1_pa, h1_em = train_and_eval(m1, train, test, ep, "Std(T=5)")
    del m1; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Time-Travel NCA (T=20, auto-rewind)
    print(f"\n[Time-Travel NCA (T=20, auto-rewind)]")
    m2 = TimeTravelNCA(11, C, 32, max_steps=20).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2_pa, h2_em = train_and_eval(m2, train, test, ep, "TimeTravel", time_travel=True)
    del m2; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    d = (h2_pa[-1] - h1_pa[-1]) * 100
    print(f"\n{'='*70}")
    print(f"Phase 201 Complete ({elapsed:.0f}s)")
    print(f"  Standard(T=5): PA={h1_pa[-1]*100:.1f}%, EM={h1_em[-1]*100:.1f}%")
    print(f"  TimeTravel:    PA={h2_pa[-1]*100:.1f}%, EM={h2_em[-1]*100:.1f}%")
    print(f"  Delta: {d:+.1f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase201_timetravel.json"), 'w', encoding='utf-8') as f:
        json.dump({'standard': {'pa': h1_pa[-1], 'em': h1_em[-1]},
                   'timetravel': {'pa': h2_pa[-1], 'em': h2_em[-1]},
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        epochs = [20*(i+1) for i in range(len(h1_pa))]
        ax.plot(epochs, [h*100 for h in h1_pa], 'o-', color='#95a5a6', lw=2, label='Standard (T=5)')
        ax.plot(epochs, [h*100 for h in h2_pa], 'D-', color='#9b59b6', lw=2, label='Time-Travel (T=20)')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test PA (%)')
        ax.set_title(f'Phase 201: Time-Travel NCA (={d:+.1f}pp)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase201_timetravel.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")
    gc.collect()
    return {'standard': h1_pa[-1], 'timetravel': h2_pa[-1]}

if __name__ == '__main__':
    main()
