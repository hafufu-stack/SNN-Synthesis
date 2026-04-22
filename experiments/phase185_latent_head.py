"""
Phase 185: Latent Confidence Head - Zero-Cost Sleep via Distillation

P184 proved: Margin (top1-top2) gives same accuracy as Entropy Sleep,
but speed doesn't improve because calling the Decoder is still required.

Solution: Train a lightweight 1x1 Conv head to predict margin directly
from latent state. During inference, skip the heavy Decoder entirely.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
EMB_DIM = 64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset
)


class ConfidenceHead(nn.Module):
    """Lightweight 1x1 Conv: latent_ch -> 1 (margin prediction)."""
    def __init__(self, latent_ch=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(latent_ch, 8, 1), nn.ReLU(),
            nn.Conv2d(8, 1, 1)
        )
    def forward(self, latent_state):
        return self.head(latent_state).squeeze(1)  # (B, H, W)


def train_confidence_head(model, head, train_tasks, n_epochs=5, lr=1e-3):
    """Distill decoder-based margin into lightweight latent head."""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    nca = model.latent_nca

    for epoch in range(n_epochs):
        total_loss = 0
        random.shuffle(train_tasks)
        for item in train_tasks[:100]:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            with torch.no_grad():
                emb = model.task_encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            B, _, H, W = ti.shape

            # Forward through NCA to get latent state at various steps
            state = nca.encoder(ti)
            te = emb.view(1, -1, 1, 1).expand(B, -1, H, W)
            for step in range(random.randint(3, 8)):
                state_ctx = torch.cat([state, te], dim=1)
                delta = nca.update(state_ctx)
                beta = nca.tau_gate(state_ctx)
                state = beta * state + (1 - beta) * delta

            # Teacher: compute actual margin from decoder (detached)
            with torch.no_grad():
                logits = nca.decoder(state)
                sorted_logits, _ = logits[:, :10].sort(dim=1, descending=True)
                true_margin = sorted_logits[:, 0] - sorted_logits[:, 1]  # (B, H, W)

            # Student: predict margin from latent state
            pred_margin = head(state)

            loss = F.mse_loss(pred_margin, true_margin)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / min(100, len(train_tasks))
        print(f"    Epoch {epoch+1}/{n_epochs}: MSE={avg_loss:.4f}")

    head.eval()
    return head


def inference_latent_sleep(model, head, x, task_embed, n_steps=8, margin_thresh=2.0):
    """Phase 185: Sleep using latent confidence head (no decoder calls mid-inference)."""
    nca = model.latent_nca
    B, _, H, W = x.shape
    state = nca.encoder(x)
    te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)
    sleep_ratio_hist = []

    for step in range(n_steps):
        state_ctx = torch.cat([state, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        new_state = beta * state + (1 - beta) * delta

        # Latent confidence head: predict margin without decoder!
        pred_margin = head(new_state)  # (B, H, W)
        confident = (pred_margin > margin_thresh).float().unsqueeze(1).expand_as(state)
        sleep_ratio_hist.append(confident[:, 0].mean().item())
        state = confident * state + (1 - confident) * new_state

    return nca.decoder(state), sleep_ratio_hist


def inference_margin_sleep(model, x, task_embed, n_steps=8, margin_thresh=2.0):
    """P184 baseline: margin via decoder each step."""
    nca = model.latent_nca
    B, _, H, W = x.shape
    state = nca.encoder(x)
    te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

    for step in range(n_steps):
        state_ctx = torch.cat([state, te], dim=1)
        delta = nca.update(state_ctx)
        beta = nca.tau_gate(state_ctx)
        new_state = beta * state + (1 - beta) * delta
        logits = nca.decoder(new_state)
        sorted_l, _ = logits[:, :10].sort(dim=1, descending=True)
        margin = sorted_l[:, 0] - sorted_l[:, 1]
        confident = (margin > margin_thresh).float().unsqueeze(1).expand_as(state)
        state = confident * state + (1 - confident) * new_state
    return nca.decoder(state)


def inference_standard(model, x, task_embed, n_steps=8):
    return model.latent_nca(x, task_embed, n_steps=n_steps)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 185: Latent Confidence Head - Zero-Cost Sleep")
    print(f"  Distill margin into 1x1 Conv, skip decoder mid-inference")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Loading Foundation Model...")
    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    ckpt_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train_tasks = all_tasks[:200]
    test_tasks = all_tasks[200:250]

    print("\n[Step 3] Training Confidence Head (distillation)...")
    head = ConfidenceHead(latent_ch=32).to(DEVICE)
    head = train_confidence_head(model, head, train_tasks, n_epochs=5)

    print("\n[Step 4] Benchmarking 3 methods on test set...")
    methods = {
        'standard': {'pas': [], 'ems': [], 'time': 0},
        'margin_decoder': {'pas': [], 'ems': [], 'time': 0},
        'latent_head': {'pas': [], 'ems': [], 'time': 0},
    }

    with torch.no_grad():
        for item in test_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.task_encoder(di, do)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            # Standard
            t1 = time.time()
            logits = inference_standard(model, ti, emb)
            methods['standard']['time'] += time.time() - t1
            pred = logits[0, :10].argmax(dim=0)
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            methods['standard']['pas'].append(pa); methods['standard']['ems'].append(em)

            # Margin via decoder
            t1 = time.time()
            logits = inference_margin_sleep(model, ti, emb)
            methods['margin_decoder']['time'] += time.time() - t1
            pred = logits[0, :10].argmax(dim=0)
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            methods['margin_decoder']['pas'].append(pa); methods['margin_decoder']['ems'].append(em)

            # Latent head
            t1 = time.time()
            logits, slp = inference_latent_sleep(model, head, ti, emb)
            methods['latent_head']['time'] += time.time() - t1
            pred = logits[0, :10].argmax(dim=0)
            pa = (pred[:oh, :ow] == gt[:oh, :ow]).float().mean().item()
            em = float((pred[:oh, :ow] == gt[:oh, :ow]).all().item())
            methods['latent_head']['pas'].append(pa); methods['latent_head']['ems'].append(em)

    elapsed = time.time() - t0
    summary = {}
    std_time = methods['standard']['time']
    print(f"\n{'='*70}")
    print(f"Phase 185 Complete ({elapsed:.0f}s)")
    for name, m in methods.items():
        avg_pa = np.mean(m['pas']); avg_em = np.mean(m['ems'])
        speedup = std_time / max(0.001, m['time'])
        summary[name] = {'pa': avg_pa, 'em': avg_em, 'time': m['time'], 'speedup': speedup}
        print(f"  {name:20s}: PA={avg_pa*100:.2f}%, Speed={speedup:.2f}x")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase185_latent_head.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 185: Latent Confidence Head',
            'timestamp': datetime.now().isoformat(),
            'summary': summary, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Save head
    torch.save(head.state_dict(), os.path.join(RESULTS_DIR, "phase185_confidence_head.pt"))
    print("  Head saved!")

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        labels = ['Standard', 'Margin\n(Decoder)', 'Latent\nHead']
        keys = ['standard', 'margin_decoder', 'latent_head']
        pa_vals = [summary[k]['pa']*100 for k in keys]
        speed_vals = [summary[k]['speedup'] for k in keys]
        colors = ['#95a5a6', '#e74c3c', '#2ecc71']
        axes[0].bar(labels, pa_vals, color=colors, alpha=0.85, edgecolor='black')
        axes[0].set_ylabel('PA (%)'); axes[0].set_title('Accuracy', fontweight='bold')
        axes[1].bar(labels, speed_vals, color=colors, alpha=0.85, edgecolor='black')
        axes[1].set_ylabel('Speedup (vs Std)'); axes[1].set_title('Speed', fontweight='bold')
        axes[1].axhline(1.0, color='black', linewidth=0.5)
        fig.suptitle('Phase 185: Latent Confidence Head', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase185_latent_head.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return summary

if __name__ == '__main__':
    main()
