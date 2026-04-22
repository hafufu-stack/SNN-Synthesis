"""
Phase 191: Generalization Scaling Law - How Does Reasoning Scale?

P188 measured memorization: M ~ P^1.33
Now measure GENERALIZATION: train on ARC tasks, test on unseen tasks.
Does PA scale with P? Where does it saturate?

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


class ScalableNCA(nn.Module):
    """NCA with scalable hidden channels + simple task conditioning."""
    def __init__(self, n_colors=11, hidden_ch=32, n_steps=5, embed_dim=32):
        super().__init__()
        self.n_steps = n_steps
        self.embed_dim = embed_dim
        C = hidden_ch

        # Task encoder: average demo outputs -> embedding
        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

        # Encoder: input grid + embedding -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )
        self.update = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 1),
        )
        self.tau_gate = nn.Sequential(
            nn.Conv2d(C, C, 1), nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

    def encode_task(self, demo_outputs):
        """Encode task from demo outputs."""
        # Average over all demos
        embeddings = []
        for do in demo_outputs:
            emb = self.demo_encoder(do.unsqueeze(0))
            embeddings.append(emb)
        return torch.stack(embeddings).mean(dim=0)  # (1, embed_dim)

    def forward(self, x, task_emb):
        B, _, H, W = x.shape
        # Broadcast embedding to spatial dims
        te = task_emb.view(B, -1, 1, 1).expand(B, -1, H, W)
        inp = torch.cat([x, te], dim=1)
        state = self.encoder(inp)
        for t in range(self.n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)
            state = beta * state + (1 - beta) * delta
        return self.decoder(state)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_and_eval(hidden_ch, train_tasks, test_tasks, n_epochs=100, n_steps=5):
    """Train at given scale, return train & test PA."""
    torch.manual_seed(SEED)
    model = ScalableNCA(11, hidden_ch, n_steps, embed_dim=32).to(DEVICE)
    n_params = model.count_params()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_pas, test_pas = [], []

    for epoch in range(n_epochs):
        model.train()
        total_loss, total_pa = 0, 0
        random.shuffle(train_tasks)
        for item in train_tasks[:50]:  # subsample per epoch for speed
            do_tensors = [d.to(DEVICE) for d in item['demo_outputs']]
            task_emb = model.encode_task(do_tensors)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            logits = model(ti, task_emb)
            loss = F.cross_entropy(logits[:, :10, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

            with torch.no_grad():
                pred = logits[0, :10, :oh, :ow].argmax(dim=0)
                pa = (pred == gt[0, :oh, :ow]).float().mean().item()
                total_pa += pa

        avg_train_pa = total_pa / min(50, len(train_tasks))

        # Evaluate on test set every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            test_pa_total = 0
            with torch.no_grad():
                for item in test_tasks:
                    do_tensors = [d.to(DEVICE) for d in item['demo_outputs']]
                    task_emb = model.encode_task(do_tensors)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    logits = model(ti, task_emb)
                    pred = logits[0, :10, :oh, :ow].argmax(dim=0)
                    pa = (pred == gt[:oh, :ow]).float().mean().item()
                    test_pa_total += pa
            avg_test_pa = test_pa_total / len(test_tasks)
            test_pas.append(avg_test_pa)
            print(f"    C={hidden_ch} Epoch {epoch+1}: TrainPA={avg_train_pa*100:.1f}%, "
                  f"TestPA={avg_test_pa*100:.1f}%")

    del model, opt
    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    return {
        'hidden_ch': hidden_ch, 'params': n_params,
        'final_train_pa': avg_train_pa,
        'final_test_pa': test_pas[-1] if test_pas else 0,
        'test_pa_history': test_pas
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 191: Generalization Scaling Law")
    print(f"  How does test PA scale with model size?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train_tasks = all_tasks[:200]
    test_tasks = all_tasks[200:250]

    print("\n[Step 2] Training at different scales...")
    channel_sizes = [16, 32, 64, 128, 256, 512]
    results = {}

    for C in channel_sizes:
        print(f"\n  --- C={C} ---", flush=True)
        # Use CPU for C>=512 to avoid VRAM issues
        device_for_size = "cpu" if C >= 512 and DEVICE == "cuda" else DEVICE
        old_device = globals().get('DEVICE')

        result = train_and_eval(C, train_tasks, test_tasks, n_epochs=100, n_steps=5)
        results[C] = result

    # Fit scaling law: PA = a - b * P^(-gamma) (saturation curve)
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 191 Complete ({elapsed:.0f}s)")
    for C in channel_sizes:
        r = results[C]
        print(f"  C={C:4d}: P={r['params']:>10,}, TrainPA={r['final_train_pa']*100:.1f}%, "
              f"TestPA={r['final_test_pa']*100:.1f}%")

    # Log-log regression on test PA
    params_list = [results[C]['params'] for C in channel_sizes]
    test_pas = [results[C]['final_test_pa'] for C in channel_sizes]

    if len(params_list) >= 3:
        log_p = np.log(params_list)
        log_pa = np.log(np.clip(test_pas, 0.01, 1.0))
        beta_gen, log_alpha_gen = np.polyfit(log_p, log_pa, 1)
        alpha_gen = np.exp(log_alpha_gen)
        r_sq = 1 - np.sum((log_pa - (log_alpha_gen + beta_gen * log_p))**2) / \
               np.sum((log_pa - np.mean(log_pa))**2)
        print(f"\n  GENERALIZATION LAW: PA ~ {alpha_gen:.4f} * P^{beta_gen:.4f} (R2={r_sq:.3f})")
    else:
        alpha_gen, beta_gen, r_sq = 0, 0, 0

    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase191_generalization.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 191: Generalization Scaling Law',
            'timestamp': datetime.now().isoformat(),
            'results': {str(C): {k: v for k, v in r.items() if k != 'test_pa_history'}
                       for C, r in results.items()},
            'scaling_law': {'alpha': alpha_gen, 'beta': beta_gen, 'r_squared': r_sq},
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        ps = [results[C]['params'] for C in channel_sizes]
        train_pa = [results[C]['final_train_pa']*100 for C in channel_sizes]
        test_pa = [results[C]['final_test_pa']*100 for C in channel_sizes]

        # Log-log: Test PA vs P
        axes[0].semilogx(ps, test_pa, 'o-', color='#2ecc71', linewidth=2, label='Test PA')
        axes[0].semilogx(ps, train_pa, 's--', color='#e74c3c', linewidth=2, label='Train PA')
        axes[0].set_xlabel('Parameters (P)'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title('Generalization Scaling', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # Generalization gap
        gap = [t - te for t, te in zip(train_pa, test_pa)]
        axes[1].bar([f'C={C}' for C in channel_sizes], gap, color='#f39c12', alpha=0.85)
        axes[1].set_ylabel('Train PA - Test PA (pp)')
        axes[1].set_title('Generalization Gap', fontweight='bold')

        # Learning curves comparison
        for idx, C in enumerate(channel_sizes[:4]):
            hist = results[C].get('test_pa_history', [])
            if hist:
                epochs = [20*(i+1) for i in range(len(hist))]
                axes[2].plot(epochs, [h*100 for h in hist], 'o-', label=f'C={C}')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Test PA (%)')
        axes[2].set_title('Learning Curves', fontweight='bold')
        axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 191: Generalization Scaling Law', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.16, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase191_generalization.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    main()
