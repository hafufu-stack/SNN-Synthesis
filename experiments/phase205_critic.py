"""
Phase 205: Auto-Encoder Critic - Geometric Aesthetics as Meta-Cognition

Train a lightweight Auto-Encoder on ARC output images ONLY.
The AE learns "what beautiful ARC grids look like" (clean lines,
symmetry, no stray pixels). At test time, NCA outputs with high
reconstruction error are "ugly" (contain stains/errors).

Use AE reconstruction loss as:
  (a) An additional TTCT loss term
  (b) A scoring function for Phase 204's Micro-Beam branches

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


class ArcAutoEncoder(nn.Module):
    """Lightweight AE that learns geometric structure of ARC grids."""
    def __init__(self, n_colors=11, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 32 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, n_colors, 1),
        )
        self.n_colors = n_colors

    def forward(self, x):
        """x: (B, n_colors, H, W) one-hot encoded grid."""
        z = self.encoder(x)
        # Decode back to fixed 4x4, then interpolate to input size
        h = self.decoder_fc(z).view(-1, 32, 4, 4)
        h = F.interpolate(h, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        recon = self.decoder(h)
        return recon, z

    def reconstruction_error(self, x):
        """Compute per-sample reconstruction error (MSE on logits)."""
        recon, _ = self(x)
        # Per-sample MSE
        err = ((recon - x) ** 2).mean(dim=(1, 2, 3))
        return err


def train_autoencoder(ae, output_grids, n_epochs=200, lr=1e-3):
    """Train AE on ARC output grids only."""
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        random.shuffle(output_grids)
        epoch_loss = 0
        for i in range(0, len(output_grids), 16):
            batch = torch.stack(output_grids[i:i+16]).to(DEVICE)
            recon, z = ae(batch)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(output_grids) // 16)
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f"    AE Epoch {epoch+1}: Loss={avg_loss:.4f}")
    return losses


def train_nca_with_critic(model, ae, train_tasks, test_tasks, n_epochs, label, critic_weight=0.1):
    """Train NCA with AE critic as auxiliary loss."""
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
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out

            # Standard CE loss
            ce_loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])

            # Critic loss: softmax output should look like a "clean" ARC grid
            soft_output = F.softmax(logits[:, :, :oh, :ow], dim=1)
            with torch.no_grad():
                ae.eval()
            recon_err = ae.reconstruction_error(soft_output)
            critic_loss = recon_err.mean()

            loss = ce_loss + critic_weight * critic_loss
            opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            tpa, tem = 0, 0
            with torch.no_grad():
                for item in test_tasks:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    out = model(ti, emb)
                    logits = out[0] if isinstance(out, tuple) else out
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
                    tem += float((pred == gt[:oh, :ow]).all().item())
            avg_pa = tpa / len(test_tasks)
            avg_em = tem / len(test_tasks)
            hist_pa.append(avg_pa); hist_em.append(avg_em)
            print(f"    {label} Ep{epoch+1}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%")
    return hist_pa, hist_em


def train_nca_standard(model, train_tasks, test_tasks, n_epochs, label):
    """Standard NCA training (no critic)."""
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
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            tpa, tem = 0, 0
            with torch.no_grad():
                for item in test_tasks:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    out = model(ti, emb)
                    logits = out[0] if isinstance(out, tuple) else out
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
                    tem += float((pred == gt[:oh, :ow]).all().item())
            avg_pa = tpa / len(test_tasks)
            avg_em = tem / len(test_tasks)
            hist_pa.append(avg_pa); hist_em.append(avg_em)
            print(f"    {label} Ep{epoch+1}: PA={avg_pa*100:.1f}%, EM={avg_em*100:.1f}%")
    return hist_pa, hist_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 205: Auto-Encoder Critic (Geometric Aesthetics)")
    print(f"  Train AE on ARC outputs, use recon error as critic loss")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    C, ep = 64, 100

    # Step 1: Collect all output grids for AE training
    print(f"\n[Step 1: Collecting ARC output grids for AE]")
    output_grids = []
    for item in train:
        gt_oh = item['test_output'][:11]  # one-hot (11, H, W)
        output_grids.append(gt_oh)
        for do in item['demo_outputs']:
            output_grids.append(do)
    print(f"  Collected {len(output_grids)} output grids")

    # Step 2: Train Auto-Encoder
    print(f"\n[Step 2: Training Auto-Encoder]")
    ae = ArcAutoEncoder(n_colors=11, latent_dim=32).to(DEVICE)
    ae_params = sum(p.numel() for p in ae.parameters())
    print(f"  AE Params: {ae_params:,}")
    ae_losses = train_autoencoder(ae, output_grids, n_epochs=200, lr=1e-3)

    # Freeze AE for critic use
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    # Step 3: Baseline Gated Hybrid (no critic)
    print(f"\n[Step 3: Gated Hybrid Baseline (no critic)]")
    m1 = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {m1.count_params():,}")
    h1_pa, h1_em = train_nca_standard(m1, train, test, ep, "Baseline")
    del m1; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Step 4: Gated Hybrid with AE Critic (weight=0.1)
    print(f"\n[Step 4: Gated Hybrid + AE Critic (w=0.1)]")
    m2 = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {m2.count_params():,}")
    h2_pa, h2_em = train_nca_with_critic(m2, ae, train, test, ep, "Critic0.1", 0.1)
    del m2; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # Step 5: Gated Hybrid with AE Critic (weight=0.5)
    print(f"\n[Step 5: Gated Hybrid + AE Critic (w=0.5)]")
    m3 = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    print(f"  Params: {m3.count_params():,}")
    h3_pa, h3_em = train_nca_with_critic(m3, ae, train, test, ep, "Critic0.5", 0.5)
    del m3; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 205 Complete ({elapsed:.0f}s)")
    print(f"  Baseline:    PA={h1_pa[-1]*100:.1f}%, EM={h1_em[-1]*100:.1f}%")
    print(f"  Critic(0.1): PA={h2_pa[-1]*100:.1f}%, EM={h2_em[-1]*100:.1f}%")
    print(f"  Critic(0.5): PA={h3_pa[-1]*100:.1f}%, EM={h3_em[-1]*100:.1f}%")
    print(f"{'='*70}")

    del ae; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase205_critic.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': {'pa': h1_pa[-1], 'em': h1_em[-1]},
            'critic_01': {'pa': h2_pa[-1], 'em': h2_em[-1]},
            'critic_05': {'pa': h3_pa[-1], 'em': h3_em[-1]},
            'ae_params': ae_params,
            'ae_final_loss': ae_losses[-1] if ae_losses else None,
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # PA curves
        epochs = [20*(i+1) for i in range(len(h1_pa))]
        for h, c, l in [(h1_pa,'#3498db','Baseline'), (h2_pa,'#e67e22','Critic(0.1)'),
                         (h3_pa,'#e74c3c','Critic(0.5)')]:
            axes[0].plot(epochs, [v*100 for v in h], 'o-', color=c, lw=2, label=l)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Test PA (%)')
        axes[0].set_title('PA: Critic vs Baseline', fontweight='bold')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # PA/EM bar chart
        labels = ['Base', 'C(0.1)', 'C(0.5)']
        pa_vals = [h1_pa[-1]*100, h2_pa[-1]*100, h3_pa[-1]*100]
        em_vals = [h1_em[-1]*100, h2_em[-1]*100, h3_em[-1]*100]
        x = np.arange(3); w = 0.35
        axes[1].bar(x-w/2, pa_vals, w, color=['#3498db','#e67e22','#e74c3c'], alpha=0.85, label='PA')
        axes[1].bar(x+w/2, em_vals, w, color=['#3498db','#e67e22','#e74c3c'], alpha=0.4,
                   hatch='//', label='EM')
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel('%'); axes[1].set_title('PA & EM', fontweight='bold')
        axes[1].legend()

        # AE training loss
        axes[2].plot(range(1, len(ae_losses)+1), ae_losses, '-', color='#9b59b6', lw=2)
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Recon Loss')
        axes[2].set_title('AE Training Loss', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 205: Auto-Encoder Critic', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase205_critic.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'baseline_pa': h1_pa[-1], 'critic01_pa': h2_pa[-1], 'critic05_pa': h3_pa[-1]}


if __name__ == '__main__':
    main()
