"""
Phase 241: Grid-MAE (Masked Autoencoder for Grid Geometry)

Self-supervised pre-training: mask 30-50% of grid pixels,
train NCA to reconstruct them from context.
This teaches "geometric common sense" (symmetry, continuity, etc.)
BEFORE supervised ARC task learning.

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
from phase239_dsl_engine import generate_dsl_batch
from phase224_synthetic import to_one_hot


class GridMAE(nn.Module):
    """Masked Autoencoder for grid geometry pre-training."""
    def __init__(self, n_colors=11, hidden_ch=64):
        super().__init__()
        C = hidden_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(n_colors + 1, C, 3, padding=1), nn.ReLU(),  # +1 for mask channel
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(),
            nn.Conv2d(C, n_colors, 1),
        )

    def forward(self, masked_input, mask):
        # mask: (B, 1, H, W) where 1=visible, 0=masked
        x = torch.cat([masked_input, mask], dim=1)
        h = self.encoder(x)
        return self.decoder(h)


def pretrain_mae(model, n_batches=500, mask_ratio=0.4):
    """Pre-train on masked reconstruction."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.RandomState(SEED + 999)
    total_loss = 0
    model.train()
    for bi in range(n_batches):
        batch = generate_dsl_batch(rng, batch_size=8)
        for inp_oh, out_oh, oh, ow in batch:
            # Use input grid for MAE (self-supervised, no labels needed)
            grid_oh = inp_oh.unsqueeze(0).to(DEVICE)  # (1, 11, H, W)
            gt = grid_oh[:, :, :oh, :ow].argmax(dim=1)  # (1, oh, ow)
            # Create random mask
            mask = (torch.rand(1, 1, oh, ow, device=DEVICE) > mask_ratio).float()
            # Masked input: zero out masked positions
            masked = grid_oh[:, :, :oh, :ow] * mask
            # Predict
            pred = model(masked, mask)
            # Loss only on masked positions (reconstruct hidden pixels)
            inv_mask = 1.0 - mask.squeeze(1)  # (1, oh, ow)
            if inv_mask.sum() > 0:
                loss = F.cross_entropy(pred, gt, reduction='none')
                loss = (loss * inv_mask).sum() / inv_mask.sum()
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()
        if (bi + 1) % (n_batches // 3) == 0:
            avg = total_loss / max(1, (bi+1) * 8)
            print(f"    MAE batch {bi+1}/{n_batches}, avg_loss={avg:.4f}")
    return total_loss / max(1, n_batches * 8)


def transfer_mae_weights(mae_model, nca_model):
    """Transfer MAE encoder weights to NCA's S2 encoder (partial)."""
    # Copy first conv layer weights
    mae_enc_layers = [m for m in mae_model.encoder if isinstance(m, nn.Conv2d)]
    nca_enc_layers = [m for m in nca_model.s2_encoder if isinstance(m, nn.Conv2d)]
    if len(mae_enc_layers) > 0 and len(nca_enc_layers) > 0:
        # Only transfer matching dimensions
        with torch.no_grad():
            # MAE encoder has n_colors+1 input channels, NCA has n_colors+embed_dim
            # Transfer the spatial convolution weights (partial)
            src_w = mae_enc_layers[0].weight  # (C, 12, 3, 3)
            dst_w = nca_enc_layers[0].weight  # (C, 43, 3, 3)
            # Copy the color channels (first 11)
            min_in = min(src_w.shape[1] - 1, dst_w.shape[1])
            min_out = min(src_w.shape[0], dst_w.shape[0])
            dst_w[:min_out, :min_in, :, :] = src_w[:min_out, :min_in, :, :]
            nca_enc_layers[0].weight.copy_(dst_w)
            if mae_enc_layers[0].bias is not None and nca_enc_layers[0].bias is not None:
                nca_enc_layers[0].bias[:min_out] = mae_enc_layers[0].bias[:min_out]
    print(f"  Transferred MAE encoder weights to NCA")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 241: Grid-MAE (Masked Autoencoder Pre-training)")
    print(f"  Learn geometric common sense via self-supervised reconstruction")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    ep = 100

    # Baseline (no pre-training)
    print(f"\n[Baseline: GatedHybrid (no pre-train)]")
    torch.manual_seed(SEED)
    m0 = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    print(f"  Params: {m0.count_params():,}")
    h0_pa, h0_em = train_and_eval(m0, train, test, ep, "Base")
    del m0; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # MAE pre-training -> transfer -> fine-tune
    mask_ratios = [0.3, 0.5]
    results = {'baseline': {'pa': h0_pa[-1], 'em': h0_em[-1]}}

    for mr in mask_ratios:
        label = f"MAE_mr{int(mr*100)}"
        print(f"\n[{label}: MAE pre-train (mask={mr}) -> transfer -> FT]")

        # Step 1: Pre-train MAE
        torch.manual_seed(SEED)
        mae = GridMAE(11, 64).to(DEVICE)
        print(f"  MAE params: {sum(p.numel() for p in mae.parameters()):,}")
        mae_loss = pretrain_mae(mae, n_batches=300, mask_ratio=mr)
        print(f"  MAE pre-training done (loss={mae_loss:.4f})")

        # Step 2: Create NCA and transfer weights
        torch.manual_seed(SEED)
        model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
        transfer_mae_weights(mae, model)
        del mae; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

        # Step 3: Fine-tune on ARC
        print(f"  Fine-tuning on ARC...")
        h_pa, h_em = train_and_eval(model, train, test, ep, label)
        results[label] = {'pa': h_pa[-1], 'em': h_em[-1]}
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  GRID-MAE RESULTS:")
    for k, r in results.items():
        print(f"  {k:15s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase241_mae.json"), 'w', encoding='utf-8') as f:
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
        ax.bar(x-w/2, pa_vals, w, color=colors[:len(labels)], alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors[:len(labels)], alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 241: Grid-MAE Pre-training', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase241_mae.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e: print(f"  Figure error: {e}")
    gc.collect()
    return results

if __name__ == '__main__':
    main()
