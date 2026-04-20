"""
Phase 157: Reaction-Diffusion NCA - True Turing Pattern Emergence

Phase 155 proved that variance alone collapses to 1 color.
Turing's reaction-diffusion requires LOCAL ACTIVATION + LONG-RANGE
INHIBITION. We implement this as a spatial correlation loss:
  - Minimize MSE between pixels at distance 1-2 (local same-color)
  - Maximize MSE between pixels at distance 4-6 (distant different-color)

This Mexican-hat filter should produce stripes, spots, or maze patterns.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
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
GRID_SIZE = 48
N_COLORS = 4


class MorphoNCA(nn.Module):
    def __init__(self, ch=32, n_colors=N_COLORS, steps=20):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(n_colors, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, n_colors, 1)

    def forward(self, x, steps=None):
        if steps is None: steps = self.steps
        h = F.relu(self.proj_in(x))
        for _ in range(steps):
            h = F.relu(h + self.rule(h))
        return self.proj_out(h)


def make_gaussian_kernel(sigma, size):
    """Create 2D Gaussian kernel."""
    ax = torch.arange(size).float() - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def activation_inhibition_loss(logits, device):
    """
    Mexican-hat spatial correlation loss:
    - Local activation: nearby pixels should be SAME color (small MSE)
    - Long-range inhibition: distant pixels should be DIFFERENT color
    """
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)

    # Local activation kernel (sigma=1.5, captures distance 1-2)
    k_act = make_gaussian_kernel(1.5, 7).to(device)
    # Long-range inhibition kernel (sigma=4.0, captures distance 4-6)
    k_inh = make_gaussian_kernel(4.0, 13).to(device)

    activation_loss = 0
    inhibition_loss = 0

    for c in range(probs.shape[1]):
        pc = probs[:, c:c+1]  # (B, 1, H, W)

        # Local average of same channel
        local_avg = F.conv2d(F.pad(pc, (3,3,3,3), mode='circular'), k_act)
        # Activation: minimize difference from local average (same color nearby)
        activation_loss += ((pc - local_avg) ** 2).mean()

        # Long-range average
        distant_avg = F.conv2d(F.pad(pc, (6,6,6,6), mode='circular'), k_inh)
        # Inhibition: maximize difference from distant average (different color far away)
        inhibition_loss += -((pc - distant_avg) ** 2).mean()

    # Sharpness: each pixel should commit to one color (low entropy)
    ent = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()

    return activation_loss + 0.5 * inhibition_loss + 0.3 * ent


def pattern_stats(logits):
    """Compute pattern quality metrics."""
    pred = logits.argmax(dim=1)  # (B, H, W)
    # Number of unique colors used
    n_colors = len(torch.unique(pred[0]))
    # Boundary density (pattern complexity)
    h_diff = (pred[:, 1:, :] != pred[:, :-1, :]).float().mean().item()
    v_diff = (pred[:, :, 1:] != pred[:, :, :-1]).float().mean().item()
    complexity = h_diff + v_diff
    # Spatial autocorrelation at different distances
    p = pred[0].float()
    auto_1 = F.mse_loss(p[1:, :], p[:-1, :]).item()  # distance 1
    auto_5 = 0
    if p.shape[0] > 5:
        auto_5 = F.mse_loss(p[5:, :], p[:-5, :]).item()  # distance 5
    return n_colors, complexity, auto_1, auto_5


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 157: Reaction-Diffusion NCA")
    print(f"  True Turing pattern via activation-inhibition loss")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, Colors: {N_COLORS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    configs = [
        (1.0, 0.5, 0.3, 20, "standard"),
        (1.0, 1.0, 0.3, 20, "strong_inhibition"),
        (1.0, 0.5, 0.1, 30, "more_steps"),
        (0.5, 1.0, 0.5, 20, "high_entropy_reg"),
    ]

    results = {}
    final_grids = {}

    for act_w, inh_w, ent_w, steps, name in configs:
        print(f"\n[Config: {name}] act={act_w}, inh={inh_w}, ent={ent_w}, steps={steps}")

        model = MorphoNCA(ch=32, n_colors=N_COLORS, steps=steps).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)

        loss_history = []
        for epoch in range(300):
            model.train()
            noise = torch.randn(4, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3
            logits = model(noise)

            probs = F.softmax(logits, dim=1)
            k_act = make_gaussian_kernel(1.5, 7).to(DEVICE)
            k_inh = make_gaussian_kernel(4.0, 13).to(DEVICE)

            act_loss = 0; inh_loss = 0
            for c in range(N_COLORS):
                pc = probs[:, c:c+1]
                local_avg = F.conv2d(F.pad(pc, (3,3,3,3), mode='circular'), k_act)
                act_loss += ((pc - local_avg) ** 2).mean()
                distant_avg = F.conv2d(F.pad(pc, (6,6,6,6), mode='circular'), k_inh)
                inh_loss += -((pc - distant_avg) ** 2).mean()

            ent = -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
            loss = act_w * act_loss + inh_w * inh_loss + ent_w * ent

            opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            loss_history.append(loss.item())

            if (epoch + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    test_noise = torch.randn(2, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3
                    test_logits = model(test_noise, steps=steps+10)
                    nc, cmplx, a1, a5 = pattern_stats(test_logits)
                print(f"  Epoch {epoch+1}/300: loss={loss.item():.4f}, "
                      f"colors={nc}, complexity={cmplx:.4f}, "
                      f"auto@1={a1:.4f}, auto@5={a5:.4f}")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_noise = torch.randn(4, N_COLORS, GRID_SIZE, GRID_SIZE, device=DEVICE) * 0.3
            final_logits = model(test_noise, steps=steps+10)
            nc, cmplx, a1, a5 = pattern_stats(final_logits)
            final_pred = final_logits[0].argmax(0).cpu().numpy()

        final_grids[name] = final_pred.tolist()
        results[name] = {
            'n_colors': int(nc), 'complexity': cmplx,
            'autocorr_1': a1, 'autocorr_5': a5,
            'loss_history': loss_history[::15],
            'turing_ratio': a5 / (a1 + 1e-8),  # >1 = Turing pattern!
        }
        print(f"  Final: colors={nc}, complexity={cmplx:.4f}, "
              f"Turing ratio={a5/(a1+1e-8):.2f}")

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Check if any config produced Turing patterns
    best = max(results.keys(), key=lambda k: results[k]['turing_ratio'])
    turing_achieved = results[best]['turing_ratio'] > 1.5 and results[best]['n_colors'] >= 2

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 157 Complete ({elapsed:.0f}s)")
    for name, r in results.items():
        marker = " ***" if name == best else ""
        print(f"  {name:>20}: colors={r['n_colors']}, "
              f"Turing={r['turing_ratio']:.2f}{marker}")
    print(f"  Turing pattern achieved: {turing_achieved}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase157_reaction_diffusion.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 157: Reaction-Diffusion NCA',
            'timestamp': datetime.now().isoformat(),
            'results': results, 'final_grids': final_grids,
            'best': best, 'turing_achieved': turing_achieved,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        cmap = plt.cm.get_cmap('Set1', N_COLORS)
        for i, (_, _, _, _, name) in enumerate(configs):
            grid = np.array(final_grids[name])
            axes[i].imshow(grid, cmap=cmap, vmin=0, vmax=N_COLORS-1, interpolation='nearest')
            r = results[name]
            axes[i].set_title(f'{name}\ncolors={r["n_colors"]}, '
                            f'Turing={r["turing_ratio"]:.2f}',
                            fontweight='bold', fontsize=9)
            axes[i].axis('off')
        fig.suptitle('Phase 157: Reaction-Diffusion NCA (Turing Pattern Emergence)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.82, bottom=0.05, left=0.02, right=0.98, wspace=0.15)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase157_reaction_diffusion.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
