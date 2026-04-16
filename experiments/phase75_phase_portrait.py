"""
Phase 75: Temporal Phase Portrait

Visual proof of WHY temporal noise is robust while spatial noise is catastrophic.
Creates 2D phase portraits of membrane potential trajectories under:
  A) No noise (reference attractor)
  B) Spatial noise (sigma = 0.1, 0.5, 1.0)
  C) Temporal noise (sigma = 0.1, 0.5, 1.0)

Uses a tiny 3-neuron Liquid-LIF to make trajectories interpretable.
Plots V1 vs V2 (membrane voltages of neuron 1 and 2).

Expected: Spatial noise scatters trajectories off the attractor.
          Temporal noise slides along the attractor (preserving causality).

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cpu"
SEED = 2026


class ATanSurrogate(torch.autograd.Function):
    alpha = 2.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = ATanSurrogate.alpha / 2 / (1 + (np.pi / 2 * ATanSurrogate.alpha * input).pow(2))
        return grad_output * grad

spike_fn = ATanSurrogate.apply


class TinyLiquidLIF(nn.Module):
    """3-neuron Liquid-LIF for trajectory visualization."""
    def __init__(self, input_dim=3, hidden_dim=3, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.b_tau = nn.Parameter(torch.ones(hidden_dim) * 1.5)
        self.spatial_sigma = 0.0
        self.temporal_sigma = 0.0

    def forward_step(self, x, mem=None):
        if mem is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        tau_input = torch.cat([x, mem], dim=-1)
        tau_logit = self.fc_tau(tau_input) + self.b_tau

        if self.temporal_sigma > 0:
            tau_logit = tau_logit + torch.randn_like(tau_logit) * self.temporal_sigma

        beta = torch.sigmoid(tau_logit)
        beta = torch.clamp(beta, 0.01, 0.99)

        mem = beta * mem + self.fc_in(x)

        if self.spatial_sigma > 0:
            mem = mem + torch.randn_like(mem) * self.spatial_sigma

        spk = spike_fn(mem - self.threshold)
        mem = mem - spk * self.threshold
        return mem


def generate_input_sequence(seq_len=50, batch_size=1, input_dim=3):
    """Generate a smooth, deterministic input pattern."""
    t = torch.linspace(0, 4 * np.pi, seq_len).unsqueeze(0).unsqueeze(-1)
    t = t.expand(batch_size, seq_len, 1)
    x = torch.cat([
        torch.sin(t),
        torch.cos(t * 0.7),
        torch.sin(t * 1.3 + 1.0),
    ], dim=-1)
    return x * 0.5


def collect_trajectory(model, x_seq, spatial_sigma=0.0, temporal_sigma=0.0, n_trials=20):
    """Run the model multiple times and collect membrane trajectories."""
    model.eval()
    model.spatial_sigma = spatial_sigma
    model.temporal_sigma = temporal_sigma

    all_trajectories = []
    seq_len = x_seq.size(1)

    for trial in range(n_trials):
        mem = None
        trajectory = []
        with torch.no_grad():
            for t in range(seq_len):
                mem = model.forward_step(x_seq[:, t, :], mem)
                trajectory.append(mem[0].numpy().copy())
        all_trajectories.append(np.array(trajectory))

    model.spatial_sigma = 0.0
    model.temporal_sigma = 0.0
    return all_trajectories


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase75_phase_portrait.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 75: Temporal Phase Portrait',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 75: Temporal Phase Portrait")
    print("  Why is temporal noise robust? Visual proof via phase portraits.")
    print("=" * 70)

    model = TinyLiquidLIF(input_dim=3, hidden_dim=3)
    # Initialize with interesting dynamics
    with torch.no_grad():
        model.fc_in.weight.copy_(torch.tensor([[0.8, -0.3, 0.5],
                                                [-0.4, 0.9, -0.2],
                                                [0.3, -0.5, 0.7]], dtype=torch.float32))
        model.fc_tau.weight.copy_(torch.randn(3, 6) * 0.5)

    x_seq = generate_input_sequence(seq_len=50, batch_size=1, input_dim=3)
    print(f"  Input sequence: {x_seq.shape}")

    NOISE_SIGMAS = [0.1, 0.5, 1.0]
    N_TRIALS = 30
    results = {}

    # Reference (no noise)
    print("\n  Collecting reference trajectory (no noise)...")
    ref_trajs = collect_trajectory(model, x_seq, 0.0, 0.0, n_trials=1)
    results['reference'] = {'trajectories': [t.tolist() for t in ref_trajs]}

    # Spatial noise
    for sigma in NOISE_SIGMAS:
        print(f"  Collecting spatial noise sigma={sigma} ({N_TRIALS} trials)...")
        trajs = collect_trajectory(model, x_seq, spatial_sigma=sigma, temporal_sigma=0.0,
                                   n_trials=N_TRIALS)
        # Compute trajectory divergence from reference
        ref = ref_trajs[0]
        divergences = [np.mean(np.sqrt(np.sum((t - ref)**2, axis=1))) for t in trajs]
        avg_div = np.mean(divergences)
        results[f'spatial_{sigma}'] = {
            'sigma': sigma,
            'avg_divergence': float(avg_div),
            'n_trials': N_TRIALS,
        }
        print(f"    Avg divergence from attractor: {avg_div:.4f}")

    # Temporal noise
    for sigma in NOISE_SIGMAS:
        print(f"  Collecting temporal noise sigma={sigma} ({N_TRIALS} trials)...")
        trajs = collect_trajectory(model, x_seq, spatial_sigma=0.0, temporal_sigma=sigma,
                                   n_trials=N_TRIALS)
        ref = ref_trajs[0]
        divergences = [np.mean(np.sqrt(np.sum((t - ref)**2, axis=1))) for t in trajs]
        avg_div = np.mean(divergences)
        results[f'temporal_{sigma}'] = {
            'sigma': sigma,
            'avg_divergence': float(avg_div),
            'n_trials': N_TRIALS,
        }
        print(f"    Avg divergence from attractor: {avg_div:.4f}")

    _save(results)

    # Summary
    print(f"\n{'='*70}")
    print("DIVERGENCE SUMMARY (lower = stays closer to attractor)")
    print(f"{'='*70}")
    print(f"  {'Condition':30s} {'Avg Divergence':>15s}")
    for key, val in results.items():
        if key == 'reference':
            print(f"  {'Reference (no noise)':30s} {'0.0000':>15s}")
        else:
            print(f"  {key:30s} {val['avg_divergence']:>15.4f}")

    # Collect full trajectories for figure generation
    all_trajs_for_fig = {'reference': ref_trajs}
    for sigma in NOISE_SIGMAS:
        all_trajs_for_fig[f'spatial_{sigma}'] = collect_trajectory(
            model, x_seq, spatial_sigma=sigma, n_trials=10)
        all_trajs_for_fig[f'temporal_{sigma}'] = collect_trajectory(
            model, x_seq, temporal_sigma=sigma, n_trials=10)

    _generate_figure(all_trajs_for_fig, NOISE_SIGMAS)
    print("\nPhase 75 complete!")


def _generate_figure(all_trajs, sigmas):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, len(sigmas), figsize=(5*len(sigmas), 10))

        for col, sigma in enumerate(sigmas):
            # Row 0: Spatial noise
            ax = axes[0, col]
            ref = all_trajs['reference'][0]
            ax.plot(ref[:, 0], ref[:, 1], 'k-', linewidth=2, alpha=0.8, label='Reference', zorder=10)
            ax.plot(ref[0, 0], ref[0, 1], 'ko', markersize=8, zorder=11)

            for traj in all_trajs[f'spatial_{sigma}']:
                ax.plot(traj[:, 0], traj[:, 1], '-', alpha=0.15, color='#EF4444', linewidth=0.8)

            ax.set_title(f'Spatial sigma={sigma}', fontweight='bold', fontsize=11)
            if col == 0:
                ax.set_ylabel('Spatial Noise\nV2 (membrane)', fontsize=11)
            ax.grid(alpha=0.2)
            ax.set_xlabel('V1 (membrane)')

            # Row 1: Temporal noise
            ax = axes[1, col]
            ax.plot(ref[:, 0], ref[:, 1], 'k-', linewidth=2, alpha=0.8, label='Reference', zorder=10)
            ax.plot(ref[0, 0], ref[0, 1], 'ko', markersize=8, zorder=11)

            for traj in all_trajs[f'temporal_{sigma}']:
                ax.plot(traj[:, 0], traj[:, 1], '-', alpha=0.15, color='#3B82F6', linewidth=0.8)

            ax.set_title(f'Temporal sigma={sigma}', fontweight='bold', fontsize=11)
            if col == 0:
                ax.set_ylabel('Temporal Noise\nV2 (membrane)', fontsize=11)
            ax.grid(alpha=0.2)
            ax.set_xlabel('V1 (membrane)')

        fig.suptitle('Phase 75: Phase Portrait - Why Temporal Noise is Robust\n'
                    'Black = reference attractor, Red = spatial noise, Blue = temporal noise',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase75_phase_portrait.png"),
                   bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
