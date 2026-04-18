"""
Phase 116: Global Energy Canary & Soft-Landing

Instead of monitoring individual channels (Phase 113: too few channels
for differentiation), monitors TOTAL state change energy:
    E(t) = ||state(t) - state(t-1)||

When E spikes (chaos onset), temporarily reduces tau gate bias to
slow down time (soft-landing), guiding the NCA back to its attractor.

Analogy: A cellular defibrillator (AED) for dying neural states.

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


# ====================================================================
# L-NCA with energy monitoring and soft-landing
# ====================================================================
class LNCA_SoftLanding(nn.Module):
    """L-NCA with built-in energy monitoring and adaptive tau."""
    def __init__(self, in_ch=3, hidden_ch=64, out_ch=10):
        super().__init__()
        self.hidden_ch = hidden_ch

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU()
        )
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, groups=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
        )
        self.tau_gate = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_ch, out_ch)
        )

    def forward(self, x, n_steps=5):
        state = self.encoder(x)
        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)
            state = beta * state + (1 - beta) * delta
        return self.decoder(state)

    def forward_with_energy(self, x, n_steps=30, soft_landing=False,
                            energy_threshold=None, slowdown_factor=3.0):
        """
        Forward with energy monitoring and optional soft-landing.

        Returns: logits, energy_trajectory, stop_step, was_healed
        """
        state = self.encoder(x)
        prev_state = state.detach().clone()

        energies = []
        best_out = None
        best_energy = float('inf')
        best_t = 0
        was_healed = False
        tau_modifier = 0.0  # additive modifier to tau gate bias

        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)

            # Apply tau modifier for soft-landing
            if tau_modifier != 0:
                # Increase beta (slow down) by biasing sigmoid input
                beta = torch.sigmoid(
                    torch.logit(beta.clamp(1e-6, 1-1e-6)) + tau_modifier
                )

            state = beta * state + (1 - beta) * delta

            # Compute state change energy
            energy = (state - prev_state).pow(2).mean().item()
            energies.append(energy)
            prev_state = state.detach().clone()

            # Track best output (lowest energy = most stable)
            out = self.decoder(state)
            if t >= 2 and energy < best_energy:
                best_energy = energy
                best_out = out.clone()
                best_t = t

            # Soft-landing: if energy spikes, slow down
            if soft_landing and energy_threshold is not None and t >= 3:
                if energy > energy_threshold:
                    if tau_modifier == 0:
                        was_healed = True
                    tau_modifier = slowdown_factor  # bias tau toward 1 (freeze)
                elif tau_modifier > 0 and energy < energy_threshold * 0.5:
                    tau_modifier = max(0, tau_modifier - 0.5)  # gradually unfreeze

        if best_out is None:
            best_out = self.decoder(state)
            best_t = n_steps - 1

        return best_out, energies, best_t, was_healed


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 116: Global Energy Canary & Soft-Landing")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_ds = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10('data', train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # Step 1: Train L-NCA (64 channels for better capacity)
    print("\n[Step 1] Training L-NCA (hidden_ch=64, T=5)...")
    model = LNCA_SoftLanding(in_ch=3, hidden_ch=64, out_ch=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

    for epoch in range(40):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=5)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            acc = 0; total = 0
            model.eval()
            with torch.no_grad():
                for xv, yv in test_loader:
                    xv, yv = xv.to(DEVICE), yv.to(DEVICE)
                    out = model(xv, n_steps=5)
                    acc += (out.argmax(1) == yv).sum().item()
                    total += yv.size(0)
            print(f"    Epoch {epoch+1}: test_acc={acc/total*100:.2f}%")

    # Step 2: Map T vs accuracy and energy trajectory
    print("\n[Step 2] Mapping T vs accuracy and energy...")
    model.eval()
    t_accs = {}
    energy_trajectories = {}

    for T in range(1, 31):
        correct = 0; total = 0
        batch_energies = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out, energies, _, _ = model.forward_with_energy(x, n_steps=T)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
                batch_energies.append(energies)
        t_accs[T] = correct / total
        # Average energy at each step
        max_len = max(len(e) for e in batch_energies)
        avg_energy = []
        for step in range(max_len):
            vals = [e[step] for e in batch_energies if step < len(e)]
            avg_energy.append(np.mean(vals))
        energy_trajectories[T] = avg_energy

        # Print key T values
        if T <= 10 or T % 5 == 0:
            last_e = avg_energy[-1] if avg_energy else 0
            print(f"    T={T:2d}: acc={t_accs[T]*100:.2f}%, "
                  f"final_energy={last_e:.6f}")

    # Find optimal T and collapse
    optimal_T = max(t_accs, key=t_accs.get)
    optimal_acc = t_accs[optimal_T]
    collapse_T = None
    for T in range(optimal_T + 1, 31):
        if t_accs.get(T, 0) < optimal_acc - 0.03:
            collapse_T = T
            break

    print(f"\n  Optimal T: {optimal_T} ({optimal_acc*100:.2f}%)")
    if collapse_T:
        print(f"  Collapse T: {collapse_T} ({t_accs[collapse_T]*100:.2f}%)")

    # Step 3: Determine energy threshold
    print("\n[Step 3] Determining energy threshold...")
    # Get energy at optimal T
    optimal_energies = energy_trajectories.get(optimal_T, [0])
    baseline_energy = np.mean(optimal_energies[-3:]) if len(optimal_energies) >= 3 else optimal_energies[-1]

    # Test different threshold multipliers
    threshold_multipliers = [1.2, 1.5, 2.0, 3.0, 5.0]
    slowdown_factors = [1.0, 2.0, 3.0, 5.0]

    print(f"  Baseline energy at optimal T: {baseline_energy:.6f}")

    # Step 4: Soft-landing experiment
    print("\n[Step 4] Soft-landing experiment...")
    results = []

    # Baseline: no soft-landing at collapse T
    collapse_test_T = collapse_T if collapse_T else 25
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=collapse_test_T)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    collapse_acc = correct / total

    # Optimal T baseline
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=optimal_T)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    optimal_acc_check = correct / total

    print(f"  Optimal T={optimal_T}: {optimal_acc_check*100:.2f}%")
    print(f"  Collapse T={collapse_test_T}: {collapse_acc*100:.2f}%")

    for mult in threshold_multipliers:
        threshold = baseline_energy * mult
        for slowdown in slowdown_factors:
            correct = 0; total = 0; healed_count = 0; total_batches = 0
            avg_stop_t = []

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out, energies, stop_t, was_healed = model.forward_with_energy(
                        x, n_steps=30, soft_landing=True,
                        energy_threshold=threshold,
                        slowdown_factor=slowdown
                    )
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
                    if was_healed:
                        healed_count += 1
                    total_batches += 1
                    avg_stop_t.append(stop_t)

            soft_acc = correct / total
            heal_pct = healed_count / total_batches * 100
            improvement = soft_acc - collapse_acc

            result = {
                'threshold_mult': mult, 'slowdown': slowdown,
                'threshold': threshold,
                'soft_acc': soft_acc,
                'optimal_acc': optimal_acc_check,
                'collapse_acc': collapse_acc,
                'improvement': improvement,
                'heal_pct': heal_pct,
                'mean_stop_t': float(np.mean(avg_stop_t))
            }
            results.append(result)

            if improvement > 0.01:
                print(f"    threshold={mult:.1f}x, slowdown={slowdown:.1f}: "
                      f"{soft_acc*100:.2f}% (improvement={improvement*100:+.2f}%, "
                      f"healed={heal_pct:.0f}%)")

    # Find best soft-landing config
    best = max(results, key=lambda r: r['soft_acc'])

    # Summary
    print(f"\n{'='*70}")
    print("  GLOBAL ENERGY CANARY RESULTS")
    print(f"{'='*70}")
    print(f"  Optimal T={optimal_T}:          {optimal_acc_check*100:.2f}%")
    print(f"  Collapse T={collapse_test_T}:         {collapse_acc*100:.2f}%")
    print(f"  Best Soft-Landing:       {best['soft_acc']*100:.2f}%")
    print(f"    threshold: {best['threshold_mult']:.1f}x baseline energy")
    print(f"    slowdown:  {best['slowdown']:.1f}x")
    print(f"    improvement vs collapse: {best['improvement']*100:+.2f}%")
    print(f"    healed batches: {best['heal_pct']:.0f}%")
    print(f"    avg stop T: {best['mean_stop_t']:.1f}")

    gap_to_optimal = best['soft_acc'] - optimal_acc_check
    print(f"    gap vs optimal: {gap_to_optimal*100:+.2f}%")

    if best['soft_acc'] >= optimal_acc_check - 0.01:
        print(f"\n  ** SOFT-LANDING MAINTAINS OPTIMAL-T ACCURACY AT T=30! **")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase116_energy_canary.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 116: Global Energy Canary',
            'timestamp': datetime.now().isoformat(),
            'optimal_T': optimal_T,
            'optimal_acc': optimal_acc_check,
            'collapse_T': collapse_T,
            'collapse_acc': collapse_acc,
            'best_config': best,
            't_vs_accuracy': {str(k): v for k, v in t_accs.items()},
            'all_results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: T vs accuracy
        axes[0, 0].plot(list(t_accs.keys()), [v*100 for v in t_accs.values()],
                        'b-o', markersize=3, label='No soft-landing')
        axes[0, 0].axhline(y=best['soft_acc']*100, color='green', linestyle='--',
                           label=f'Soft-landing ({best["soft_acc"]*100:.1f}%)')
        axes[0, 0].axvline(x=optimal_T, color='green', linestyle=':', alpha=0.5)
        if collapse_T:
            axes[0, 0].axvline(x=collapse_T, color='red', linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel('T (steps)'); axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('NCA Accuracy: With vs Without Soft-Landing')
        axes[0, 0].legend()

        # Plot 2: Energy trajectory
        for T in [optimal_T, collapse_test_T, 30]:
            et = energy_trajectories.get(T, [])
            if et:
                axes[0, 1].plot(range(len(et)), et, '-', label=f'T={T}', alpha=0.7)
        if baseline_energy > 0:
            axes[0, 1].axhline(y=baseline_energy * best['threshold_mult'],
                               color='red', linestyle='--', alpha=0.5,
                               label=f'Threshold ({best["threshold_mult"]}x)')
        axes[0, 1].set_xlabel('Step'); axes[0, 1].set_ylabel('Energy ||ds||^2')
        axes[0, 1].set_title('State Change Energy Trajectory')
        axes[0, 1].legend(fontsize=8)

        # Plot 3: Heatmap threshold x slowdown
        heat = np.zeros((len(threshold_multipliers), len(slowdown_factors)))
        for i, m in enumerate(threshold_multipliers):
            for j, s in enumerate(slowdown_factors):
                r = [x for x in results if x['threshold_mult'] == m and x['slowdown'] == s]
                if r:
                    heat[i, j] = r[0]['improvement'] * 100
        im = axes[1, 0].imshow(heat, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=10)
        axes[1, 0].set_xticks(range(len(slowdown_factors)))
        axes[1, 0].set_xticklabels([f'{s}x' for s in slowdown_factors])
        axes[1, 0].set_yticks(range(len(threshold_multipliers)))
        axes[1, 0].set_yticklabels([f'{m}x' for m in threshold_multipliers])
        axes[1, 0].set_xlabel('Slowdown factor')
        axes[1, 0].set_ylabel('Threshold multiplier')
        axes[1, 0].set_title('Improvement vs Collapse (%)')
        plt.colorbar(im, ax=axes[1, 0])
        for i in range(len(threshold_multipliers)):
            for j in range(len(slowdown_factors)):
                axes[1, 0].text(j, i, f'{heat[i,j]:.1f}',
                               ha='center', va='center', fontsize=8)

        # Plot 4: Comparison bar
        methods = ['Optimal\nT', 'Collapse\nT', 'Soft-\nLanding']
        accs = [optimal_acc_check*100, collapse_acc*100, best['soft_acc']*100]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        axes[1, 1].bar(methods, accs, color=colors)
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Self-Healing via Energy Canary')
        for i, v in enumerate(accs):
            axes[1, 1].text(i, v + 0.3, f'{v:.1f}%', ha='center')

        plt.suptitle('Phase 116: Global Energy Canary & Soft-Landing', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase116_energy_canary.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 116 complete!")


if __name__ == '__main__':
    main()
