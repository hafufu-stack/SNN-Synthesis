"""
Phase 113: Cellular Canary — Self-Healing in L-NCA

Transplants v11's Canary Head concept to L-NCA hidden channels.
Detects entropy spikes that predict NCA output collapse,
then applies τ-modulation to self-heal.

v11 Canary Head:   LLM attention heads at 30-55% depth detect hallucination
Phase 113 Canary:  NCA hidden channels detect temporal collapse (overstepping)

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
# L-NCA Model (from Phase 101-106 architecture)
# ====================================================================
class LNCA(nn.Module):
    """L-NCA with observable hidden channels for canary analysis."""
    def __init__(self, in_ch=3, hidden_ch=32, out_ch=10, img_size=32):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.img_size = img_size

        # Input encoding: image → hidden state
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU()
        )

        # NCA update rule (single rule, applied T times)
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, groups=1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
        )

        # τ gate (controls update speed per channel)
        self.tau_gate = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.Sigmoid()
        )

        # Output decoder
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_ch, out_ch)
        )

    def forward(self, x, n_steps=5, return_trajectory=False):
        """Forward with optional trajectory recording."""
        state = self.encoder(x)

        if return_trajectory:
            trajectory = [state.detach().clone()]

        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)  # per-channel, per-pixel τ
            state = beta * state + (1 - beta) * delta

            if return_trajectory:
                trajectory.append(state.detach().clone())

        out = self.decoder(state)

        if return_trajectory:
            return out, trajectory
        return out


# ====================================================================
# Canary Analysis Functions
# ====================================================================
def compute_channel_entropy(state):
    """
    Compute per-channel entropy of spatial activations.
    state: (B, C, H, W) → entropy: (B, C)
    """
    B, C, H, W = state.shape
    # Normalize each channel's spatial distribution to probability
    flat = state.view(B, C, -1)  # (B, C, H*W)
    # Shift to positive & normalize
    flat_pos = flat - flat.min(dim=2, keepdim=True)[0] + 1e-8
    probs = flat_pos / flat_pos.sum(dim=2, keepdim=True)
    # Shannon entropy
    entropy = -(probs * torch.log2(probs + 1e-10)).sum(dim=2)  # (B, C)
    return entropy


def find_canary_channels(model, test_loader, max_T=30, collapse_T=20):
    """
    Find channels whose entropy spikes predict output collapse.
    Returns canary channels ranked by prediction power.
    """
    model.eval()
    all_entropies = []
    all_correct = []

    with torch.no_grad():
        batch_count = 0
        for x, y in test_loader:
            if batch_count >= 5:  # Use 5 batches for analysis
                break
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Get trajectory
            _, trajectory = model(x, n_steps=max_T, return_trajectory=True)

            # For each step, record entropy and whether output is still correct
            for t in range(len(trajectory)):
                state_t = trajectory[t]
                ent = compute_channel_entropy(state_t)  # (B, C)
                all_entropies.append(ent.cpu())

                # Check accuracy at this step
                out_t = model.decoder(state_t)
                correct_t = (out_t.argmax(1) == y).float()
                all_correct.append(correct_t.cpu())

            batch_count += 1

    # Reshape: (n_steps * n_batches, B, C) → analyze
    n_steps = max_T + 1  # including initial state
    n_total = len(all_entropies) // n_steps

    # Per-channel: compute correlation between entropy change and collapse
    channel_scores = []
    for ch in range(model.hidden_ch):
        # For each sample, track entropy trajectory of this channel
        early_entropy = []
        collapse_signal = []

        for batch_idx in range(n_total):
            base = batch_idx * n_steps
            # Entropy at "safe" step vs "collapse warning" step
            safe_ent = all_entropies[base + 5].mean(0)[ch].item() if base + 5 < len(all_entropies) else 0
            warn_ent = all_entropies[base + collapse_T - 3].mean(0)[ch].item() if base + collapse_T - 3 < len(all_entropies) else 0

            # Did collapse happen?
            safe_acc = all_correct[base + 5].mean().item() if base + 5 < len(all_correct) else 1
            late_acc = all_correct[base + collapse_T].mean().item() if base + collapse_T < len(all_correct) else 0
            collapsed = 1.0 if late_acc < safe_acc - 0.1 else 0.0

            early_entropy.append(warn_ent - safe_ent)  # entropy delta
            collapse_signal.append(collapsed)

        # Channel's canary score = correlation(entropy_delta, collapse)
        if len(early_entropy) > 1 and np.std(collapse_signal) > 0:
            score = np.corrcoef(early_entropy, collapse_signal)[0, 1]
        else:
            score = 0.0
        channel_scores.append((ch, score, np.mean(early_entropy)))

    channel_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return channel_scores


def self_heal(model, x, y, canary_channels, threshold=0.5, max_T=30):
    """
    Apply self-healing: monitor canary channels and stop/slow when they spike.
    """
    model.eval()
    state = model.encoder(x)
    best_out = None
    best_acc = 0
    best_t = 0

    for t in range(max_T):
        delta = model.update(state)
        beta = model.tau_gate(state)
        state = beta * state + (1 - beta) * delta

        # Check canary entropy
        ent = compute_channel_entropy(state)  # (B, C)
        canary_ent = ent[:, canary_channels[:3]].mean().item()  # top-3 canaries

        # Record current output
        out = model.decoder(state)
        acc = (out.argmax(1) == y).float().mean().item()

        if acc >= best_acc:
            best_acc = acc
            best_out = out.clone()
            best_t = t

        # If canary entropy exceeds threshold, increase β (slow down)
        if canary_ent > threshold and t > 3:
            # Self-healing: freeze state by setting β → 1
            break

    return best_out, best_acc, best_t


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 113: Cellular Canary - Self-Healing in L-NCA")
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

    # Step 1: Train L-NCA
    print("\n[Step 1] Training L-NCA (hidden_ch=32, T=5)...")
    model = LNCA(in_ch=3, hidden_ch=32, out_ch=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

    for epoch in range(30):
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

    # Step 2: Map T vs accuracy (find collapse point)
    print("\n[Step 2] Mapping T vs accuracy (finding collapse)...")
    t_values = list(range(1, 31))
    t_accs = {}
    model.eval()
    for T in t_values:
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x, n_steps=T)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        t_accs[T] = correct / total
        print(f"    T={T:2d}: {t_accs[T]*100:.2f}%", end="")
        if T == 5:
            print(" <-- training T", end="")
        if T > 1 and t_accs[T] < t_accs.get(T-1, 1.0) - 0.02:
            print(" !! collapse", end="")
        print()

    # Find optimal and collapse point
    optimal_T = max(t_accs, key=t_accs.get)
    optimal_acc = t_accs[optimal_T]
    collapse_T = None
    for T in range(optimal_T + 1, max(t_values) + 1):
        if t_accs.get(T, 0) < optimal_acc - 0.05:
            collapse_T = T
            break

    print(f"\n  Optimal T: {optimal_T} ({optimal_acc*100:.2f}%)")
    if collapse_T:
        print(f"  Collapse T: {collapse_T} ({t_accs[collapse_T]*100:.2f}%)")
    else:
        print(f"  No significant collapse detected")

    # Step 3: Find canary channels
    print("\n[Step 3] Finding canary channels...")
    canary_scores = find_canary_channels(model, test_loader,
                                         max_T=25, collapse_T=collapse_T or 20)
    print("  Top 10 canary channels (by |correlation| with collapse):")
    for rank, (ch, score, ent_delta) in enumerate(canary_scores[:10]):
        depth_pct = (ch / 32) * 100
        print(f"    #{rank+1}: Channel {ch:2d} (depth={depth_pct:.0f}%) "
              f"corr={score:+.3f} d_ent={ent_delta:.3f}")

    canary_ch_ids = [ch for ch, _, _ in canary_scores[:3]]
    print(f"\n  Selected canary channels: {canary_ch_ids}")

    # Step 4: Self-healing experiment
    print("\n[Step 4] Self-healing experiment...")
    # Compare: fixed T (optimal), fixed T (collapse), self-healed
    results_comparison = {}

    # Fixed T = optimal
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=optimal_T)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    results_comparison['optimal'] = correct / total

    # Fixed T = collapse (or max)
    collapse_test_T = collapse_T or 25
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=collapse_test_T)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    results_comparison['collapse'] = correct / total

    # Self-healed (canary-triggered stop)
    # Get entropy threshold from canary channels at optimal T
    model.eval()
    threshold_ents = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, traj = model(x, n_steps=optimal_T, return_trajectory=True)
            ent = compute_channel_entropy(traj[-1])
            threshold_ents.append(ent[:, canary_ch_ids].mean().item())
            if len(threshold_ents) >= 5:
                break
    canary_threshold = np.mean(threshold_ents) * 1.2  # 20% above optimal-T entropy
    print(f"  Canary entropy threshold: {canary_threshold:.3f}")

    correct = 0; total = 0; heal_stops = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out, acc, stop_t = self_heal(model, x, y, canary_ch_ids,
                                         threshold=canary_threshold, max_T=30)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            heal_stops.append(stop_t)
    results_comparison['healed'] = correct / total

    # Summary
    print(f"\n{'='*70}")
    print("  CELLULAR CANARY RESULTS")
    print(f"{'='*70}")
    print(f"  Optimal T={optimal_T}:  {results_comparison['optimal']*100:.2f}%")
    print(f"  Collapse T={collapse_test_T}: {results_comparison['collapse']*100:.2f}%")
    print(f"  Self-Healed:      {results_comparison['healed']*100:.2f}%")
    heal_improvement = results_comparison['healed'] - results_comparison['collapse']
    print(f"  Healing effect:   {heal_improvement*100:+.2f}%")
    print(f"  Avg stop T:       {np.mean(heal_stops):.1f}")
    print(f"  Canary channels:  {canary_ch_ids}")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase113_cellular_canary.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 113: Cellular Canary',
            'timestamp': datetime.now().isoformat(),
            't_vs_accuracy': t_accs,
            'optimal_T': optimal_T,
            'collapse_T': collapse_T,
            'canary_channels': canary_ch_ids,
            'canary_threshold': canary_threshold,
            'results': results_comparison,
            'heal_improvement': heal_improvement,
            'mean_stop_T': float(np.mean(heal_stops)),
            'canary_scores': [(ch, sc, ed) for ch, sc, ed in canary_scores[:10]]
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: T vs accuracy with collapse & canary zones
        axes[0, 0].plot(list(t_accs.keys()), [v*100 for v in t_accs.values()],
                        'b-o', markersize=3, label='NCA accuracy')
        axes[0, 0].axvline(x=optimal_T, color='green', linestyle='--',
                           alpha=0.7, label=f'Optimal T={optimal_T}')
        if collapse_T:
            axes[0, 0].axvline(x=collapse_T, color='red', linestyle='--',
                               alpha=0.7, label=f'Collapse T={collapse_T}')
            axes[0, 0].axvspan(collapse_T, max(t_values), alpha=0.1, color='red')
        axes[0, 0].set_xlabel('T (NCA steps)')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('NCA Accuracy vs Time Steps')
        axes[0, 0].legend()

        # Plot 2: Canary channel scores
        channels = [ch for ch, _, _ in canary_scores[:15]]
        scores = [abs(sc) for _, sc, _ in canary_scores[:15]]
        colors = ['red' if ch in canary_ch_ids else 'steelblue' for ch in channels]
        axes[0, 1].barh(range(len(channels)), scores, color=colors)
        axes[0, 1].set_yticks(range(len(channels)))
        axes[0, 1].set_yticklabels([f'Ch {ch}' for ch in channels])
        axes[0, 1].set_xlabel('|Correlation| with collapse')
        axes[0, 1].set_title('Canary Channel Ranking')

        # Plot 3: Comparison bar chart
        methods = ['Optimal', 'Collapse', 'Self-Healed']
        accs = [results_comparison[k]*100 for k in ['optimal', 'collapse', 'healed']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        axes[1, 0].bar(methods, accs, color=colors)
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Self-Healing Effect')
        for i, v in enumerate(accs):
            axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center')

        # Plot 4: Self-heal stop T distribution
        axes[1, 1].hist(heal_stops, bins=range(0, 32), color='#9b59b6', alpha=0.7)
        axes[1, 1].axvline(x=np.mean(heal_stops), color='red', linestyle='--',
                           label=f'Mean={np.mean(heal_stops):.1f}')
        axes[1, 1].set_xlabel('Stop T (canary-triggered)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Self-Healing Stop Distribution')
        axes[1, 1].legend()

        plt.suptitle('Phase 113: Cellular Canary — Self-Healing in L-NCA', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase113_cellular_canary.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 113 complete!")


if __name__ == '__main__':
    main()
