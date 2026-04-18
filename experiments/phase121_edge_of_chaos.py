"""
Phase 121: Edge of Chaos Regularization

Phase 118 showed noise has no effect when beta saturates to 0/1.
Solution: Force beta to stay near 0.5 ("liquid") during training,
then temporal noise can actually explore.

L_total = L_task + lambda * L_liquid
where L_liquid = mean((beta - 0.5)^2)

Also trains WITH noise injection for robustness.

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


class LiquidNCA(nn.Module):
    """NCA with controllable beta regularization."""
    def __init__(self, in_ch=3, hidden_ch=64, out_ch=10, use_liquid_reg=False,
                 lambda_liquid=0.1, train_noise=0.0):
        super().__init__()
        self.use_liquid_reg = use_liquid_reg
        self.lambda_liquid = lambda_liquid
        self.train_noise = train_noise

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch), nn.ReLU()
        )
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
        )
        self.tau_gate = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.Sigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_ch, out_ch)

    def forward(self, x, n_steps=5, noise_sigma=0.0, return_beta_stats=False):
        state = self.stem(x)
        all_betas = []

        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)

            # Inject noise at inference or during training
            sigma = noise_sigma if noise_sigma > 0 else (
                self.train_noise if self.training else 0)
            if sigma > 0:
                noise = torch.randn_like(beta) * sigma
                beta_input = torch.logit(beta.clamp(1e-6, 1 - 1e-6)) + noise
                beta = torch.sigmoid(beta_input)

            all_betas.append(beta)
            state = beta * state + (1 - beta) * delta

        logits = self.fc(self.pool(state).view(state.size(0), -1))

        if return_beta_stats:
            beta_cat = torch.cat([b.view(-1) for b in all_betas])
            beta_mean = beta_cat.mean().item()
            beta_std = beta_cat.std().item()
            return logits, beta_mean, beta_std

        return logits

    def get_liquid_loss(self, x, n_steps=5):
        """Compute liquid regularization: penalize beta away from 0.5."""
        state = self.stem(x)
        liquid_loss = 0

        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau_gate(state)

            if self.train_noise > 0 and self.training:
                noise = torch.randn_like(beta) * self.train_noise
                beta_input = torch.logit(beta.clamp(1e-6, 1 - 1e-6)) + noise
                beta = torch.sigmoid(beta_input)

            liquid_loss += ((beta - 0.5) ** 2).mean()
            state = beta * state + (1 - beta) * delta

        logits = self.fc(self.pool(state).view(state.size(0), -1))
        return logits, liquid_loss / n_steps

    def forward_majority_vote(self, x, n_steps=5, noise_sigma=0.1, K=11):
        votes = torch.zeros(x.size(0), 10, device=x.device)
        for k in range(K):
            out = self.forward(x, n_steps=n_steps, noise_sigma=noise_sigma)
            preds = out.argmax(1)
            for i in range(x.size(0)):
                votes[i, preds[i]] += 1
        return votes


def train_and_eval(config_name, model, train_loader, test_loader,
                   n_steps=5, epochs=40, lr=1e-3):
    """Train model and return comprehensive results."""
    print(f"\n  [{config_name}] Training...")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if model.use_liquid_reg:
                logits, liq_loss = model.get_liquid_loss(x, n_steps=n_steps)
                loss = F.cross_entropy(logits, y) + model.lambda_liquid * liq_loss
            else:
                loss = F.cross_entropy(model(x, n_steps=n_steps), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}")

    # Evaluate
    model.eval()
    results = {}

    # Baseline (no noise)
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=n_steps, return_beta_stats=False)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    base_acc = correct / total
    results['base'] = base_acc

    # Beta statistics
    model.eval()
    with torch.no_grad():
        x_sample = next(iter(test_loader))[0][:32].to(DEVICE)
        _, beta_mean, beta_std = model(x_sample, n_steps=n_steps, return_beta_stats=True)
    results['beta_mean'] = beta_mean
    results['beta_std'] = beta_std

    # Noise sweep
    noise_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for sigma in noise_values:
        # Single pass
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x, n_steps=n_steps, noise_sigma=sigma)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        results[f'noise_{sigma}'] = correct / total

        # Majority vote K=11
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model.forward_majority_vote(x, n_steps=n_steps,
                                                   noise_sigma=sigma, K=11)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        results[f'vote_{sigma}'] = correct / total

    # Print summary
    print(f"    Base: {base_acc*100:.2f}%")
    print(f"    Beta: mean={beta_mean:.3f}, std={beta_std:.3f}")
    best_noise = max([(s, results.get(f'noise_{s}', 0)) for s in noise_values],
                     key=lambda x: x[1])
    best_vote = max([(s, results.get(f'vote_{s}', 0)) for s in noise_values],
                    key=lambda x: x[1])
    print(f"    Best noise: sigma={best_noise[0]}, acc={best_noise[1]*100:.2f}%")
    print(f"    Best vote:  sigma={best_vote[0]}, K=11, acc={best_vote[1]*100:.2f}%")

    gap_noise = best_noise[1] - base_acc
    gap_vote = best_vote[1] - base_acc
    if gap_noise > 0.002:
        print(f"    ** NOISE EXCEEDS BASE by {gap_noise*100:+.2f}% **")
    if gap_vote > 0.002:
        print(f"    ** VOTE EXCEEDS BASE by {gap_vote*100:+.2f}% **")

    return results


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 121: Edge of Chaos Regularization")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
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
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    configs = [
        ("Vanilla NCA", dict(use_liquid_reg=False, lambda_liquid=0, train_noise=0)),
        ("+ Train Noise (0.1)", dict(use_liquid_reg=False, lambda_liquid=0, train_noise=0.1)),
        ("+ Liquid Reg (0.1)", dict(use_liquid_reg=True, lambda_liquid=0.1, train_noise=0)),
        ("+ Liquid + Noise", dict(use_liquid_reg=True, lambda_liquid=0.1, train_noise=0.1)),
        ("+ Liquid Strong (0.5)", dict(use_liquid_reg=True, lambda_liquid=0.5, train_noise=0.1)),
    ]

    all_results = {}

    for name, kwargs in configs:
        torch.manual_seed(SEED)
        model = LiquidNCA(in_ch=3, hidden_ch=64, out_ch=10, **kwargs).to(DEVICE)
        results = train_and_eval(name, model, train_loader, test_loader,
                                n_steps=5, epochs=40)
        all_results[name] = results
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*70}")
    print("  EDGE OF CHAOS RESULTS")
    print(f"{'='*70}")
    for name, res in all_results.items():
        base = res['base']
        best_vote = max(res.get(f'vote_{s}', 0) for s in [0.01,0.05,0.1,0.2,0.5,1.0])
        gap = best_vote - base
        marker = " ** SR WORKS **" if gap > 0.005 else ""
        print(f"  {name:30s}: base={base*100:.1f}%, "
              f"best_vote={best_vote*100:.1f}%, "
              f"beta={res['beta_mean']:.3f}+/-{res['beta_std']:.3f}, "
              f"gap={gap*100:+.2f}%{marker}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase121_edge_of_chaos.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 121: Edge of Chaos',
                   'timestamp': datetime.now().isoformat(),
                   'all_results': all_results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Beta distributions
        names = list(all_results.keys())
        beta_means = [all_results[n]['beta_mean'] for n in names]
        beta_stds = [all_results[n]['beta_std'] for n in names]
        x_pos = range(len(names))
        axes[0].bar(x_pos, beta_means, yerr=beta_stds, color=['#e74c3c','#e67e22','#3498db','#9b59b6','#2ecc71'])
        axes[0].axhline(y=0.5, color='black', linestyle='--', label='Liquid target')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([n[:15] for n in names], fontsize=7, rotation=20)
        axes[0].set_ylabel('Beta mean'); axes[0].set_title('Tau Gate Distribution')
        axes[0].legend()

        # Noise effect curves
        sigmas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        for name, res in all_results.items():
            accs = [res.get(f'vote_{s}', 0) * 100 for s in sigmas]
            axes[1].plot(sigmas, accs, 'o-', label=name[:15], markersize=3)
        axes[1].set_xlabel('Noise sigma'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Majority Vote (K=11) vs Noise')
        axes[1].legend(fontsize=6)

        # Gap chart
        gaps = []
        for name, res in all_results.items():
            best_v = max(res.get(f'vote_{s}', 0) for s in sigmas)
            gaps.append((best_v - res['base']) * 100)
        colors = ['#e74c3c' if g <= 0 else '#2ecc71' for g in gaps]
        axes[2].barh(range(len(names)), gaps, color=colors)
        axes[2].set_yticks(range(len(names)))
        axes[2].set_yticklabels([n[:20] for n in names], fontsize=7)
        axes[2].set_xlabel('Gap: Best Vote - Base (%)')
        axes[2].set_title('Stochastic Resonance Effect')
        axes[2].axvline(x=0, color='black', linewidth=0.5)

        plt.suptitle('Phase 121: Edge of Chaos Regularization', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase121_edge_of_chaos.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 121 complete!")


if __name__ == '__main__':
    main()
