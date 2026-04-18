"""
Phase 128: Vector-Quantized L-NCA (VQ-NCA)

The "Articulation Bottleneck" problem:
  Latent-NCA pixel accuracy = 72.6% but exact match = 0/50.
  The continuous latent space causes "smearing" - outputs
  are slightly blurry/shifted, failing exact match.

Solution: Vector Quantization forces latent states into
discrete codebook entries (like VQ-VAE), eliminating blur.

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

# Reuse tasks
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase122_meta_nca import TASK_RULES, generate_meta_dataset


class VectorQuantizer(nn.Module):
    """VQ layer: quantize latent states to discrete codebook entries."""
    def __init__(self, n_embeddings=32, embedding_dim=16, commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(n_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/n_embeddings, 1/n_embeddings)

    def forward(self, z):
        """
        z: (B, C, H, W) where C = embedding_dim
        Returns: quantized z, VQ loss, codebook usage stats
        """
        B, C, H, W = z.shape
        # Reshape to (B*H*W, C)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)

        # Compute distances
        dists = (z_flat ** 2).sum(dim=1, keepdim=True) + \
                (self.codebook.weight ** 2).sum(dim=1) - \
                2 * z_flat @ self.codebook.weight.t()

        # Find nearest codebook entry
        indices = dists.argmin(dim=1)
        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        # Losses
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        # Codebook usage
        usage = len(indices.unique()) / self.n_embeddings

        return z_q_st, vq_loss, usage


class ContinuousNCA(nn.Module):
    """Standard continuous NCA (baseline)."""
    def __init__(self, hidden_ch=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_ch, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1), nn.Sigmoid())
        self.decoder = nn.Conv2d(hidden_ch, 1, 1)

    def forward(self, x, n_steps=5):
        state = self.stem(x)
        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau(state)
            state = beta * state + (1 - beta) * delta
        return torch.sigmoid(self.decoder(state))


class VQNCA(nn.Module):
    """NCA with Vector Quantization at each step."""
    def __init__(self, hidden_ch=16, n_codes=32, vq_every=1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_ch, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1), nn.Sigmoid())
        self.vq = VectorQuantizer(n_codes, hidden_ch)
        self.decoder = nn.Conv2d(hidden_ch, 1, 1)
        self.vq_every = vq_every

    def forward(self, x, n_steps=5, return_vq_loss=False):
        state = self.stem(x)
        total_vq_loss = 0
        total_usage = 0
        vq_count = 0

        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau(state)
            state = beta * state + (1 - beta) * delta

            # Quantize every vq_every steps
            if (t + 1) % self.vq_every == 0:
                state, vq_loss, usage = self.vq(state)
                total_vq_loss += vq_loss
                total_usage += usage
                vq_count += 1

        output = torch.sigmoid(self.decoder(state))
        if return_vq_loss:
            avg_loss = total_vq_loss / max(vq_count, 1)
            avg_usage = total_usage / max(vq_count, 1)
            return output, avg_loss, avg_usage
        return output


class VQFinalNCA(nn.Module):
    """NCA with VQ only at the FINAL step (softer variant)."""
    def __init__(self, hidden_ch=16, n_codes=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_ch, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1), nn.Sigmoid())
        self.vq = VectorQuantizer(n_codes, hidden_ch)
        self.decoder = nn.Conv2d(hidden_ch, 1, 1)

    def forward(self, x, n_steps=5, return_vq_loss=False):
        state = self.stem(x)
        for t in range(n_steps):
            delta = self.update(state)
            beta = self.tau(state)
            state = beta * state + (1 - beta) * delta

        # Quantize ONLY at the end
        state, vq_loss, usage = self.vq(state)
        output = torch.sigmoid(self.decoder(state))
        if return_vq_loss:
            return output, vq_loss, usage
        return output


def train_model(name, model, train_data, is_vq=False, epochs=60):
    """Train a model."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        for item in train_data:
            inp = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

            if is_vq:
                pred, vq_loss, _ = model(inp, n_steps=5, return_vq_loss=True)
                loss = F.binary_cross_entropy(pred, out) + vq_loss
            else:
                pred = model(inp, n_steps=5)
                loss = F.binary_cross_entropy(pred, out)

            opt.zero_grad(); loss.backward(); opt.step()

    return model


def evaluate_model(name, model, test_data, is_vq=False):
    """Evaluate model with pixel accuracy AND exact match."""
    model.eval()
    rule_results = {}
    total_px_correct = 0; total_px = 0
    total_exact = 0; total_n = 0
    avg_usage = 0; usage_n = 0

    with torch.no_grad():
        for item in test_data:
            inp = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

            if is_vq:
                pred, _, usage = model(inp, n_steps=5, return_vq_loss=True)
                avg_usage += usage; usage_n += 1
            else:
                pred = model(inp, n_steps=5)

            pred_b = (pred > 0.5).float()
            px_correct = (pred_b == out).float().mean().item()
            exact = (pred_b == out).all().item()

            total_px_correct += px_correct
            total_px += 1
            total_exact += exact
            total_n += 1

            rule = item['rule']
            if rule not in rule_results:
                rule_results[rule] = {'px': [], 'exact': []}
            rule_results[rule]['px'].append(px_correct)
            rule_results[rule]['exact'].append(exact)

    px_acc = total_px_correct / max(total_px, 1)
    exact_rate = total_exact / max(total_n, 1)
    avg_u = avg_usage / max(usage_n, 1) if usage_n > 0 else 0

    return {
        'pixel_acc': px_acc,
        'exact_rate': exact_rate,
        'exact_count': total_exact,
        'total': total_n,
        'codebook_usage': avg_u,
        'per_rule': {r: {'px': np.mean(v['px']), 'exact': np.mean(v['exact'])}
                     for r, v in rule_results.items()},
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 128: Vector-Quantized L-NCA (VQ-NCA)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Generate dataset
    print("\n[Step 1] Generating dataset...")
    all_rules = list(TASK_RULES.keys())
    dataset = generate_meta_dataset(grid_sz=8, n_tasks_per_rule=100, n_demos=2)

    # Use individual tasks for training (not meta)
    train_data = dataset[:int(len(dataset)*0.8)]
    test_data = dataset[int(len(dataset)*0.8):]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # Compare architectures
    configs = [
        ('Continuous-NCA', lambda: ContinuousNCA(hidden_ch=16).to(DEVICE), False),
        ('VQ-NCA (every step)', lambda: VQNCA(hidden_ch=16, n_codes=32, vq_every=1).to(DEVICE), True),
        ('VQ-NCA (every 2)', lambda: VQNCA(hidden_ch=16, n_codes=32, vq_every=2).to(DEVICE), True),
        ('VQ-NCA (final only)', lambda: VQFinalNCA(hidden_ch=16, n_codes=32).to(DEVICE), True),
        ('VQ-NCA (64 codes)', lambda: VQNCA(hidden_ch=16, n_codes=64, vq_every=1).to(DEVICE), True),
        ('VQ-NCA (128 codes)', lambda: VQNCA(hidden_ch=16, n_codes=128, vq_every=1).to(DEVICE), True),
    ]

    all_results = {}

    for name, model_fn, is_vq in configs:
        print(f"\n  === {name} ===")
        torch.manual_seed(SEED)
        model = model_fn()
        n_params = sum(p.numel() for p in model.parameters())

        model = train_model(name, model, train_data, is_vq=is_vq, epochs=60)
        res = evaluate_model(name, model, test_data, is_vq=is_vq)
        res['n_params'] = n_params
        all_results[name] = res

        em = res['exact_count']
        tt = res['total']
        print(f"    pixel={res['pixel_acc']*100:.2f}%, "
              f"exact={em}/{tt} ({res['exact_rate']*100:.1f}%), "
              f"params={n_params:,}")
        if is_vq:
            print(f"    codebook usage: {res['codebook_usage']*100:.1f}%")

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("  VQ-NCA RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Config':30s} {'Pixel':>8s} {'Exact':>8s} {'Exact#':>7s} {'CBUsage':>8s}")
    print(f"  {'-'*65}")
    for name, res in all_results.items():
        cb = f"{res.get('codebook_usage',0)*100:.0f}%" if res.get('codebook_usage',0) > 0 else "N/A"
        print(f"  {name:30s} {res['pixel_acc']*100:7.2f}% "
              f"{res['exact_rate']*100:7.1f}% "
              f"{res['exact_count']:>3d}/{res['total']:<3d} {cb:>7s}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase128_vq_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 128: VQ-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': all_results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        names = list(all_results.keys())
        px_accs = [all_results[n]['pixel_acc'] * 100 for n in names]
        exact_rates = [all_results[n]['exact_rate'] * 100 for n in names]

        colors = ['#e74c3c', '#2ecc71', '#27ae60', '#1abc9c', '#3498db', '#2980b9']

        axes[0].bar(range(len(names)), px_accs, color=colors)
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(names, fontsize=6, rotation=30, ha='right')
        axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title('Pixel Accuracy')

        axes[1].bar(range(len(names)), exact_rates, color=colors)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, fontsize=6, rotation=30, ha='right')
        axes[1].set_ylabel('Exact Match Rate (%)')
        axes[1].set_title('Exact Match Rate (The Goal!)')

        # Per-rule for best VQ
        best_vq_name = max(
            [(n, r['exact_rate']) for n, r in all_results.items() if 'VQ' in n],
            key=lambda x: x[1]
        )[0]
        cont_pr = all_results['Continuous-NCA'].get('per_rule', {})
        vq_pr = all_results[best_vq_name].get('per_rule', {})
        rules = sorted(set(list(cont_pr.keys()) + list(vq_pr.keys())))
        cont_exact = [cont_pr.get(r, {}).get('exact', 0) * 100 for r in rules]
        vq_exact = [vq_pr.get(r, {}).get('exact', 0) * 100 for r in rules]
        x = np.arange(len(rules))
        axes[2].bar(x - 0.18, cont_exact, 0.35, label='Continuous', color='#e74c3c')
        axes[2].bar(x + 0.18, vq_exact, 0.35, label=best_vq_name, color='#2ecc71')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(rules, fontsize=6, rotation=30, ha='right')
        axes[2].set_ylabel('Exact Match (%)')
        axes[2].set_title('Per-Rule Exact Match')
        axes[2].legend(fontsize=7)

        plt.suptitle('Phase 128: Vector-Quantized NCA (Quantum Cells)', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase128_vq_nca.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 128 complete!")


if __name__ == '__main__':
    main()
