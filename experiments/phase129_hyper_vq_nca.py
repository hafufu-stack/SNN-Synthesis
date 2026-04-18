"""
Phase 129: Hyper-VQ-NCA (Instant Quantum Cell Generation)

Fusion of Phase 127 (Hypernetwork) + Phase 128 (VQ-NCA):
  Encoder → HyperNet → VQ-NCA weights → Discrete inference

"See demo → instantly generate a pixel-perfect cell"

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

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase122_meta_nca import TASK_RULES, generate_meta_dataset


class VectorQuantizer(nn.Module):
    """VQ layer with straight-through estimator."""
    def __init__(self, n_codes=32, dim=16, commitment=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.commitment = commitment
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1/n_codes, 1/n_codes)

    def forward(self, z, gumbel_noise=0.0):
        """z: (B, C, H, W). Returns quantized, vq_loss, indices."""
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)

        dists = (z_flat ** 2).sum(1, keepdim=True) + \
                (self.codebook.weight ** 2).sum(1) - \
                2 * z_flat @ self.codebook.weight.t()

        # Optional Gumbel noise for exploration
        if gumbel_noise > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(dists) + 1e-20) + 1e-20)
            dists = dists - gumbel_noise * gumbel

        indices = dists.argmin(dim=1)
        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)

        commit_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment * commit_loss

        z_q_st = z + (z_q - z).detach()
        usage = len(indices.unique()) / self.n_codes
        return z_q_st, vq_loss, usage, indices.view(B, H, W)


class TaskEncoder(nn.Module):
    """Encode demo pairs into task embedding."""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, demo_inputs, demo_outputs):
        pairs = torch.cat([demo_inputs, demo_outputs], dim=1)
        return self.net(pairs).mean(dim=0)


class HyperVQNCA(nn.Module):
    """Full Hyper-VQ-NCA system."""
    def __init__(self, embed_dim=64, nca_hidden=16, n_codes=32):
        super().__init__()
        self.nca_hidden = nca_hidden
        self.encoder = TaskEncoder(embed_dim=embed_dim)

        # Hypernetwork: generate NCA weights
        update_size = nca_hidden * nca_hidden * 3 * 3
        tau_size = nca_hidden * nca_hidden
        self.gen_update = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, update_size),
        )
        self.gen_tau = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, tau_size),
        )

        # Shared components
        self.stem = nn.Sequential(
            nn.Conv2d(1, nca_hidden, 3, padding=1), nn.ReLU())
        self.vq = VectorQuantizer(n_codes, nca_hidden)
        self.decoder = nn.Sequential(
            nn.Conv2d(nca_hidden, 1, 1), nn.Sigmoid())

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5,
                return_vq_loss=False, gumbel_noise=0.0):
        # Generate weights from demos
        te = self.encoder(demo_inputs, demo_outputs)
        W_update = self.gen_update(te).view(
            self.nca_hidden, self.nca_hidden, 3, 3)
        W_tau = self.gen_tau(te).view(
            self.nca_hidden, self.nca_hidden, 1, 1)

        # Run VQ-NCA with generated weights
        state = self.stem(test_input)
        total_vq = 0; total_usage = 0

        for t in range(n_steps):
            delta = F.relu(F.conv2d(state, W_update, padding=1))
            beta = torch.sigmoid(F.conv2d(state, W_tau))
            state = beta * state + (1 - beta) * delta

            # Quantize at every step
            state, vq_loss, usage, _ = self.vq(state, gumbel_noise=gumbel_noise)
            total_vq += vq_loss
            total_usage += usage

        output = self.decoder(state)
        if return_vq_loss:
            return output, total_vq / n_steps, total_usage / n_steps
        return output


# Baselines for comparison
class ContextNCA(nn.Module):
    """Context-injection NCA (no VQ)."""
    def __init__(self, embed_dim=64, nca_hidden=16):
        super().__init__()
        self.encoder = TaskEncoder(embed_dim=embed_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(1, nca_hidden, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(nca_hidden + embed_dim, nca_hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(nca_hidden, nca_hidden, 1))
        self.tau = nn.Sequential(
            nn.Conv2d(nca_hidden + embed_dim, nca_hidden, 1), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Conv2d(nca_hidden, 1, 1), nn.Sigmoid())

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5, **kw):
        te = self.encoder(demo_inputs, demo_outputs)
        B, _, H, W = test_input.shape
        te_s = te.view(1, -1, 1, 1).expand(B, -1, H, W)
        state = self.stem(test_input)
        for t in range(n_steps):
            ctx = torch.cat([state, te_s], dim=1)
            delta = self.update(ctx)
            beta = self.tau(ctx)
            state = beta * state + (1 - beta) * delta
        return self.decoder(state)


class HyperNCA(nn.Module):
    """Hyper-NCA without VQ (Phase 127)."""
    def __init__(self, embed_dim=64, nca_hidden=16):
        super().__init__()
        self.nca_hidden = nca_hidden
        self.encoder = TaskEncoder(embed_dim=embed_dim)
        update_size = nca_hidden * nca_hidden * 3 * 3
        tau_size = nca_hidden * nca_hidden
        self.gen_update = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, update_size))
        self.gen_tau = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, tau_size))
        self.stem = nn.Sequential(
            nn.Conv2d(1, nca_hidden, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(nca_hidden, 1, 1), nn.Sigmoid())

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5, **kw):
        te = self.encoder(demo_inputs, demo_outputs)
        W_u = self.gen_update(te).view(self.nca_hidden, self.nca_hidden, 3, 3)
        W_t = self.gen_tau(te).view(self.nca_hidden, self.nca_hidden, 1, 1)
        state = self.stem(test_input)
        for t in range(n_steps):
            delta = F.relu(F.conv2d(state, W_u, padding=1))
            beta = torch.sigmoid(F.conv2d(state, W_t))
            state = beta * state + (1 - beta) * delta
        return self.decoder(state)


def train_and_eval(name, model, train_data, test_data, is_vq=False, epochs=60):
    """Train and evaluate."""
    print(f"\n  [{name}] Training...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        for item in train_data:
            demos = item['demos']
            di = torch.stack([torch.tensor(d[0]).unsqueeze(0) for d in demos]).to(DEVICE)
            do = torch.stack([torch.tensor(d[1]).unsqueeze(0) for d in demos]).to(DEVICE)
            ti = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            to = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

            if is_vq:
                pred, vq_loss, _ = model(di, do, ti, n_steps=5, return_vq_loss=True)
                loss = F.binary_cross_entropy(pred, to) + vq_loss
            else:
                pred = model(di, do, ti, n_steps=5)
                loss = F.binary_cross_entropy(pred, to)

            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}")

    # Evaluate
    model.eval()
    rule_results = {}
    total_px = 0; total_exact = 0; n = 0

    with torch.no_grad():
        for item in test_data:
            demos = item['demos']
            di = torch.stack([torch.tensor(d[0]).unsqueeze(0) for d in demos]).to(DEVICE)
            do = torch.stack([torch.tensor(d[1]).unsqueeze(0) for d in demos]).to(DEVICE)
            ti = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            to = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

            pred = model(di, do, ti, n_steps=5)
            pred_b = (pred > 0.5).float()
            px = (pred_b == to).float().mean().item()
            exact = (pred_b == to).all().item()
            total_px += px; total_exact += exact; n += 1

            rule = item['rule']
            if rule not in rule_results:
                rule_results[rule] = {'px': [], 'exact': []}
            rule_results[rule]['px'].append(px)
            rule_results[rule]['exact'].append(exact)

    px_acc = total_px / max(n, 1)
    exact_rate = total_exact / max(n, 1)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"    pixel={px_acc*100:.2f}%, exact={total_exact}/{n} ({exact_rate*100:.1f}%), "
          f"params={n_params:,}")

    return {
        'pixel_acc': px_acc, 'exact_rate': exact_rate,
        'exact_count': total_exact, 'total': n,
        'n_params': n_params,
        'per_rule': {r: {'px': np.mean(v['px']), 'exact': np.mean(v['exact'])}
                     for r, v in rule_results.items()},
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 129: Hyper-VQ-NCA (Instant Quantum Cell Generation)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Generating meta-dataset...")
    dataset = generate_meta_dataset(grid_sz=8, n_tasks_per_rule=80, n_demos=2)

    train_rules = ['invert', 'flip_h', 'flip_v', 'rotate90', 'fill_border', 'hollow']
    test_rules = ['dilate', 'erode']

    train_data = [d for d in dataset if d['rule'] in train_rules]
    test_seen = [d for d in dataset if d['rule'] in train_rules][-80:]
    test_unseen = [d for d in dataset if d['rule'] in test_rules]
    test_all = test_seen + test_unseen

    print(f"  Train: {len(train_data)}, Test: {len(test_all)} "
          f"(seen={len(test_seen)}, unseen={len(test_unseen)})")

    configs = [
        ('Context-NCA', lambda: ContextNCA(embed_dim=64, nca_hidden=16).to(DEVICE), False),
        ('Hyper-NCA', lambda: HyperNCA(embed_dim=64, nca_hidden=16).to(DEVICE), False),
        ('Hyper-VQ-NCA', lambda: HyperVQNCA(embed_dim=64, nca_hidden=16, n_codes=32).to(DEVICE), True),
        ('Hyper-VQ-NCA-64', lambda: HyperVQNCA(embed_dim=64, nca_hidden=16, n_codes=64).to(DEVICE), True),
    ]

    all_results = {}
    for name, model_fn, is_vq in configs:
        torch.manual_seed(SEED)
        model = model_fn()
        all_results[name] = train_and_eval(name, model, train_data, test_all, is_vq, epochs=60)
        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("  HYPER-VQ-NCA COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Name':25s} {'Pixel':>8s} {'Exact':>10s} {'Params':>10s}")
    for name, res in all_results.items():
        print(f"  {name:25s} {res['pixel_acc']*100:7.2f}% "
              f"{res['exact_count']:>3d}/{res['total']:<3d} "
              f"{res['n_params']:>9,}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase129_hyper_vq_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 129: Hyper-VQ-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': all_results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        names = list(all_results.keys())
        px = [all_results[n]['pixel_acc']*100 for n in names]
        ex = [all_results[n]['exact_rate']*100 for n in names]
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#27ae60']

        axes[0].bar(range(len(names)), px, color=colors)
        axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, fontsize=7, rotation=15)
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('Pixel Accuracy')

        axes[1].bar(range(len(names)), ex, color=colors)
        axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, fontsize=7, rotation=15)
        axes[1].set_ylabel('Exact Match (%)'); axes[1].set_title('Exact Match Rate')

        plt.suptitle('Phase 129: Hyper-VQ-NCA', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase129_hyper_vq_nca.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 129 complete!")


if __name__ == '__main__':
    main()
