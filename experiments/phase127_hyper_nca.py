"""
Phase 127: Hyper-NCA Auto-Compiler

Instead of context injection (Phase 122), a Hypernetwork
directly generates NCA weights from task embedding.

"See demo → instantly generate a specialized cell"

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

# Reuse synthetic tasks from Phase 122
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase122_meta_nca import TASK_RULES, generate_meta_dataset


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


class HyperNetwork(nn.Module):
    """Generate NCA weights from task embedding."""
    def __init__(self, embed_dim=64, hidden_ch=16, nca_hidden=16):
        super().__init__()
        self.nca_hidden = nca_hidden

        # Generate update conv weights: (nca_hidden, nca_hidden, 3, 3)
        update_size = nca_hidden * nca_hidden * 3 * 3
        self.gen_update = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, update_size),
        )

        # Generate tau gate weights: (nca_hidden, nca_hidden, 1, 1)
        tau_size = nca_hidden * nca_hidden
        self.gen_tau = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, tau_size),
        )

    def forward(self, task_embed):
        """Generate NCA weights from embedding."""
        W_update = self.gen_update(task_embed).view(
            self.nca_hidden, self.nca_hidden, 3, 3)
        W_tau = self.gen_tau(task_embed).view(
            self.nca_hidden, self.nca_hidden, 1, 1)
        return W_update, W_tau


class GeneratedNCA(nn.Module):
    """NCA that uses externally generated weights."""
    def __init__(self, nca_hidden=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, nca_hidden, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Conv2d(nca_hidden, 1, 1), nn.Sigmoid())

    def forward(self, x, W_update, W_tau, n_steps=5):
        state = self.stem(x)
        for t in range(n_steps):
            delta = F.relu(F.conv2d(state, W_update, padding=1))
            beta = torch.sigmoid(F.conv2d(state, W_tau))
            state = beta * state + (1 - beta) * delta
        return self.decoder(state)


class HyperNCASystem(nn.Module):
    """Full system: Encoder → HyperNetwork → Generated NCA."""
    def __init__(self, embed_dim=64, nca_hidden=16):
        super().__init__()
        self.encoder = TaskEncoder(embed_dim=embed_dim)
        self.hypernet = HyperNetwork(embed_dim=embed_dim, nca_hidden=nca_hidden)
        self.nca = GeneratedNCA(nca_hidden=nca_hidden)

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5):
        task_embed = self.encoder(demo_inputs, demo_outputs)
        W_update, W_tau = self.hypernet(task_embed)
        return self.nca(test_input, W_update, W_tau, n_steps=n_steps)


# Context-injection baseline (same as Phase 122 but with same capacity)
class ContextNCA(nn.Module):
    """Phase 122-style context injection for fair comparison."""
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

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5):
        te = self.encoder(demo_inputs, demo_outputs)
        B, _, H, W = test_input.shape
        te_spatial = te.view(1, -1, 1, 1).expand(B, -1, H, W)

        state = self.stem(test_input)
        for t in range(n_steps):
            ctx = torch.cat([state, te_spatial], dim=1)
            delta = self.update(ctx)
            beta = self.tau(ctx)
            state = beta * state + (1 - beta) * delta
        return self.decoder(state)


def train_and_eval(name, model, train_data, test_seen, test_unseen, epochs=60):
    """Train and evaluate a meta-learning model."""
    print(f"\n  [{name}] Training...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        for item in train_data:
            demos = item['demos']
            demo_in = torch.stack([torch.tensor(d[0]).unsqueeze(0) for d in demos]).to(DEVICE)
            demo_out = torch.stack([torch.tensor(d[1]).unsqueeze(0) for d in demos]).to(DEVICE)
            test_in = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            test_out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

            pred = model(demo_in, demo_out, test_in, n_steps=5)
            loss = F.binary_cross_entropy(pred, test_out)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}")

    # Evaluate
    model.eval()
    results = {}
    for split_name, data in [('seen', test_seen), ('unseen', test_unseen)]:
        iou_sum = 0; n = 0
        rule_results = {}
        with torch.no_grad():
            for item in data:
                demos = item['demos']
                demo_in = torch.stack([torch.tensor(d[0]).unsqueeze(0) for d in demos]).to(DEVICE)
                demo_out = torch.stack([torch.tensor(d[1]).unsqueeze(0) for d in demos]).to(DEVICE)
                test_in = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
                test_out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

                pred = model(demo_in, demo_out, test_in, n_steps=5)
                pred_b = (pred > 0.5).float()
                inter = (pred_b * test_out).sum()
                union = ((pred_b + test_out) > 0).float().sum()
                iou = (inter / (union + 1e-8)).item()
                iou_sum += iou; n += 1

                rule = item['rule']
                if rule not in rule_results:
                    rule_results[rule] = []
                rule_results[rule].append(iou)

        avg_iou = iou_sum / max(n, 1)
        results[split_name] = {
            'iou': avg_iou,
            'per_rule': {r: np.mean(v) for r, v in rule_results.items()}
        }
        print(f"    {split_name}: IoU={avg_iou:.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    results['n_params'] = n_params
    print(f"    Params: {n_params:,}")
    return results


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 127: Hyper-NCA Auto-Compiler")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Generate dataset
    print("\n[Step 1] Generating meta-dataset...")
    dataset = generate_meta_dataset(grid_sz=8, n_tasks_per_rule=60, n_demos=2)

    train_rules = ['invert', 'flip_h', 'flip_v', 'rotate90', 'fill_border', 'hollow']
    test_rules = ['dilate', 'erode']

    train_data = [d for d in dataset if d['rule'] in train_rules]
    test_seen = [d for d in dataset if d['rule'] in train_rules][-60:]
    test_unseen = [d for d in dataset if d['rule'] in test_rules]

    print(f"  Train: {len(train_data)}, Test seen: {len(test_seen)}, "
          f"Test unseen: {len(test_unseen)}")

    # Compare: Hyper-NCA vs Context-NCA
    all_results = {}

    # Context-NCA (Phase 122 style)
    torch.manual_seed(SEED)
    ctx_model = ContextNCA(embed_dim=64, nca_hidden=16).to(DEVICE)
    all_results['Context-NCA'] = train_and_eval(
        'Context-NCA', ctx_model, train_data, test_seen, test_unseen)
    del ctx_model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # Hyper-NCA
    torch.manual_seed(SEED)
    hyper_model = HyperNCASystem(embed_dim=64, nca_hidden=16).to(DEVICE)
    all_results['Hyper-NCA'] = train_and_eval(
        'Hyper-NCA', hyper_model, train_data, test_seen, test_unseen)

    # Summary
    print(f"\n{'='*70}")
    print("  HYPER-NCA vs CONTEXT-NCA RESULTS")
    print(f"{'='*70}")
    for name, res in all_results.items():
        s = res['seen']['iou']
        u = res['unseen']['iou']
        p = res['n_params']
        print(f"  {name:20s}: seen={s:.4f}, unseen={u:.4f}, params={p:,}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase127_hyper_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 127: Hyper-NCA Auto-Compiler',
                   'timestamp': datetime.now().isoformat(),
                   'results': all_results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, split in enumerate(['seen', 'unseen']):
            names = list(all_results.keys())
            for ni, name in enumerate(names):
                pr = all_results[name][split].get('per_rule', {})
                rules = sorted(pr.keys())
                vals = [pr[r] for r in rules]
                x = np.arange(len(rules))
                w = 0.35
                offset = (ni - 0.5) * w
                axes[idx].bar(x + offset, vals, w, label=name,
                            color=['#3498db', '#e74c3c'][ni])
            axes[idx].set_xticks(np.arange(len(rules)))
            axes[idx].set_xticklabels(rules, fontsize=7, rotation=30)
            axes[idx].set_ylabel('IoU')
            axes[idx].set_title(f'{split.title()} Rules')
            axes[idx].legend(fontsize=8)

        plt.suptitle('Phase 127: Hyper-NCA vs Context-NCA', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase127_hyper_nca.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 127 complete!")


if __name__ == '__main__':
    main()
