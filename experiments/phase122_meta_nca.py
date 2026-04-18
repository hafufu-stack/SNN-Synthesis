"""
Phase 122: In-Context Meta-NCA (Zero-Shot Rule Extraction)

Instead of TTT (backprop at test time), extract task rules from
demo pairs in a single forward pass.

Architecture:
  1. Task Encoder: processes (demo_input, demo_output) pairs
     -> produces a task embedding vector
  2. Task-Conditioned NCA: runs L-NCA with task embedding
     broadcast to every cell
  3. Output: solves test input using extracted rules

No gradient update at test time -> fits in 500ms budget.

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
# Synthetic ARC-like multi-task dataset
# ====================================================================
TASK_RULES = {
    'invert': lambda g: 1.0 - g,
    'flip_h': lambda g: np.flip(g, axis=1).copy(),
    'flip_v': lambda g: np.flip(g, axis=0).copy(),
    'rotate90': lambda g: np.rot90(g, k=1).copy(),
    'fill_border': lambda g: _fill_border(g),
    'hollow': lambda g: _hollow(g),
    'dilate': lambda g: _dilate(g),
    'erode': lambda g: _erode(g),
}


def _fill_border(g):
    out = g.copy()
    h, w = g.shape
    out[0, :] = 1; out[-1, :] = 1; out[:, 0] = 1; out[:, -1] = 1
    return out


def _hollow(g):
    out = np.zeros_like(g)
    h, w = g.shape
    for y in range(h):
        for x in range(w):
            if g[y, x] > 0.5:
                is_edge = False
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w or g[ny, nx] < 0.5:
                        is_edge = True; break
                if is_edge:
                    out[y, x] = 1.0
    return out


def _dilate(g):
    out = g.copy()
    h, w = g.shape
    for y in range(h):
        for x in range(w):
            if g[y, x] > 0.5:
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        out[ny, nx] = 1.0
    return out


def _erode(g):
    out = np.zeros_like(g)
    h, w = g.shape
    for y in range(h):
        for x in range(w):
            if g[y, x] > 0.5:
                all_neighbors = True
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w or g[ny, nx] < 0.5:
                        all_neighbors = False; break
                if all_neighbors:
                    out[y, x] = 1.0
    return out


def generate_meta_dataset(grid_sz=8, n_tasks_per_rule=50, n_demos=2):
    """Generate a meta-learning dataset with multiple ARC-like tasks."""
    dataset = []
    for rule_name, rule_fn in TASK_RULES.items():
        for _ in range(n_tasks_per_rule):
            # Generate random grids for this task
            demos = []
            for _ in range(n_demos):
                # Random input grid
                inp = np.zeros((grid_sz, grid_sz), dtype=np.float32)
                # Random shapes
                n_shapes = random.randint(1, 4)
                for _ in range(n_shapes):
                    cx = random.randint(1, grid_sz-2)
                    cy = random.randint(1, grid_sz-2)
                    r = random.randint(1, 2)
                    for dy in range(-r, r+1):
                        for dx in range(-r, r+1):
                            ny, nx = cy+dy, cx+dx
                            if 0 <= ny < grid_sz and 0 <= nx < grid_sz:
                                if random.random() > 0.3:
                                    inp[ny, nx] = 1.0
                out = rule_fn(inp)
                demos.append((inp, out))

            # Test example (different grid, same rule)
            test_inp = np.zeros((grid_sz, grid_sz), dtype=np.float32)
            n_shapes = random.randint(1, 4)
            for _ in range(n_shapes):
                cx = random.randint(1, grid_sz-2)
                cy = random.randint(1, grid_sz-2)
                r = random.randint(1, 2)
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < grid_sz and 0 <= nx < grid_sz:
                            if random.random() > 0.3:
                                test_inp[ny, nx] = 1.0
            test_out = rule_fn(test_inp)

            dataset.append({
                'rule': rule_name,
                'demos': demos,
                'test_inp': test_inp,
                'test_out': test_out,
            })

    random.shuffle(dataset)
    return dataset


# ====================================================================
# In-Context Meta-NCA
# ====================================================================
class TaskEncoder(nn.Module):
    """Encode demo (input, output) pairs into a task embedding."""
    def __init__(self, grid_ch=1, embed_dim=32):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, demo_inputs, demo_outputs):
        """
        demo_inputs: (n_demos, 1, H, W)
        demo_outputs: (n_demos, 1, H, W)
        Returns: task embedding (embed_dim,)
        """
        pairs = torch.cat([demo_inputs, demo_outputs], dim=1)  # (n_demos, 2, H, W)
        embeddings = self.pair_encoder(pairs)  # (n_demos, embed_dim)
        # Average across demos
        return embeddings.mean(dim=0)  # (embed_dim,)


class MetaNCA(nn.Module):
    """NCA conditioned on task embedding."""
    def __init__(self, hidden_ch=32, task_embed_dim=32):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_ch, 3, padding=1), nn.ReLU()
        )
        # Update conditioned on task
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch + task_embed_dim, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
        )
        self.tau_gate = nn.Sequential(
            nn.Conv2d(hidden_ch + task_embed_dim, hidden_ch, 1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, task_embed, n_steps=5):
        """
        x: (B, 1, H, W) test input
        task_embed: (embed_dim,) task context
        """
        B, _, H, W = x.shape
        state = self.stem(x)

        # Broadcast task embedding to spatial dims
        te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

        for t in range(n_steps):
            state_ctx = torch.cat([state, te], dim=1)
            delta = self.update(state_ctx)
            beta = self.tau_gate(state_ctx)
            state = beta * state + (1 - beta) * delta

        return self.decoder(state)


class InContextMetaNCA(nn.Module):
    """Full system: TaskEncoder + MetaNCA."""
    def __init__(self, embed_dim=32, hidden_ch=32):
        super().__init__()
        self.task_encoder = TaskEncoder(embed_dim=embed_dim)
        self.meta_nca = MetaNCA(hidden_ch=hidden_ch, task_embed_dim=embed_dim)

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5):
        task_embed = self.task_encoder(demo_inputs, demo_outputs)
        return self.meta_nca(test_input, task_embed, n_steps=n_steps)


# ====================================================================
# Baseline: Direct NCA (no context)
# ====================================================================
class DirectNCA(nn.Module):
    """NCA trained on all tasks without context (memorization baseline)."""
    def __init__(self, hidden_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden_ch, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_ch, 1, 1), nn.Sigmoid())

    def forward(self, x, n_steps=5):
        s = self.stem(x)
        for t in range(n_steps):
            s = F.relu(s + self.update(s))
        return self.decoder(s)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 122: In-Context Meta-NCA")
    print(f"  Device: {DEVICE}")
    print(f"  Rules: {list(TASK_RULES.keys())}")
    print("=" * 70)

    # Generate dataset
    print("\n[Step 1] Generating meta-dataset...")
    dataset = generate_meta_dataset(grid_sz=8, n_tasks_per_rule=60, n_demos=2)
    print(f"  Total tasks: {len(dataset)}")

    # Split: train on some rules, test on others (meta-generalization)
    train_rules = ['invert', 'flip_h', 'flip_v', 'rotate90', 'fill_border', 'hollow']
    test_rules = ['dilate', 'erode']  # unseen rules at test time!

    train_data = [d for d in dataset if d['rule'] in train_rules]
    test_data_seen = [d for d in dataset if d['rule'] in train_rules][-60:]  # last 60 of seen rules
    test_data_unseen = [d for d in dataset if d['rule'] in test_rules]

    print(f"  Train: {len(train_data)} tasks ({train_rules})")
    print(f"  Test (seen rules): {len(test_data_seen)}")
    print(f"  Test (unseen rules): {len(test_data_unseen)} ({test_rules})")

    # Train In-Context Meta-NCA
    print("\n[Step 2] Training In-Context Meta-NCA...")
    model = InContextMetaNCA(embed_dim=32, hidden_ch=32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

    for epoch in range(80):
        model.train()
        random.shuffle(train_data)
        epoch_loss = 0; n = 0
        for item in train_data:
            demos = item['demos']
            demo_in = torch.stack([torch.tensor(d[0]).unsqueeze(0) for d in demos]).to(DEVICE)
            demo_out = torch.stack([torch.tensor(d[1]).unsqueeze(0) for d in demos]).to(DEVICE)
            test_in = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            test_out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

            pred = model(demo_in, demo_out, test_in, n_steps=5)
            loss = F.binary_cross_entropy(pred, test_out)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n += 1

        sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/80: loss={epoch_loss/n:.4f}")

    # Train baseline (Direct NCA, no context)
    print("\n[Step 3] Training Direct NCA (no context baseline)...")
    baseline = DirectNCA(hidden_ch=32).to(DEVICE)
    opt_b = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    for epoch in range(80):
        baseline.train()
        random.shuffle(train_data)
        for item in train_data:
            test_in = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            test_out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred = baseline(test_in, n_steps=5)
            loss = F.binary_cross_entropy(pred, test_out)
            opt_b.zero_grad(); loss.backward(); opt_b.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/80")

    # Evaluate
    print("\n[Step 4] Evaluation...")
    results = {}

    for split_name, test_data in [('seen', test_data_seen), ('unseen', test_data_unseen)]:
        model.eval(); baseline.eval()
        meta_iou = 0; base_iou = 0; n = 0
        rule_results = {}

        with torch.no_grad():
            for item in test_data:
                demos = item['demos']
                demo_in = torch.stack([torch.tensor(d[0]).unsqueeze(0) for d in demos]).to(DEVICE)
                demo_out = torch.stack([torch.tensor(d[1]).unsqueeze(0) for d in demos]).to(DEVICE)
                test_in = torch.tensor(item['test_inp']).unsqueeze(0).unsqueeze(0).to(DEVICE)
                test_out = torch.tensor(item['test_out']).unsqueeze(0).unsqueeze(0).to(DEVICE)

                # Meta-NCA
                pred_meta = model(demo_in, demo_out, test_in, n_steps=5)
                pred_meta_b = (pred_meta > 0.5).float()
                inter = (pred_meta_b * test_out).sum()
                union = ((pred_meta_b + test_out) > 0).float().sum()
                m_iou = (inter / (union + 1e-8)).item()
                meta_iou += m_iou

                # Baseline
                pred_base = baseline(test_in, n_steps=5)
                pred_base_b = (pred_base > 0.5).float()
                inter_b = (pred_base_b * test_out).sum()
                union_b = ((pred_base_b + test_out) > 0).float().sum()
                b_iou = (inter_b / (union_b + 1e-8)).item()
                base_iou += b_iou

                rule = item['rule']
                if rule not in rule_results:
                    rule_results[rule] = {'meta': [], 'base': []}
                rule_results[rule]['meta'].append(m_iou)
                rule_results[rule]['base'].append(b_iou)
                n += 1

        meta_avg = meta_iou / max(n, 1)
        base_avg = base_iou / max(n, 1)

        results[split_name] = {
            'meta_iou': meta_avg,
            'base_iou': base_avg,
            'per_rule': {r: {'meta': np.mean(v['meta']), 'base': np.mean(v['base'])}
                         for r, v in rule_results.items()}
        }

        print(f"\n  {split_name.upper()} rules:")
        print(f"    Meta-NCA: IoU={meta_avg:.4f}")
        print(f"    Baseline: IoU={base_avg:.4f}")
        print(f"    Gap:      {(meta_avg-base_avg)*100:+.2f}%")
        for rule, rv in sorted(rule_results.items()):
            m = np.mean(rv['meta'])
            b = np.mean(rv['base'])
            print(f"      {rule:15s}: meta={m:.3f}, base={b:.3f}, gap={m-b:+.3f}")

    # Summary
    print(f"\n{'='*70}")
    print("  IN-CONTEXT META-NCA RESULTS")
    print(f"{'='*70}")
    seen = results['seen']
    unseen = results['unseen']
    print(f"  Seen rules:   Meta={seen['meta_iou']:.4f}, Base={seen['base_iou']:.4f}")
    print(f"  Unseen rules: Meta={unseen['meta_iou']:.4f}, Base={unseen['base_iou']:.4f}")

    if unseen['meta_iou'] > unseen['base_iou'] + 0.05:
        print(f"\n  ** META-NCA GENERALIZES TO UNSEEN RULES! **")
    elif unseen['meta_iou'] > unseen['base_iou']:
        print(f"\n  Meta-NCA slightly better on unseen rules")
    else:
        print(f"\n  Meta-NCA did not generalize to unseen rules")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase122_meta_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 122: In-Context Meta-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Per-rule comparison
        for idx, split in enumerate(['seen', 'unseen']):
            pr = results[split]['per_rule']
            rules = sorted(pr.keys())
            x = np.arange(len(rules))
            meta_vals = [pr[r]['meta'] for r in rules]
            base_vals = [pr[r]['base'] for r in rules]
            w = 0.35
            axes[idx].bar(x - w/2, meta_vals, w, label='Meta-NCA', color='#3498db')
            axes[idx].bar(x + w/2, base_vals, w, label='Baseline', color='#e74c3c')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(rules, fontsize=7, rotation=30)
            axes[idx].set_ylabel('IoU')
            axes[idx].set_title(f'{split.title()} Rules')
            axes[idx].legend(fontsize=8)

        # Summary comparison
        categories = ['Seen\nMeta', 'Seen\nBase', 'Unseen\nMeta', 'Unseen\nBase']
        vals = [seen['meta_iou'], seen['base_iou'],
                unseen['meta_iou'], unseen['base_iou']]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#e67e22']
        axes[2].bar(range(4), vals, color=colors)
        axes[2].set_xticks(range(4))
        axes[2].set_xticklabels(categories, fontsize=8)
        axes[2].set_ylabel('IoU')
        axes[2].set_title('Meta-Generalization')
        for i, v in enumerate(vals):
            axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

        plt.suptitle('Phase 122: In-Context Meta-NCA', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase122_meta_nca.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 122 complete!")


if __name__ == '__main__':
    main()
