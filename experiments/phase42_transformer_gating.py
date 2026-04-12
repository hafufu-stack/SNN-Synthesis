"""
Phase 42: Transformer ID Gating (LoRA-based Task Routing)
Test if discrete ID gating works on Transformers via LoRA routing.
Compare: shared LoRA vs per-task LoRA vs ID-gated LoRA.

Author: Hiroto Funasaki
"""
import os, json, gc, time, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Lightweight LoRA Module
# ==============================================================
class LoRALayer(nn.Module):
    """Low-rank adaptation layer."""
    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        return x @ self.lora_A @ self.lora_B


class IDGatedLoRA(nn.Module):
    """LoRA with discrete ID gating: different LoRA weights per task ID."""
    def __init__(self, in_dim, out_dim, rank=8, n_tasks=2):
        super().__init__()
        self.n_tasks = n_tasks
        self.loras = nn.ModuleList([
            LoRALayer(in_dim, out_dim, rank) for _ in range(n_tasks)
        ])

    def forward(self, x, task_id):
        """Apply task-specific LoRA."""
        outputs = []
        for i in range(x.shape[0]):
            tid = task_id[i].item()
            outputs.append(self.loras[tid](x[i:i+1]))
        return torch.cat(outputs, dim=0)


# ==============================================================
# Small Transformer with LoRA
# ==============================================================
class SmallTransformerWithLoRA(nn.Module):
    """Small Transformer (~10M params) with LoRA injection."""
    def __init__(self, vocab_size=1000, d_model=256, nhead=4, n_layers=4,
                 n_tasks=2, lora_mode='none', lora_rank=8):
        super().__init__()
        self.d_model = d_model
        self.lora_mode = lora_mode

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(128, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output = nn.Linear(d_model, vocab_size)

        # LoRA injection
        if lora_mode == 'shared':
            self.lora = LoRALayer(d_model, d_model, lora_rank)
        elif lora_mode == 'per_task':
            self.loras = nn.ModuleList([
                LoRALayer(d_model, d_model, lora_rank) for _ in range(n_tasks)
            ])
        elif lora_mode == 'id_gated':
            self.id_gate = IDGatedLoRA(d_model, d_model, lora_rank, n_tasks)

    def forward(self, x, task_id=None):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device)
        h = self.embed(x) + self.pos_embed(pos).unsqueeze(0)

        h = self.transformer(h)

        # Apply LoRA
        if self.lora_mode == 'shared':
            h = h + self.lora(h)
        elif self.lora_mode == 'per_task' and task_id is not None:
            # Use first task_id in batch (assuming homogeneous batch)
            tid = task_id[0].item()
            h = h + self.loras[tid](h)
        elif self.lora_mode == 'id_gated' and task_id is not None:
            # Per-sample gating
            last_h = h[:, -1, :]
            gated = self.id_gate(last_h, task_id)
            h[:, -1, :] = h[:, -1, :] + gated

        return self.output(h[:, -1, :])


# ==============================================================
# Synthetic Multi-Task Data
# ==============================================================
def generate_multi_task_data(n_samples=500, seq_len=20, vocab_size=1000, seed=42):
    """Generate two distinct sequence prediction tasks.
    Task 0: Arithmetic progression (predict next number)
    Task 1: Reverse the pattern
    """
    rng = np.random.RandomState(seed)
    X_all, y_all, task_ids = [], [], []

    for _ in range(n_samples):
        for task_id in range(2):
            if task_id == 0:
                # Arithmetic: a, a+d, a+2d, ... -> predict a+nd
                a = rng.randint(1, 100)
                d = rng.randint(1, 10)
                seq = [(a + i * d) % vocab_size for i in range(seq_len)]
                target = (a + seq_len * d) % vocab_size
            else:
                # Modular: (a * i + b) % vocab_size
                a = rng.randint(2, 20)
                b = rng.randint(0, 50)
                seq = [(a * i + b) % vocab_size for i in range(seq_len)]
                target = (a * seq_len + b) % vocab_size

            X_all.append(seq)
            y_all.append(target)
            task_ids.append(task_id)

    return (
        torch.tensor(X_all, dtype=torch.long),
        torch.tensor(y_all, dtype=torch.long),
        torch.tensor(task_ids, dtype=torch.long),
    )


def train_and_eval_lora(lora_mode, X, y, task_ids, vocab_size=1000,
                         epochs=80, lr=0.001):
    """Train and evaluate a specific LoRA configuration."""
    n = len(y)
    split = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx, test_idx = perm[:split], perm[split:]

    model = SmallTransformerWithLoRA(
        vocab_size=vocab_size, d_model=128, nhead=4, n_layers=2,
        n_tasks=2, lora_mode=lora_mode, lora_rank=8)
    n_params = sum(p.numel() for p in model.parameters())

    opt = optim.Adam(model.parameters(), lr=lr)

    best_results = {'overall': 0, 'task_0': 0, 'task_1': 0}

    for epoch in range(epochs):
        model.train()
        batch = train_idx[torch.randperm(len(train_idx))[:min(128, len(train_idx))]]

        logits = model(X[batch], task_id=task_ids[batch])
        loss = F.cross_entropy(logits, y[batch])
        opt.zero_grad(); loss.backward(); opt.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(X[test_idx], task_id=task_ids[test_idx])
                preds = test_logits.argmax(1)

                overall = (preds == y[test_idx]).float().mean().item()
                t0_mask = task_ids[test_idx] == 0
                t1_mask = task_ids[test_idx] == 1

                t0_acc = (preds[t0_mask] == y[test_idx][t0_mask]).float().mean().item() if t0_mask.any() else 0
                t1_acc = (preds[t1_mask] == y[test_idx][t1_mask]).float().mean().item() if t1_mask.any() else 0

                if overall > best_results['overall']:
                    best_results = {'overall': overall, 'task_0': t0_acc, 'task_1': t1_acc}

                # Cross-task interference test
                if lora_mode in ['per_task', 'id_gated']:
                    wrong_ids = 1 - task_ids[test_idx]
                    wrong_logits = model(X[test_idx], task_id=wrong_ids)
                    wrong_preds = wrong_logits.argmax(1)
                    wrong_acc = (wrong_preds == y[test_idx]).float().mean().item()
                    best_results['wrong_id_acc'] = wrong_acc
                    best_results['separation'] = overall - wrong_acc

    best_results['n_params'] = n_params
    return best_results


def main():
    print("=" * 60)
    print("Phase 42: Transformer ID Gating (LoRA-based Task Routing)")
    print("=" * 60)

    torch.manual_seed(42)

    print("\nGenerating multi-task data...")
    X, y, task_ids = generate_multi_task_data(n_samples=500, seq_len=20, seed=42)
    print(f"  Data: {len(y)} samples, {(task_ids == 0).sum()} task_0, {(task_ids == 1).sum()} task_1")

    modes = ['none', 'shared', 'per_task', 'id_gated']
    mode_names = {
        'none': 'No LoRA (baseline)',
        'shared': 'Shared LoRA',
        'per_task': 'Per-Task LoRA',
        'id_gated': 'ID-Gated LoRA (ours)',
    }

    all_results = {}
    print(f"\n{'Mode':>25s} | {'Params':>8s} {'Overall':>8s} {'T0':>6s} {'T1':>6s} {'Sep':>6s}")
    print("-" * 70)

    for mode in modes:
        r = train_and_eval_lora(mode, X, y, task_ids, epochs=80)
        sep = r.get('separation', 0)
        print(f"{mode_names[mode]:>25s} | {r['n_params']:>8,} {r['overall']*100:>7.1f}% "
              f"{r['task_0']*100:>5.1f}% {r['task_1']*100:>5.1f}% {sep*100:>+5.1f}%")
        all_results[mode] = {**r, 'mode_name': mode_names[mode]}

    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    gated = all_results.get('id_gated', {})
    shared = all_results.get('shared', {})
    if gated and shared:
        print(f"  ID-Gated vs Shared: {(gated['overall'] - shared['overall'])*100:+.1f}pp overall")
        print(f"  ID-Gated separation: {gated.get('separation', 0)*100:+.1f}pp")
        if gated.get('separation', 0) > 0.05:
            print("  → DISCRETE GATING WORKS ON TRANSFORMERS! ✓")
        else:
            print("  → Gating effect is weak on Transformers (null result)")

    save_path = os.path.join(RESULTS_DIR, "phase42_transformer_gating.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 42: Transformer ID Gating',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
