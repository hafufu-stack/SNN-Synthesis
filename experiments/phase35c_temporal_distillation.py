"""
Phase 35c: Temporal Knowledge Distillation
Distill knowledge from a "teacher's successful trajectories"
into a student model using noise-conditioned training.

Each trajectory is tagged with its noise condition (sigma, temperature),
and the student learns to reproduce trajectories when given the
matching noise condition — enabling "multi-personality" from one model.

Author: Hiroto Funasaki
Theory: SNN-Comprypto temporal coding + SNN-Synthesis ExIt
"""
import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Synthetic "Teacher" Trajectory Generation
# ==============================================================
def generate_teacher_trajectories(n_tasks=3, n_trajectories=100,
                                   seq_len=20, seed=42):
    """Simulate successful trajectories from a "teacher model".

    Each task represents a different domain of knowledge:
    Task 0: Math-like (arithmetic patterns)
    Task 1: Language-like (sequential patterns)
    Task 2: Spatial-like (geometric patterns)

    Returns: list of (task_id, sigma_used, trajectory)
    """
    rng = np.random.RandomState(seed)
    trajectories = []

    for task_id in range(n_tasks):
        sigma_optimal = [0.05, 0.15, 0.30][task_id]  # task-specific σ*

        for _ in range(n_trajectories):
            if task_id == 0:
                # Math: fibonacci-like sequences
                a, b = rng.randint(1, 5), rng.randint(1, 5)
                seq = []
                for _ in range(seq_len):
                    seq.append(a % 10)
                    a, b = b, (a + b) % 100
            elif task_id == 1:
                # Language: repeating pattern with variations
                base = rng.randint(0, 5, size=4)
                seq = []
                for i in range(seq_len):
                    val = base[i % 4] + rng.randint(-1, 2)
                    seq.append(max(0, min(9, val)))
            else:
                # Spatial: sine-wave based patterns
                freq = rng.uniform(0.1, 0.5)
                phase = rng.uniform(0, 2 * math.pi)
                seq = [int(5 + 4 * math.sin(freq * i + phase)) for i in range(seq_len)]

            trajectories.append({
                'task_id': task_id,
                'sigma_optimal': sigma_optimal,
                'temperature': [0.001, 0.01, 0.15][task_id],
                'trajectory': seq,
            })

    return trajectories


# ==============================================================
# Student Model with Noise-Conditioned Training
# ==============================================================
class NoiseConditionedStudent(nn.Module):
    """Student model that accepts a "noise condition" embedding.

    The noise condition (sigma, temperature) tells the model
    which "personality" / knowledge mode to activate.
    """
    def __init__(self, seq_len=20, hidden=64, n_classes=10, n_conditions=3):
        super().__init__()
        # Condition embedding: maps noise condition to hidden state modifier
        self.condition_embed = nn.Embedding(n_conditions, hidden)

        # Sequence encoder
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x, condition_id=None, noise=None):
        h = F.relu(self.fc1(x))

        # Apply condition-specific modulation
        if condition_id is not None:
            cond = self.condition_embed(condition_id)
            h = h * torch.sigmoid(cond)  # gating mechanism

        # Apply noise
        if noise is not None:
            h = h + noise

        h = F.relu(self.fc2(h))
        return self.fc3(h)


class UnconditionedStudent(nn.Module):
    """Control: student without noise conditioning (standard distillation)."""
    def __init__(self, seq_len=20, hidden=64, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x, noise=None):
        h = F.relu(self.fc1(x))
        if noise is not None:
            h = h + noise
        h = F.relu(self.fc2(h))
        return self.fc3(h)


# ==============================================================
# Training
# ==============================================================
def prepare_training_data(trajectories, seq_len=20):
    """Convert trajectories to (input, target, condition) tensors."""
    X = []
    y = []
    conditions = []

    for traj in trajectories:
        seq = traj['trajectory']
        task_id = traj['task_id']
        # Input: full sequence, Target: predict next value
        x = torch.tensor(seq[:-1] + [0], dtype=torch.float32)  # pad
        target = seq[-1]  # predict last element
        X.append(x)
        y.append(target)
        conditions.append(task_id)

    return (
        torch.stack(X),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(conditions, dtype=torch.long)
    )


def train_conditioned(model, X, y, conditions, epochs=100, lr=0.01):
    """Train noise-conditioned student."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        model.train()
        logits = model(X, condition_id=conditions)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Per-task accuracy
        model.eval()
        with torch.no_grad():
            pred = model(X, condition_id=conditions).argmax(1)
            accs = {}
            for task_id in range(3):
                mask = conditions == task_id
                if mask.sum() > 0:
                    accs[f'task_{task_id}'] = (pred[mask] == y[mask]).float().mean().item()

            # Cross-condition test: use wrong condition
            cross_accs = {}
            for task_id in range(3):
                mask = conditions == task_id
                if mask.sum() > 0:
                    wrong_cond = torch.full_like(conditions[mask], (task_id + 1) % 3)
                    pred_wrong = model(X[mask], condition_id=wrong_cond).argmax(1)
                    cross_accs[f'task_{task_id}_wrong'] = (
                        (pred_wrong == y[mask]).float().mean().item()
                    )

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}  "
                  f"| T0={accs.get('task_0', 0):.3f} "
                  f"T1={accs.get('task_1', 0):.3f} "
                  f"T2={accs.get('task_2', 0):.3f} "
                  f"| wrong: T0={cross_accs.get('task_0_wrong', 0):.3f}")

        history.append({
            'epoch': epoch,
            'loss': loss.item(),
            **accs,
            **cross_accs,
        })

    return history


def train_unconditioned(model, X, y, epochs=100, lr=0.01):
    """Train standard student (no conditioning)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []
    conditions_tensor = None  # not used for evaluation indices

    for epoch in range(epochs):
        model.train()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X).argmax(1)
            acc = (pred == y).float().mean().item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, acc={acc:.3f}")

        history.append({'epoch': epoch, 'loss': loss.item(), 'overall_acc': acc})

    return history


# ==============================================================
# Main
# ==============================================================
def main():
    print("=" * 60)
    print("Phase 35c: Temporal Knowledge Distillation")
    print("=" * 60)

    torch.manual_seed(42)

    # Generate teacher trajectories
    print("\nGenerating teacher trajectories...")
    trajectories = generate_teacher_trajectories(
        n_tasks=3, n_trajectories=200, seed=42
    )
    print(f"Total trajectories: {len(trajectories)}")

    # Prepare data
    X, y, conditions = prepare_training_data(trajectories)
    print(f"Training data: X={X.shape}, y={y.shape}, conditions={conditions.shape}")

    # ---- Experiment 1: Unconditioned baseline ----
    print("\n--- Exp 1: Unconditioned student (standard distillation) ---")
    model_uncond = UnconditionedStudent(seq_len=20, hidden=64, n_classes=10)
    hist_uncond = train_unconditioned(model_uncond, X, y, epochs=100)

    # ---- Experiment 2: Noise-conditioned student ----
    print("\n--- Exp 2: Noise-conditioned student (temporal routing) ---")
    model_cond = NoiseConditionedStudent(seq_len=20, hidden=64, n_classes=10)
    hist_cond = train_conditioned(model_cond, X, y, conditions, epochs=100)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nUnconditioned: final acc = {hist_uncond[-1]['overall_acc']:.3f}")
    print(f"\nConditioned (correct mode):")
    for task_id in range(3):
        key = f'task_{task_id}'
        print(f"  Task {task_id}: {hist_cond[-1].get(key, 0):.3f}")
    print(f"\nConditioned (wrong mode):")
    for task_id in range(3):
        key = f'task_{task_id}_wrong'
        print(f"  Task {task_id}: {hist_cond[-1].get(key, 0):.3f}")

    # Knowledge separation score
    correct_avg = np.mean([hist_cond[-1].get(f'task_{i}', 0) for i in range(3)])
    wrong_avg = np.mean([hist_cond[-1].get(f'task_{i}_wrong', 0) for i in range(3)])
    separation = correct_avg - wrong_avg
    print(f"\nKnowledge Separation Score: {separation:.3f}")
    print(f"  (>0 means conditioning helps separate knowledge)")

    # Save
    save_path = os.path.join(RESULTS_DIR, "phase35c_temporal_distillation.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 35c: Temporal Knowledge Distillation',
            'timestamp': datetime.now().isoformat(),
            'unconditioned_final': hist_uncond[-1],
            'conditioned_final': hist_cond[-1],
            'knowledge_separation': separation,
        }, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
