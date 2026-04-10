"""
Phase 36a: Continuous Wave Gating
Instead of discrete condition IDs (embedding), use actual physical
sine waves as gating signals. The model learns to "tune in" to
different frequencies to access different knowledge.

Author: Hiroto Funasaki
Theory: SNN-Comprypto temporal coding → continuous frequency routing
"""
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Synthetic Task (same as 35c)
# ==============================================================
def generate_teacher_trajectories(n_tasks=3, n_trajectories=200,
                                   seq_len=20, seed=42):
    rng = np.random.RandomState(seed)
    trajectories = []

    for task_id in range(n_tasks):
        for _ in range(n_trajectories):
            if task_id == 0:
                a, b = rng.randint(1, 5), rng.randint(1, 5)
                seq = []
                for _ in range(seq_len):
                    seq.append(a % 10)
                    a, b = b, (a + b) % 100
            elif task_id == 1:
                base = rng.randint(0, 5, size=4)
                seq = []
                for i in range(seq_len):
                    val = base[i % 4] + rng.randint(-1, 2)
                    seq.append(max(0, min(9, val)))
            else:
                freq = rng.uniform(0.1, 0.5)
                phase = rng.uniform(0, 2 * math.pi)
                seq = [int(5 + 4 * math.sin(freq * i + phase))
                       for i in range(seq_len)]

            trajectories.append({
                'task_id': task_id,
                'frequency': [0.1, 0.5, 0.9][task_id],
                'trajectory': seq,
            })

    return trajectories


# ==============================================================
# Wave-Gated Model (continuous frequency input)
# ==============================================================
class WaveGatedStudent(nn.Module):
    """Student that uses a continuous wave signal for gating.

    Instead of Embedding(condition_id), receives a 1D wave value
    sin(2*pi*freq*t) and learns to route knowledge based on it.
    """
    def __init__(self, seq_len=20, hidden=64, n_classes=10):
        super().__init__()
        # Wave-to-gate: maps 1D wave signal → hidden-dim gate mask
        self.wave_to_gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Sigmoid()
        )

        # Sequence encoder
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x, wave_value=None):
        h = F.relu(self.fc1(x))

        # Apply wave-based gating (multiplicative modulation)
        if wave_value is not None:
            gate = self.wave_to_gate(wave_value)
            h = h * gate  # neural modulation

        h = F.relu(self.fc2(h))
        return self.fc3(h)


class EmbeddingGatedStudent(nn.Module):
    """Control: same architecture but with discrete embedding (Phase 35c)."""
    def __init__(self, seq_len=20, hidden=64, n_classes=10, n_conditions=3):
        super().__init__()
        self.condition_embed = nn.Embedding(n_conditions, hidden)
        self.fc1 = nn.Linear(seq_len, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x, condition_id=None):
        h = F.relu(self.fc1(x))
        if condition_id is not None:
            cond = self.condition_embed(condition_id)
            h = h * torch.sigmoid(cond)
        h = F.relu(self.fc2(h))
        return self.fc3(h)


# ==============================================================
# Training
# ==============================================================
def prepare_data(trajectories, seq_len=20):
    X, y, freqs, task_ids = [], [], [], []
    for traj in trajectories:
        seq = traj['trajectory']
        x = torch.tensor(seq[:-1] + [0], dtype=torch.float32)
        target = seq[-1]
        X.append(x)
        y.append(target)
        freqs.append(traj['frequency'])
        task_ids.append(traj['task_id'])

    return (
        torch.stack(X),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(freqs, dtype=torch.float32),
        torch.tensor(task_ids, dtype=torch.long),
    )


def generate_wave_values(freqs, t_step):
    """Generate sine wave values for each sample at time step t."""
    wave = torch.sin(2 * math.pi * freqs * t_step)
    return wave.unsqueeze(1)  # shape: (N, 1)


def main():
    print("=" * 60)
    print("Phase 36a: Continuous Wave Gating")
    print("=" * 60)

    torch.manual_seed(42)

    trajectories = generate_teacher_trajectories(n_tasks=3, n_trajectories=200)
    X, y, freqs, task_ids = prepare_data(trajectories)
    print(f"Data: {X.shape}, Frequencies: {freqs.unique()}")

    # ==============================
    # Exp 1: Wave-gated model
    # ==============================
    print("\n--- Exp 1: Continuous Wave Gating ---")
    model_wave = WaveGatedStudent(seq_len=20, hidden=64, n_classes=10)
    optimizer = optim.Adam(model_wave.parameters(), lr=0.01)

    for epoch in range(150):
        model_wave.train()
        # Use different time steps per epoch for temporal diversity
        t = float(epoch)
        wave_vals = generate_wave_values(freqs, t)
        logits = model_wave(X, wave_value=wave_vals)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            model_wave.eval()
            with torch.no_grad():
                # Correct frequency
                wave_correct = generate_wave_values(freqs, t)
                pred_correct = model_wave(X, wave_value=wave_correct).argmax(1)

                # Wrong frequency (shift all by one task)
                wrong_freqs = torch.tensor(
                    [([0.5, 0.9, 0.1][tid]) for tid in task_ids],
                    dtype=torch.float32
                )
                wave_wrong = generate_wave_values(wrong_freqs, t)
                pred_wrong = model_wave(X, wave_value=wave_wrong).argmax(1)

                # Per-task accuracies
                for tid in range(3):
                    mask = task_ids == tid
                    acc_c = (pred_correct[mask] == y[mask]).float().mean().item()
                    acc_w = (pred_wrong[mask] == y[mask]).float().mean().item()
                    if epoch % 30 == 0 and tid == 0:
                        print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}  "
                              f"T0(c={acc_c:.3f}/w={acc_w:.3f})", end="")
                    elif epoch % 30 == 0 and tid == 1:
                        print(f" T1(c={acc_c:.3f}/w={acc_w:.3f})", end="")
                    elif epoch % 30 == 0:
                        print(f" T2(c={acc_c:.3f}/w={acc_w:.3f})")

    # Final evaluation with fixed time step
    model_wave.eval()
    t_eval = 50.0
    results_wave = {}
    with torch.no_grad():
        for tid in range(3):
            mask = task_ids == tid

            # Correct freq
            wave_c = generate_wave_values(freqs[mask], t_eval)
            pred_c = model_wave(X[mask], wave_value=wave_c).argmax(1)
            acc_c = (pred_c == y[mask]).float().mean().item()

            # Wrong freq (next task's freq)
            wrong_f = torch.full((mask.sum(),), [0.5, 0.9, 0.1][tid])
            wave_w = generate_wave_values(wrong_f, t_eval)
            pred_w = model_wave(X[mask], wave_value=wave_w).argmax(1)
            acc_w = (pred_w == y[mask]).float().mean().item()

            # Novel freq (never seen: 0.3)
            novel_f = torch.full((mask.sum(),), 0.3)
            wave_n = generate_wave_values(novel_f, t_eval)
            pred_n = model_wave(X[mask], wave_value=wave_n).argmax(1)
            acc_n = (pred_n == y[mask]).float().mean().item()

            results_wave[f'task_{tid}'] = {
                'correct_freq': acc_c,
                'wrong_freq': acc_w,
                'novel_freq': acc_n,
            }

    # ==============================
    # Exp 2: Embedding baseline (Phase 35c reproduction)
    # ==============================
    print("\n--- Exp 2: Discrete Embedding Gating (baseline) ---")
    model_emb = EmbeddingGatedStudent(seq_len=20, hidden=64, n_classes=10)
    optimizer2 = optim.Adam(model_emb.parameters(), lr=0.01)

    for epoch in range(150):
        model_emb.train()
        logits = model_emb(X, condition_id=task_ids)
        loss = F.cross_entropy(logits, y)
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        if epoch % 30 == 0:
            model_emb.eval()
            with torch.no_grad():
                pred = model_emb(X, condition_id=task_ids).argmax(1)
                acc = (pred == y).float().mean().item()
                print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, acc={acc:.3f}")

    results_emb = {}
    model_emb.eval()
    with torch.no_grad():
        for tid in range(3):
            mask = task_ids == tid
            cond_c = task_ids[mask]
            pred_c = model_emb(X[mask], condition_id=cond_c).argmax(1)
            acc_c = (pred_c == y[mask]).float().mean().item()

            cond_w = torch.full_like(cond_c, (tid + 1) % 3)
            pred_w = model_emb(X[mask], condition_id=cond_w).argmax(1)
            acc_w = (pred_w == y[mask]).float().mean().item()

            results_emb[f'task_{tid}'] = {
                'correct_id': acc_c,
                'wrong_id': acc_w,
            }

    # ==============================
    # Summary
    # ==============================
    print("\n" + "=" * 60)
    print("SUMMARY: Continuous Wave vs Discrete Embedding")
    print("=" * 60)

    print("\nWave-Gated Model (continuous frequency):")
    wave_sep = 0
    for tid in range(3):
        r = results_wave[f'task_{tid}']
        print(f"  Task {tid}: correct={r['correct_freq']:.3f}  "
              f"wrong={r['wrong_freq']:.3f}  "
              f"novel={r['novel_freq']:.3f}")
        wave_sep += r['correct_freq'] - r['wrong_freq']
    wave_sep /= 3
    print(f"  → Knowledge Separation: {wave_sep:.3f}")

    print("\nEmbedding-Gated Model (discrete ID):")
    emb_sep = 0
    for tid in range(3):
        r = results_emb[f'task_{tid}']
        print(f"  Task {tid}: correct={r['correct_id']:.3f}  "
              f"wrong={r['wrong_id']:.3f}")
        emb_sep += r['correct_id'] - r['wrong_id']
    emb_sep /= 3
    print(f"  → Knowledge Separation: {emb_sep:.3f}")

    print(f"\n{'='*60}")
    print(f"Wave Gating Separation:     {wave_sep:.3f}")
    print(f"Embedding Gating Separation: {emb_sep:.3f}")
    winner = "Wave" if wave_sep >= emb_sep else "Embedding"
    print(f"Winner: {winner}")

    # Save
    save_path = os.path.join(RESULTS_DIR, "phase36a_continuous_wave_gating.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 36a: Continuous Wave Gating',
            'timestamp': datetime.now().isoformat(),
            'wave_gated': results_wave,
            'embedding_gated': results_emb,
            'wave_separation': wave_sep,
            'embedding_separation': emb_sep,
        }, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
