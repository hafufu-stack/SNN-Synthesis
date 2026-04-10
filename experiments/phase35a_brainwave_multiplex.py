"""
Phase 35a: Brainwave Multiplexing (脳波多重化)
Two different "brainwave frequencies" encode two different tasks
in the same small model, testing if temporal noise patterns
can prevent catastrophic forgetting.

Author: Hiroto Funasaki
Theory: SNN-Synthesis + SNN-Comprypto fusion
"""
import os
import math
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# MicroBrain: Small CNN for ARC-like grid tasks
# ==============================================================
class MicroBrainSmall(nn.Module):
    """Tiny CNN (~6K params) for knowledge capacity testing."""
    def __init__(self, in_dim=64, hidden=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x, noise_fn=None, step=0):
        h = F.relu(self.fc1(x))
        # Inject brainwave noise at hidden layer
        if noise_fn is not None:
            h = h + noise_fn(h, step)
        h = F.relu(self.fc2(h))
        return self.fc3(h)


# ==============================================================
# Brainwave Noise Generators
# ==============================================================
def static_noise(sigma):
    """Standard SNN-Synthesis: static Gaussian noise."""
    def _noise(h, step):
        return torch.randn_like(h) * sigma
    return _noise


def brainwave_noise(sigma, freq):
    """Brainwave-modulated noise: sine wave * Gaussian.

    Different frequencies encode different "brain modes".
    Low freq (0.1) = slow wave (memory consolidation mode)
    High freq (0.8) = fast wave (active reasoning mode)
    """
    def _noise(h, step):
        wave = math.sin(2 * math.pi * freq * step)
        return torch.randn_like(h) * sigma * wave
    return _noise


def burst_noise(sigma, freq, duty=0.3):
    """Burst noise: periodic bursts of noise then silence.

    Mimics hippocampal sharp-wave ripples.
    """
    def _noise(h, step):
        phase = (step * freq) % 1.0
        if phase < duty:
            return torch.randn_like(h) * sigma * 2.0
        else:
            return torch.zeros_like(h)
    return _noise


# ==============================================================
# Synthetic Task Generation
# ==============================================================
def generate_task_data(task_type, n_samples=500, in_dim=64, seed=42):
    """Generate synthetic classification data for two different tasks.

    Task A (pattern detection): Classify based on sum of features
    Task B (spatial detection): Classify based on specific feature positions
    """
    rng = np.random.RandomState(seed)

    if task_type == "A":
        # Task A: Sum-based classification (8 classes based on total sum)
        X = rng.randn(n_samples, in_dim).astype(np.float32)
        y = np.clip((X.sum(axis=1) + 8) / 2, 0, 7).astype(np.int64)
    elif task_type == "B":
        # Task B: Position-based classification
        X = rng.randn(n_samples, in_dim).astype(np.float32)
        # Classify based on first 8 features
        y = np.argmax(X[:, :8], axis=1).astype(np.int64)
    else:
        raise ValueError(f"Unknown task: {task_type}")

    return torch.tensor(X), torch.tensor(y)


# ==============================================================
# Training Loop
# ==============================================================
def train_with_brainwave(model, X_a, y_a, X_b, y_b,
                         noise_fn_a, noise_fn_b,
                         epochs=50, lr=0.01):
    """Train one model on TWO tasks with different brainwave frequencies.

    Each epoch alternates between Task A (with freq_a noise)
    and Task B (with freq_b noise).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        step = epoch  # global step for brainwave phase

        # Train on Task A with brainwave A
        logits_a = model(X_a, noise_fn=noise_fn_a, step=step)
        loss_a = F.cross_entropy(logits_a, y_a)

        # Train on Task B with brainwave B
        logits_b = model(X_b, noise_fn=noise_fn_b, step=step + 0.5)
        loss_b = F.cross_entropy(logits_b, y_b)

        # Combined loss
        loss = loss_a + loss_b
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            # Evaluate Task A with brainwave A
            pred_a = model(X_a, noise_fn=noise_fn_a, step=step).argmax(1)
            acc_a = (pred_a == y_a).float().mean().item()

            # Evaluate Task B with brainwave B
            pred_b = model(X_b, noise_fn=noise_fn_b, step=step).argmax(1)
            acc_b = (pred_b == y_b).float().mean().item()

            # Cross-evaluation: wrong brainwave
            pred_a_wrong = model(X_a, noise_fn=noise_fn_b, step=step).argmax(1)
            acc_a_wrong = (pred_a_wrong == y_a).float().mean().item()

            pred_b_wrong = model(X_b, noise_fn=noise_fn_a, step=step).argmax(1)
            acc_b_wrong = (pred_b_wrong == y_b).float().mean().item()

        history.append({
            'epoch': epoch,
            'loss': loss.item(),
            'acc_a_correct_wave': acc_a,
            'acc_b_correct_wave': acc_b,
            'acc_a_wrong_wave': acc_a_wrong,
            'acc_b_wrong_wave': acc_b_wrong,
        })

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}  "
                  f"A(correct)={acc_a:.3f}  B(correct)={acc_b:.3f}  "
                  f"A(wrong)={acc_a_wrong:.3f}  B(wrong)={acc_b_wrong:.3f}")

    return history


# ==============================================================
# Main Experiment
# ==============================================================
def main():
    print("=" * 60)
    print("Phase 35a: Brainwave Multiplexing Experiment")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device('cpu')  # tiny model, CPU is fine

    # Generate tasks
    X_a, y_a = generate_task_data("A", n_samples=500)
    X_b, y_b = generate_task_data("B", n_samples=500)

    results = {}

    # ---- Experiment 1: Baseline (no noise) ----
    print("\n--- Exp 1: No noise baseline ---")
    model1 = MicroBrainSmall(in_dim=64, hidden=32, out_dim=8)
    hist1 = train_with_brainwave(
        model1, X_a, y_a, X_b, y_b,
        noise_fn_a=None, noise_fn_b=None,
        epochs=100
    )
    results['no_noise'] = hist1

    # ---- Experiment 2: Same noise for both tasks ----
    print("\n--- Exp 2: Same static noise (sigma=0.15) ---")
    model2 = MicroBrainSmall(in_dim=64, hidden=32, out_dim=8)
    hist2 = train_with_brainwave(
        model2, X_a, y_a, X_b, y_b,
        noise_fn_a=static_noise(0.15),
        noise_fn_b=static_noise(0.15),
        epochs=100
    )
    results['same_noise'] = hist2

    # ---- Experiment 3: Different brainwave frequencies ----
    print("\n--- Exp 3: Different brainwaves (f=0.1 vs f=0.8) ---")
    model3 = MicroBrainSmall(in_dim=64, hidden=32, out_dim=8)
    hist3 = train_with_brainwave(
        model3, X_a, y_a, X_b, y_b,
        noise_fn_a=brainwave_noise(0.15, freq=0.1),
        noise_fn_b=brainwave_noise(0.15, freq=0.8),
        epochs=100
    )
    results['brainwave_multiplex'] = hist3

    # ---- Experiment 4: Burst noise ----
    print("\n--- Exp 4: Burst noise (hippocampal ripples) ---")
    model4 = MicroBrainSmall(in_dim=64, hidden=32, out_dim=8)
    hist4 = train_with_brainwave(
        model4, X_a, y_a, X_b, y_b,
        noise_fn_a=burst_noise(0.15, freq=0.1),
        noise_fn_b=burst_noise(0.15, freq=0.8),
        epochs=100
    )
    results['burst_multiplex'] = hist4

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY: Final epoch results")
    print("=" * 60)
    for name, hist in results.items():
        final = hist[-1]
        separation = (
            (final['acc_a_correct_wave'] + final['acc_b_correct_wave']) / 2
            - (final['acc_a_wrong_wave'] + final['acc_b_wrong_wave']) / 2
        )
        print(f"\n{name}:")
        print(f"  Task A (correct wave): {final['acc_a_correct_wave']:.3f}")
        print(f"  Task B (correct wave): {final['acc_b_correct_wave']:.3f}")
        print(f"  Task A (wrong wave):   {final['acc_a_wrong_wave']:.3f}")
        print(f"  Task B (wrong wave):   {final['acc_b_wrong_wave']:.3f}")
        print(f"  Knowledge Separation:  {separation:+.3f}")

    # Save results
    save_path = os.path.join(RESULTS_DIR, "phase35a_brainwave_multiplex.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 35a: Brainwave Multiplexing',
            'timestamp': datetime.now().isoformat(),
            'results': {k: v[-1] for k, v in results.items()},
        }, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
