"""
Phase 35b: SNN Chaotic Noise for NBS
Replace white Gaussian noise in NBS with SNN reservoir chaotic noise.
Tests if biologically-realistic noise structures improve Noisy Beam Search.

Author: Hiroto Funasaki
Theory: SNN-Comprypto reservoir dynamics as NBS noise generator
"""
import os
import math
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# SNN-Comprypto Reservoir (LIF Neurons)
# ==============================================================
class SNNReservoir:
    """Leaky Integrate-and-Fire reservoir for chaotic noise generation.

    Based on SNN-Comprypto v5 architecture.
    Produces structured chaotic noise with:
    - Near-perfect entropy (7.998/8.0)
    - Minimal autocorrelation (0.008)
    - Positive Lyapunov exponents
    """
    def __init__(self, n_neurons=100, temperature=1.0,
                 spectral_radius=1.4, density=0.1, seed=42):
        self.n = n_neurons
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)

        # LIF parameters
        self.tau = 20.0        # membrane time constant (ms)
        self.v_rest = -65.0    # resting potential (mV)
        self.v_threshold = -55.0  # firing threshold (mV)
        self.v_reset = -70.0   # reset after spike (mV)
        self.dt = 1.0          # time step (ms)

        # Recurrent connectivity (sparse)
        W = self.rng.randn(n_neurons, n_neurons) * (
            self.rng.random((n_neurons, n_neurons)) < density
        ).astype(np.float64)

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigenvalues)) + 1e-10
        self.W = W * (spectral_radius / max_eig)

        # State
        self.membrane = np.full(n_neurons, self.v_rest)
        self.spikes = np.zeros(n_neurons)

    def step(self, external_input=None):
        """Advance one time step. Returns membrane potentials."""
        # Synaptic input
        I_syn = self.W @ self.spikes * 10.0

        if external_input is not None:
            I_syn += external_input

        # Thermal noise (temperature-dependent)
        eta = self.rng.randn(self.n) * 0.5 * self.temperature

        # LIF dynamics
        dv = (-(self.membrane - self.v_rest) + I_syn + eta) * (self.dt / self.tau)
        self.membrane += dv

        # Spike detection
        self.spikes = (self.membrane >= self.v_threshold).astype(np.float64)
        self.membrane[self.spikes > 0] = self.v_reset

        return self.membrane.copy()

    def generate_noise(self, shape, n_steps=10):
        """Generate structured chaotic noise for a given tensor shape.

        Runs the reservoir for n_steps, collects membrane potentials,
        and reshapes to match the target tensor shape.
        """
        total_elements = 1
        for s in shape:
            total_elements *= s

        # Collect membrane potential trajectories
        collected = []
        for _ in range(n_steps):
            v = self.step()
            collected.append(v)

        # Flatten and normalize
        all_v = np.concatenate(collected)

        # Repeat if needed
        while len(all_v) < total_elements:
            v = self.step()
            all_v = np.concatenate([all_v, v])

        # Take what we need and normalize to zero-mean unit-variance
        noise = all_v[:total_elements]
        noise = (noise - noise.mean()) / (noise.std() + 1e-10)

        return torch.tensor(noise.reshape(shape), dtype=torch.float32)


# ==============================================================
# Noise Generators for Comparison
# ==============================================================
def white_noise_generator(sigma):
    """Standard Gaussian white noise (current SNN-Synthesis)."""
    def generate(shape):
        return torch.randn(shape) * sigma
    return generate


def snn_noise_generator(reservoir, sigma):
    """SNN reservoir chaotic noise."""
    def generate(shape):
        noise = reservoir.generate_noise(shape)
        return noise * sigma
    return generate


def pink_noise_generator(sigma, seed=42):
    """1/f (pink) noise - intermediate between white and brown."""
    rng = np.random.RandomState(seed)

    def generate(shape):
        total = 1
        for s in shape:
            total *= s
        white = rng.randn(total)

        # Simple 1/f filter via FFT
        freqs = np.fft.rfftfreq(total)
        freqs[0] = 1  # avoid division by zero
        fft = np.fft.rfft(white)
        fft = fft / np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=total)
        pink = (pink - pink.mean()) / (pink.std() + 1e-10)

        return torch.tensor(pink.reshape(shape), dtype=torch.float32) * sigma
    return generate


# ==============================================================
# Simple Task: Tower of Hanoi State Evaluation
# ==============================================================
class SimpleReasoningTask:
    """A simple reasoning task to compare noise qualities.

    Task: Given a sequence, predict the next element.
    Tests if structured noise helps exploration.
    """
    def __init__(self, n_samples=1000, seq_len=8, n_classes=10, seed=42):
        rng = np.random.RandomState(seed)
        # Generate patterns: each sample is a sequence following a rule
        self.X = []
        self.y = []
        for _ in range(n_samples):
            # Random rule: arithmetic sequence with noise
            start = rng.randint(0, n_classes)
            step = rng.randint(1, 3)
            seq = [(start + i * step) % n_classes for i in range(seq_len)]
            target = (start + seq_len * step) % n_classes
            self.X.append(seq)
            self.y.append(target)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)


class SmallPredictor(nn.Module):
    def __init__(self, seq_len=8, hidden=32, n_classes=10):
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
# NBS with Different Noise Types
# ==============================================================
def nbs_evaluate(model, X, y, noise_gen, K=11, n_eval=200):
    """Run Noisy Beam Search with a specific noise generator.

    Returns: accuracy (best of K noisy trajectories)
    """
    model.eval()
    correct = 0
    total = min(n_eval, len(X))

    with torch.no_grad():
        for i in range(total):
            x_i = X[i:i+1]
            target = y[i].item()

            # K noisy forward passes
            preds = []
            for _ in range(K):
                noise = noise_gen((1, 32))  # hidden dim = 32
                logits = model(x_i, noise=noise)
                preds.append(logits.argmax(1).item())

            # Majority vote
            from collections import Counter
            best = Counter(preds).most_common(1)[0][0]
            if best == target:
                correct += 1

    return correct / total


def main():
    print("=" * 60)
    print("Phase 35b: SNN Chaotic Noise for NBS")
    print("=" * 60)

    torch.manual_seed(42)

    # Create task
    task = SimpleReasoningTask(n_samples=1000, seed=42)

    # Train a base model (no noise during training)
    model = SmallPredictor(seq_len=8, hidden=32, n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("\nTraining base model...")
    for epoch in range(50):
        model.train()
        logits = model(task.X)
        loss = F.cross_entropy(logits, task.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            acc = (logits.argmax(1) == task.y).float().mean().item()
            print(f"  Epoch {epoch}: loss={loss.item():.4f}, acc={acc:.3f}")

    # Evaluate with different noise types
    print("\n--- NBS with different noise types ---")
    sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]
    results = {}

    for sigma in sigmas:
        print(f"\nsigma = {sigma}")

        # White noise (baseline)
        white_gen = white_noise_generator(sigma)
        acc_white = nbs_evaluate(model, task.X, task.y, white_gen, K=11)
        print(f"  White noise:  {acc_white:.3f}")

        # SNN chaotic noise
        reservoir = SNNReservoir(n_neurons=100, temperature=1.0, seed=42)
        snn_gen = snn_noise_generator(reservoir, sigma)
        acc_snn = nbs_evaluate(model, task.X, task.y, snn_gen, K=11)
        print(f"  SNN chaotic:  {acc_snn:.3f}")

        # Pink (1/f) noise
        pink_gen = pink_noise_generator(sigma)
        acc_pink = nbs_evaluate(model, task.X, task.y, pink_gen, K=11)
        print(f"  Pink (1/f):   {acc_pink:.3f}")

        results[f"sigma_{sigma}"] = {
            'white': acc_white,
            'snn_chaotic': acc_snn,
            'pink_1f': acc_pink,
        }

    # No noise baseline
    model.eval()
    with torch.no_grad():
        acc_no_noise = (model(task.X).argmax(1) == task.y).float().mean().item()
    print(f"\nNo noise baseline: {acc_no_noise:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sigma_key, accs in results.items():
        print(f"\n{sigma_key}:")
        for noise_type, acc in accs.items():
            improvement = acc - acc_no_noise
            print(f"  {noise_type:15s}: {acc:.3f} ({improvement:+.3f} vs no noise)")

    # Save
    save_path = os.path.join(RESULTS_DIR, "phase35b_snn_chaotic_nbs.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 35b: SNN Chaotic Noise for NBS',
            'timestamp': datetime.now().isoformat(),
            'no_noise_baseline': acc_no_noise,
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
