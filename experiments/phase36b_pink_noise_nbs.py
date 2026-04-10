"""
Phase 36b: Pink Noise Beam Search - Robustness Test
Test if structured noise (pink/chaotic) maintains search quality
at extreme sigma values where white noise fails.

Uses a HARDER task than 35b to expose noise-quality differences.

Author: Hiroto Funasaki
Theory: 1/f noise (biological fluctuation) maintains coherent
exploration even at high noise levels, unlike white noise.
"""
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# SNN Reservoir (from 35b)
# ==============================================================
class SNNReservoir:
    """LIF reservoir for structured chaotic noise."""
    def __init__(self, n_neurons=100, temperature=1.0,
                 spectral_radius=1.4, density=0.1, seed=42):
        self.n = n_neurons
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)
        self.tau = 20.0
        self.v_rest = -65.0
        self.v_threshold = -55.0
        self.v_reset = -70.0
        self.dt = 1.0

        W = self.rng.randn(n_neurons, n_neurons) * (
            self.rng.random((n_neurons, n_neurons)) < density
        ).astype(np.float64)
        eigenvalues = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigenvalues)) + 1e-10
        self.W = W * (spectral_radius / max_eig)
        self.membrane = np.full(n_neurons, self.v_rest)
        self.spikes = np.zeros(n_neurons)

    def step(self, external_input=None):
        I_syn = self.W @ self.spikes * 10.0
        if external_input is not None:
            I_syn += external_input
        eta = self.rng.randn(self.n) * 0.5 * self.temperature
        dv = (-(self.membrane - self.v_rest) + I_syn + eta) * (self.dt / self.tau)
        self.membrane += dv
        self.spikes = (self.membrane >= self.v_threshold).astype(np.float64)
        self.membrane[self.spikes > 0] = self.v_reset
        return self.membrane.copy()

    def generate_noise(self, shape, n_steps=10):
        total_elements = 1
        for s in shape:
            total_elements *= s
        collected = []
        for _ in range(n_steps):
            collected.append(self.step())
        all_v = np.concatenate(collected)
        while len(all_v) < total_elements:
            all_v = np.concatenate([all_v, self.step()])
        noise = all_v[:total_elements]
        noise = (noise - noise.mean()) / (noise.std() + 1e-10)
        return torch.tensor(noise.reshape(shape), dtype=torch.float32)


# ==============================================================
# Hard Reasoning Task (multi-step logic)
# ==============================================================
class HardReasoningTask:
    """A harder task where noise quality matters more.

    Task: Multi-step arithmetic reasoning.
    Given [a, op1, b, op2, c], compute ((a op1 b) op2 c) mod 10.
    ops: 0=add, 1=subtract, 2=multiply
    """
    def __init__(self, n_samples=2000, seed=42):
        rng = np.random.RandomState(seed)
        self.X = []
        self.y = []

        for _ in range(n_samples):
            a = rng.randint(1, 10)
            b = rng.randint(1, 10)
            c = rng.randint(1, 10)
            op1 = rng.randint(0, 3)
            op2 = rng.randint(0, 3)

            # Compute result
            if op1 == 0:
                r1 = a + b
            elif op1 == 1:
                r1 = a - b
            else:
                r1 = a * b

            if op2 == 0:
                result = r1 + c
            elif op2 == 1:
                result = r1 - c
            else:
                result = r1 * c

            target = result % 10

            # Encode as features: [a, b, c, op1_onehot(3), op2_onehot(3)]
            features = [a / 10.0, b / 10.0, c / 10.0]
            for op in [op1, op2]:
                for j in range(3):
                    features.append(1.0 if op == j else 0.0)

            self.X.append(features)
            self.y.append(target)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)


class ReasoningModel(nn.Module):
    """Slightly larger model for harder task."""
    def __init__(self, in_dim=9, hidden=64, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, n_classes)

    def forward(self, x, noise=None, noise_layer=1):
        h = F.relu(self.fc1(x))
        if noise is not None and noise_layer == 1:
            h = h + noise
        h = F.relu(self.fc2(h))
        if noise is not None and noise_layer == 2:
            h = h + noise
        h = F.relu(self.fc3(h))
        return self.fc4(h)


# ==============================================================
# Noise Generators
# ==============================================================
def white_noise(shape, sigma, rng=None):
    return torch.randn(shape) * sigma


def pink_noise(shape, sigma, rng_state=None):
    """1/f pink noise via FFT filtering."""
    total = 1
    for s in shape:
        total *= s
    rng = np.random.RandomState(rng_state) if rng_state else np.random
    white = rng.randn(total)
    freqs = np.fft.rfftfreq(total)
    freqs[0] = 1
    fft = np.fft.rfft(white)
    fft = fft / np.sqrt(freqs)
    pink_arr = np.fft.irfft(fft, n=total)
    pink_arr = (pink_arr - pink_arr.mean()) / (pink_arr.std() + 1e-10)
    return torch.tensor(pink_arr.reshape(shape), dtype=torch.float32) * sigma


def snn_noise(shape, sigma, reservoir):
    """SNN chaotic reservoir noise."""
    return reservoir.generate_noise(shape) * sigma


# ==============================================================
# NBS Evaluation
# ==============================================================
def nbs_eval(model, X, y, noise_fn, K=11, n_eval=500):
    """Noisy Beam Search: K forward passes, majority vote."""
    model.eval()
    correct = 0
    total = min(n_eval, len(X))

    with torch.no_grad():
        for i in range(total):
            x_i = X[i:i+1]
            target = y[i].item()

            preds = []
            for _ in range(K):
                noise = noise_fn((1, 64))
                logits = model(x_i, noise=noise)
                preds.append(logits.argmax(1).item())

            best = Counter(preds).most_common(1)[0][0]
            if best == target:
                correct += 1

    return correct / total


def main():
    print("=" * 60)
    print("Phase 36b: Pink Noise Beam Search - High Sigma Robustness")
    print("=" * 60)

    torch.manual_seed(42)

    # Create hard task
    task = HardReasoningTask(n_samples=2000, seed=42)
    print(f"Task: {task.X.shape}, Classes: {task.y.unique().shape[0]}")

    # Train base model
    model = ReasoningModel(in_dim=9, hidden=64, n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    print("\nTraining base model (no noise)...")
    for epoch in range(100):
        model.train()
        logits = model(task.X)
        loss = F.cross_entropy(logits, task.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            acc = (logits.argmax(1) == task.y).float().mean().item()
            print(f"  Epoch {epoch}: loss={loss.item():.4f}, acc={acc:.3f}")

    # Baseline (no noise)
    model.eval()
    with torch.no_grad():
        acc_baseline = (model(task.X[:500]).argmax(1) == task.y[:500]).float().mean().item()
    print(f"\nBaseline (no noise): {acc_baseline:.3f}")

    # Test sigmas from conservative to extreme
    sigmas = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.50, 2.00]
    results = {'baseline': acc_baseline}

    reservoir = SNNReservoir(n_neurons=100, temperature=1.0, seed=42)

    for sigma in sigmas:
        print(f"\nσ = {sigma:.2f}:")

        # White noise NBS
        acc_white = nbs_eval(
            model, task.X, task.y,
            lambda shape: white_noise(shape, sigma),
            K=11, n_eval=300
        )

        # Pink noise NBS
        acc_pink = nbs_eval(
            model, task.X, task.y,
            lambda shape, s=sigma: pink_noise(shape, s, rng_state=42),
            K=11, n_eval=300
        )

        # SNN chaotic NBS
        reservoir_fresh = SNNReservoir(n_neurons=100, temperature=1.0, seed=42)
        acc_snn = nbs_eval(
            model, task.X, task.y,
            lambda shape, r=reservoir_fresh, s=sigma: snn_noise(shape, s, r),
            K=11, n_eval=300
        )

        print(f"  White:  {acc_white:.3f}  "
              f"Pink:  {acc_pink:.3f}  "
              f"SNN:   {acc_snn:.3f}")

        results[f'sigma_{sigma}'] = {
            'white': acc_white,
            'pink': acc_pink,
            'snn_chaotic': acc_snn,
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Noise Robustness at Extreme σ")
    print("=" * 60)
    print(f"\n{'σ':>6s}  {'White':>8s}  {'Pink':>8s}  {'SNN':>8s}  {'Best':>8s}")
    print("-" * 45)
    for sigma in sigmas:
        r = results[f'sigma_{sigma}']
        best = max(r.values())
        best_name = [k for k, v in r.items() if v == best][0]
        marker = " ←" if best_name != 'white' else ""
        print(f"{sigma:6.2f}  {r['white']:8.3f}  {r['pink']:8.3f}  "
              f"{r['snn_chaotic']:8.3f}  {best_name:>8s}{marker}")

    # Collapse point analysis
    print(f"\nCollapse analysis (where accuracy drops below baseline):")
    print(f"  Baseline: {acc_baseline:.3f}")
    for noise_type in ['white', 'pink', 'snn_chaotic']:
        collapse_sigma = None
        for sigma in sigmas:
            acc = results[f'sigma_{sigma}'][noise_type]
            if acc < acc_baseline * 0.9:  # 10% degradation threshold
                collapse_sigma = sigma
                break
        if collapse_sigma:
            print(f"  {noise_type:15s}: collapses at σ={collapse_sigma:.2f}")
        else:
            print(f"  {noise_type:15s}: no collapse (robust!)")

    # Save
    save_path = os.path.join(RESULTS_DIR, "phase36b_pink_noise_nbs.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 36b: Pink Noise Beam Search',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    main()
