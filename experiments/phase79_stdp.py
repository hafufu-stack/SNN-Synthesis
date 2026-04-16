"""
Phase 79: Reward-Modulated STDP - Backprop-Free Learning

Implements biologically-plausible learning for Liquid-LIF:
No gradient computation (backward pass). Only forward spikes + local traces.

Algorithm:
  1. Forward pass: compute spikes and eligibility traces (pre*post timing)
  2. If output is correct: positive dopamine -> strengthen active synapses
  3. If output is wrong: negative dopamine -> weaken active synapses
  4. O(1) per-sample update (no backprop, no autograd)

Task: Simple pattern classification (4 patterns -> 4 classes)
Compare: Backprop-trained Liquid-LIF vs STDP-trained Liquid-LIF

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cpu"  # STDP is inherently sequential
SEED = 2026

HIDDEN_DIM = 16
INPUT_DIM = 8
N_CLASSES = 4
SEQ_LEN = 10
N_TRAIN = 2000
N_TEST = 500
STDP_LR = 0.01
STDP_EPOCHS = 50
BACKPROP_EPOCHS = 30
TRACE_DECAY = 0.95


class ATanSurrogate(torch.autograd.Function):
    alpha = 2.0
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = ATanSurrogate.alpha / 2 / (1 + (np.pi / 2 * ATanSurrogate.alpha * input).pow(2))
        return grad_output * grad

spike_fn = ATanSurrogate.apply


# ==============================================================
# Synthetic Pattern Dataset
# ==============================================================
def generate_patterns(n_samples, input_dim, n_classes, seq_len, seed=None):
    """Generate temporal patterns: each class has a distinct spike template."""
    rng = np.random.RandomState(seed)

    # Create class templates (unique temporal patterns)
    templates = []
    for c in range(n_classes):
        t = np.zeros((seq_len, input_dim), dtype=np.float32)
        # Each class fires specific neurons at specific times
        active_neurons = rng.choice(input_dim, size=input_dim // 2, replace=False)
        for step in range(seq_len):
            phase = (step + c * 2) % seq_len
            if phase < seq_len // 2:
                for n in active_neurons[:len(active_neurons) // 2]:
                    t[step, n] = 1.0
            else:
                for n in active_neurons[len(active_neurons) // 2:]:
                    t[step, n] = 1.0
        templates.append(t)

    X = []
    y = []
    for _ in range(n_samples):
        c = rng.randint(0, n_classes)
        noise = rng.randn(seq_len, input_dim).astype(np.float32) * 0.1
        X.append(templates[c] + noise)
        y.append(c)

    return torch.tensor(np.array(X)), torch.tensor(np.array(y))


# ==============================================================
# Liquid-LIF with Eligibility Traces (for STDP)
# ==============================================================
class LiquidLIFWithTraces(nn.Module):
    """Liquid-LIF that tracks eligibility traces for STDP learning."""
    def __init__(self, input_dim, hidden_dim, n_classes, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        # Learnable weights (manually managed for STDP)
        self.W_in = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.W_tau = nn.Parameter(torch.randn(hidden_dim, input_dim + hidden_dim) * 0.1)
        self.b_tau = nn.Parameter(torch.ones(hidden_dim) * 1.5)
        self.W_out = nn.Parameter(torch.randn(n_classes, hidden_dim) * 0.1)

    def forward_with_traces(self, x_seq):
        """Forward pass that also computes eligibility traces.
        x_seq: (seq_len, input_dim) - single sample"""
        seq_len = x_seq.size(0)
        mem = torch.zeros(self.hidden_dim, device=x_seq.device)

        # Traces for STDP
        pre_trace = torch.zeros(x_seq.size(1), device=x_seq.device)
        post_trace = torch.zeros(self.hidden_dim, device=x_seq.device)
        eligibility = torch.zeros_like(self.W_in)
        tau_eligibility = torch.zeros_like(self.W_tau)

        spike_count = 0
        for t in range(seq_len):
            x = x_seq[t]

            # Update pre-synaptic trace
            pre_trace = TRACE_DECAY * pre_trace + x

            # Liquid tau gate
            tau_input = torch.cat([x, mem])
            beta = torch.sigmoid(self.W_tau @ tau_input + self.b_tau)
            beta = torch.clamp(beta, 0.01, 0.99)

            # LIF dynamics
            mem = beta * mem + self.W_in @ x
            spk = spike_fn(mem - self.threshold)
            mem = mem - spk * self.threshold

            # Update post-synaptic trace
            post_trace = TRACE_DECAY * post_trace + spk

            # Eligibility: outer product of post and pre traces (STDP rule)
            eligibility += torch.outer(post_trace, pre_trace)
            tau_eligibility += torch.outer(post_trace, torch.cat([pre_trace,
                                           torch.zeros(self.hidden_dim, device=x.device)]))

            spike_count += spk.sum().item()

        # Readout
        out = self.W_out @ mem
        sparsity = 1.0 - (spike_count / (seq_len * self.hidden_dim))
        return out, eligibility, tau_eligibility, sparsity

    def forward(self, x_seq):
        """Standard forward for evaluation. x_seq: (batch, seq, input)"""
        batch_size = x_seq.size(0)
        all_logits = []
        total_sparsity = 0

        for b in range(batch_size):
            logits, _, _, sparsity = self.forward_with_traces(x_seq[b])
            all_logits.append(logits)
            total_sparsity += sparsity

        return torch.stack(all_logits), total_sparsity / batch_size

    def stdp_update(self, eligibility, tau_eligibility, reward, lr=STDP_LR):
        """Apply STDP update modulated by reward (dopamine)."""
        with torch.no_grad():
            self.W_in += lr * reward * eligibility
            self.W_tau += lr * reward * tau_eligibility * 0.5


# ==============================================================
# Training Functions
# ==============================================================
def train_stdp(model, X_train, y_train, epochs):
    """Train with reward-modulated STDP (no backprop)."""
    n = X_train.size(0)
    history = []
    for epoch in range(epochs):
        correct = 0
        perm = torch.randperm(n)
        for i in perm:
            x = X_train[i]
            target = y_train[i].item()

            logits, eligibility, tau_elig, _ = model.forward_with_traces(x)
            pred = logits.argmax().item()

            if pred == target:
                reward = 1.0
                correct += 1
            else:
                reward = -0.5

            model.stdp_update(eligibility, tau_elig, reward)

        acc = correct / n
        history.append(acc)
        if (epoch + 1) % 10 == 0:
            print(f"      STDP Epoch {epoch+1}: {acc*100:.1f}%")

    return history


def train_backprop(X_train, y_train, epochs):
    """Standard backprop training for comparison."""
    model = LiquidLIFWithTraces(INPUT_DIM, HIDDEN_DIM, N_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = (logits.argmax(1) == y_train).float().mean().item()
        history.append(acc)
        if (epoch + 1) % 10 == 0:
            print(f"      Backprop Epoch {epoch+1}: {acc*100:.1f}%")

    return model, history


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits, sparsity = model(X_test)
        acc = (logits.argmax(1) == y_test).float().mean().item()
    return acc, sparsity


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase79_stdp.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 79: Reward-Modulated STDP',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 79: Reward-Modulated STDP - Backprop-Free Learning")
    print("  Can Liquid-LIF learn with only forward spikes + dopamine?")
    print("=" * 70)

    X_train, y_train = generate_patterns(N_TRAIN, INPUT_DIM, N_CLASSES, SEQ_LEN, seed=SEED)
    X_test, y_test = generate_patterns(N_TEST, INPUT_DIM, N_CLASSES, SEQ_LEN, seed=SEED+1)
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}, Classes: {N_CLASSES}")

    results = {}

    # 1) Backprop baseline
    print(f"\n  === Backprop Training ===")
    t0 = time.time()
    bp_model, bp_history = train_backprop(X_train, y_train, BACKPROP_EPOCHS)
    bp_time = time.time() - t0
    bp_acc, bp_sparsity = evaluate(bp_model, X_test, y_test)
    print(f"    Test: {bp_acc*100:.1f}%, Sparsity: {bp_sparsity:.3f}, Time: {bp_time:.1f}s")
    results['backprop'] = {
        'test_acc': bp_acc, 'sparsity': bp_sparsity,
        'train_time': bp_time, 'history': bp_history
    }
    _save(results)

    # 2) STDP training
    print(f"\n  === STDP Training (No Backprop) ===")
    stdp_model = LiquidLIFWithTraces(INPUT_DIM, HIDDEN_DIM, N_CLASSES)
    t0 = time.time()
    stdp_history = train_stdp(stdp_model, X_train, y_train, STDP_EPOCHS)
    stdp_time = time.time() - t0
    stdp_acc, stdp_sparsity = evaluate(stdp_model, X_test, y_test)
    print(f"    Test: {stdp_acc*100:.1f}%, Sparsity: {stdp_sparsity:.3f}, Time: {stdp_time:.1f}s")
    results['stdp'] = {
        'test_acc': stdp_acc, 'sparsity': stdp_sparsity,
        'train_time': stdp_time, 'history': stdp_history
    }
    _save(results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Backprop vs STDP")
    print(f"{'='*70}")
    print(f"  Backprop: {bp_acc*100:.1f}% (time={bp_time:.1f}s)")
    print(f"  STDP:     {stdp_acc*100:.1f}% (time={stdp_time:.1f}s)")
    ratio = stdp_acc / max(0.001, bp_acc) * 100
    print(f"  STDP retains {ratio:.0f}% of backprop performance")
    print(f"  STDP uses ZERO gradient computation!")

    del bp_model, stdp_model; gc.collect()
    _generate_figure(results)
    print("\nPhase 79 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Learning curves
        bp_h = results['backprop']['history']
        stdp_h = results['stdp']['history']
        ax1.plot(range(1, len(bp_h)+1), [a*100 for a in bp_h],
                '-', color='#3B82F6', label='Backprop', linewidth=2)
        ax1.plot(range(1, len(stdp_h)+1), [a*100 for a in stdp_h],
                '-', color='#EC4899', label='STDP (no gradients)', linewidth=2)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Accuracy (%)')
        ax1.set_title('Learning Curves', fontweight='bold')
        ax1.legend(); ax1.grid(alpha=0.3)

        # Final comparison
        names = ['Backprop', 'STDP']
        accs = [results['backprop']['test_acc']*100, results['stdp']['test_acc']*100]
        colors = ['#3B82F6', '#EC4899']
        bars = ax2.bar(names, accs, color=colors, edgecolor='white', linewidth=1.5)
        for b, a in zip(bars, accs):
            ax2.text(b.get_x()+b.get_width()/2, a+1, f'{a:.1f}%',
                    ha='center', fontweight='bold')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Backprop vs STDP', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        fig.suptitle('Phase 79: Reward-Modulated STDP\n'
                    'Can Liquid-LIF learn without ANY gradient computation?',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase79_stdp.png"),
                   bbox_inches='tight', dpi=200)
        plt.close(); print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
