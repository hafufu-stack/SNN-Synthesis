"""
Phase 74: Event-Driven Liquid ARC Agent - The Micro-Liquid Limit

Two experiments in one:
1) Micro-Liquid Limit: Sweep hidden_dim = {128, 64, 32, 16, 8, 4, 2}
   on Sequential MNIST to find the minimum neurons needed for temporal reasoning.
2) Event-Driven Benchmark: Measure inference latency of Liquid-LIF
   in event-driven mode (skip steps with zero input) vs. standard mode.
   Target: break the 0.5ms barrier from Phase 46.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

N_CLASSES = 10
SEQ_LEN = 28
INPUT_DIM = 28
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

HIDDEN_DIMS = [128, 64, 32, 16, 8, 4, 2]


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


class LiquidLIFCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_tau = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.b_tau = nn.Parameter(torch.ones(hidden_dim) * 1.5)

    def forward(self, x, state=None):
        if state is None:
            mem = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        else:
            mem = state
        tau_input = torch.cat([x, mem], dim=-1)
        beta = torch.sigmoid(self.fc_tau(tau_input) + self.b_tau)
        beta = torch.clamp(beta, 0.01, 0.99)
        mem = beta * mem + self.fc_in(x)
        spk = spike_fn(mem - self.threshold)
        mem = mem - spk * self.threshold
        return spk, mem


class MicroLiquidClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.cell = LiquidLIFCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, n_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        state = None
        spike_count = 0
        total_neurons = 0
        for t in range(seq_len):
            spk, state = self.cell(x[:, t, :], state)
            spike_count += spk.sum().item()
            total_neurons += spk.numel()
        sparsity = 1.0 - (spike_count / max(1, total_neurons))
        return self.readout(state), sparsity

    def forward_event_driven(self, x, input_threshold=0.01):
        """Event-driven: skip timesteps where input magnitude < threshold."""
        batch_size, seq_len, _ = x.shape
        state = None
        steps_computed = 0
        for t in range(seq_len):
            step_input = x[:, t, :]
            max_input = step_input.abs().max().item()
            if max_input < input_threshold and state is not None:
                continue  # Skip: no meaningful input event
            _, state = self.cell(step_input, state)
            steps_computed += 1
        return self.readout(state), steps_computed


def get_sequential_mnist(batch_size):
    from torchvision import datasets, transforms
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "mnist")
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        train_ds = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
        test_ds = datasets.MNIST(data_dir, train=False, download=False, transform=transform)
    except RuntimeError:
        train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_and_evaluate(hidden_dim, train_loader, test_loader, device):
    model = MicroLiquidClassifier(INPUT_DIM, hidden_dim, N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train
    model.train()
    for epoch in range(EPOCHS):
        total_correct, total_n = 0, 0
        for images, labels in train_loader:
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += x.size(0)

    # Evaluate accuracy + sparsity
    model.eval()
    total_correct, total_n = 0, 0
    total_sparsity = 0; n_batches = 0
    with torch.no_grad():
        for images, labels in test_loader:
            x = images.squeeze(1).to(device)
            labels = labels.to(device)
            logits, sparsity = model(x)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += x.size(0)
            total_sparsity += sparsity; n_batches += 1
    test_acc = total_correct / total_n
    avg_sparsity = total_sparsity / max(1, n_batches)

    # Measure inference latency (standard mode)
    x_single = torch.randn(1, SEQ_LEN, INPUT_DIM).to(device)
    # Warmup
    for _ in range(10):
        with torch.no_grad(): model(x_single)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad(): model(x_single)
    if device == "cuda": torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) / 100 * 1000

    # Measure event-driven latency
    for _ in range(10):
        with torch.no_grad(): model.forward_event_driven(x_single)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_steps = 0
    for _ in range(100):
        with torch.no_grad():
            _, steps = model.forward_event_driven(x_single)
            total_steps += steps
    if device == "cuda": torch.cuda.synchronize()
    event_latency_ms = (time.perf_counter() - t0) / 100 * 1000
    avg_steps = total_steps / 100

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        'hidden_dim': hidden_dim,
        'n_params': n_params,
        'test_acc': test_acc,
        'sparsity': avg_sparsity,
        'latency_ms': latency_ms,
        'event_latency_ms': event_latency_ms,
        'event_avg_steps': avg_steps,
    }


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase74_micro_liquid.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 74: Micro-Liquid Limit',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    print("=" * 70)
    print("Phase 74: Micro-Liquid Limit & Event-Driven Benchmark")
    print("  How few neurons can handle temporal reasoning?")
    print(f"  Device: {DEVICE}, Hidden dims: {HIDDEN_DIMS}")
    print("=" * 70)

    train_loader, test_loader = get_sequential_mnist(BATCH_SIZE)
    all_results = []

    for hd in HIDDEN_DIMS:
        print(f"\n  --- Hidden dim = {hd} ---")
        t0 = time.time()
        result = train_and_evaluate(hd, train_loader, test_loader, DEVICE)
        elapsed = time.time() - t0

        print(f"    Params:       {result['n_params']:,}")
        print(f"    Test acc:     {result['test_acc']*100:.1f}%")
        print(f"    Sparsity:     {result['sparsity']:.3f}")
        print(f"    Latency:      {result['latency_ms']:.3f}ms (standard)")
        print(f"    Event latency: {result['event_latency_ms']:.3f}ms "
              f"(avg {result['event_avg_steps']:.0f}/{SEQ_LEN} steps)")
        print(f"    Train time:   {elapsed:.0f}s")

        result['train_time_sec'] = elapsed
        all_results.append(result)
        _save(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Micro-Liquid Limit")
    print(f"{'='*70}")
    print(f"  {'H_dim':>6s} {'Params':>8s} {'Acc':>8s} {'Sparsity':>10s} "
          f"{'Latency':>10s} {'Event':>10s} {'Steps':>8s}")
    for r in all_results:
        print(f"  {r['hidden_dim']:>6d} {r['n_params']:>8,} {r['test_acc']*100:>7.1f}% "
              f"{r['sparsity']:>9.3f} {r['latency_ms']:>9.3f}ms "
              f"{r['event_latency_ms']:>9.3f}ms {r['event_avg_steps']:>7.0f}")

    # Find minimum viable
    for r in sorted(all_results, key=lambda x: x['hidden_dim']):
        if r['test_acc'] >= 0.90:
            print(f"\n  >>> Minimum viable: {r['hidden_dim']} neurons ({r['n_params']:,} params) "
                  f"= {r['test_acc']*100:.1f}% accuracy")
            break
    else:
        print(f"\n  >>> No configuration reached 90% threshold")

    sub_half_ms = [r for r in all_results if r['event_latency_ms'] < 0.5]
    if sub_half_ms:
        best = max(sub_half_ms, key=lambda x: x['test_acc'])
        print(f"  >>> Best sub-0.5ms: {best['hidden_dim']} neurons = "
              f"{best['test_acc']*100:.1f}% @ {best['event_latency_ms']:.3f}ms")

    _generate_figure(all_results)
    print("\nPhase 74 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        dims = [r['hidden_dim'] for r in results]
        accs = [r['test_acc'] * 100 for r in results]
        latencies = [r['latency_ms'] for r in results]
        event_latencies = [r['event_latency_ms'] for r in results]
        sparsities = [r['sparsity'] * 100 for r in results]

        # 1) Accuracy vs Hidden Dim
        ax = axes[0]
        ax.plot(dims, accs, '-o', color='#EC4899', markersize=8, linewidth=2)
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
        ax.set_xlabel('Hidden Dimension'); ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Micro-Liquid Limit', fontweight='bold')
        ax.set_xscale('log', base=2); ax.legend(); ax.grid(alpha=0.3)
        for d, a in zip(dims, accs):
            ax.annotate(f'{a:.0f}%', (d, a+1), ha='center', fontsize=9)

        # 2) Latency: Standard vs Event-Driven
        ax = axes[1]
        x = np.arange(len(dims))
        ax.bar(x - 0.18, latencies, 0.35, label='Standard', color='#3B82F6')
        ax.bar(x + 0.18, event_latencies, 0.35, label='Event-Driven', color='#10B981')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='0.5ms barrier')
        ax.set_xticks(x); ax.set_xticklabels([str(d) for d in dims])
        ax.set_xlabel('Hidden Dimension'); ax.set_ylabel('Latency (ms)')
        ax.set_title('Inference Speed', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)

        # 3) Accuracy vs Latency (Pareto)
        ax = axes[2]
        scatter = ax.scatter(event_latencies, accs, c=dims, cmap='plasma',
                            s=150, edgecolors='white', linewidth=1.5, zorder=5)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='0.5ms barrier')
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
        plt.colorbar(scatter, ax=ax, label='Hidden Dim')
        for d, lat, a in zip(dims, event_latencies, accs):
            ax.annotate(f'H={d}', (lat+0.01, a+0.5), fontsize=8)
        ax.set_xlabel('Event-Driven Latency (ms)'); ax.set_ylabel('Accuracy (%)')
        ax.set_title('Pareto Frontier', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        fig.suptitle('Phase 74: Micro-Liquid Limit & Event-Driven Benchmark',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase74_micro_liquid.png"),
                   bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
