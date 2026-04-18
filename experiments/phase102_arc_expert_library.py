"""
Phase 102: ARC-Native Expert Library

Train one L-NCA specialist per real ARC training task (~400 tasks).
Uses D8 geometric augmentation (8x data: 4 rotations x 2 flips).
Total library: ~400 x 2.8K = ~1.1M parameters.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
ARC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "training")
EXPERT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "experts")
DEVICE = "cpu"; SEED = 2026; NC = 10; HC = 32; NCA_STEPS = 10; CTX_CH = 10
SPECIALIST_EPOCHS = 100; BS = 16; MAX_GS = 30

# ====================================================================
# L-NCA Architecture
# ====================================================================
class LiquidNCA(nn.Module):
    def __init__(self, nc=10, hc=32):
        super().__init__()
        self.hc = hc
        self.perceive = nn.Conv2d(nc + hc, hc * 2, 3, padding=1)
        self.update = nn.Sequential(nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(nc + hc * 2, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)

    def forward(self, x, n_steps=NCA_STEPS, ctx=None):
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        x_in = x
        if ctx is not None:
            x_in = x + ctx.expand(-1, -1, h, w) if ctx.dim() == 4 else x
        for _ in range(n_steps):
            combined = torch.cat([x_in, state], 1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x_in, state, delta], 1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)

# ====================================================================
# Color-Frequency Mapping
# ====================================================================
def freq_remap(grid):
    flat = grid.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    mapping = {unique[order[i]]: i for i in range(len(unique))}
    return np.vectorize(lambda c: mapping.get(c, c))(grid)

def one_hot(grid, nc=10):
    h, w = grid.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc):
        o[c] = (grid == c).astype(np.float32)
    return o

# ====================================================================
# D8 Geometric Augmentation
# ====================================================================
def d8_augment(inp_grid, out_grid):
    """Apply all 8 D8 symmetries (4 rotations x 2 flips) to both grids."""
    pairs = []
    for k in range(4):
        ri = np.rot90(inp_grid, k)
        ro = np.rot90(out_grid, k)
        pairs.append((ri.copy(), ro.copy()))
        # Also flip
        fi = np.flipud(ri)
        fo = np.flipud(ro)
        pairs.append((fi.copy(), fo.copy()))
    return pairs

# ====================================================================
# Pad grids to uniform size
# ====================================================================
def pad_to(grid, max_h, max_w, val=0):
    h, w = grid.shape
    out = np.full((max_h, max_w), val, dtype=grid.dtype)
    out[:h, :w] = grid
    return out

# ====================================================================
# Train one expert per ARC task
# ====================================================================
def train_expert(task_data, epochs=SPECIALIST_EPOCHS):
    demos = task_data['train']
    if not demos:
        return None, None, 0.0

    # Collect all augmented demo pairs with freq-remap
    all_ins, all_outs = [], []
    for d in demos:
        inp = freq_remap(np.array(d['input']))
        out = freq_remap(np.array(d['output']))
        augmented = d8_augment(inp, out)
        for ai, ao in augmented:
            all_ins.append(ai)
            all_outs.append(ao)

    if not all_ins:
        return None, None, 0.0

    # Find max grid size (input and output may differ)
    max_h = max(max(g.shape[0] for g in all_ins), max(g.shape[0] for g in all_outs))
    max_w = max(max(g.shape[1] for g in all_ins), max(g.shape[1] for g in all_outs))

    if max_h > MAX_GS or max_w > MAX_GS:
        return None, None, 0.0

    # Check: if input/output sizes differ, L-NCA (same-size only) can't handle it
    for ai, ao in zip(all_ins, all_outs):
        if ai.shape != ao.shape:
            return None, None, -1.0   # Size mismatch

    # Pad and one-hot encode
    x = torch.tensor(np.array([one_hot(pad_to(g, max_h, max_w)) for g in all_ins]))
    y = torch.tensor(np.array([pad_to(g, max_h, max_w) for g in all_outs]))
    n = len(x)

    model = LiquidNCA(NC, HC).to(DEVICE)
    ctx = nn.Parameter(torch.randn(1, CTX_CH, 1, 1) * 0.01)
    opt = torch.optim.Adam(list(model.parameters()) + [ctx], lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_loss = float('inf')
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            xb, yb = x[idx].to(DEVICE), y[idx].to(DEVICE)
            opt.zero_grad()
            out = model(xb, ctx=ctx.expand(xb.size(0), -1, -1, -1))
            loss = F.cross_entropy(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [ctx], 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        best_loss = min(best_loss, ep_loss / max(n // BS, 1))

    # Evaluate on training data (check memorization)
    model.eval()
    with torch.no_grad():
        pred = model(x.to(DEVICE), ctx=ctx.expand(n, -1, -1, -1)).argmax(1).numpy()
    exact_match = sum(1 for i in range(n) if np.array_equal(pred[i], y[i].numpy())) / n

    return model, ctx.data.clone(), exact_match


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 102: ARC-Native Expert Library")
    print("=" * 70)

    if not os.path.exists(ARC_DIR):
        print(f"  ERROR: ARC data not found at {ARC_DIR}")
        return

    # Load all tasks
    files = sorted([f for f in os.listdir(ARC_DIR) if f.endswith('.json')])
    print(f"  Found {len(files)} ARC tasks")

    os.makedirs(EXPERT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    trained = 0; skipped_size = 0; skipped_mismatch = 0; total_params = 0

    for i, fname in enumerate(files):
        tid = fname.replace('.json', '')
        expert_path = os.path.join(EXPERT_DIR, f"{tid}.pt")

        # Skip if already trained
        if os.path.exists(expert_path):
            trained += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1:3d}/{len(files)}] {tid[:12]:12s} (cached)")
            continue

        with open(os.path.join(ARC_DIR, fname), 'r', encoding='utf-8') as f:
            task = json.load(f)

        t0 = time.time()
        model, ctx, em = train_expert(task)
        elapsed = time.time() - t0

        if model is None:
            if em == -1.0:
                skipped_mismatch += 1
                reason = "size_mismatch"
            else:
                skipped_size += 1
                reason = "too_large"
            results.append({'task_id': tid, 'status': reason})
            if (i + 1) % 50 == 0:
                print(f"  [{i+1:3d}/{len(files)}] {tid[:12]:12s} SKIP ({reason})")
            continue

        # Save expert
        torch.save({'model': model.state_dict(), 'ctx': ctx}, expert_path)
        n_params = sum(p.numel() for p in model.parameters())
        total_params += n_params
        trained += 1

        results.append({
            'task_id': tid,
            'status': 'trained',
            'exact_match': em,
            'params': n_params,
            'time_s': elapsed
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(files)}] {tid[:12]:12s} EM={em*100:.0f}%  "
                  f"{elapsed:.1f}s  ({trained} trained, {skipped_mismatch} size_mismatch, "
                  f"{skipped_size} too_large)")
            gc.collect()

    # Summary
    trained_results = [r for r in results if r.get('status') == 'trained']
    avg_em = np.mean([r['exact_match'] for r in trained_results]) if trained_results else 0
    print(f"\n{'=' * 70}")
    print(f"  Expert Library Complete!")
    print(f"  Trained: {len(trained_results)}/{len(files)}")
    print(f"  Skipped (size mismatch): {skipped_mismatch}")
    print(f"  Skipped (too large): {skipped_size}")
    print(f"  Avg Exact Match: {avg_em*100:.1f}%")
    print(f"  Total params: {total_params:,}")
    print(f"{'=' * 70}")

    with open(os.path.join(RESULTS_DIR, "phase102_arc_expert_library.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 102: ARC-Native Expert Library',
            'timestamp': datetime.now().isoformat(),
            'trained': len(trained_results),
            'total_tasks': len(files),
            'skipped_mismatch': skipped_mismatch,
            'skipped_too_large': skipped_size,
            'avg_exact_match': avg_em,
            'total_params': total_params,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Status pie chart
        labels = ['Trained', 'Size Mismatch', 'Too Large']
        sizes = [len(trained_results), skipped_mismatch, skipped_size]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        axes[0].pie([s for s, l in zip(sizes, labels) if s > 0],
                    labels=[l for s, l in zip(sizes, labels) if s > 0],
                    colors=[c for s, c in zip(sizes, colors) if s > 0],
                    autopct='%1.0f%%')
        axes[0].set_title(f'Task Coverage ({len(trained_results)}/{len(files)})')

        # 2. EM distribution
        ems = [r['exact_match'] for r in trained_results]
        if ems:
            axes[1].hist(ems, bins=20, color='steelblue', edgecolor='black')
            axes[1].axvline(x=np.mean(ems), color='red', linestyle='--', label=f'Mean={np.mean(ems)*100:.0f}%')
            axes[1].legend()
        axes[1].set_xlabel('Exact Match')
        axes[1].set_title('Training Accuracy Distribution')

        # 3. Time distribution
        times = [r['time_s'] for r in trained_results if 'time_s' in r]
        if times:
            axes[2].hist(times, bins=20, color='coral', edgecolor='black')
            axes[2].axvline(x=np.mean(times), color='red', linestyle='--', label=f'Mean={np.mean(times):.1f}s')
            axes[2].legend()
        axes[2].set_xlabel('Training Time (s)')
        axes[2].set_title('Per-Task Training Time')

        plt.suptitle('Phase 102: ARC-Native Expert Library', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase102_arc_expert_library.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 102 complete!")

if __name__ == '__main__':
    main()
