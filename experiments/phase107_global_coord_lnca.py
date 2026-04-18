"""
Phase 107: Global-Coordinate L-NCA

Add global context (Global Average Pooling + absolute coordinates) to L-NCA
to overcome local receptive field limitation on real ARC tasks.

Key insight from Phase 102: 3x3 local rules can't learn ARC's global logic.
Solution: Give each cell awareness of "where it is" and "what the whole grid looks like".

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
DEVICE = "cpu"; SEED = 2026; NC = 10; HC = 32; NCA_STEPS = 10
EPOCHS = 100; BS = 16; MAX_GS = 30

# ====================================================================
# Global-Coordinate L-NCA
# ====================================================================
class GlobalCoordLNCA(nn.Module):
    """L-NCA with global average pooling + absolute coordinates.
    
    Extra input channels:
    - NC channels: global average pooling (broadcast to all cells)
    - 2 channels: normalized (x/W, y/H) coordinates
    Total input: NC (one-hot) + NC (global) + 2 (coords) + HC (state) = 2*NC + 2 + HC
    """
    def __init__(self, nc=10, hc=32):
        super().__init__()
        self.nc = nc
        self.hc = hc
        in_ch = nc + nc + 2 + hc  # one-hot + global_avg + coords + state
        self.perceive = nn.Conv2d(in_ch, hc * 2, 3, padding=1)
        self.update = nn.Sequential(nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(in_ch + hc, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)

    def _make_coords(self, b, h, w, device):
        """Create normalized coordinate channels (y/H, x/W)."""
        yy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([yy, xx], dim=1)  # (b, 2, h, w)

    def forward(self, x, n_steps=NCA_STEPS, ctx=None):
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        coords = self._make_coords(b, h, w, x.device)

        for _ in range(n_steps):
            # Global average pooling → broadcast
            global_avg = x.mean(dim=[2, 3], keepdim=True).expand(b, self.nc, h, w)
            
            # Combine: one-hot + global + coords + state
            combined = torch.cat([x, global_avg, coords, state], dim=1)
            
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([combined, delta], 1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta

        return self.readout(state)


# ====================================================================
# Vanilla L-NCA (baseline)
# ====================================================================
class VanillaLNCA(nn.Module):
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
        for _ in range(n_steps):
            combined = torch.cat([x, state], 1)
            delta = self.update(self.perceive(combined))
            tau_in = torch.cat([x, state, delta], 1)
            beta = torch.sigmoid(self.tau_gate(tau_in) + self.b_tau).clamp(0.01, 0.99)
            state = beta * state + (1 - beta) * delta
        return self.readout(state)

# ====================================================================
# Helpers
# ====================================================================
def one_hot(grid, nc=10):
    h, w = grid.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): o[c] = (grid == c).astype(np.float32)
    return o

def freq_remap(grid):
    flat = grid.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    mapping = {unique[order[i]]: i for i in range(len(unique))}
    return np.vectorize(lambda c: mapping.get(c, c))(grid)

def d8_augment(inp_grid, out_grid):
    pairs = []
    for k in range(4):
        ri, ro = np.rot90(inp_grid, k), np.rot90(out_grid, k)
        pairs.append((ri.copy(), ro.copy()))
        pairs.append((np.flipud(ri).copy(), np.flipud(ro).copy()))
    return pairs

def pad_to(grid, max_h, max_w, val=0):
    h, w = grid.shape
    out = np.full((max_h, max_w), val, dtype=grid.dtype)
    out[:h, :w] = grid
    return out

# ====================================================================
# Train and evaluate on real ARC (same-size tasks only)
# ====================================================================
def load_same_size_tasks(arc_dir, max_tasks=50):
    """Load ARC tasks where all input/output pairs have same dimensions."""
    tasks = {}
    files = sorted([f for f in os.listdir(arc_dir) if f.endswith('.json')])
    for fname in files:
        tid = fname.replace('.json', '')
        with open(os.path.join(arc_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Check all pairs are same-size and within limits
        valid = True
        for pair in data['train'] + data['test']:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            if inp.shape != out.shape or max(inp.shape) > MAX_GS:
                valid = False; break
        if valid:
            tasks[tid] = data
        if len(tasks) >= max_tasks:
            break
    return tasks

def train_and_eval(ModelClass, task_data, epochs=EPOCHS):
    """Train model on demo pairs, evaluate on test pairs."""
    demos = task_data['train']
    tests = task_data['test']
    
    # Prepare augmented training data
    all_ins, all_outs = [], []
    for d in demos:
        inp = freq_remap(np.array(d['input']))
        out = freq_remap(np.array(d['output']))
        for ai, ao in d8_augment(inp, out):
            all_ins.append(ai); all_outs.append(ao)
    
    if not all_ins:
        return None, 0.0
    
    max_h = max(g.shape[0] for g in all_ins + all_outs)
    max_w = max(g.shape[1] for g in all_ins + all_outs)
    
    x = torch.tensor(np.array([one_hot(pad_to(g, max_h, max_w)) for g in all_ins]))
    y = torch.tensor(np.array([pad_to(g, max_h, max_w) for g in all_outs]))
    n = len(x)
    
    model = ModelClass(NC, HC).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            out = model(x[idx])
            F.cross_entropy(out, y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    
    # Evaluate on test
    model.eval()
    n_correct = 0
    for test_pair in tests:
        ti = freq_remap(np.array(test_pair['input']))
        to = np.array(test_pair['output'])
        ti_t = torch.tensor(one_hot(pad_to(ti, max_h, max_w))).unsqueeze(0)
        with torch.no_grad():
            pred = model(ti_t).argmax(1).squeeze(0).numpy()
        oh, ow = to.shape
        if np.array_equal(pred[:oh, :ow], to):
            n_correct += 1
    
    em = n_correct / max(len(tests), 1)
    # Also check training accuracy
    with torch.no_grad():
        train_pred = model(x).argmax(1).numpy()
    train_em = sum(1 for i in range(n) if np.array_equal(train_pred[i], y[i].numpy())) / n
    
    return model, em, train_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 107: Global-Coordinate L-NCA")
    print("=" * 70)

    # Load same-size ARC tasks
    print("  Loading same-size ARC tasks...")
    tasks = load_same_size_tasks(ARC_DIR, max_tasks=30)
    print(f"  Found {len(tasks)} same-size tasks")

    results = []
    vanilla_ems = []; global_ems = []
    vanilla_train = []; global_train = []

    for i, (tid, task) in enumerate(tasks.items()):
        n_demos = len(task['train'])
        inp_shape = np.array(task['train'][0]['input']).shape

        # Train Vanilla L-NCA
        _, v_em, v_tr = train_and_eval(VanillaLNCA, task, epochs=EPOCHS)
        
        # Train Global-Coord L-NCA
        _, g_em, g_tr = train_and_eval(GlobalCoordLNCA, task, epochs=EPOCHS)

        vanilla_ems.append(v_em); global_ems.append(g_em)
        vanilla_train.append(v_tr); global_train.append(g_tr)
        
        marker = "<<< WINNER" if g_em > v_em else ("  TIE" if g_em == v_em else "")
        print(f"  [{i+1:2d}] {tid[:12]:12s} {inp_shape} demos={n_demos}  "
              f"Vanilla={v_em*100:.0f}%/{v_tr*100:.0f}%  "
              f"Global={g_em*100:.0f}%/{g_tr*100:.0f}% {marker}")

        results.append({
            'task_id': tid, 'shape': list(inp_shape), 'n_demos': n_demos,
            'vanilla_test_em': v_em, 'vanilla_train_em': v_tr,
            'global_test_em': g_em, 'global_train_em': g_tr
        })
        gc.collect()

    # Summary
    avg_v = np.mean(vanilla_ems) * 100; avg_g = np.mean(global_ems) * 100
    avg_vt = np.mean(vanilla_train) * 100; avg_gt = np.mean(global_train) * 100
    wins = sum(1 for v, g in zip(vanilla_ems, global_ems) if g > v)
    
    print(f"\n{'=' * 70}")
    print(f"  Vanilla L-NCA:       Test={avg_v:.1f}%  Train={avg_vt:.1f}%")
    print(f"  Global-Coord L-NCA:  Test={avg_g:.1f}%  Train={avg_gt:.1f}%")
    print(f"  Global wins: {wins}/{len(tasks)}")
    print(f"  Improvement: {avg_g - avg_v:+.1f}pp")
    print(f"{'=' * 70}")

    n_params_v = sum(p.numel() for p in VanillaLNCA(NC, HC).parameters())
    n_params_g = sum(p.numel() for p in GlobalCoordLNCA(NC, HC).parameters())
    
    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase107_global_coord_lnca.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 107: Global-Coordinate L-NCA',
            'timestamp': datetime.now().isoformat(),
            'avg_vanilla_test': avg_v, 'avg_global_test': avg_g,
            'avg_vanilla_train': avg_vt, 'avg_global_train': avg_gt,
            'global_wins': wins, 'total_tasks': len(tasks),
            'vanilla_params': n_params_v, 'global_params': n_params_g,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Comparison bar chart
        x_pos = np.arange(len(tasks))
        w = 0.35
        axes[0].bar(x_pos - w/2, [r['vanilla_test_em']*100 for r in results], w, label='Vanilla', color='#e74c3c', alpha=0.8)
        axes[0].bar(x_pos + w/2, [r['global_test_em']*100 for r in results], w, label='Global-Coord', color='#2ecc71', alpha=0.8)
        axes[0].set_xlabel('Task'); axes[0].set_ylabel('Test EM (%)')
        axes[0].set_title('Per-Task Test Accuracy'); axes[0].legend()

        # 2. Train accuracy comparison
        axes[1].scatter([r['vanilla_train_em']*100 for r in results],
                       [r['global_train_em']*100 for r in results], alpha=0.7, s=60)
        axes[1].plot([0, 100], [0, 100], 'k--', alpha=0.3)
        axes[1].set_xlabel('Vanilla Train EM (%)'); axes[1].set_ylabel('Global Train EM (%)')
        axes[1].set_title('Training Accuracy: Global vs Vanilla\n(Above diagonal = Global better)')

        # 3. Summary
        axes[2].bar(['Vanilla\nL-NCA', 'Global-Coord\nL-NCA'],
                    [avg_v, avg_g], color=['#e74c3c', '#2ecc71'], edgecolor='black')
        axes[2].set_ylabel('Avg Test EM (%)')
        axes[2].set_title(f'Global-Coord: {avg_g - avg_v:+.1f}pp improvement\n'
                          f'({n_params_v} vs {n_params_g} params)')

        plt.suptitle('Phase 107: Global-Coordinate L-NCA on Real ARC', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase107_global_coord_lnca.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 107 complete!")

if __name__ == '__main__':
    main()
