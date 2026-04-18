"""
Phase 107b: Size-Agnostic Global L-NCA (30x30 Fixed Canvas)

Fix Phase 107's crash: always pad ALL grids to 30x30 canvas first,
then compare Vanilla vs Global-Coord L-NCA on real ARC tasks.

Key fix: pad to CANVAS_SIZE (30), not max(train grids).

Author: Hiroto Funasaki
"""
import os, json, time, gc, random
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
EPOCHS = 80; BS = 8; CANVAS = 30

# ====================================================================
# Global-Coordinate L-NCA (with absolute coords + global avg pooling)
# ====================================================================
class GlobalCoordLNCA(nn.Module):
    def __init__(self, nc=10, hc=32):
        super().__init__()
        self.nc = nc; self.hc = hc
        in_ch = nc + nc + 2 + hc  # one-hot + global_avg + coords + state
        self.perceive = nn.Conv2d(in_ch, hc * 2, 3, padding=1)
        self.update = nn.Sequential(nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(in_ch + hc, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)
    
    def _make_coords(self, b, h, w, device):
        yy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([yy, xx], dim=1)

    def forward(self, x, n_steps=NCA_STEPS):
        b, c, h, w = x.shape
        state = torch.zeros(b, self.hc, h, w, device=x.device)
        coords = self._make_coords(b, h, w, x.device)
        for _ in range(n_steps):
            global_avg = x.mean(dim=[2, 3], keepdim=True).expand(b, self.nc, h, w)
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
    def forward(self, x, n_steps=NCA_STEPS):
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
# Helpers (all padded to CANVAS x CANVAS)
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

def pad_canvas(grid, canvas=CANVAS):
    """Always pad to fixed canvas size."""
    h, w = grid.shape
    out = np.zeros((canvas, canvas), dtype=grid.dtype)
    out[:min(h, canvas), :min(w, canvas)] = grid[:min(h, canvas), :min(w, canvas)]
    return out

def d8_augment_canvas(inp_grid, out_grid, canvas=CANVAS):
    """D8 augment then pad to canvas."""
    pairs = []
    for k in range(4):
        ri, ro = np.rot90(inp_grid, k), np.rot90(out_grid, k)
        pairs.append((pad_canvas(ri.copy(), canvas), pad_canvas(ro.copy(), canvas)))
        fi, fo = np.flipud(ri), np.flipud(ro)
        pairs.append((pad_canvas(fi.copy(), canvas), pad_canvas(fo.copy(), canvas)))
    return pairs

# ====================================================================
# Load same-size ARC tasks
# ====================================================================
def load_same_size_tasks(arc_dir, max_tasks=40):
    tasks = {}
    files = sorted([f for f in os.listdir(arc_dir) if f.endswith('.json')])
    for fname in files:
        tid = fname.replace('.json', '')
        with open(os.path.join(arc_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid = True
        for pair in data['train'] + data['test']:
            inp = np.array(pair['input']); out = np.array(pair['output'])
            if inp.shape != out.shape or max(inp.shape) > CANVAS:
                valid = False; break
        if valid:
            tasks[tid] = data
        if len(tasks) >= max_tasks:
            break
    return tasks

# ====================================================================
# Train and evaluate (FIXED: always use CANVAS padding)
# ====================================================================
def train_and_eval(ModelClass, task_data, epochs=EPOCHS):
    demos = task_data['train']; tests = task_data['test']
    
    # All grids padded to CANVAS x CANVAS
    all_ins, all_outs = [], []
    for d in demos:
        inp = freq_remap(np.array(d['input']))
        out = freq_remap(np.array(d['output']))
        for ai, ao in d8_augment_canvas(inp, out, CANVAS):
            all_ins.append(one_hot(ai)); all_outs.append(ao)
    
    if not all_ins: return 0.0, 0.0
    
    x = torch.tensor(np.array(all_ins))
    y = torch.tensor(np.array(all_outs))
    n = len(x)
    
    model = ModelClass(NC, HC).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]; opt.zero_grad()
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
    
    # Evaluate test (also padded to CANVAS)
    model.eval(); n_correct = 0
    for test_pair in tests:
        ti = freq_remap(np.array(test_pair['input']))
        to = np.array(test_pair['output'])
        oh, ow = to.shape
        ti_pad = pad_canvas(ti, CANVAS)
        ti_t = torch.tensor(one_hot(ti_pad)).unsqueeze(0)
        with torch.no_grad():
            pred = model(ti_t).argmax(1).squeeze(0).numpy()
        if np.array_equal(pred[:oh, :ow], to):
            n_correct += 1
    
    test_em = n_correct / max(len(tests), 1)
    
    # Training accuracy check
    with torch.no_grad():
        train_pred = model(x).argmax(1).numpy()
    train_em = sum(1 for i in range(n) if np.array_equal(train_pred[i], y[i].numpy())) / n
    
    return test_em, train_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 107b: Size-Agnostic Global L-NCA (30x30 Canvas)")
    print("=" * 70)

    print("  Loading same-size ARC tasks...")
    tasks = load_same_size_tasks(ARC_DIR, max_tasks=30)
    print(f"  Found {len(tasks)} same-size tasks")

    results = []
    vanilla_test = []; global_test = []
    vanilla_train = []; global_train = []

    for i, (tid, task) in enumerate(tasks.items()):
        n_demos = len(task['train'])
        inp_shape = np.array(task['train'][0]['input']).shape

        v_te, v_tr = train_and_eval(VanillaLNCA, task, epochs=EPOCHS)
        g_te, g_tr = train_and_eval(GlobalCoordLNCA, task, epochs=EPOCHS)

        vanilla_test.append(v_te); global_test.append(g_te)
        vanilla_train.append(v_tr); global_train.append(g_tr)
        
        marker = "<<< GLOBAL WINS" if g_te > v_te else ("  TIE" if g_te == v_te else "")
        print(f"  [{i+1:2d}] {tid[:12]:12s} {str(inp_shape):10s} d={n_demos}  "
              f"V={v_te*100:.0f}%/{v_tr*100:.0f}%  "
              f"G={g_te*100:.0f}%/{g_tr*100:.0f}% {marker}")

        results.append({
            'task_id': tid, 'shape': list(inp_shape), 'n_demos': n_demos,
            'vanilla_test': v_te, 'vanilla_train': v_tr,
            'global_test': g_te, 'global_train': g_tr
        })
        gc.collect()

    # Summary
    avg_vt = np.mean(vanilla_test)*100; avg_gt = np.mean(global_test)*100
    avg_vtr = np.mean(vanilla_train)*100; avg_gtr = np.mean(global_train)*100
    wins = sum(1 for v, g in zip(vanilla_test, global_test) if g > v)
    
    print(f"\n{'='*70}")
    print(f"  Vanilla L-NCA:       Test={avg_vt:.1f}%  Train={avg_vtr:.1f}%")
    print(f"  Global-Coord L-NCA:  Test={avg_gt:.1f}%  Train={avg_gtr:.1f}%")
    print(f"  Global wins: {wins}/{len(tasks)}, Improvement: {avg_gt - avg_vt:+.1f}pp")
    
    n_params_v = sum(p.numel() for p in VanillaLNCA(NC, HC).parameters())
    n_params_g = sum(p.numel() for p in GlobalCoordLNCA(NC, HC).parameters())
    print(f"  Params: Vanilla={n_params_v}, Global={n_params_g}")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase107b_global_coord_lnca.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 107b: Size-Agnostic Global L-NCA',
            'timestamp': datetime.now().isoformat(),
            'canvas_size': CANVAS,
            'avg_vanilla_test': avg_vt, 'avg_global_test': avg_gt,
            'avg_vanilla_train': avg_vtr, 'avg_global_train': avg_gtr,
            'global_wins': wins, 'total_tasks': len(tasks),
            'vanilla_params': n_params_v, 'global_params': n_params_g,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x_pos = np.arange(len(tasks)); w = 0.35
        axes[0].bar(x_pos - w/2, [r['vanilla_test']*100 for r in results], w,
                    label='Vanilla', color='#e74c3c', alpha=0.8)
        axes[0].bar(x_pos + w/2, [r['global_test']*100 for r in results], w,
                    label='Global', color='#2ecc71', alpha=0.8)
        axes[0].set_xlabel('Task'); axes[0].set_ylabel('Test EM (%)')
        axes[0].set_title('Per-Task Test Accuracy'); axes[0].legend()

        axes[1].scatter([r['vanilla_train']*100 for r in results],
                       [r['global_train']*100 for r in results], alpha=0.7, s=60)
        axes[1].plot([0, 100], [0, 100], 'k--', alpha=0.3)
        axes[1].set_xlabel('Vanilla Train EM (%)'); axes[1].set_ylabel('Global Train EM (%)')
        axes[1].set_title('Training: Global vs Vanilla')

        axes[2].bar(['Vanilla', 'Global-Coord'],
                    [avg_vt, avg_gt], color=['#e74c3c', '#2ecc71'], edgecolor='black')
        axes[2].set_ylabel('Avg Test EM (%)')
        axes[2].set_title(f'Global: {avg_gt - avg_vt:+.1f}pp\n({n_params_v} vs {n_params_g} params)')

        plt.suptitle('Phase 107b: Global-Coordinate L-NCA on Real ARC (30x30 Canvas)', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase107b_global_coord_lnca.png'), dpi=150)
        plt.close(); print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 107b complete!")

if __name__ == '__main__':
    main()
