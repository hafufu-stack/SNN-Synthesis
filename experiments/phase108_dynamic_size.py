"""
Phase 108: Dynamic Size Cropper

Solve ARC's size mismatch problem (34.5% of tasks) by:
1. Always compute on max canvas (30x30, zero-padded)
2. MLP head predicts output dimensions (H, W) from demo context
3. Crop NCA output to predicted size

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
EPOCHS = 100; BS = 8; CANVAS = 30

# ====================================================================
# Dynamic Size L-NCA
# ====================================================================
class DynamicSizeLNCA(nn.Module):
    """L-NCA that works on fixed canvas and predicts output size."""
    def __init__(self, nc=10, hc=32, canvas=30):
        super().__init__()
        self.nc = nc; self.hc = hc; self.canvas = canvas
        
        # NCA core (with global context)
        in_ch = nc + nc + 2 + hc  # one-hot + global + coords + state
        self.perceive = nn.Conv2d(in_ch, hc * 2, 3, padding=1)
        self.update = nn.Sequential(nn.Conv2d(hc * 2, hc, 1), nn.ReLU(), nn.Conv2d(hc, hc, 1))
        self.tau_gate = nn.Conv2d(in_ch + hc, hc, 3, padding=1)
        self.b_tau = nn.Parameter(torch.ones(1, hc, 1, 1) * 1.5)
        self.readout = nn.Conv2d(hc, nc, 1)
        
        # Size predictor: from global features → (H, W)
        self.size_head = nn.Sequential(
            nn.Linear(nc * 2 + 2, 64),  # input_dims(H,W) + global stats
            nn.ReLU(),
            nn.Linear(64, 2)  # predict (out_H, out_W)
        )
    
    def _make_coords(self, b, h, w, device):
        yy = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([yy, xx], dim=1)

    def forward(self, x, n_steps=NCA_STEPS, input_hw=None):
        """
        Args:
            x: (B, NC, canvas, canvas) padded input
            input_hw: (B, 2) original input H, W (for size prediction)
        Returns:
            output: (B, NC, canvas, canvas) full canvas output
            pred_hw: (B, 2) predicted output H, W
        """
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
        
        output = self.readout(state)
        
        # Predict output size
        if input_hw is not None:
            global_feat = x.mean(dim=[2, 3])  # (B, NC)
            out_feat = output.mean(dim=[2, 3])  # (B, NC)
            size_input = torch.cat([global_feat, out_feat, input_hw.float() / self.canvas], dim=1)
            pred_hw = torch.sigmoid(self.size_head(size_input)) * self.canvas
        else:
            pred_hw = None
        
        return output, pred_hw

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

def pad_canvas(grid, canvas_size, nc=10):
    """Pad grid to (canvas, canvas) and one-hot encode."""
    h, w = grid.shape
    padded = np.zeros((canvas_size, canvas_size), dtype=grid.dtype)
    padded[:h, :w] = grid
    return one_hot(padded, nc)

# ====================================================================
# Load tasks (including size-changing!)
# ====================================================================
def load_all_tasks(arc_dir, max_tasks=40):
    tasks = {}
    files = sorted([f for f in os.listdir(arc_dir) if f.endswith('.json')])
    for fname in files:
        tid = fname.replace('.json', '')
        with open(os.path.join(arc_dir, fname), 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Check all grids fit in canvas
        valid = True
        has_size_change = False
        for pair in data['train'] + data['test']:
            inp = np.array(pair['input']); out = np.array(pair['output'])
            if max(inp.shape) > CANVAS or max(out.shape) > CANVAS:
                valid = False; break
            if inp.shape != out.shape:
                has_size_change = True
        if valid:
            tasks[tid] = {'data': data, 'size_change': has_size_change}
        if len(tasks) >= max_tasks:
            break
    return tasks


def train_and_eval_dynamic(task_data):
    """Train DynamicSizeLNCA on one task."""
    demos = task_data['train']; tests = task_data['test']
    
    # Prepare training data (no D8 for size-changing tasks to keep size mapping)
    all_ins, all_outs, all_in_hw, all_out_hw = [], [], [], []
    for d in demos:
        inp = freq_remap(np.array(d['input']))
        out = freq_remap(np.array(d['output']))
        ih, iw = inp.shape; oh, ow = out.shape
        all_ins.append(pad_canvas(inp, CANVAS))
        all_outs.append(np.pad(out, ((0, CANVAS-oh), (0, CANVAS-ow)), constant_values=0))
        all_in_hw.append([ih, iw])
        all_out_hw.append([oh, ow])
    
    if not all_ins: return 0.0, 0.0, 0.0
    
    x = torch.tensor(np.array(all_ins))
    y = torch.tensor(np.array(all_outs))
    hw_in = torch.tensor(np.array(all_in_hw, dtype=np.float32))
    hw_out = torch.tensor(np.array(all_out_hw, dtype=np.float32))
    n = len(x)
    
    model = DynamicSizeLNCA(NC, HC, CANVAS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        output, pred_hw = model(x, input_hw=hw_in)
        
        # Pixel loss (on full canvas, target is zero-padded)
        pixel_loss = F.cross_entropy(output, y)
        
        # Size prediction loss
        size_loss = F.mse_loss(pred_hw, hw_out) if pred_hw is not None else 0
        
        loss = pixel_loss + 0.1 * size_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    
    # Evaluate
    model.eval()
    n_correct_pixel = 0; n_correct_size = 0; n_correct_both = 0
    
    for test_pair in tests:
        ti = freq_remap(np.array(test_pair['input']))
        to = np.array(test_pair['output'])
        ih, iw = ti.shape; oh, ow = to.shape
        
        ti_t = torch.tensor(pad_canvas(ti, CANVAS)).unsqueeze(0)
        hw_t = torch.tensor([[ih, iw]], dtype=torch.float32)
        
        with torch.no_grad():
            pred_out, pred_hw = model(ti_t, input_hw=hw_t)
            pred_grid = pred_out.argmax(1).squeeze(0).numpy()
        
        # Check size prediction
        if pred_hw is not None:
            ph = int(pred_hw[0, 0].round().clamp(1, CANVAS).item())
            pw = int(pred_hw[0, 1].round().clamp(1, CANVAS).item())
        else:
            ph, pw = ih, iw
        
        size_correct = (ph == oh and pw == ow)
        if size_correct: n_correct_size += 1
        
        # Check pixel accuracy (cropped to true size)
        pixel_correct = np.array_equal(pred_grid[:oh, :ow], to)
        if pixel_correct: n_correct_pixel += 1
        if size_correct and pixel_correct: n_correct_both += 1
    
    nt = max(len(tests), 1)
    return n_correct_pixel / nt, n_correct_size / nt, n_correct_both / nt


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 108: Dynamic Size Cropper")
    print("=" * 70)

    print("  Loading ARC tasks (including size-changing)...")
    tasks = load_all_tasks(ARC_DIR, max_tasks=30)
    same_size = sum(1 for t in tasks.values() if not t['size_change'])
    diff_size = sum(1 for t in tasks.values() if t['size_change'])
    print(f"  Loaded {len(tasks)} tasks ({same_size} same-size, {diff_size} size-changing)")

    results = []
    for i, (tid, task_info) in enumerate(tasks.items()):
        task = task_info['data']
        sc = task_info['size_change']
        
        t0 = time.time()
        pixel_em, size_em, both_em = train_and_eval_dynamic(task)
        elapsed = time.time() - t0
        
        tag = "SIZE_CHANGE" if sc else "SAME_SIZE"
        print(f"  [{i+1:2d}] {tid[:12]:12s} [{tag:11s}]  "
              f"pixel={pixel_em*100:.0f}%  size={size_em*100:.0f}%  "
              f"both={both_em*100:.0f}%  ({elapsed:.1f}s)")
        
        results.append({
            'task_id': tid, 'size_change': sc,
            'pixel_em': pixel_em, 'size_em': size_em, 'both_em': both_em,
            'time_s': elapsed
        })
        gc.collect()

    # Summary
    same_results = [r for r in results if not r['size_change']]
    diff_results = [r for r in results if r['size_change']]
    
    avg_same = np.mean([r['pixel_em'] for r in same_results]) * 100 if same_results else 0
    avg_diff_pixel = np.mean([r['pixel_em'] for r in diff_results]) * 100 if diff_results else 0
    avg_diff_size = np.mean([r['size_em'] for r in diff_results]) * 100 if diff_results else 0
    avg_diff_both = np.mean([r['both_em'] for r in diff_results]) * 100 if diff_results else 0

    print(f"\n{'=' * 70}")
    print(f"  Same-size tasks:     pixel EM = {avg_same:.1f}%")
    print(f"  Size-changing tasks: pixel EM = {avg_diff_pixel:.1f}%, "
          f"size EM = {avg_diff_size:.1f}%, both EM = {avg_diff_both:.1f}%")
    print(f"{'=' * 70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase108_dynamic_size.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 108: Dynamic Size Cropper',
            'timestamp': datetime.now().isoformat(),
            'avg_same_size_em': avg_same,
            'avg_diff_size_pixel_em': avg_diff_pixel,
            'avg_diff_size_size_em': avg_diff_size,
            'results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Same vs Different size
        axes[0].bar(['Same-size\n(Pixel EM)', 'Size-change\n(Pixel EM)', 'Size-change\n(Size EM)', 'Size-change\n(Both EM)'],
                    [avg_same, avg_diff_pixel, avg_diff_size, avg_diff_both],
                    color=['#2ecc71', '#3498db', '#e67e22', '#9b59b6'], edgecolor='black')
        axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Dynamic Size Cropper Performance')

        # 2. Per-task pixel EM
        for r in same_results:
            axes[1].bar(r['task_id'][:6], r['pixel_em']*100, color='#2ecc71', alpha=0.7)
        for r in diff_results:
            axes[1].bar(r['task_id'][:6], r['pixel_em']*100, color='#e74c3c', alpha=0.7)
        axes[1].set_ylabel('Pixel EM (%)'); axes[1].set_title('Per-Task (Green=same, Red=size-change)')
        axes[1].tick_params(axis='x', rotation=90, labelsize=6)

        # 3. Size prediction accuracy
        if diff_results:
            axes[2].scatter([r['size_em']*100 for r in diff_results],
                          [r['pixel_em']*100 for r in diff_results], s=80, c='#e74c3c')
            axes[2].set_xlabel('Size Prediction (%)'); axes[2].set_ylabel('Pixel EM (%)')
            axes[2].set_title('Size Pred vs Pixel Accuracy\n(Size-changing tasks)')

        plt.suptitle('Phase 108: Dynamic Size Cropper on Real ARC', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase108_dynamic_size.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 108 complete!")

if __name__ == '__main__':
    main()
