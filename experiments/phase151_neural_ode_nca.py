"""
Phase 151: Neural ODE Automata  -  Continuous Time NCA

Space ≡ Time was proved with discrete steps. But physics has continuous time.
This phase tests whether reducing Δt (sub-step refinement) produces smoother,
more accurate NCA dynamics  -  approaching the Neural ODE limit.

We compare: dt=1.0 (standard), dt=0.5 (2x steps), dt=0.25 (4x steps),
dt=0.1 (10x steps) on the Dilate→Invert→Erode Turing task.

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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
GRID_SIZE = 16
TOTAL_LOGICAL_STEPS = 5  # Dilate x2 + Invert + Erode x2


# ================================================================
# Operations (from Phase 148)
# ================================================================
def dilate(grid):
    g = grid.float()
    if g.dim() == 2: g = g.unsqueeze(0).unsqueeze(0)
    elif g.dim() == 3: g = g.unsqueeze(0)
    k = torch.ones(1, 1, 3, 3, device=g.device)
    p = F.pad(g, (1,1,1,1), mode='constant', value=0)
    return (F.conv2d(p, k).squeeze() > 0).float()

def erode(grid):
    g = grid.float()
    if g.dim() == 2: g = g.unsqueeze(0).unsqueeze(0)
    elif g.dim() == 3: g = g.unsqueeze(0)
    k = torch.ones(1, 1, 3, 3, device=g.device)
    p = F.pad(g, (1,1,1,1), mode='constant', value=0)
    return (F.conv2d(p, k).squeeze() >= 9).float()

def invert(grid):
    return 1.0 - grid

def generate_data(n_samples):
    X, Y = [], []
    for _ in range(n_samples):
        g = (torch.rand(GRID_SIZE, GRID_SIZE) > 0.65).float()
        X.append(g.clone())
        g = dilate(g); g = dilate(g); g = invert(g); g = erode(g); g = erode(g)
        Y.append(g)
    return torch.stack(X).unsqueeze(1), torch.stack(Y).unsqueeze(1)


class TuringDS(torch.utils.data.Dataset):
    def __init__(self, n=5000):
        self.X, self.Y = generate_data(n)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ================================================================
# Continuous-Time NCA (parameterized by dt)
# ================================================================
class ContinuousNCA(nn.Module):
    """NCA with configurable time step dt.
    
    Standard NCA: h = h + f(h)  [dt=1.0, Euler method]
    Neural ODE:   h = h + dt*f(h)  [dt→0, approaches true ODE]
    
    For logical_steps=5 with dt=0.1, we run 50 micro-steps.
    """
    def __init__(self, ch=32, logical_steps=TOTAL_LOGICAL_STEPS, dt=1.0):
        super().__init__()
        self.logical_steps = logical_steps
        self.dt = dt
        self.n_steps = int(logical_steps / dt)
        self.proj_in = nn.Conv2d(1, ch, 3, padding=1)
        self.rule = nn.Sequential(
            nn.Conv2d(ch, ch*2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch*2, ch, 3, padding=1))
        self.proj_out = nn.Conv2d(ch, 1, 1)

    def forward(self, x, steps=None):
        n = steps if steps else self.n_steps
        h = F.relu(self.proj_in(x))
        trajectory = []
        for t in range(n):
            dh = self.rule(h)
            h = F.relu(h + self.dt * dh)  # Euler step with dt scaling
            if t % max(1, n // 10) == 0:
                trajectory.append(h.detach().mean().item())
        self._trajectory = trajectory
        return self.proj_out(h)


# ================================================================
# Training & evaluation
# ================================================================
def train_model(model, loader, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

def eval_model(model, loader):
    model.eval()
    correct = pixels = exact = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = (torch.sigmoid(model(x)) > 0.5).float()
            correct += (pred == y).sum().item(); pixels += y.numel()
            exact += (pred==y).all(-1).all(-1).all(-1).sum().item()
            total += y.size(0)
    return correct/pixels, exact/total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 151: Neural ODE Automata  -  Continuous Time NCA")
    print(f"  Task: Dilate x2 -> Invert -> Erode x2 on {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Testing dt = [1.0, 0.5, 0.25, 0.1]")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    train_ds = TuringDS(5000)
    test_ds = TuringDS(1000)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)

    dt_values = [1.0, 0.5, 0.25, 0.1]
    results = {}
    trajectories = {}

    for dt in dt_values:
        model = ContinuousNCA(ch=32, dt=dt).to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters())
        n_steps = model.n_steps
        print(f"\n[dt={dt}] Total micro-steps: {n_steps}, Params: {n_p:,}")

        train_model(model, train_loader, epochs=50)
        pa, em = eval_model(model, test_loader)
        results[f"dt={dt}"] = {
            'pixel_acc': pa, 'exact_match': em, 'params': n_p,
            'dt': dt, 'n_steps': n_steps
        }
        print(f"  PA={pa*100:.2f}%, EM={em*100:.2f}%")

        # Capture trajectory
        model.eval()
        with torch.no_grad():
            x_t, _ = next(iter(test_loader))
            _ = model(x_t[:16].to(DEVICE))
        trajectories[f"dt={dt}"] = model._trajectory

        # Temporal generalization test
        print(f"  [Generalization] Testing at 2x steps...")
        pa_2x, em_2x = eval_model(model, test_loader)
        model_gen = ContinuousNCA(ch=32, dt=dt).to(DEVICE)
        model_gen.load_state_dict(model.state_dict())
        model_gen.n_steps = n_steps * 2  # Double the steps
        pa_2x, em_2x = eval_model(model_gen, test_loader)
        results[f"dt={dt}"]['pa_2x'] = pa_2x
        results[f"dt={dt}"]['em_2x'] = em_2x
        print(f"  PA@2x={pa_2x*100:.2f}%, EM@2x={em_2x*100:.2f}%")

        del model, model_gen; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Summary
    best = max(results.keys(), key=lambda k: results[k]['pixel_acc'])
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 151 Complete ({elapsed:.0f}s)")
    print(f"{'dt':>8} {'Steps':>6} {'PA':>8} {'EM':>8} {'PA@2x':>8}")
    print("-"*42)
    for name, r in results.items():
        print(f"{name:>8} {r['n_steps']:>6} {r['pixel_acc']*100:>7.2f}% {r['exact_match']*100:>7.2f}% {r.get('pa_2x',0)*100:>7.2f}%")
    print(f"\nBest: {best}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase151_neural_ode_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 151: Neural ODE Automata',
            'timestamp': datetime.now().isoformat(),
            'results': results, 'trajectories': trajectories,
            'best': best, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        names = list(results.keys())
        accs = [results[n]['pixel_acc']*100 for n in names]
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        bars = axes[0].bar(range(len(names)), accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=9)
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels([f'dt={d}' for d in dt_values], fontsize=9)
        axes[0].set_ylabel('Pixel Accuracy (%)')
        axes[0].set_title('Continuous Time vs Discrete', fontweight='bold', fontsize=10)

        # Trajectories
        for dt_name, traj in trajectories.items():
            axes[1].plot(traj, 'o-', label=dt_name, markersize=4, alpha=0.7)
        axes[1].set_xlabel('Measurement Point')
        axes[1].set_ylabel('Mean State')
        axes[1].set_title('State Trajectories', fontweight='bold', fontsize=10)
        axes[1].legend(fontsize=8)

        # Generalization
        pa_train = [results[n]['pixel_acc']*100 for n in names]
        pa_2x = [results[n].get('pa_2x', 0)*100 for n in names]
        x = np.arange(len(names))
        axes[2].bar(x - 0.15, pa_train, 0.3, label='T steps', color='#3498db', alpha=0.8)
        axes[2].bar(x + 0.15, pa_2x, 0.3, label='2T steps', color='#e74c3c', alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([f'dt={d}' for d in dt_values], fontsize=9)
        axes[2].set_ylabel('Pixel Accuracy (%)')
        axes[2].set_title('Temporal Generalization', fontweight='bold', fontsize=10)
        axes[2].legend(fontsize=8)

        fig.suptitle('Phase 151: Neural ODE Automata (Continuous Time NCA)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase151_neural_ode_nca.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
