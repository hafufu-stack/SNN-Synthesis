"""
Phase 103: Compute-Optimal Scaling for ARC

With 432s per task (12h / 100 tasks), scale TTT steps and NBS beams
to find the Pareto frontier of solve rate vs compute.

TTT steps: 15, 100, 500, 1000
NBS beams: K=1, K=7, K=21, K=51

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
MAX_GS = 30; MAX_TASKS = 50

# L-NCA
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

def freq_remap(grid):
    flat = grid.flatten()
    unique, counts = np.unique(flat, return_counts=True)
    order = np.argsort(-counts)
    mapping = {unique[order[i]]: i for i in range(len(unique))}
    return np.vectorize(lambda c: mapping.get(c, c))(grid)

def one_hot(grid, nc=10):
    h, w = grid.shape
    o = np.zeros((nc, h, w), dtype=np.float32)
    for c in range(nc): o[c] = (grid == c).astype(np.float32)
    return o

def pad_to(grid, max_h, max_w, val=0):
    h, w = grid.shape
    out = np.full((max_h, max_w), val, dtype=grid.dtype)
    out[:h, :w] = grid
    return out


def load_experts():
    """Load trained experts from Phase 102."""
    experts = {}
    if not os.path.exists(EXPERT_DIR):
        return experts
    for f in os.listdir(EXPERT_DIR):
        if not f.endswith('.pt'):
            continue
        tid = f.replace('.pt', '')
        data = torch.load(os.path.join(EXPERT_DIR, f), map_location=DEVICE, weights_only=False)
        model = LiquidNCA(NC, HC)
        model.load_state_dict(data['model'])
        model.eval()
        experts[tid] = {'model': model, 'ctx': data['ctx']}
    return experts


def ttt_solve(model, ctx, demos, test_in, ttt_steps, n_beams):
    """TTT with variable steps and NBS with variable K."""
    demo_ins = [freq_remap(np.array(d['input'])) for d in demos]
    demo_outs = [freq_remap(np.array(d['output'])) for d in demos]

    # Check size compatibility
    for di, do in zip(demo_ins, demo_outs):
        if di.shape != do.shape:
            return None

    test_in_r = freq_remap(np.array(test_in))

    # Determine padding size
    all_grids = demo_ins + demo_outs + [test_in_r]
    max_h = max(g.shape[0] for g in all_grids)
    max_w = max(g.shape[1] for g in all_grids)
    if max_h > MAX_GS or max_w > MAX_GS:
        return None

    # TTT: optimize context
    best_pred = None
    best_loss = float('inf')

    for beam_idx in range(n_beams):
        m = copy.deepcopy(model)
        ttt_ctx = nn.Parameter(ctx.clone() + torch.randn_like(ctx) * 0.05 * beam_idx)
        opt = torch.optim.Adam([ttt_ctx], lr=0.1)

        for step in range(ttt_steps):
            opt.zero_grad()
            total_loss = 0
            for di, do in zip(demo_ins, demo_outs):
                di_t = torch.tensor(one_hot(pad_to(di, max_h, max_w))).unsqueeze(0)
                do_t = torch.tensor(pad_to(do, max_h, max_w)).unsqueeze(0)
                pred = m(di_t, ctx=ttt_ctx.expand(1, -1, -1, -1))
                total_loss += F.cross_entropy(pred, do_t)
            avg_loss = total_loss / len(demos)
            avg_loss.backward()
            opt.step()

            # Auto-T: early stop if loss is near zero
            if avg_loss.item() < 0.01:
                break

        # Evaluate on demos
        m.eval()
        with torch.no_grad():
            final_loss = 0
            for di, do in zip(demo_ins, demo_outs):
                di_t = torch.tensor(one_hot(pad_to(di, max_h, max_w))).unsqueeze(0)
                do_t = torch.tensor(pad_to(do, max_h, max_w)).unsqueeze(0)
                pred = m(di_t, ctx=ttt_ctx.expand(1, -1, -1, -1))
                final_loss += F.cross_entropy(pred, do_t).item()
            final_loss /= len(demos)

        if final_loss < best_loss:
            best_loss = final_loss
            with torch.no_grad():
                ti_t = torch.tensor(one_hot(pad_to(test_in_r, max_h, max_w))).unsqueeze(0)
                best_pred = m(ti_t, ctx=ttt_ctx.expand(1, -1, -1, -1))
                best_pred = best_pred.argmax(1).squeeze(0).numpy()

        del m, ttt_ctx, opt
        gc.collect()

    return best_pred


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 103: Compute-Optimal Scaling for ARC")
    print("=" * 70)

    # Load experts
    print("  Loading Phase 102 expert library...")
    experts = load_experts()
    print(f"  Loaded {len(experts)} experts")
    if not experts:
        print("  ERROR: No experts found. Run Phase 102 first.")
        return

    # Load ARC tasks (only matching experts)
    files = sorted([f for f in os.listdir(ARC_DIR) if f.endswith('.json')])
    tasks = {}
    for fname in files[:MAX_TASKS * 3]:  # Load extra to find matches
        tid = fname.replace('.json', '')
        if tid in experts:
            with open(os.path.join(ARC_DIR, fname), 'r', encoding='utf-8') as f:
                tasks[tid] = json.load(f)
        if len(tasks) >= MAX_TASKS:
            break
    print(f"  Matched {len(tasks)} tasks with experts")

    # Scaling experiments
    ttt_steps_list = [15, 100, 500]
    k_beams_list = [1, 7, 21]
    all_results = {}

    for ttt_steps in ttt_steps_list:
        for k_beams in k_beams_list:
            config = f"TTT={ttt_steps}_K={k_beams}"
            print(f"\n  === {config} ===")

            solved = 0; tested = 0; total_time = 0
            task_results = []

            for tid, task in list(tasks.items())[:30]:
                if tid not in experts:
                    continue
                demos = task['train']
                tests = task['test']

                t0 = time.time()
                n_correct = 0
                for test_pair in tests:
                    test_in = test_pair['input']
                    test_out = np.array(test_pair['output'])

                    pred = ttt_solve(
                        experts[tid]['model'], experts[tid]['ctx'],
                        demos, test_in, ttt_steps, k_beams
                    )

                    if pred is not None:
                        # Crop prediction to match output size
                        oh, ow = test_out.shape
                        pred_crop = pred[:oh, :ow]
                        if np.array_equal(pred_crop, test_out):
                            n_correct += 1

                elapsed = time.time() - t0
                total_time += elapsed
                tested += 1
                task_solved = (n_correct == len(tests))
                if task_solved:
                    solved += 1

                task_results.append({
                    'task_id': tid,
                    'solved': task_solved,
                    'time_s': elapsed,
                    'n_correct': n_correct,
                    'n_test': len(tests)
                })

            sr = solved / max(tested, 1)
            avg_time = total_time / max(tested, 1)
            print(f"    Solve Rate: {solved}/{tested} = {sr*100:.1f}%  Avg: {avg_time:.1f}s")

            all_results[config] = {
                'ttt_steps': ttt_steps,
                'k_beams': k_beams,
                'solve_rate': sr,
                'solved': solved,
                'tested': tested,
                'avg_time_s': avg_time,
                'tasks': task_results
            }
            gc.collect()

    # Save results
    print(f"\n{'=' * 70}")
    print("  Compute-Optimal Scaling Summary:")
    for config, r in all_results.items():
        print(f"    {config:20s}: {r['solve_rate']*100:5.1f}%  ({r['avg_time_s']:.1f}s/task)")
    print(f"{'=' * 70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase103_compute_scaling.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 103: Compute-Optimal Scaling',
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Solve rate heatmap
        ttt_vals = sorted(set(r['ttt_steps'] for r in all_results.values()))
        k_vals = sorted(set(r['k_beams'] for r in all_results.values()))
        matrix = np.zeros((len(ttt_vals), len(k_vals)))
        for ci, config in enumerate(all_results):
            r = all_results[config]
            ti = ttt_vals.index(r['ttt_steps'])
            ki = k_vals.index(r['k_beams'])
            matrix[ti, ki] = r['solve_rate'] * 100

        im = axes[0].imshow(matrix, cmap='YlOrRd', aspect='auto')
        axes[0].set_xticks(range(len(k_vals)))
        axes[0].set_xticklabels([f'K={k}' for k in k_vals])
        axes[0].set_yticks(range(len(ttt_vals)))
        axes[0].set_yticklabels([f'TTT={t}' for t in ttt_vals])
        for i in range(len(ttt_vals)):
            for j in range(len(k_vals)):
                axes[0].text(j, i, f'{matrix[i, j]:.0f}%', ha='center', va='center', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=axes[0])
        axes[0].set_title('Solve Rate (%) by TTT Steps × NBS Beams')

        # 2. Time vs solve rate
        for config, r in all_results.items():
            axes[1].scatter(r['avg_time_s'], r['solve_rate'] * 100, s=100, zorder=5)
            axes[1].annotate(config, (r['avg_time_s'], r['solve_rate'] * 100), fontsize=7)
        axes[1].set_xlabel('Avg Time per Task (s)')
        axes[1].set_ylabel('Solve Rate (%)')
        axes[1].set_title('Compute-Accuracy Pareto Frontier')
        axes[1].axvline(x=432, color='red', linestyle='--', alpha=0.5, label='432s budget')
        axes[1].legend()

        plt.suptitle('Phase 103: Compute-Optimal Scaling for ARC', fontsize=14)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase103_compute_scaling.png'), dpi=150)
        plt.close()
        print("  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("Phase 103 complete!")

if __name__ == '__main__':
    main()
