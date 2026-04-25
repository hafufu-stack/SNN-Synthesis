"""
Phase 206: The Causal Volume Law - Intelligence Equation

Systematically measure NCA reasoning ability (Test PA) as a function of:
  1. Synapse count K^2 (kernel size: K=1, 3, 5)
  2. Information capacity C (hidden channels: 16, 32, 64)
  3. Computation time T (inference steps: 1, 3, 5)

Fit: I ~ (K^2 * C)^alpha * T^beta  via log-linear regression.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import load_arc_training, prepare_arc_meta_dataset


class ParametricNCA(nn.Module):
    """NCA with configurable kernel size, channels, and steps."""
    def __init__(self, n_colors=11, hidden_ch=64, kernel_size=3, steps=5, embed_dim=32):
        super().__init__()
        self.steps = steps
        self.kernel_size = kernel_size
        pad = kernel_size // 2

        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(n_colors + embed_dim, hidden_ch, kernel_size, padding=pad), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=pad), nn.ReLU(),
        )
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=pad), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
        )
        self.tau = nn.Sequential(nn.Conv2d(hidden_ch, hidden_ch, 1), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size, padding=pad), nn.ReLU(),
            nn.Conv2d(hidden_ch, n_colors, 1),
        )

    def encode_task(self, demo_outputs):
        embeddings = []
        for do in demo_outputs:
            emb = self.demo_encoder(do.unsqueeze(0))
            embeddings.append(emb)
        return torch.stack(embeddings).mean(dim=0)

    def forward(self, x, task_emb):
        B, _, H, W = x.shape
        te = task_emb.view(B, -1, 1, 1).expand(B, -1, H, W)
        inp = torch.cat([x, te], dim=1)
        state = self.encoder(inp)
        for t in range(self.steps):
            delta = self.update(state)
            beta = self.tau(state)
            state = beta * state + (1 - beta) * delta
        return self.decoder(state)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_and_eval(model, train_tasks, test_tasks, n_epochs=80):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_pa, best_em = 0, 0
    for epoch in range(n_epochs):
        model.train()
        random.shuffle(train_tasks)
        for item in train_tasks[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            logits = model(ti, emb)
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            model.eval()
            tpa, tem = 0, 0
            with torch.no_grad():
                for item in test_tasks:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = model.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    logits = model(ti, emb)
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
                    tem += float((pred == gt[:oh, :ow]).all().item())
            pa = tpa / len(test_tasks)
            em = tem / len(test_tasks)
            if pa > best_pa:
                best_pa = pa
                best_em = em
    return best_pa, best_em


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 206: The Causal Volume Law")
    print(f"  Sweep K, C, T -> fit I ~ (K^2*C)^a * T^b")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    K_vals = [1, 3, 5]
    C_vals = [16, 32, 64]
    T_vals = [1, 3, 5]

    results = []
    for K in K_vals:
        for C in C_vals:
            for T in T_vals:
                print(f"\n  [K={K}, C={C}, T={T}]")
                torch.manual_seed(SEED)
                model = ParametricNCA(11, C, K, T, 32).to(DEVICE)
                params = model.count_params()
                print(f"    Params: {params:,}")
                pa, em = train_and_eval(model, train, test, n_epochs=80)
                synapse = K * K * C
                entry = {'K': K, 'C': C, 'T': T, 'K2': K*K, 'synapse': synapse,
                         'params': params, 'pa': pa, 'em': em}
                results.append(entry)
                print(f"    PA={pa*100:.1f}%, EM={em*100:.1f}%, Synapse(K2*C)={synapse}")
                del model; gc.collect()
                if DEVICE == "cuda": torch.cuda.empty_cache()

    # Fit power law: log(PA) = alpha * log(K^2 * C) + beta * log(T) + const
    # Use only entries where PA > 0.3 (meaningful)
    valid = [r for r in results if r['pa'] > 0.3 and r['T'] > 0]
    if len(valid) >= 5:
        log_synapse = np.array([np.log(r['synapse'] + 1) for r in valid])
        log_T = np.array([np.log(r['T'] + 1) for r in valid])
        log_PA = np.array([np.log(r['pa']) for r in valid])

        # log(PA) = alpha * log(synapse) + beta * log(T) + c
        A = np.column_stack([log_synapse, log_T, np.ones(len(valid))])
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, log_PA, rcond=None)
        alpha, beta, const = coeffs
        r2 = 1 - np.sum((log_PA - A @ coeffs)**2) / np.sum((log_PA - log_PA.mean())**2)

        print(f"\n{'='*70}")
        print(f"  THE CAUSAL VOLUME LAW:")
        print(f"  I ~ (K^2 * C)^{alpha:.3f} * T^{beta:.3f}")
        print(f"  alpha = {alpha:.4f} (spatial synapse exponent)")
        print(f"  beta  = {beta:.4f} (temporal computation exponent)")
        print(f"  R^2   = {r2:.4f}")
        print(f"{'='*70}")
    else:
        alpha, beta, const, r2 = 0, 0, 0, 0

    elapsed = time.time() - t0
    print(f"\nPhase 206 Complete ({elapsed:.0f}s)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase206_causal_volume.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'equation': {'alpha': float(alpha), 'beta': float(beta),
                         'const': float(const), 'r2': float(r2)},
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(18, 5))

        # 3D scatter: K^2*C vs T vs PA
        ax1 = fig.add_subplot(131, projection='3d')
        for r in results:
            color = '#e74c3c' if r['K'] == 1 else '#3498db' if r['K'] == 3 else '#2ecc71'
            ax1.scatter(np.log2(r['synapse']+1), r['T'], r['pa']*100,
                       s=80, c=color, edgecolor='black', alpha=0.8)
        ax1.set_xlabel('log2(K^2*C)'); ax1.set_ylabel('T (steps)')
        ax1.set_zlabel('Test PA (%)')
        ax1.set_title('Causal Volume Law\n3D Landscape', fontweight='bold', fontsize=10)

        # Heatmap: C vs T for K=3
        ax2 = fig.add_subplot(132)
        k3_data = [r for r in results if r['K'] == 3]
        if k3_data:
            pa_grid = np.zeros((len(C_vals), len(T_vals)))
            for r in k3_data:
                ci = C_vals.index(r['C'])
                ti = T_vals.index(r['T'])
                pa_grid[ci, ti] = r['pa'] * 100
            im = ax2.imshow(pa_grid, cmap='viridis', aspect='auto', origin='lower')
            ax2.set_xticks(range(len(T_vals))); ax2.set_xticklabels(T_vals)
            ax2.set_yticks(range(len(C_vals))); ax2.set_yticklabels(C_vals)
            ax2.set_xlabel('T (steps)'); ax2.set_ylabel('C (channels)')
            ax2.set_title('PA Heatmap (K=3)', fontweight='bold')
            for i in range(len(C_vals)):
                for j in range(len(T_vals)):
                    ax2.text(j, i, f'{pa_grid[i,j]:.1f}', ha='center', va='center',
                            color='white' if pa_grid[i,j] < 55 else 'black', fontsize=9)
            plt.colorbar(im, ax=ax2, label='PA (%)')

        # Equation fit: predicted vs actual
        ax3 = fig.add_subplot(133)
        if valid:
            actual = [r['pa']*100 for r in valid]
            predicted = [np.exp(alpha*np.log(r['synapse']+1) + beta*np.log(r['T']+1) + const)*100
                        for r in valid]
            ax3.scatter(actual, predicted, s=80, c='#3498db', edgecolor='black', zorder=5)
            mn, mx = min(min(actual), min(predicted)), max(max(actual), max(predicted))
            ax3.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Perfect fit')
            ax3.set_xlabel('Actual PA (%)'); ax3.set_ylabel('Predicted PA (%)')
            ax3.set_title(f'Fit: I~(K2C)^{alpha:.2f}*T^{beta:.2f}\nR2={r2:.3f}',
                         fontweight='bold', fontsize=10)
            ax3.legend(); ax3.grid(True, alpha=0.3)

        fig.suptitle('Phase 206: The Causal Volume Law', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.82, bottom=0.12, left=0.06, right=0.96, wspace=0.35)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase206_causal_volume.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'alpha': float(alpha), 'beta': float(beta), 'r2': float(r2)}


if __name__ == '__main__':
    main()
