"""
Phase 152: Sleep-Phase Consolidation  -  NCA Dreams

Phase 149 proved that entropy minimization sharpens outputs (+3.51%).
This phase tests the ultimate extension: can NCA "dream"?

Feed Foundation NCA random noise as input (no target), apply ONLY
entropy minimization loss via TTCT. If the NCA autonomously organizes
noise into structured, ARC-like patterns, it demonstrates generative
capability  -  creating order from chaos without supervision.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
PAD_SIZE = 32
N_COLORS = 11
MAX_GRID = 30


# ================================================================
# ARC utilities (from phase149)
# ================================================================
def load_arc():
    path = os.path.join(DATA_DIR, "arc_training.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def grid_to_tensor(grid, ps=PAD_SIZE):
    h, w = len(grid), len(grid[0])
    t = torch.zeros(N_COLORS, ps, ps)
    for y in range(h):
        for x in range(w):
            t[grid[y][x], y, x] = 1.0
    t[10, :h, :w] = 1.0
    return t

def prep_tasks(data, max_tasks=400):
    tasks = []
    tids = list(data.keys()); random.shuffle(tids)
    for tid in tids[:max_tasks]:
        task = data[tid]
        if 'train' not in task or 'test' not in task: continue
        valid = True
        for p in task['train'] + task['test']:
            if len(p['input']) > MAX_GRID or len(p['input'][0]) > MAX_GRID: valid = False
            if len(p['output']) > MAX_GRID or len(p['output'][0]) > MAX_GRID: valid = False
        if not valid: continue
        di = [grid_to_tensor(p['input']) for p in task['train']]
        do = [grid_to_tensor(p['output']) for p in task['train']]
        for tp in task['test']:
            tasks.append({
                'task_id': tid, 'demo_inputs': di, 'demo_outputs': do,
                'test_input': grid_to_tensor(tp['input']),
                'test_output': grid_to_tensor(tp['output']),
                'out_h': len(tp['output']), 'out_w': len(tp['output'][0])})
    return tasks


# ================================================================
# Foundation Model (reuse ChronosSystem from phase149)
# ================================================================
class DreamEncoder(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.pair_enc = nn.Sequential(
            nn.Conv2d(N_COLORS * 2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, embed_dim))
    def forward(self, di, do):
        pairs = torch.stack([torch.cat([i,o], 0) for i,o in zip(di, do)])
        return self.pair_enc(pairs).mean(0)


class DreamNCA(nn.Module):
    def __init__(self, hidden_ch=64, embed_dim=64, latent_ch=32):
        super().__init__()
        self.latent_ch = latent_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(N_COLORS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, latent_ch, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim + 1, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1))
        self.tau_gate = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim + 1, latent_ch, 1), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, N_COLORS, 1))

    def forward(self, x, task_embed, n_steps=5):
        B, _, H, W = x.shape
        state = self.encoder(x)
        te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)
        for t in range(n_steps):
            clock = torch.full((B, 1, H, W), t / max(n_steps-1, 1), device=x.device)
            ctx = torch.cat([state, te, clock], dim=1)
            delta = self.update(ctx)
            beta = self.tau_gate(ctx)
            state = beta * state + (1-beta) * delta
        return self.decoder(state)


class DreamSystem(nn.Module):
    def __init__(self, embed_dim=64, hidden_ch=64, latent_ch=32):
        super().__init__()
        self.task_encoder = DreamEncoder(embed_dim=embed_dim)
        self.nca = DreamNCA(hidden_ch=hidden_ch, embed_dim=embed_dim, latent_ch=latent_ch)
    def forward(self, di, do, ti, n_steps=5):
        te = self.task_encoder(di, do)
        return self.nca(ti.unsqueeze(0), te, n_steps)


# ================================================================
# Dream Protocol
# ================================================================
def entropy_of_logits(logits):
    probs = F.softmax(logits[:, :10], dim=1)
    ent = -(probs * (probs + 1e-8).log()).sum(dim=1)
    return ent.mean()


def dream(model, task_embed, n_steps=10, dream_steps=300, lr=0.01):
    """Feed noise → entropy minimize → observe crystallization."""
    # Random noise input (uniform across all color channels)
    noise_input = torch.randn(1, N_COLORS, PAD_SIZE, PAD_SIZE, device=DEVICE) * 0.5
    noise_input[:, 10, :, :] = 1.0  # mask channel active

    te = task_embed.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([te], lr=lr)

    entropy_history = []
    snapshots = []

    for step in range(dream_steps):
        opt.zero_grad()
        logits = model.nca(noise_input, te, n_steps)
        ent = entropy_of_logits(logits)
        ent.backward()
        opt.step()

        entropy_history.append(ent.item())

        if step in [0, dream_steps//4, dream_steps//2, 3*dream_steps//4, dream_steps-1]:
            with torch.no_grad():
                pred = logits[0, :10].argmax(0).cpu().numpy()
                snapshots.append({
                    'step': step,
                    'entropy': ent.item(),
                    'grid': pred[:16, :16].tolist(),
                    'n_unique_colors': int(len(np.unique(pred[:16, :16]))),
                    'spatial_variance': float(np.var(pred[:16, :16].astype(float)))
                })

    return entropy_history, snapshots, te.detach()


def dream_from_random_embed(model, n_steps=10, dream_steps=300, lr=0.01):
    """Dream with fully random embedding (no task context)."""
    te = torch.randn(64, device=DEVICE) * 0.1
    te = te.requires_grad_(True)
    opt = torch.optim.Adam([te], lr=lr)

    noise_input = torch.randn(1, N_COLORS, PAD_SIZE, PAD_SIZE, device=DEVICE) * 0.5
    noise_input[:, 10, :, :] = 1.0

    entropy_history = []
    snapshots = []

    for step in range(dream_steps):
        opt.zero_grad()
        logits = model.nca(noise_input, te, n_steps)
        ent = entropy_of_logits(logits)
        ent.backward()
        opt.step()
        entropy_history.append(ent.item())

        if step in [0, dream_steps//4, dream_steps//2, 3*dream_steps//4, dream_steps-1]:
            with torch.no_grad():
                pred = logits[0, :10].argmax(0).cpu().numpy()
                snapshots.append({
                    'step': step, 'entropy': ent.item(),
                    'grid': pred[:16, :16].tolist(),
                    'n_unique_colors': int(len(np.unique(pred[:16, :16]))),
                    'spatial_variance': float(np.var(pred[:16, :16].astype(float)))
                })

    return entropy_history, snapshots


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 152: Sleep-Phase Consolidation  -  NCA Dreams")
    print(f"  Does entropy minimization create order from chaos?")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load ARC & train foundation
    print("\n[Step 1] Training Foundation Model...")
    arc = load_arc()
    all_tasks = prep_tasks(arc, max_tasks=400)
    random.shuffle(all_tasks)
    split = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split]
    test_tasks = all_tasks[split:]

    model = DreamSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    for epoch in range(60):
        model.train()
        random.shuffle(train_tasks)
        eloss = 0; n = 0
        for item in train_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do_ = [d.to(DEVICE) for d in item['demo_outputs']]
            ti = item['test_input'].to(DEVICE)
            to_ = item['test_output'].to(DEVICE)
            logits = model(di, do_, ti, n_steps=5)
            target = to_[:10].argmax(0).unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        sched.step()
        if (epoch+1) % 20 == 0:
            print(f"    Epoch {epoch+1}/60: loss={eloss/n:.4f}")

    model.eval()

    # Dream Experiment 1: Task-conditioned dreams
    print("\n[Step 2] Dreaming with task embeddings...")
    dream_results = {}
    for i in range(min(5, len(test_tasks))):
        item = test_tasks[i]
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do_ = [d.to(DEVICE) for d in item['demo_outputs']]
        with torch.no_grad():
            te = model.task_encoder(di, do_)

        ent_hist, snaps, _ = dream(model, te, n_steps=10, dream_steps=300)
        dream_results[f"task_{i}"] = {
            'entropy_history': ent_hist,
            'snapshots': snaps,
            'task_id': item['task_id'],
            'entropy_drop': ent_hist[0] - ent_hist[-1],
            'final_entropy': ent_hist[-1],
            'initial_entropy': ent_hist[0],
            'final_unique_colors': snaps[-1]['n_unique_colors']
        }
        print(f"  Task {i}: entropy {ent_hist[0]:.3f} -> {ent_hist[-1]:.3f} "
              f"(drop={ent_hist[0]-ent_hist[-1]:.3f}), "
              f"colors={snaps[-1]['n_unique_colors']}")

    # Dream Experiment 2: Random embedding (no task)
    print("\n[Step 3] Dreaming with random embeddings (no task context)...")
    random_dreams = {}
    for i in range(3):
        torch.manual_seed(SEED + i)
        ent_hist, snaps = dream_from_random_embed(model, n_steps=10, dream_steps=300)
        random_dreams[f"random_{i}"] = {
            'entropy_history': ent_hist,
            'snapshots': snaps,
            'entropy_drop': ent_hist[0] - ent_hist[-1],
            'final_entropy': ent_hist[-1],
            'final_unique_colors': snaps[-1]['n_unique_colors']
        }
        print(f"  Random {i}: entropy {ent_hist[0]:.3f} -> {ent_hist[-1]:.3f} "
              f"(drop={ent_hist[0]-ent_hist[-1]:.3f})")

    # Analysis
    avg_ent_drop_task = np.mean([d['entropy_drop'] for d in dream_results.values()])
    avg_ent_drop_rand = np.mean([d['entropy_drop'] for d in random_dreams.values()])
    avg_final_colors_task = np.mean([d['final_unique_colors'] for d in dream_results.values()])
    avg_final_colors_rand = np.mean([d['final_unique_colors'] for d in random_dreams.values()])

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 152 Complete ({elapsed:.0f}s)")
    print(f"  Task-conditioned dreams: avg entropy drop = {avg_ent_drop_task:.3f}")
    print(f"  Random dreams: avg entropy drop = {avg_ent_drop_rand:.3f}")
    print(f"  Task dreams: avg final colors = {avg_final_colors_task:.1f}")
    print(f"  Random dreams: avg final colors = {avg_final_colors_rand:.1f}")
    print(f"  Dreaming creates order: {avg_ent_drop_task > 0.1}")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase152_sleep_dreams.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 152: Sleep-Phase Consolidation',
            'timestamp': datetime.now().isoformat(),
            'dream_results': dream_results,
            'random_dreams': random_dreams,
            'avg_entropy_drop_task': avg_ent_drop_task,
            'avg_entropy_drop_random': avg_ent_drop_rand,
            'avg_final_colors_task': avg_final_colors_task,
            'avg_final_colors_random': avg_final_colors_rand,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Entropy trajectories
        for name, d in dream_results.items():
            axes[0].plot(d['entropy_history'], alpha=0.5, label=name)
        for name, d in random_dreams.items():
            axes[0].plot(d['entropy_history'], '--', alpha=0.5, label=name)
        axes[0].set_xlabel('Dream Step')
        axes[0].set_ylabel('Entropy')
        axes[0].set_title('Entropy During Dreaming', fontweight='bold', fontsize=10)
        axes[0].legend(fontsize=6, ncol=2)

        # Final state visualization (dream grids as heatmaps)
        dream_key = list(dream_results.keys())[0]
        final_grid = np.array(dream_results[dream_key]['snapshots'][-1]['grid'])
        im = axes[1].imshow(final_grid, cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
        axes[1].set_title(f'Dream Output (Task-conditioned)', fontweight='bold', fontsize=10)
        axes[1].set_xlabel(f"Colors: {dream_results[dream_key]['final_unique_colors']}")

        # Comparison bar
        categories = ['Task\nDreams', 'Random\nDreams']
        ent_drops = [avg_ent_drop_task, avg_ent_drop_rand]
        bars = axes[2].bar(categories, ent_drops,
                          color=['#2ecc71', '#e74c3c'], alpha=0.85, edgecolor='black')
        for bar, val in zip(bars, ent_drops):
            axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{val:.3f}', ha='center', fontweight='bold')
        axes[2].set_ylabel('Entropy Drop')
        axes[2].set_title('Order from Chaos', fontweight='bold', fontsize=10)

        fig.suptitle('Phase 152: NCA Dreams  -  Entropy Minimization Creates Order',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase152_sleep_dreams.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return dream_results


if __name__ == '__main__':
    main()
