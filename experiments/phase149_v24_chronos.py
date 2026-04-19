"""
Phase 149: v24 Chronos Agent — External Clock + Soft Crystallization

Upgrades v23 (best agent: 83.5% PA) with:
  1. External Clock (t/T) in NCA loop — proven powerful on grid tasks
  2. Entropy Minimization during TTCT — soft-crystallize outputs to
     pixel-perfect 0/1 without VQ's gradient destruction

Goal: Break through the Exact Match barrier.

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
# ARC utilities (from phase123)
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

def tensor_to_grid(t, h=None, w=None):
    pred = t[:10].argmax(dim=0)
    if h and w: return pred[:h, :w].cpu().numpy().tolist()
    mask = t[10]
    hh = max(1, int(mask.sum(1).gt(0).sum().item()))
    ww = max(1, int(mask.sum(0).gt(0).sum().item()))
    return pred[:hh, :ww].cpu().numpy().tolist()

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
# v24 Foundation Model (with External Clock)
# ================================================================
class ChronosEncoder(nn.Module):
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


class ChronosLatentNCA(nn.Module):
    """Latent NCA with External Clock (t/T) channel."""
    def __init__(self, hidden_ch=64, embed_dim=64, latent_ch=32):
        super().__init__()
        self.latent_ch = latent_ch
        self.lambda_liquid = 0.1
        self.encoder = nn.Sequential(
            nn.Conv2d(N_COLORS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, latent_ch, 3, padding=1), nn.ReLU())
        # +1 for clock channel
        self.update = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim + 1, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1))
        self.tau_gate = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim + 1, latent_ch, 1), nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, N_COLORS, 1))

    def forward(self, x, task_embed, n_steps=5, return_liquid=False):
        B, _, H, W = x.shape
        state = self.encoder(x)
        te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)
        liq = 0
        for t in range(n_steps):
            # External clock channel
            clock = torch.full((B, 1, H, W), t / max(n_steps-1, 1), device=x.device)
            ctx = torch.cat([state, te, clock], dim=1)
            delta = self.update(ctx)
            beta = self.tau_gate(ctx)
            if return_liquid: liq += ((beta - 0.5)**2).mean()
            state = beta * state + (1-beta) * delta
        logits = self.decoder(state)
        if return_liquid: return logits, liq / n_steps
        return logits


class ChronosSystem(nn.Module):
    def __init__(self, embed_dim=64, hidden_ch=64, latent_ch=32):
        super().__init__()
        self.task_encoder = ChronosEncoder(embed_dim=embed_dim)
        self.latent_nca = ChronosLatentNCA(hidden_ch=hidden_ch, embed_dim=embed_dim, latent_ch=latent_ch)
    def forward(self, di, do, ti, n_steps=5, return_liquid=False):
        te = self.task_encoder(di, do)
        return self.latent_nca(ti.unsqueeze(0), te, n_steps, return_liquid)


# ================================================================
# TTCT with Entropy Minimization (Soft Crystallization)
# ================================================================
def entropy_loss(logits):
    """Entropy of softmax output — minimize to sharpen predictions."""
    probs = F.softmax(logits[:, :10], dim=1)  # (B, 10, H, W)
    ent = -(probs * (probs + 1e-8).log()).sum(dim=1)  # (B, H, W)
    return ent.mean()


def ttct_chronos(model, di, do, n_steps=5, ttct_steps=200, lr=0.01, ent_weight=0.1):
    """Test-Time Context Tuning with entropy minimization."""
    te = model.task_encoder(di, do).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([te], lr=lr)
    best_loss = float('inf'); best_te = te.data.clone()

    for step in range(ttct_steps):
        opt.zero_grad()
        total_loss = 0
        for inp, out in zip(di, do):
            logits = model.latent_nca(inp.unsqueeze(0), te, n_steps)
            target = out[:10].argmax(0).unsqueeze(0)
            demo_loss = F.cross_entropy(logits, target)
            ent = entropy_loss(logits)
            total_loss += demo_loss + ent_weight * ent

        total_loss /= len(di)
        total_loss.backward()
        opt.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_te = te.data.clone()

    return best_te, best_loss


def nbs_vote(model, test_in, te, K=11, n_steps=5, noise_sigma=0.1):
    """Noise Beam Search with majority voting."""
    votes = torch.zeros(N_COLORS, PAD_SIZE, PAD_SIZE, device=DEVICE)
    for k in range(K):
        state = model.latent_nca.encoder(test_in.unsqueeze(0))
        te_exp = te.view(1,-1,1,1).expand(1,-1,PAD_SIZE,PAD_SIZE)
        for t in range(n_steps):
            clock = torch.full((1,1,PAD_SIZE,PAD_SIZE), t/max(n_steps-1,1), device=DEVICE)
            ctx = torch.cat([state, te_exp, clock], dim=1)
            delta = model.latent_nca.update(ctx)
            beta = model.latent_nca.tau_gate(ctx)
            if noise_sigma > 0 and k > 0:
                noise = torch.randn_like(beta) * noise_sigma
                beta = torch.sigmoid(torch.logit(beta.clamp(1e-6, 1-1e-6)) + noise)
            state = beta * state + (1-beta) * delta
        logits = model.latent_nca.decoder(state)
        pred = logits[0,:10].argmax(0)
        for c in range(10):
            votes[c] += (pred == c).float()
    return votes[:10].argmax(0)


# ================================================================
# Main
# ================================================================
def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 149: v24 Chronos Agent")
    print(f"  External Clock + Soft Crystallization")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load ARC
    print("\n[Step 1] Loading ARC data...")
    arc = load_arc()
    all_tasks = prep_tasks(arc, max_tasks=400)
    random.shuffle(all_tasks)
    split = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split]
    test_tasks = all_tasks[split:]
    print(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Build v24
    model = ChronosSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    print("\n[Step 2] Training v24 Chronos Foundation...")
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

            logits, liq = model(di, do_, ti, n_steps=5, return_liquid=True)
            target = to_[:10].argmax(0).unsqueeze(0)
            task_loss = F.cross_entropy(logits, target)
            ent = entropy_loss(logits)
            loss = task_loss + 0.1 * liq + 0.05 * ent

            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item(); n += 1
        sched.step()
        if (epoch+1) % 15 == 0:
            print(f"    Epoch {epoch+1}/60: loss={eloss/n:.4f}")

    # Evaluate
    print("\n[Step 3] Evaluating v24 Chronos Agent...")
    model.eval()

    zs_px = zs_ex = v24_px = v24_ex = 0
    total_px = total_tasks_n = 0
    times = []

    for i, item in enumerate(test_tasks[:50]):
        di = [d.to(DEVICE) for d in item['demo_inputs']]
        do_ = [d.to(DEVICE) for d in item['demo_outputs']]
        ti = item['test_input'].to(DEVICE)
        to_ = item['test_output'].to(DEVICE)
        oh, ow = item['out_h'], item['out_w']
        gt = to_[:10].argmax(0)[:oh, :ow]

        # Zero-shot
        with torch.no_grad():
            logits = model(di, do_, ti, n_steps=5)
            pred_zs = logits[0,:10].argmax(0)[:oh, :ow]
            zs_px += (pred_zs == gt).sum().item()
            zs_ex += (pred_zs == gt).all().item()

        # TTCT + NBS
        t_start = time.time()
        best_te, _ = ttct_chronos(model, di, do_, ttct_steps=100, ent_weight=0.1)
        with torch.no_grad():
            pred_v24 = nbs_vote(model, ti, best_te, K=11, n_steps=5)
        pred_crop = pred_v24[:oh, :ow]
        v24_px += (pred_crop == gt).sum().item()
        v24_ex += (pred_crop == gt).all().item()
        elapsed_task = time.time() - t_start
        times.append(elapsed_task)

        total_px += oh * ow; total_tasks_n += 1
        if (i+1) % 10 == 0:
            print(f"    {i+1}/50: ZS_ex={zs_ex}, v24_ex={v24_ex}, avg_t={np.mean(times):.1f}s")

    zs_pa = zs_px / max(total_px, 1)
    v24_pa = v24_px / max(total_px, 1)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 149: v24 Chronos Agent ({elapsed:.0f}s)")
    print(f"  Tasks: {total_tasks_n}")
    print(f"  Zero-Shot: PA={zs_pa*100:.2f}%, Exact={zs_ex}/{total_tasks_n}")
    print(f"  v24 Agent: PA={v24_pa*100:.2f}%, Exact={v24_ex}/{total_tasks_n}")
    print(f"  Improvement: PA={((v24_pa-zs_pa)*100):+.2f}%, Exact={v24_ex-zs_ex:+d}")
    print(f"  Avg time/task: {np.mean(times):.1f}s")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase149_v24_chronos.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 149: v24 Chronos Agent',
            'timestamp': datetime.now().isoformat(),
            'n_params': n_params,
            'zs_pa': zs_pa, 'zs_exact': zs_ex,
            'v24_pa': v24_pa, 'v24_exact': v24_ex,
            'total_tasks': total_tasks_n,
            'avg_time': float(np.mean(times)),
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # PA comparison
        bars = axes[0].bar([0,1], [zs_pa*100, v24_pa*100],
                          color=['#e74c3c', '#2ecc71'], alpha=0.85, edgecolor='black')
        axes[0].set_xticks([0,1]); axes[0].set_xticklabels(['Zero-Shot', 'v24 Chronos'])
        axes[0].set_ylabel('Pixel Accuracy (%)'); axes[0].set_title('v24 vs Zero-Shot', fontweight='bold')
        for bar, val in zip(bars, [zs_pa*100, v24_pa*100]):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f'{val:.1f}%', ha='center', fontweight='bold')

        # Exact match
        bars = axes[1].bar([0,1], [zs_ex, v24_ex],
                          color=['#e74c3c', '#2ecc71'], alpha=0.85, edgecolor='black')
        axes[1].set_xticks([0,1]); axes[1].set_xticklabels(['Zero-Shot', 'v24 Chronos'])
        axes[1].set_ylabel('Exact Matches'); axes[1].set_title('Exact Match Count', fontweight='bold')
        for bar, val in zip(bars, [zs_ex, v24_ex]):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        str(val), ha='center', fontweight='bold')

        # Time distribution
        axes[2].hist(times, bins=15, color='#3498db', alpha=0.7)
        axes[2].set_xlabel('Time per task (s)'); axes[2].set_ylabel('Count')
        axes[2].set_title(f'Runtime (avg={np.mean(times):.1f}s)', fontweight='bold')

        plt.suptitle('Phase 149: v24 Chronos Agent (Clock + Entropy Minimization)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase149_v24_chronos.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return {'zs_pa': zs_pa, 'v24_pa': v24_pa, 'zs_ex': zs_ex, 'v24_ex': v24_ex}


if __name__ == '__main__':
    main()
