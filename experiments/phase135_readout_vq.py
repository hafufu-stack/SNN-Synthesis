"""
Phase 135: Readout-Only VQ-NCA

"Think in continuous, speak in discrete"

Key insight from Season 10's failure:
  - VQ at every step kills expressiveness (72.6% -> 60%)
  - STE over T loops destroys TTCT gradients (gap=0%)

Solution: Remove VQ from the NCA loop entirely.
Apply VQ ONLY at the final readout (before decoder).

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
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
PAD_SIZE = 32
N_COLORS = 11
MAX_GRID = 30
IN_CH = N_COLORS + 2  # 11 + coord

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase132_foundation_vq import (
    load_arc_training, prepare_arc_meta_dataset, grid_to_tensor, add_coord_channels
)


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes=256, dim=32, commitment=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.commitment = commitment
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1/n_codes, 1/n_codes)

    def forward(self, z):
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
        dists = (z_flat ** 2).sum(1, keepdim=True) + \
                (self.codebook.weight ** 2).sum(1) - \
                2 * z_flat @ self.codebook.weight.t()
        indices = dists.argmin(dim=1)
        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        commit_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment * commit_loss
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss


class ContextEncoder(nn.Module):
    def __init__(self, in_ch=IN_CH, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch * 2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, demo_inputs, demo_outputs):
        pairs = torch.stack([
            torch.cat([di, do], dim=0)
            for di, do in zip(demo_inputs, demo_outputs)
        ])
        return self.net(pairs).mean(dim=0)


class ReadoutVQNCA(nn.Module):
    """
    Continuous NCA loop + VQ only at final readout.
    Think: continuous. Speak: discrete.
    """
    def __init__(self, in_ch=IN_CH, embed_dim=64, hidden_ch=32,
                 n_codes=256, use_vq=True):
        super().__init__()
        self.use_vq = use_vq
        self.embed_dim = embed_dim
        self.encoder = ContextEncoder(in_ch=in_ch, embed_dim=embed_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1), nn.ReLU())

        # Continuous NCA update (NO VQ inside loop!)
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch + embed_dim, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau = nn.Sequential(
            nn.Conv2d(hidden_ch + embed_dim, hidden_ch, 1), nn.Sigmoid())

        # VQ at the readout only
        if use_vq:
            self.vq = VectorQuantizer(n_codes, hidden_ch)

        self.decoder = nn.Conv2d(hidden_ch, N_COLORS, 1)

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5,
                return_vq_loss=False, task_embed_override=None):
        if task_embed_override is not None:
            te = task_embed_override
        else:
            te = self.encoder(demo_inputs, demo_outputs)

        B = 1
        x = test_input.unsqueeze(0)
        _, _, H, W = x.shape
        te_spatial = te.view(1, -1, 1, 1).expand(B, -1, H, W)

        # Continuous NCA loop (no VQ!)
        state = self.stem(x)
        for t in range(n_steps):
            ctx = torch.cat([state, te_spatial], dim=1)
            delta = self.update(ctx)
            beta = self.tau(ctx)
            state = beta * state + (1 - beta) * delta

        # Readout: apply VQ ONLY here
        vq_loss = torch.tensor(0.0, device=state.device)
        if self.use_vq:
            state, vq_loss = self.vq(state)

        logits = self.decoder(state)

        if return_vq_loss:
            return logits, vq_loss
        return logits


def ttct_readout_vq(model, demo_inputs, demo_outputs, n_steps=5,
                    ttct_steps=100, ttct_lr=0.01):
    """TTCT works because no STE in the loop!"""
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        te_init = model.encoder(demo_inputs, demo_outputs)

    te = te_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([te], lr=ttct_lr)

    best_loss = float('inf')
    best_embed = te.clone().detach()

    for step in range(ttct_steps):
        opt.zero_grad()
        total_loss = 0
        for di, do_gt in zip(demo_inputs, demo_outputs):
            logits, vq_loss = model(
                [di], [do_gt], di, n_steps=n_steps,
                return_vq_loss=True, task_embed_override=te)
            target = do_gt[:10].argmax(dim=0).unsqueeze(0)
            loss = F.cross_entropy(logits, target) + vq_loss
            total_loss += loss
        total_loss /= len(demo_inputs)
        total_loss.backward()
        opt.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_embed = te.clone().detach()

    for p in model.parameters():
        p.requires_grad = True
    return best_embed, best_loss


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 135: Readout-Only VQ-NCA")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=350)
    random.shuffle(all_tasks)
    split = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split]
    test_tasks = all_tasks[split:]
    print(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Compare: Continuous-only vs Readout-VQ
    configs = [
        ('Continuous (no VQ)', False, 0),
        ('Readout-VQ-256', True, 256),
        ('Readout-VQ-512', True, 512),
    ]

    all_results = {}

    for name, use_vq, n_codes in configs:
        print(f"\n  === {name} ===")
        torch.manual_seed(SEED)
        model = ReadoutVQNCA(
            embed_dim=64, hidden_ch=32,
            n_codes=n_codes if n_codes > 0 else 256,
            use_vq=use_vq
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        # Train
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        for epoch in range(80):
            model.train()
            random.shuffle(train_tasks)
            for item in train_tasks:
                di = [d.to(DEVICE) for d in item['demo_inputs']]
                do = [d.to(DEVICE) for d in item['demo_outputs']]
                ti = item['test_input'].to(DEVICE)
                to_gt = item['test_output'].to(DEVICE)
                logits, vq_loss = model(di, do, ti, n_steps=5, return_vq_loss=True)
                target = to_gt[:10].argmax(dim=0).unsqueeze(0)
                loss = F.cross_entropy(logits, target) + vq_loss
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/80")

        # Eval: Zero-shot
        model.eval()
        zs_px = 0; zs_exact = 0; total_px = 0; n = 0
        with torch.no_grad():
            for item in test_tasks:
                di = [d.to(DEVICE) for d in item['demo_inputs']]
                do = [d.to(DEVICE) for d in item['demo_outputs']]
                ti = item['test_input'].to(DEVICE)
                to_gt = item['test_output'].to(DEVICE)
                oh, ow = item['out_h'], item['out_w']
                logits = model(di, do, ti, n_steps=5)
                pred = logits[0, :10].argmax(dim=0)[:oh, :ow]
                gt = to_gt[:10].argmax(dim=0)[:oh, :ow]
                zs_px += (pred == gt).sum().item()
                zs_exact += (pred == gt).all().item()
                total_px += oh * ow; n += 1

        # Eval: TTCT
        tt_px = 0; tt_exact = 0; total_px2 = 0; n2 = 0
        for item in test_tasks[:20]:  # Subset for speed
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            ti = item['test_input'].to(DEVICE)
            to_gt = item['test_output'].to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            gt = to_gt[:10].argmax(dim=0)[:oh, :ow]

            best_embed, _ = ttct_readout_vq(model, di, do, n_steps=5,
                                              ttct_steps=80, ttct_lr=0.01)
            with torch.no_grad():
                logits = model(di, do, ti, n_steps=5, task_embed_override=best_embed)
                pred = logits[0, :10].argmax(dim=0)[:oh, :ow]
                tt_px += (pred == gt).sum().item()
                tt_exact += (pred == gt).all().item()
                total_px2 += oh * ow; n2 += 1

        zs_acc = zs_px / max(total_px, 1)
        tt_acc = tt_px / max(total_px2, 1)

        all_results[name] = {
            'zs_pixel': zs_acc, 'zs_exact': zs_exact, 'zs_total': n,
            'tt_pixel': tt_acc, 'tt_exact': tt_exact, 'tt_total': n2,
            'n_params': n_params,
        }
        print(f"  ZS: pixel={zs_acc*100:.2f}%, exact={zs_exact}/{n}")
        print(f"  TTCT: pixel={tt_acc*100:.2f}%, exact={tt_exact}/{n2}, "
              f"gap={((tt_acc-zs_acc)*100):+.2f}%")

        # Save model if best
        if use_vq:
            torch.save(model.state_dict(),
                      os.path.join(RESULTS_DIR, f"phase135_model_{n_codes}.pt"))

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("  READOUT-VQ RESULTS")
    print(f"{'='*70}")
    for name, res in all_results.items():
        print(f"  {name:25s}: ZS pixel={res['zs_pixel']*100:.2f}%, "
              f"ZS exact={res['zs_exact']}/{res['zs_total']}, "
              f"TTCT pixel={res['tt_pixel']*100:.2f}%, "
              f"TTCT exact={res['tt_exact']}/{res['tt_total']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase135_readout_vq.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 135: Readout-Only VQ-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': all_results}, f, indent=2, default=str)

    print("\nPhase 135 complete!")
    return all_results


if __name__ == '__main__':
    main()
