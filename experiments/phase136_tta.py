"""
Phase 136: TTA Pixel Majority Vote

No VQ needed! Use D8 symmetry group (8 geometric transforms)
to create 8 views of each test input, infer on all, then
majority-vote per-pixel to cancel out stochastic noise/blur.

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
PAD_SIZE = 32
N_COLORS = 11
IN_CH = N_COLORS + 2

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase132_foundation_vq import (
    load_arc_training, prepare_arc_meta_dataset, add_coord_channels,
    grid_to_tensor
)


# ====================================================================
# D8 transforms (rotation + flip)
# ====================================================================
def d8_transforms(x):
    """Generate 8 D8 augmented versions of x (B,C,H,W)."""
    views = [x]                                    # identity
    views.append(torch.rot90(x, 1, [2, 3]))        # 90
    views.append(torch.rot90(x, 2, [2, 3]))        # 180
    views.append(torch.rot90(x, 3, [2, 3]))        # 270
    views.append(torch.flip(x, [3]))                # flip-H
    views.append(torch.flip(torch.rot90(x, 1, [2, 3]), [3]))  # 90+flipH
    views.append(torch.flip(x, [2]))                # flip-V
    views.append(torch.flip(torch.rot90(x, 1, [2, 3]), [2]))  # 90+flipV
    return views


def d8_inverse_2d(pred, idx):
    """Inverse transform on 2D tensor (H, W). Operates on FULL padded space."""
    # pred is (H, W) where H=W=PAD_SIZE
    if idx == 0: return pred
    elif idx == 1: return torch.rot90(pred, -1, [0, 1])
    elif idx == 2: return torch.rot90(pred, -2, [0, 1])
    elif idx == 3: return torch.rot90(pred, -3, [0, 1])
    elif idx == 4: return torch.flip(pred, [1])
    elif idx == 5: return torch.flip(torch.rot90(pred, -1, [0, 1]), [1])
    elif idx == 6: return torch.flip(pred, [0])
    elif idx == 7: return torch.flip(torch.rot90(pred, -1, [0, 1]), [0])
    return pred


# ====================================================================
# Continuous NCA (v21-style) for TTA
# ====================================================================
class ContinuousContextNCA(nn.Module):
    """v21-style continuous NCA (no VQ) with context injection."""
    def __init__(self, in_ch=IN_CH, embed_dim=64, hidden_ch=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch * 2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, embed_dim),
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch + embed_dim, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau = nn.Sequential(
            nn.Conv2d(hidden_ch + embed_dim, hidden_ch, 1), nn.Sigmoid())
        self.decoder = nn.Conv2d(hidden_ch, N_COLORS, 1)

    def encode_demos(self, demo_inputs, demo_outputs):
        pairs = torch.stack([
            torch.cat([di, do], dim=0) for di, do in zip(demo_inputs, demo_outputs)
        ])
        return self.encoder(pairs).mean(dim=0)

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5,
                task_embed=None):
        if task_embed is None:
            task_embed = self.encode_demos(demo_inputs, demo_outputs)

        B = 1
        x = test_input.unsqueeze(0)
        _, _, H, W = x.shape
        te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

        state = self.stem(x)
        for t in range(n_steps):
            ctx = torch.cat([state, te], dim=1)
            delta = self.update(ctx)
            beta = self.tau(ctx)
            state = beta * state + (1 - beta) * delta

        return self.decoder(state)


def tta_predict(model, test_input, task_embed, oh, ow, n_steps=5):
    """TTA with D8 symmetry + majority vote.
    Key fix: inverse-transform on FULL padded space (PAD_SIZE x PAD_SIZE),
    then crop AFTER voting. This avoids size mismatch on non-square grids."""
    x = test_input.unsqueeze(0)  # (1, C, H, W) where H=W=PAD_SIZE
    views = d8_transforms(x)
    H, W = PAD_SIZE, PAD_SIZE

    # Collect predictions in FULL padded space
    all_preds = []
    for i, v in enumerate(views):
        logits = model.forward(None, None, v.squeeze(0), n_steps=n_steps,
                              task_embed=task_embed)
        pred = logits[0, :10].argmax(dim=0)  # (PAD_SIZE, PAD_SIZE)

        # Inverse transform on FULL padded space (always square!)
        pred_inv = d8_inverse_2d(pred, i)
        all_preds.append(pred_inv)

    # Majority vote per pixel on FULL space
    stacked = torch.stack(all_preds)  # (8, PAD_SIZE, PAD_SIZE)
    votes = torch.zeros(10, H, W, device=DEVICE)
    for c in range(10):
        votes[c] = (stacked == c).float().sum(dim=0)

    # Crop AFTER voting
    return votes.argmax(dim=0)[:oh, :ow]  # (oh, ow)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 136: TTA Pixel Majority Vote (D8 Symmetry)")
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

    # Train continuous model
    print("\n[Step 2] Training Continuous Context-NCA...")
    model = ContinuousContextNCA(embed_dim=64, hidden_ch=32).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

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
            logits = model(di, do, ti, n_steps=5)
            target = to_gt[:10].argmax(dim=0).unsqueeze(0)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/80")

    # Save for Phase 137
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "phase136_model.pt"))

    # Evaluate: No TTA vs TTA
    print("\n[Step 3] Evaluation: Standard vs TTA...")
    model.eval()

    std_px = 0; std_exact = 0; tta_px = 0; tta_exact = 0
    total_px = 0; n = 0

    with torch.no_grad():
        for i, item in enumerate(test_tasks):
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            ti = item['test_input'].to(DEVICE)
            to_gt = item['test_output'].to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            gt = to_gt[:10].argmax(dim=0)[:oh, :ow]

            # Standard
            logits = model(di, do, ti, n_steps=5)
            pred_std = logits[0, :10].argmax(dim=0)[:oh, :ow]
            std_px += (pred_std == gt).sum().item()
            std_exact += (pred_std == gt).all().item()

            # TTA with D8
            task_embed = model.encode_demos(di, do)
            pred_tta = tta_predict(model, ti, task_embed, oh, ow, n_steps=5)
            tta_px += (pred_tta == gt).sum().item()
            tta_exact += (pred_tta == gt).all().item()

            total_px += oh * ow; n += 1
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(test_tasks)}: "
                      f"std_ex={std_exact}, tta_ex={tta_exact}")

    std_acc = std_px / max(total_px, 1)
    tta_acc = tta_px / max(total_px, 1)

    print(f"\n{'='*70}")
    print("  TTA RESULTS")
    print(f"{'='*70}")
    print(f"  Standard:  pixel={std_acc*100:.2f}%, exact={std_exact}/{n}")
    print(f"  TTA (D8):  pixel={tta_acc*100:.2f}%, exact={tta_exact}/{n}")
    print(f"  Gap:       pixel={((tta_acc-std_acc)*100):+.2f}%, exact={tta_exact-std_exact:+d}")

    results = {
        'std_pixel': std_acc, 'std_exact': std_exact,
        'tta_pixel': tta_acc, 'tta_exact': tta_exact,
        'total': n, 'n_params': n_params,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase136_tta.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 136: TTA Pixel Vote',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)

    print("\nPhase 136 complete!")
    return results


if __name__ == '__main__':
    main()
