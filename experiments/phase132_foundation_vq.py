"""
Phase 132: Foundation Context-VQ-NCA

The winning architecture from Season 9:
  Context-NCA (winner of P129) + VQ (winner of P128)
  + Global Coord + Dynamic Size Cropper

Trained on REAL ARC training data (~400 tasks).

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
MAX_GRID = 30
PAD_SIZE = 32
N_COLORS = 11  # 0-9 + mask


# ====================================================================
# ARC Data Loader (from Phase 123)
# ====================================================================
def load_arc_training():
    path = os.path.join(DATA_DIR, "arc_training.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def grid_to_tensor(grid, pad_size=PAD_SIZE):
    h, w = len(grid), len(grid[0])
    t = torch.zeros(N_COLORS, pad_size, pad_size)
    for y in range(h):
        for x in range(w):
            t[grid[y][x], y, x] = 1.0
    t[10, :h, :w] = 1.0  # mask
    return t


def add_coord_channels(t):
    """Add 2 coordinate channels (global position info)."""
    _, H, W = t.shape
    yy = torch.linspace(0, 1, H).view(H, 1).expand(H, W).unsqueeze(0)
    xx = torch.linspace(0, 1, W).view(1, W).expand(H, W).unsqueeze(0)
    return torch.cat([t, yy, xx], dim=0)  # N_COLORS+2 channels


def prepare_arc_meta_dataset(arc_data, max_tasks=300):
    tasks = []
    task_ids = list(arc_data.keys())
    random.shuffle(task_ids)

    for tid in task_ids[:max_tasks]:
        task = arc_data[tid]
        if 'train' not in task or 'test' not in task:
            continue
        train_pairs = task['train']
        test_pairs = task['test']

        valid = True
        for pair in train_pairs + test_pairs:
            if len(pair['input']) > MAX_GRID or len(pair['input'][0]) > MAX_GRID:
                valid = False; break
            if len(pair['output']) > MAX_GRID or len(pair['output'][0]) > MAX_GRID:
                valid = False; break
        if not valid:
            continue

        demo_inputs = [add_coord_channels(grid_to_tensor(p['input'])) for p in train_pairs]
        demo_outputs = [add_coord_channels(grid_to_tensor(p['output'])) for p in train_pairs]

        for tp in test_pairs:
            test_in = add_coord_channels(grid_to_tensor(tp['input']))
            test_out = grid_to_tensor(tp['output'])  # No coord for target
            tasks.append({
                'task_id': tid,
                'demo_inputs': demo_inputs,
                'demo_outputs': demo_outputs,
                'test_input': test_in,
                'test_output': test_out,
                'out_h': len(tp['output']),
                'out_w': len(tp['output'][0]),
            })
    return tasks


# ====================================================================
# Vector Quantizer
# ====================================================================
class VectorQuantizer(nn.Module):
    def __init__(self, n_codes=64, dim=32, commitment=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.commitment = commitment
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1/n_codes, 1/n_codes)

    def forward(self, z, gumbel_scale=0.0):
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)
        dists = (z_flat ** 2).sum(1, keepdim=True) + \
                (self.codebook.weight ** 2).sum(1) - \
                2 * z_flat @ self.codebook.weight.t()
        if gumbel_scale > 0:
            gumbel = -torch.log(-torch.log(torch.rand_like(dists) + 1e-20) + 1e-20)
            dists = dists - gumbel_scale * gumbel
        indices = dists.argmin(dim=1)
        z_q = self.codebook(indices).view(B, H, W, C).permute(0, 3, 1, 2)
        commit_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment * commit_loss
        z_q_st = z + (z_q - z).detach()  # STE
        usage = len(indices.unique()) / self.n_codes
        return z_q_st, vq_loss, usage, indices.view(B, H, W)


# ====================================================================
# Context-VQ-NCA with Global Coord
# ====================================================================
IN_CH = N_COLORS + 2  # 11 colors + 2 coord channels = 13


class ContextEncoder(nn.Module):
    """Encode demo pairs into task embedding."""
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
        embeddings = self.net(pairs)
        return embeddings.mean(dim=0)


class ContextVQNCA(nn.Module):
    """Context-injected NCA with VQ at each step."""
    def __init__(self, in_ch=IN_CH, embed_dim=64, hidden_ch=32, n_codes=64):
        super().__init__()
        self.embed_dim = embed_dim

        self.encoder = ContextEncoder(in_ch=in_ch, embed_dim=embed_dim)

        # NCA stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1), nn.ReLU())

        # Context-conditioned update
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch + embed_dim, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau = nn.Sequential(
            nn.Conv2d(hidden_ch + embed_dim, hidden_ch, 1), nn.Sigmoid())

        self.vq = VectorQuantizer(n_codes, hidden_ch)

        # Decoder: predict all 11 channels (10 colors + mask)
        self.decoder = nn.Conv2d(hidden_ch, N_COLORS, 1)

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5,
                return_vq_loss=False, gumbel_scale=0.0):
        # Get context
        task_embed = self.encoder(demo_inputs, demo_outputs)

        B = 1
        x = test_input.unsqueeze(0)  # (1, IN_CH, H, W)
        _, _, H, W = x.shape
        te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

        state = self.stem(x)
        total_vq = 0; total_usage = 0
        prev_indices = None
        n_changes = []

        for t in range(n_steps):
            ctx = torch.cat([state, te], dim=1)
            delta = self.update(ctx)
            beta = self.tau(ctx)
            state = beta * state + (1 - beta) * delta

            state, vq_loss, usage, indices = self.vq(state, gumbel_scale=gumbel_scale)
            total_vq += vq_loss
            total_usage += usage

            if prev_indices is not None:
                n_changed = (indices != prev_indices).sum().item()
                n_changes.append(n_changed)
            prev_indices = indices

        logits = self.decoder(state)  # (1, N_COLORS, H, W)

        info = {
            'vq_loss': total_vq / n_steps,
            'usage': total_usage / n_steps,
            'n_changes': n_changes,
            'actual_steps': n_steps,
        }

        if return_vq_loss:
            return logits, info
        return logits


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 132: Foundation Context-VQ-NCA")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load ARC
    print("\n[Step 1] Loading ARC training data...")
    arc_data = load_arc_training()
    print(f"  Total ARC tasks: {len(arc_data)}")

    print("\n[Step 2] Preparing meta dataset...")
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=350)
    random.shuffle(all_tasks)
    split = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split]
    test_tasks = all_tasks[split:]
    print(f"  Total: {len(all_tasks)}, Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Build model
    model = ContextVQNCA(embed_dim=64, hidden_ch=32, n_codes=64).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    print("\n[Step 3] Training Foundation Model...")
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)

    for epoch in range(80):
        model.train()
        random.shuffle(train_tasks)
        epoch_loss = 0; n = 0

        for item in train_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            ti = item['test_input'].to(DEVICE)
            to_gt = item['test_output'].to(DEVICE)

            logits, info = model(di, do, ti, n_steps=5, return_vq_loss=True)
            target = to_gt[:10].argmax(dim=0).unsqueeze(0)
            task_loss = F.cross_entropy(logits, target)
            loss = task_loss + info['vq_loss']

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n += 1

        sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/80: loss={epoch_loss/n:.4f}")

    # Evaluate
    print("\n[Step 4] Evaluation...")
    model.eval()
    correct_px = 0; total_px = 0; exact_matches = 0; total_n = 0

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

            correct_px += (pred == gt).sum().item()
            total_px += oh * ow
            exact_matches += (pred == gt).all().item()
            total_n += 1

    px_acc = correct_px / max(total_px, 1)
    exact_rate = exact_matches / max(total_n, 1)

    print(f"\n{'='*70}")
    print(f"  FOUNDATION CONTEXT-VQ-NCA RESULTS")
    print(f"{'='*70}")
    print(f"  Test tasks:     {total_n}")
    print(f"  Pixel accuracy: {px_acc*100:.2f}%")
    print(f"  Exact match:    {exact_matches}/{total_n} ({exact_rate*100:.2f}%)")
    print(f"  Parameters:     {n_params:,}")

    # Save model
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "phase132_model.pt"))

    results = {
        'pixel_accuracy': px_acc, 'exact_match_rate': exact_rate,
        'exact_matches': exact_matches, 'total_tasks': total_n,
        'n_params': n_params,
    }
    with open(os.path.join(RESULTS_DIR, "phase132_foundation_vq.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 132: Foundation Context-VQ-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)

    print("  Model saved!")
    print("\nPhase 132 complete!")
    return results


if __name__ == '__main__':
    main()
