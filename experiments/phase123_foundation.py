"""
Phase 123: Foundation Latent-Meta-NCA

Merges all Season 6 discoveries:
  - Latent-NCA (Phase 120): encoder-decoder for continuous computation
  - Liquid Reg (Phase 121): lambda=0.1 for edge-of-chaos beta
  - In-Context Meta (Phase 122): task embedding from demo pairs

Trains on REAL ARC training data (~400 tasks).

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
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
MAX_GRID = 30  # ARC max grid size
PAD_SIZE = 32  # Padded to power of 2 for CNN
N_COLORS = 11  # 0-9 colors + padding


# ====================================================================
# ARC Data Loader
# ====================================================================
def load_arc_training():
    """Load ARC training tasks from merged JSON."""
    path = os.path.join(DATA_DIR, "arc_training.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def grid_to_tensor(grid, pad_size=PAD_SIZE):
    """Convert ARC grid (list of lists) to one-hot tensor (N_COLORS, H, W)."""
    h, w = len(grid), len(grid[0])
    t = torch.zeros(N_COLORS, pad_size, pad_size)
    for y in range(h):
        for x in range(w):
            c = grid[y][x]
            t[c, y, x] = 1.0
    # Mark valid region with last channel
    t[10, :h, :w] = 1.0  # mask channel
    return t


def tensor_to_grid(t, out_h=None, out_w=None):
    """Convert one-hot tensor back to ARC grid."""
    # t shape: (N_COLORS, H, W)
    pred = t[:10].argmax(dim=0)  # (H, W)
    if out_h and out_w:
        return pred[:out_h, :out_w].cpu().numpy().tolist()
    # Use mask to determine size
    mask = t[10]
    h = max(1, int(mask.sum(dim=1).gt(0).sum().item()))
    w = max(1, int(mask.sum(dim=0).gt(0).sum().item()))
    return pred[:h, :w].cpu().numpy().tolist()


def prepare_arc_meta_dataset(arc_data, max_tasks=200):
    """Convert ARC tasks into meta-learning format."""
    tasks = []
    task_ids = list(arc_data.keys())
    random.shuffle(task_ids)

    for tid in task_ids[:max_tasks]:
        task = arc_data[tid]
        if 'train' not in task or 'test' not in task:
            continue
        train_pairs = task['train']
        test_pairs = task['test']

        # Skip tasks with grids > MAX_GRID
        valid = True
        for pair in train_pairs + test_pairs:
            inp = pair['input']
            out = pair['output']
            if len(inp) > MAX_GRID or len(inp[0]) > MAX_GRID:
                valid = False; break
            if len(out) > MAX_GRID or len(out[0]) > MAX_GRID:
                valid = False; break
        if not valid:
            continue

        demo_inputs = [grid_to_tensor(p['input']) for p in train_pairs]
        demo_outputs = [grid_to_tensor(p['output']) for p in train_pairs]

        for tp in test_pairs:
            test_in = grid_to_tensor(tp['input'])
            test_out = grid_to_tensor(tp['output'])
            out_h = len(tp['output'])
            out_w = len(tp['output'][0])
            tasks.append({
                'task_id': tid,
                'demo_inputs': demo_inputs,
                'demo_outputs': demo_outputs,
                'test_input': test_in,
                'test_output': test_out,
                'out_h': out_h,
                'out_w': out_w,
            })

    return tasks


# ====================================================================
# Foundation Model: Latent-Meta-NCA with Liquid Reg
# ====================================================================
class FoundationEncoder(nn.Module):
    """Encode demo (input, output) pairs into task embedding."""
    def __init__(self, in_ch=N_COLORS, embed_dim=64):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Conv2d(in_ch * 2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, demo_inputs, demo_outputs):
        """
        demo_inputs: list of (N_COLORS, H, W)
        demo_outputs: list of (N_COLORS, H, W)
        Returns: (embed_dim,) task embedding
        """
        pairs = torch.stack([
            torch.cat([di, do], dim=0)
            for di, do in zip(demo_inputs, demo_outputs)
        ])  # (n_demos, 2*N_COLORS, H, W)
        embeddings = self.pair_encoder(pairs)  # (n_demos, embed_dim)
        return embeddings.mean(dim=0)


class FoundationLatentNCA(nn.Module):
    """Latent NCA conditioned on task embedding with liquid regularization."""
    def __init__(self, in_ch=N_COLORS, hidden_ch=64, embed_dim=64, latent_ch=32):
        super().__init__()
        self.latent_ch = latent_ch
        self.lambda_liquid = 0.1  # Edge of chaos

        # Encoder: input grid -> latent
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, latent_ch, 3, padding=1), nn.ReLU(),
        )

        # NCA update conditioned on task embedding
        self.update = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim, hidden_ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden_ch, latent_ch, 1),
        )
        self.tau_gate = nn.Sequential(
            nn.Conv2d(latent_ch + embed_dim, latent_ch, 1),
            nn.Sigmoid()
        )

        # Decoder: latent -> output grid
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, N_COLORS, 1),
        )

    def forward(self, x, task_embed, n_steps=5, return_liquid_loss=False):
        """
        x: (B, N_COLORS, PAD_SIZE, PAD_SIZE)
        task_embed: (embed_dim,) from encoder
        """
        B, _, H, W = x.shape
        state = self.encoder(x)

        # Broadcast task embedding
        te = task_embed.view(1, -1, 1, 1).expand(B, -1, H, W)

        liquid_loss = 0
        for t in range(n_steps):
            state_ctx = torch.cat([state, te], dim=1)
            delta = self.update(state_ctx)
            beta = self.tau_gate(state_ctx)

            if return_liquid_loss:
                liquid_loss += ((beta - 0.5) ** 2).mean()

            state = beta * state + (1 - beta) * delta

        logits = self.decoder(state)

        if return_liquid_loss:
            return logits, liquid_loss / n_steps

        return logits


class FoundationSystem(nn.Module):
    """Full Encoder + LatentNCA + Decoder system."""
    def __init__(self, embed_dim=64, hidden_ch=64, latent_ch=32):
        super().__init__()
        self.task_encoder = FoundationEncoder(embed_dim=embed_dim)
        self.latent_nca = FoundationLatentNCA(
            hidden_ch=hidden_ch, embed_dim=embed_dim, latent_ch=latent_ch)

    def forward(self, demo_inputs, demo_outputs, test_input, n_steps=5,
                return_liquid_loss=False):
        task_embed = self.task_encoder(demo_inputs, demo_outputs)
        return self.latent_nca(
            test_input.unsqueeze(0), task_embed, n_steps=n_steps,
            return_liquid_loss=return_liquid_loss)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 123: Foundation Latent-Meta-NCA")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load ARC data
    print("\n[Step 1] Loading ARC training data...")
    arc_data = load_arc_training()
    print(f"  Total ARC tasks: {len(arc_data)}")

    # Prepare meta-learning dataset
    print("\n[Step 2] Preparing meta dataset...")
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    print(f"  Total meta-examples: {len(all_tasks)}")

    # Split 80/20
    random.shuffle(all_tasks)
    split = int(len(all_tasks) * 0.8)
    train_tasks = all_tasks[:split]
    test_tasks = all_tasks[split:]
    print(f"  Train: {len(train_tasks)}, Test: {len(test_tasks)}")

    # Build model
    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    print("\n[Step 3] Training Foundation Model...")
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    for epoch in range(60):
        model.train()
        random.shuffle(train_tasks)
        epoch_loss = 0; n = 0

        for item in train_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            ti = item['test_input'].to(DEVICE)
            to_gt = item['test_output'].to(DEVICE)

            logits, liq_loss = model(di, do, ti, n_steps=5,
                                      return_liquid_loss=True)

            # Cross-entropy over color channels
            target = to_gt[:10].argmax(dim=0).unsqueeze(0)  # (1, H, W)
            task_loss = F.cross_entropy(logits, target)
            loss = task_loss + model.latent_nca.lambda_liquid * liq_loss

            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n += 1

        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/60: loss={epoch_loss/n:.4f}")

    # Evaluate
    print("\n[Step 4] Evaluation on held-out ARC tasks...")
    model.eval()
    correct_pixels = 0; total_pixels = 0
    exact_matches = 0; total_tasks = 0

    with torch.no_grad():
        for item in test_tasks:
            di = [d.to(DEVICE) for d in item['demo_inputs']]
            do = [d.to(DEVICE) for d in item['demo_outputs']]
            ti = item['test_input'].to(DEVICE)
            to_gt = item['test_output'].to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            logits = model(di, do, ti, n_steps=5)
            pred = logits[0, :10].argmax(dim=0)  # (H, W)
            gt = to_gt[:10].argmax(dim=0)

            # Compare in valid region
            p_crop = pred[:oh, :ow]
            g_crop = gt[:oh, :ow]

            correct_pixels += (p_crop == g_crop).sum().item()
            total_pixels += oh * ow
            if (p_crop == g_crop).all():
                exact_matches += 1
            total_tasks += 1

    pixel_acc = correct_pixels / max(total_pixels, 1)
    exact_rate = exact_matches / max(total_tasks, 1)

    print(f"\n{'='*70}")
    print(f"  FOUNDATION LATENT-META-NCA RESULTS")
    print(f"{'='*70}")
    print(f"  Test tasks:     {total_tasks}")
    print(f"  Pixel accuracy: {pixel_acc*100:.2f}%")
    print(f"  Exact match:    {exact_matches}/{total_tasks} ({exact_rate*100:.2f}%)")
    print(f"  Parameters:     {n_params:,}")

    results = {
        'pixel_accuracy': pixel_acc,
        'exact_match_rate': exact_rate,
        'exact_matches': exact_matches,
        'total_tasks': total_tasks,
        'n_params': n_params,
    }

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase123_foundation.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 123: Foundation Latent-Meta-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': results}, f, indent=2, default=str)

    # Save model for Phase 124/125
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "phase123_model.pt"))
    print("  Model saved!")

    print("\nPhase 123 complete!")
    return results


if __name__ == '__main__':
    main()
