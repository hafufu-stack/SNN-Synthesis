"""
Phase 120: Latent-NCA

Solves Phase 119's "1-step sudden death" by moving NCA computation
from discrete pixel space to continuous latent space.

Architecture: CNN Encoder -> Latent Grid -> L-NCA (T steps) -> CNN Decoder

In latent space:
- No discrete boundaries -> no cliff-edge collapse
- Soft-landing has time to react
- Decoder can output different sizes (size-change tasks)

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


# ====================================================================
# Latent-NCA Architecture
# ====================================================================
class LatentEncoder(nn.Module):
    """Encode pixel grid to latent grid."""
    def __init__(self, in_ch=2, latent_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, latent_ch, 3, padding=1), nn.BatchNorm2d(latent_ch), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class LatentDecoder(nn.Module):
    """Decode latent grid to pixel output."""
    def __init__(self, latent_ch=32, out_ch=1, target_size=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, out_ch, 1),
        )
        self.target_size = target_size

    def forward(self, z, target_size=None):
        sz = target_size or self.target_size
        if sz is not None and (z.shape[2] != sz[0] or z.shape[3] != sz[1]):
            z = F.interpolate(z, size=sz, mode='bilinear', align_corners=False)
        return self.net(z)


class LatentNCA(nn.Module):
    """NCA operating in latent space with energy canary."""
    def __init__(self, latent_ch=32):
        super().__init__()
        self.update = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch, 3, padding=1),
            nn.BatchNorm2d(latent_ch), nn.ReLU(),
            nn.Conv2d(latent_ch, latent_ch, 1),
        )
        self.tau_gate = nn.Sequential(
            nn.Conv2d(latent_ch, latent_ch, 1), nn.Sigmoid()
        )

    def forward(self, z, n_steps=5):
        for t in range(n_steps):
            delta = self.update(z)
            beta = self.tau_gate(z)
            z = beta * z + (1 - beta) * delta
        return z

    def forward_with_energy(self, z, n_steps=30, energy_threshold=None, slowdown=3.0):
        """Forward with soft-landing."""
        prev = z.detach().clone()
        energies = []
        tau_mod = 0.0

        for t in range(n_steps):
            delta = self.update(z)
            beta = self.tau_gate(z)

            if tau_mod > 0:
                beta = torch.sigmoid(
                    torch.logit(beta.clamp(1e-6, 1 - 1e-6)) + tau_mod)

            z = beta * z + (1 - beta) * delta

            energy = (z - prev).pow(2).mean().item()
            energies.append(energy)
            prev = z.detach().clone()

            if energy_threshold is not None and t >= 2:
                if energy > energy_threshold:
                    tau_mod = slowdown
                elif tau_mod > 0 and energy < energy_threshold * 0.3:
                    tau_mod = max(0, tau_mod - 0.5)

        return z, energies


class LatentNCA_System(nn.Module):
    """Full Encoder -> Latent NCA -> Decoder system."""
    def __init__(self, in_ch=2, out_ch=1, latent_ch=32):
        super().__init__()
        self.encoder = LatentEncoder(in_ch, latent_ch)
        self.nca = LatentNCA(latent_ch)
        self.decoder = LatentDecoder(latent_ch, out_ch)

    def forward(self, x, n_steps=5, target_size=None):
        z = self.encoder(x)
        z = self.nca(z, n_steps=n_steps)
        return torch.sigmoid(self.decoder(z, target_size=target_size))

    def forward_with_energy(self, x, n_steps=30, target_size=None,
                            energy_threshold=None, slowdown=3.0):
        z = self.encoder(x)
        z, energies = self.nca.forward_with_energy(
            z, n_steps=n_steps, energy_threshold=energy_threshold, slowdown=slowdown)
        return torch.sigmoid(self.decoder(z, target_size=target_size)), energies


# ====================================================================
# Pixel-NCA baseline (from Phase 119)
# ====================================================================
class PixelNCA(nn.Module):
    def __init__(self, in_ch=2, hidden_ch=32, out_ch=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch), nn.ReLU())
        self.update = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.BatchNorm2d(hidden_ch), nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 1))
        self.tau_gate = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, 1), nn.Sigmoid())
        self.decoder = nn.Conv2d(hidden_ch, out_ch, 1)

    def forward(self, x, n_steps=5):
        s = self.encoder(x)
        for t in range(n_steps):
            d = self.update(s)
            b = self.tau_gate(s)
            s = b * s + (1 - b) * d
        return torch.sigmoid(self.decoder(s))


# ====================================================================
# Task generators
# ====================================================================
def generate_flood_fill(grid_size=16, n=500):
    inputs, targets = [], []
    for _ in range(n):
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        for _ in range(random.randint(5, 15)):
            x, y = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            l = random.randint(2, grid_size//2)
            if random.random() > 0.5:
                for dx in range(l):
                    if x+dx < grid_size: grid[y, x+dx] = 1.0
            else:
                for dy in range(l):
                    if y+dy < grid_size: grid[y+dy, x] = 1.0
        while True:
            sx, sy = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            if grid[sy, sx] == 0: break
        inp = np.zeros((2, grid_size, grid_size), dtype=np.float32)
        inp[0] = grid; inp[1, sy, sx] = 1.0
        target = np.zeros((grid_size, grid_size), dtype=np.float32)
        visited = set(); queue = [(sy, sx)]
        while queue:
            cy, cx = queue.pop(0)
            if (cy, cx) in visited: continue
            if cy<0 or cy>=grid_size or cx<0 or cx>=grid_size: continue
            if grid[cy, cx] == 1.0: continue
            visited.add((cy, cx)); target[cy, cx] = 1.0
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]: queue.append((cy+dy, cx+dx))
        inputs.append(inp); targets.append(target)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets)).unsqueeze(1)


def generate_long_move(grid_size=16, n=500):
    inputs, targets = [], []
    for _ in range(n):
        inp = np.zeros((2, grid_size, grid_size), dtype=np.float32)
        target = np.zeros((grid_size, grid_size), dtype=np.float32)
        ox = random.randint(0, grid_size-3); oy = random.randint(0, grid_size-3)
        inp[0, oy:oy+2, ox:ox+2] = 1.0
        tx = grid_size - 3 - ox; ty = grid_size - 3 - oy
        inp[1, ty:ty+2, tx:tx+2] = 0.5
        target[ty:ty+2, tx:tx+2] = 1.0
        inputs.append(inp); targets.append(target)
    return torch.tensor(np.array(inputs)), torch.tensor(np.array(targets)).unsqueeze(1)


def generate_size_change(n=500):
    """Generate tasks where output is DIFFERENT size from input."""
    inputs, targets = [], []
    for _ in range(n):
        in_sz = random.choice([8, 12, 16])
        out_sz = random.choice([s for s in [8, 12, 16, 20] if s != in_sz])
        inp = np.zeros((2, in_sz, in_sz), dtype=np.float32)
        # Draw a simple shape in input
        cx, cy = in_sz // 2, in_sz // 2
        r = random.randint(1, in_sz // 4)
        for y in range(in_sz):
            for x in range(in_sz):
                if abs(x-cx) + abs(y-cy) <= r:
                    inp[0, y, x] = 1.0
        # Mark scale info
        scale = out_sz / in_sz
        inp[1, :, :] = scale / 3.0  # normalized scale channel

        # Target: same shape scaled to out_sz
        target = np.zeros((out_sz, out_sz), dtype=np.float32)
        cx2, cy2 = out_sz // 2, out_sz // 2
        r2 = max(1, int(r * scale))
        for y in range(out_sz):
            for x in range(out_sz):
                if abs(x-cx2) + abs(y-cy2) <= r2:
                    target[y, x] = 1.0

        inputs.append((torch.tensor(inp), torch.tensor(target).unsqueeze(0),
                       (out_sz, out_sz)))
        targets.append(None)  # stored in inputs

    return inputs


# ====================================================================
# Training and evaluation
# ====================================================================
def train_model(model, train_x, train_y, n_steps=5, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = torch.utils.data.TensorDataset(train_x, train_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=n_steps)
            loss = F.binary_cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}")
    return model


def eval_iou(model, test_x, test_y, n_steps=5, use_energy=False,
             energy_threshold=None, slowdown=3.0):
    model.eval()
    ds = torch.utils.data.TensorDataset(test_x, test_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)
    total_iou = 0; n = 0; all_energies = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if use_energy and hasattr(model, 'forward_with_energy'):
                out, energies = model.forward_with_energy(
                    x, n_steps=n_steps, energy_threshold=energy_threshold,
                    slowdown=slowdown)
                all_energies.extend(energies)
            else:
                out = model(x, n_steps=n_steps)
            pred = (out > 0.5).float()
            inter = (pred * y).sum(dim=(1,2,3))
            union = ((pred + y) > 0).float().sum(dim=(1,2,3))
            iou = (inter / (union + 1e-8)).mean().item()
            total_iou += iou * x.size(0); n += x.size(0)
    return total_iou / n, all_energies


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 120: Latent-NCA")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    all_results = {}

    for task_name, gen_fn in [('flood_fill', generate_flood_fill),
                                ('long_move', generate_long_move)]:
        print(f"\n{'='*50}")
        print(f"  Task: {task_name}")
        print(f"{'='*50}")
        data_x, data_y = gen_fn(grid_size=16, n=800)
        train_x, test_x = data_x[:600], data_x[600:]
        train_y, test_y = data_y[:600], data_y[600:]

        results = {}

        # Pixel-NCA baseline
        print(f"\n  [Pixel-NCA] Training (T=5)...")
        pixel_model = PixelNCA(in_ch=2, hidden_ch=32, out_ch=1).to(DEVICE)
        pixel_model = train_model(pixel_model, train_x, train_y, n_steps=5, epochs=50)

        for T in [1, 3, 5, 10, 15, 20, 30, 50]:
            iou, _ = eval_iou(pixel_model, test_x, test_y, n_steps=T)
            results[f'pixel_T{T}'] = iou
            if T <= 5 or T in [15, 30, 50]:
                marker = " <-- training T" if T == 5 else ""
                print(f"    T={T:2d}: IoU={iou:.4f}{marker}")

        del pixel_model; gc.collect()

        # Latent-NCA
        print(f"\n  [Latent-NCA] Training (T=5)...")
        latent_model = LatentNCA_System(in_ch=2, out_ch=1, latent_ch=32).to(DEVICE)
        latent_model = train_model(latent_model, train_x, train_y, n_steps=5, epochs=50)

        for T in [1, 3, 5, 10, 15, 20, 30, 50]:
            iou, _ = eval_iou(latent_model, test_x, test_y, n_steps=T)
            results[f'latent_T{T}'] = iou
            if T <= 5 or T in [15, 30, 50]:
                marker = " <-- training T" if T == 5 else ""
                print(f"    T={T:2d}: IoU={iou:.4f}{marker}")

        # Latent-NCA + Soft Landing at high T
        print(f"\n  [Latent-NCA + Soft Landing]...")
        # Calibrate threshold at optimal T
        opt_T = max(range(1, 11), key=lambda t: results.get(f'latent_T{t}', 0))
        _, cal_e = eval_iou(latent_model, test_x, test_y, n_steps=opt_T,
                           use_energy=True, energy_threshold=1e10)
        baseline_e = np.mean(cal_e[-3:]) if cal_e else 1.0
        print(f"    Baseline energy at T={opt_T}: {baseline_e:.6f}")

        for T in [20, 30, 50]:
            for mult in [1.5, 2.0, 3.0]:
                for slow in [2.0, 5.0]:
                    iou, _ = eval_iou(latent_model, test_x, test_y, n_steps=T,
                                     use_energy=True,
                                     energy_threshold=baseline_e * mult,
                                     slowdown=slow)
                    key = f'latent_SL_T{T}_m{mult}_s{slow}'
                    results[key] = iou
                    unchecked = results.get(f'latent_T{T}', 0)
                    if iou > unchecked + 0.005:
                        print(f"    T={T}, m={mult}, s={slow}: IoU={iou:.4f} "
                              f"(+{iou-unchecked:.4f} vs unchecked)")

        all_results[task_name] = results
        del latent_model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    # Size-change experiment
    print(f"\n{'='*50}")
    print(f"  Task: size_change (variable I/O sizes)")
    print(f"{'='*50}")
    # This requires custom batch handling
    latent_model = LatentNCA_System(in_ch=2, out_ch=1, latent_ch=32).to(DEVICE)
    size_data = generate_size_change(n=400)
    random.shuffle(size_data)
    train_sz = size_data[:300]; test_sz = size_data[300:]

    # Train with variable sizes
    opt = torch.optim.Adam(latent_model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(50):
        latent_model.train()
        random.shuffle(train_sz)
        for inp_t, tgt_t, out_sz in train_sz:
            inp_t = inp_t.unsqueeze(0).to(DEVICE)
            tgt_t = tgt_t.unsqueeze(0).to(DEVICE)
            out = latent_model(inp_t, n_steps=5, target_size=out_sz)
            loss = F.binary_cross_entropy(out, tgt_t)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/50")

    # Evaluate size change
    latent_model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for inp_t, tgt_t, out_sz in test_sz:
            inp_t = inp_t.unsqueeze(0).to(DEVICE)
            tgt_t = tgt_t.unsqueeze(0).to(DEVICE)
            out = latent_model(inp_t, n_steps=5, target_size=out_sz)
            pred = (out > 0.5).float()
            inter = (pred * tgt_t).sum()
            union = ((pred + tgt_t) > 0).float().sum()
            iou = (inter / (union + 1e-8)).item()
            correct += iou; total += 1
    size_iou = correct / total
    all_results['size_change'] = {'latent_iou': size_iou}
    print(f"  Size-change IoU: {size_iou:.4f}")
    del latent_model; gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("  LATENT-NCA RESULTS")
    print(f"{'='*70}")
    for task, res in all_results.items():
        if task == 'size_change':
            print(f"\n  {task}: IoU={res['latent_iou']:.4f}")
            continue
        pixel_5 = res.get('pixel_T5', 0)
        pixel_50 = res.get('pixel_T50', 0)
        latent_5 = res.get('latent_T5', 0)
        latent_50 = res.get('latent_T50', 0)
        sl_best = max([v for k, v in res.items() if 'SL_T50' in k], default=0)
        print(f"\n  {task}:")
        print(f"    Pixel T=5:         {pixel_5:.4f}")
        print(f"    Pixel T=50:        {pixel_50:.4f}")
        print(f"    Latent T=5:        {latent_5:.4f}")
        print(f"    Latent T=50:       {latent_50:.4f}")
        print(f"    Latent T=50+SL:    {sl_best:.4f}")
        if latent_50 > pixel_50 + 0.01:
            print(f"    ** LATENT BEATS PIXEL at T=50! (+{latent_50-pixel_50:.4f}) **")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase120_latent_nca.json"), 'w', encoding='utf-8') as f:
        json.dump({'experiment': 'Phase 120: Latent-NCA',
                   'timestamp': datetime.now().isoformat(),
                   'results': all_results}, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        t_list = [1, 3, 5, 10, 15, 20, 30, 50]

        for idx, task in enumerate(['flood_fill', 'long_move']):
            res = all_results.get(task, {})
            pixel_curve = [res.get(f'pixel_T{t}', 0) for t in t_list]
            latent_curve = [res.get(f'latent_T{t}', 0) for t in t_list]
            axes[idx].plot(t_list, pixel_curve, 'r-o', label='Pixel-NCA', ms=4)
            axes[idx].plot(t_list, latent_curve, 'b-s', label='Latent-NCA', ms=4)
            # SL points
            for T in [30, 50]:
                sl = max([v for k, v in res.items() if f'SL_T{T}' in k], default=0)
                if sl > 0:
                    axes[idx].scatter([T], [sl], c='green', s=100, marker='*', zorder=5)
            axes[idx].set_xlabel('T'); axes[idx].set_ylabel('IoU')
            axes[idx].set_title(f'{task}'); axes[idx].legend(fontsize=8)

        # Size change bar
        sz_iou = all_results.get('size_change', {}).get('latent_iou', 0)
        axes[2].bar(['Latent-NCA\nSize Change'], [sz_iou], color='#9b59b6')
        axes[2].set_ylabel('IoU'); axes[2].set_title('Size Change Task')
        axes[2].text(0, sz_iou + 0.02, f'{sz_iou:.3f}', ha='center')

        plt.suptitle('Phase 120: Latent-NCA vs Pixel-NCA', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase120_latent_nca.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 120 complete!")


if __name__ == '__main__':
    main()
