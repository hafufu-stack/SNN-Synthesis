"""
Phase 117: Time-Conditioned Knowledge Distillation

Phase 114 recovered 29.9% of teacher knowledge via distillation.
The bottleneck: NCA uses the SAME weights at every step, but CNN
uses DIFFERENT weights at each layer.

Solution: Give the NCA a CLOCK -- a time channel (t/T) that lets
the tau gate know which "layer" to emulate at each step.

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
# Teacher CNN (same as Phase 114)
# ====================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class TeacherCNN(nn.Module):
    def __init__(self, n_blocks=8, channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)
        self.n_blocks = n_blocks; self.channels = channels

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

    def forward_with_intermediates(self, x):
        intermediates = []
        x = self.stem(x)
        intermediates.append(x.detach())
        for block in self.blocks:
            x = block(x)
            intermediates.append(x.detach())
        pooled = self.pool(x).view(x.size(0), -1)
        return self.fc(pooled), intermediates


# ====================================================================
# Student NCA variants
# ====================================================================
class StudentNCA_NoTime(nn.Module):
    """NCA WITHOUT time conditioning (Phase 114 baseline)."""
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU()
        )
        self.update = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)

    def forward(self, x, n_steps=8):
        state = self.stem(x)
        for t in range(n_steps):
            state = F.relu(state + self.update(state))
        return self.fc(self.pool(state).view(state.size(0), -1))

    def forward_with_intermediates(self, x, n_steps=8):
        intermediates = []
        state = self.stem(x)
        intermediates.append(state)
        for t in range(n_steps):
            state = F.relu(state + self.update(state))
            intermediates.append(state)
        return self.fc(self.pool(state).view(state.size(0), -1)), intermediates


class StudentNCA_TimeCond(nn.Module):
    """NCA WITH time conditioning -- gives the cell a clock."""
    def __init__(self, channels=64, time_embed_dim=16):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU()
        )

        # Time embedding: scalar t/T -> time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Update rule takes channels + time_embed_dim -> channels
        self.update_conv1 = nn.Conv2d(channels + time_embed_dim, channels, 3, padding=1, bias=False)
        self.update_bn1 = nn.BatchNorm2d(channels)
        self.update_conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.update_bn2 = nn.BatchNorm2d(channels)

        # Tau gate also time-conditioned
        self.tau_conv = nn.Conv2d(channels + time_embed_dim, channels, 1)
        self.tau_sigmoid = nn.Sigmoid()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)

    def _get_time_embed(self, t, T, batch_size, H, W, device):
        """Create time embedding broadcast to spatial dims."""
        t_normalized = torch.tensor([[t / T]], device=device, dtype=torch.float32)
        t_embed = self.time_mlp(t_normalized)  # (1, time_embed_dim)
        # Broadcast to (B, time_embed_dim, H, W)
        t_spatial = t_embed.view(1, self.time_embed_dim, 1, 1)
        t_spatial = t_spatial.expand(batch_size, -1, H, W)
        return t_spatial

    def forward(self, x, n_steps=8):
        state = self.stem(x)
        B, C, H, W = state.shape

        for t in range(n_steps):
            t_embed = self._get_time_embed(t, n_steps, B, H, W, state.device)
            # Concatenate state + time
            state_time = torch.cat([state, t_embed], dim=1)

            # Update with time awareness
            delta = F.relu(self.update_bn1(self.update_conv1(state_time)))
            delta = self.update_bn2(self.update_conv2(delta))

            # Tau gate with time awareness
            beta = self.tau_sigmoid(self.tau_conv(state_time))

            state = beta * state + (1 - beta) * F.relu(state + delta)

        return self.fc(self.pool(state).view(B, -1))

    def forward_with_intermediates(self, x, n_steps=8):
        intermediates = []
        state = self.stem(x)
        B, C, H, W = state.shape
        intermediates.append(state)

        for t in range(n_steps):
            t_embed = self._get_time_embed(t, n_steps, B, H, W, state.device)
            state_time = torch.cat([state, t_embed], dim=1)
            delta = F.relu(self.update_bn1(self.update_conv1(state_time)))
            delta = self.update_bn2(self.update_conv2(delta))
            beta = self.tau_sigmoid(self.tau_conv(state_time))
            state = beta * state + (1 - beta) * F.relu(state + delta)
            intermediates.append(state)

        return self.fc(self.pool(state).view(B, -1)), intermediates


# ====================================================================
# Distillation
# ====================================================================
def distill(student, teacher, train_loader, n_steps, epochs=40,
            lr=0.001, temperature=4.0, alpha_kd=0.7):
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    teacher.eval()

    for epoch in range(epochs):
        student.train()
        total_loss = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_logits, t_feats = teacher.forward_with_intermediates(x)
            s_logits, s_feats = student.forward_with_intermediates(x, n_steps=n_steps)

            # KD loss
            t_soft = F.log_softmax(t_logits / temperature, dim=1)
            s_soft = F.log_softmax(s_logits / temperature, dim=1)
            kd_loss = F.kl_div(s_soft, t_soft.exp(), reduction='batchmean') * (temperature ** 2)
            ce_loss = F.cross_entropy(s_logits, y)
            loss = alpha_kd * kd_loss + (1 - alpha_kd) * ce_loss

            # Intermediate feature matching (proportional mapping)
            n_match = min(len(s_feats), len(t_feats))
            feat_loss = 0
            for i in range(n_match):
                si = min(i, len(s_feats) - 1)
                ti = min(i, len(t_feats) - 1)
                feat_loss += F.mse_loss(s_feats[si], t_feats[ti])
            loss += 0.1 * feat_loss / max(n_match, 1)

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * y.size(0); n += y.size(0)

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}")

    return student


def evaluate(model, test_loader, n_steps=None):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x, n_steps=n_steps) if n_steps else model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 117: Time-Conditioned Knowledge Distillation")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_ds = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10('data', train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    n_blocks = 8
    T = 8

    # Step 1: Train teacher
    print(f"\n[Step 1] Training teacher ResNet-{n_blocks}...")
    teacher = TeacherCNN(n_blocks=n_blocks, channels=64).to(DEVICE)
    opt = torch.optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for epoch in range(30):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(teacher(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Teacher epoch {epoch+1}/30")
    teacher_acc = evaluate(teacher, test_loader)
    print(f"  Teacher accuracy: {teacher_acc*100:.2f}%")

    # Step 2: Distill WITHOUT time (Phase 114 baseline)
    print(f"\n[Step 2] Distilling NCA WITHOUT time conditioning...")
    student_no_time = StudentNCA_NoTime(channels=64).to(DEVICE)
    student_no_time = distill(student_no_time, teacher, train_loader, n_steps=T, epochs=40)
    results = {}

    for test_T in [T//2, T, T*2]:
        test_T = max(1, test_T)
        acc = evaluate(student_no_time, test_loader, n_steps=test_T)
        results[f'no_time_T{test_T}'] = acc
        print(f"    NoTime T={test_T}: {acc*100:.2f}% (gap={((acc-teacher_acc)*100):+.2f}%)")

    del student_no_time; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # Step 3: Distill WITH time conditioning (THE KEY EXPERIMENT)
    print(f"\n[Step 3] Distilling NCA WITH time conditioning (Phase 117)...")
    student_time = StudentNCA_TimeCond(channels=64, time_embed_dim=16).to(DEVICE)
    student_time = distill(student_time, teacher, train_loader, n_steps=T, epochs=40)

    for test_T in [T//2, T, T*2]:
        test_T = max(1, test_T)
        acc = evaluate(student_time, test_loader, n_steps=test_T)
        results[f'time_cond_T{test_T}'] = acc
        print(f"    TimeCond T={test_T}: {acc*100:.2f}% (gap={((acc-teacher_acc)*100):+.2f}%)")

    del student_time; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    # Step 4: Direct CE training comparison
    print(f"\n[Step 4] NCA trained directly (no distillation, T={T})...")
    direct_no_time = StudentNCA_NoTime(channels=64).to(DEVICE)
    opt = torch.optim.Adam(direct_no_time.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    for epoch in range(40):
        direct_no_time.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(direct_no_time(x, n_steps=T), y)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
    direct_acc = evaluate(direct_no_time, test_loader, n_steps=T)
    results['direct_no_kd'] = direct_acc
    print(f"    Direct NCA (no KD): {direct_acc*100:.2f}%")

    direct_time = StudentNCA_TimeCond(channels=64, time_embed_dim=16).to(DEVICE)
    opt = torch.optim.Adam(direct_time.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    for epoch in range(40):
        direct_time.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(direct_time(x, n_steps=T), y)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
    direct_time_acc = evaluate(direct_time, test_loader, n_steps=T)
    results['direct_time_cond'] = direct_time_acc
    print(f"    Direct NCA + Time: {direct_time_acc*100:.2f}%")

    del direct_no_time, direct_time; gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print("  TIME-CONDITIONED KD RESULTS")
    print(f"{'='*70}")
    print(f"  Teacher (ResNet-{n_blocks}):      {teacher_acc*100:.2f}%")
    print(f"  Direct NCA (no time):       {results['direct_no_kd']*100:.2f}%")
    print(f"  Direct NCA (+ time):        {results['direct_time_cond']*100:.2f}%")
    notime_best = results.get(f'no_time_T{T}', 0)
    time_best = results.get(f'time_cond_T{T}', 0)
    print(f"  KD NCA (no time, T={T}):     {notime_best*100:.2f}%")
    print(f"  KD NCA (+ time, T={T}):      {time_best*100:.2f}%")

    improvement = (time_best - notime_best) * 100
    print(f"\n  Time conditioning effect: {improvement:+.2f}%")

    notime_recovery = (notime_best - results['direct_no_kd']) / (teacher_acc - results['direct_no_kd']) * 100 if teacher_acc > results['direct_no_kd'] else 0
    time_recovery = (time_best - results['direct_time_cond']) / (teacher_acc - results['direct_time_cond']) * 100 if teacher_acc > results['direct_time_cond'] else 0
    print(f"  Recovery (no time):  {notime_recovery:.1f}%")
    print(f"  Recovery (+ time):   {time_recovery:.1f}%")
    print(f"{'='*70}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase117_time_cond_kd.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 117: Time-Conditioned KD',
            'timestamp': datetime.now().isoformat(),
            'teacher_acc': teacher_acc,
            'results': results,
            'time_improvement': improvement,
            'notime_recovery': notime_recovery,
            'time_recovery': time_recovery
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Comparison bar
        methods = ['Teacher\nResNet-8', 'Direct\nNCA', 'Direct\nNCA+Time', 'KD\nNCA', 'KD\nNCA+Time']
        vals = [teacher_acc*100, results['direct_no_kd']*100, results['direct_time_cond']*100,
                notime_best*100, time_best*100]
        colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db', '#9b59b6']
        axes[0].bar(range(len(methods)), vals, color=colors)
        axes[0].set_xticks(range(len(methods)))
        axes[0].set_xticklabels(methods, fontsize=8)
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Time-Conditioning Effect on KD')
        for i, v in enumerate(vals):
            axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)

        # T-curve comparison
        for prefix, label, color in [('no_time', 'KD (no time)', '#3498db'),
                                       ('time_cond', 'KD (+ time)', '#9b59b6')]:
            ts = []; accs = []
            for k, v in results.items():
                if k.startswith(prefix + '_T'):
                    t = int(k.split('T')[-1])
                    ts.append(t); accs.append(v * 100)
            if ts:
                axes[1].plot(sorted(ts), [a for _, a in sorted(zip(ts, accs))],
                            'o-', label=label, color=color, markersize=6)
        axes[1].axhline(y=teacher_acc*100, color='green', linestyle='--', label='Teacher')
        axes[1].set_xlabel('T (steps)'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy vs T')
        axes[1].legend()

        plt.suptitle('Phase 117: Time-Conditioned Knowledge Distillation', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase117_time_cond_kd.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 117 complete!")


if __name__ == '__main__':
    main()
