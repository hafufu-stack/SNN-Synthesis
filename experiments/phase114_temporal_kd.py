"""
Phase 114: Temporal Knowledge Distillation

Recovers Phase 112's failure by using LEARNING (not algebra) to
compress spatial depth into temporal steps.

Teacher: Trained deep CNN (ResNet-N)
Student: 1-layer L-NCA run for T steps
Loss: KL divergence on output logits + MSE on intermediate features

Proves: "Spatial depth CAN be folded into temporal loops via distillation"

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
# Teacher: ResNet with intermediate feature extraction
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
    def __init__(self, n_blocks=4, channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)
        self.n_blocks = n_blocks
        self.channels = channels

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

    def forward_with_intermediates(self, x):
        """Return logits + intermediate feature maps after each block."""
        intermediates = []
        x = self.stem(x)
        intermediates.append(x.detach())
        for block in self.blocks:
            x = block(x)
            intermediates.append(x.detach())
        pooled = self.pool(x).view(x.size(0), -1)
        logits = self.fc(pooled)
        return logits, intermediates


# ====================================================================
# Student: L-NCA with temporal unrolling
# ====================================================================
class StudentNCA(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        # SINGLE update rule (the whole point: 1 rule, T steps)
        self.update = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)

    def forward(self, x, n_steps=4):
        state = self.stem(x)
        for t in range(n_steps):
            state = F.relu(state + self.update(state))  # residual NCA step
        pooled = self.pool(state).view(state.size(0), -1)
        return self.fc(pooled)

    def forward_with_intermediates(self, x, n_steps=4):
        """Return logits + state after each step."""
        intermediates = []
        state = self.stem(x)
        intermediates.append(state)
        for t in range(n_steps):
            state = F.relu(state + self.update(state))
            intermediates.append(state)
        pooled = self.pool(state).view(state.size(0), -1)
        logits = self.fc(pooled)
        return logits, intermediates


# ====================================================================
# Training functions
# ====================================================================
def train_teacher(model, train_loader, epochs=30, lr=0.01):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Teacher epoch {epoch+1}/{epochs}")
    return model


def distill_student(student, teacher, train_loader, n_steps, epochs=30,
                    lr=0.001, temperature=4.0, alpha_kd=0.7, use_intermediate=False):
    """
    Train student NCA to mimic teacher CNN via knowledge distillation.

    Loss = alpha_kd * KL(teacher_soft || student_soft) + (1-alpha_kd) * CE(student, labels)
         + beta * MSE(intermediate features)  [if use_intermediate]
    """
    opt = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    teacher.eval()

    for epoch in range(epochs):
        student.train()
        total_loss = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Teacher forward (no grad)
            with torch.no_grad():
                if use_intermediate:
                    t_logits, t_feats = teacher.forward_with_intermediates(x)
                else:
                    t_logits = teacher(x)

            # Student forward
            if use_intermediate:
                s_logits, s_feats = student.forward_with_intermediates(x, n_steps=n_steps)
            else:
                s_logits = student(x, n_steps=n_steps)

            # KD loss (soft targets)
            t_soft = F.log_softmax(t_logits / temperature, dim=1)
            s_soft = F.log_softmax(s_logits / temperature, dim=1)
            kd_loss = F.kl_div(s_soft, t_soft.exp(), reduction='batchmean') * (temperature ** 2)

            # Hard label loss
            ce_loss = F.cross_entropy(s_logits, y)

            loss = alpha_kd * kd_loss + (1 - alpha_kd) * ce_loss

            # Intermediate feature matching (if enabled)
            if use_intermediate:
                # Match student step t with teacher block t (proportional mapping)
                n_match = min(len(s_feats), len(t_feats))
                feat_loss = 0
                for i in range(n_match):
                    si = min(i, len(s_feats) - 1)
                    ti = min(i, len(t_feats) - 1)
                    feat_loss += F.mse_loss(s_feats[si], t_feats[ti])
                loss += 0.1 * feat_loss / n_match

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * y.size(0); n += y.size(0)

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Distill epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f}")

    return student


def evaluate(model, test_loader, n_steps=None):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if n_steps is not None:
                out = model(x, n_steps=n_steps)
            else:
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    print("=" * 70)
    print("Phase 114: Temporal Knowledge Distillation")
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

    results = []

    for n_blocks in [4, 8]:
        print(f"\n{'='*50}")
        print(f"  Teacher: ResNet-{n_blocks}")
        print(f"{'='*50}")

        # Train teacher
        print(f"  [1/3] Training teacher CNN (ResNet-{n_blocks})...")
        teacher = TeacherCNN(n_blocks=n_blocks, channels=64).to(DEVICE)
        teacher = train_teacher(teacher, train_loader, epochs=30)
        teacher_acc = evaluate(teacher, test_loader)
        print(f"  Teacher accuracy: {teacher_acc*100:.2f}%")

        # Phase 112 baseline (no distillation, just random NCA)
        print(f"\n  [2/3] Phase 112 baseline (NCA without distillation)...")
        naive_nca = StudentNCA(channels=64).to(DEVICE)
        # Train NCA directly on labels (no teacher)
        opt = torch.optim.Adam(naive_nca.parameters(), lr=0.001)
        for epoch in range(30):
            naive_nca.train()
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = F.cross_entropy(naive_nca(x, n_steps=n_blocks), y)
                opt.zero_grad(); loss.backward(); opt.step()
        naive_acc = evaluate(naive_nca, test_loader, n_steps=n_blocks)
        print(f"  NCA (no distillation, T={n_blocks}): {naive_acc*100:.2f}%")
        del naive_nca; gc.collect()

        # Distilled NCA variants
        for use_intermediate in [False, True]:
            method = "KD+intermediate" if use_intermediate else "KD only"
            print(f"\n  [3/3] Distilling NCA from teacher ({method})...")

            student = StudentNCA(channels=64).to(DEVICE)
            student = distill_student(
                student, teacher, train_loader,
                n_steps=n_blocks, epochs=30,
                use_intermediate=use_intermediate
            )

            # Test at different T values
            for T in [n_blocks // 2, n_blocks, n_blocks * 2]:
                T = max(1, T)
                acc = evaluate(student, test_loader, n_steps=T)
                gap_teacher = acc - teacher_acc
                gap_naive = acc - naive_acc
                result = {
                    'teacher_blocks': n_blocks, 'method': method,
                    'T': T, 'teacher_acc': teacher_acc,
                    'naive_nca_acc': naive_acc, 'distilled_acc': acc,
                    'gap_vs_teacher': gap_teacher,
                    'gap_vs_naive': gap_naive,
                    'recovery_pct': (acc - naive_acc) / (teacher_acc - naive_acc) * 100
                    if teacher_acc > naive_acc else 0
                }
                results.append(result)
                recovery = result['recovery_pct']
                marker = " [STAR]" if recovery > 50 else ""
                print(f"    T={T:2d}: {acc*100:.2f}% "
                      f"(vs teacher: {gap_teacher*100:+.2f}%, "
                      f"vs naive: {gap_naive*100:+.2f}%, "
                      f"recovery: {recovery:.1f}%){marker}")

            del student; gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        del teacher; gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("  TEMPORAL KNOWLEDGE DISTILLATION RESULTS")
    print(f"{'='*70}")

    for n_blocks in [4, 8]:
        dr = [r for r in results if r['teacher_blocks'] == n_blocks]
        if not dr:
            continue
        teacher_a = dr[0]['teacher_acc']
        naive_a = dr[0]['naive_nca_acc']
        best = max(dr, key=lambda r: r['distilled_acc'])
        print(f"\n  ResNet-{n_blocks} -> NCA:")
        print(f"    Teacher:            {teacher_a*100:.2f}%")
        print(f"    Naive NCA:          {naive_a*100:.2f}%")
        print(f"    Best Distilled:     {best['distilled_acc']*100:.2f}% "
              f"({best['method']}, T={best['T']})")
        print(f"    Knowledge Recovery: {best['recovery_pct']:.1f}%")
        gap_112 = -(teacher_a - naive_a) * 100
        gap_114 = -(teacher_a - best['distilled_acc']) * 100
        print(f"    Phase 112 gap:      {gap_112:.1f}%")
        print(f"    Phase 114 gap:      {gap_114:.1f}%")
        if abs(gap_112) > 0:
            improvement = (1 - abs(gap_114) / abs(gap_112)) * 100
            print(f"    Gap reduction:      {improvement:.1f}%")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase114_temporal_kd.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 114: Temporal Knowledge Distillation',
            'timestamp': datetime.now().isoformat(),
            'all_results': results
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Comparison bar chart
        for idx, n_blocks in enumerate([4, 8]):
            dr = [r for r in results if r['teacher_blocks'] == n_blocks]
            if not dr:
                continue
            categories = ['Teacher\nCNN', 'Naive\nNCA', 'Distilled\n(KD)', 'Distilled\n(KD+feat)']
            vals = [
                dr[0]['teacher_acc'] * 100,
                dr[0]['naive_nca_acc'] * 100,
                max([r['distilled_acc'] for r in dr if r['method'] == 'KD only'], default=0) * 100,
                max([r['distilled_acc'] for r in dr if r['method'] == 'KD+intermediate'], default=0) * 100,
            ]
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
            x = np.arange(len(categories))
            axes[idx].bar(x, vals, color=colors)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(categories, fontsize=9)
            axes[idx].set_ylabel('Accuracy (%)')
            axes[idx].set_title(f'ResNet-{n_blocks} Distillation')
            for i, v in enumerate(vals):
                axes[idx].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=8)

        # Plot 3: Recovery percentage
        methods = sorted(set(r['method'] for r in results))
        for method in methods:
            mr = [r for r in results if r['method'] == method]
            teachers = sorted(set(r['teacher_blocks'] for r in mr))
            recoveries = []
            for tb in teachers:
                best = max([r for r in mr if r['teacher_blocks'] == tb],
                          key=lambda r: r['recovery_pct'])
                recoveries.append(best['recovery_pct'])
            axes[2].plot(teachers, recoveries, 'o-', label=method, markersize=8)
        axes[2].set_xlabel('Teacher depth (blocks)')
        axes[2].set_ylabel('Knowledge Recovery (%)')
        axes[2].set_title('Distillation Recovery Rate')
        axes[2].legend()
        axes[2].axhline(y=100, color='green', linestyle='--', alpha=0.3, label='Perfect')
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.3)

        plt.suptitle('Phase 114: Temporal Knowledge Distillation\n'
                     '(Recovering Phase 112 failure via learning)', fontsize=13)
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase114_temporal_kd.png'), dpi=150)
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    print("\nPhase 114 complete!")


if __name__ == '__main__':
    main()
