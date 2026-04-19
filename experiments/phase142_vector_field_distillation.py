"""
Phase 142: Vector Field Distillation — Flow Matching for CNN→NCA

Instead of matching outputs (Phase 114: 29.9%), match the RATE OF CHANGE:
  - Teacher CNN: residual Δ_l = x_{l+1} - x_l at each layer l
  - Student NCA: update Δ_t = s_{t+1} - s_t at each step t

By matching the "velocity field" of computation, we perform Flow Matching
that smoothly folds spatial depth into temporal iteration.

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
DEPTH = 8
CHANNELS = 32


# ================================================================
# Teacher: Normal CNN with residual connections
# ================================================================
class TeacherCNN(nn.Module):
    def __init__(self, channels=CHANNELS, depth=DEPTH):
        super().__init__()
        self.proj_in = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
            ) for _ in range(depth)
        ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 10))

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        for block in self.blocks:
            h = F.relu(h + block(h))
        return self.classifier(h)

    def get_residuals(self, x):
        """Return list of residuals Δ_l = block(h) for each layer."""
        h = F.relu(self.proj_in(x))
        residuals = []
        states = [h.clone()]
        for block in self.blocks:
            delta = block(h)
            residuals.append(delta)
            h = F.relu(h + delta)
            states.append(h.clone())
        return residuals, states


# ================================================================
# Student: Single-block NCA
# ================================================================
class StudentNCA(nn.Module):
    def __init__(self, channels=CHANNELS, steps=DEPTH):
        super().__init__()
        self.steps = steps
        self.proj_in = nn.Conv2d(1, channels, 3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 10))

    def forward(self, x):
        h = F.relu(self.proj_in(x))
        for _ in range(self.steps):
            h = F.relu(h + self.block(h))
        return self.classifier(h)

    def get_updates(self, x):
        """Return list of updates Δ_t = block(h) for each step."""
        h = F.relu(self.proj_in(x))
        updates = []
        states = [h.clone()]
        for _ in range(self.steps):
            delta = self.block(h)
            updates.append(delta)
            h = F.relu(h + delta)
            states.append(h.clone())
        return updates, states


# ================================================================
# Training
# ================================================================
def train_teacher(model, train_loader, epochs=10):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()


def distill_output_only(student, teacher, train_loader, epochs=15, lr=1e-3):
    """Baseline: match only the final logits (Phase 114 approach)."""
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    for epoch in range(epochs):
        student.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            # KD loss + CE loss
            loss_kd = F.mse_loss(student_logits, teacher_logits)
            loss_ce = F.cross_entropy(student_logits, y)
            loss = 0.7 * loss_kd + 0.3 * loss_ce
            opt.zero_grad(); loss.backward(); opt.step()


def distill_vector_field(student, teacher, train_loader, epochs=15, lr=1e-3):
    """NEW: match the velocity field (residuals) at each step."""
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    teacher.eval()
    for epoch in range(epochs):
        student.train()
        total_vf_loss = 0; total_ce_loss = 0; n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Get teacher residuals and student updates
            with torch.no_grad():
                teacher_residuals, teacher_states = teacher.get_residuals(x)
            student_updates, student_states = student.get_updates(x)

            # Vector field matching: align Δ_l with Δ_t
            vf_loss = 0
            n_match = min(len(teacher_residuals), len(student_updates))
            for i in range(n_match):
                vf_loss += F.mse_loss(student_updates[i], teacher_residuals[i])
            vf_loss /= n_match

            # Also match final states
            state_loss = F.mse_loss(student_states[-1], teacher_states[-1])

            # Classification loss
            student_logits = student(x)
            ce_loss = F.cross_entropy(student_logits, y)

            loss = 0.4 * vf_loss + 0.3 * state_loss + 0.3 * ce_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_vf_loss += vf_loss.item()
            total_ce_loss += ce_loss.item(); n += 1


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 142: Vector Field Distillation (Flow Matching)")
    print("=" * 70)

    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # Step 1: Train teacher CNN
    print(f"\n[Step 1] Training Teacher CNN (depth={DEPTH})...")
    teacher = TeacherCNN().to(DEVICE)
    n_teacher = sum(p.numel() for p in teacher.parameters())
    train_teacher(teacher, train_loader, epochs=10)
    teacher_acc = evaluate(teacher, test_loader)
    print(f"  Teacher accuracy: {teacher_acc*100:.2f}% ({n_teacher:,} params)")

    # Step 2: Baseline — output-only distillation
    print(f"\n[Step 2] Output-only distillation (Phase 114 approach)...")
    student_baseline = StudentNCA().to(DEVICE)
    n_student = sum(p.numel() for p in student_baseline.parameters())
    distill_output_only(student_baseline, teacher, train_loader, epochs=15)
    baseline_acc = evaluate(student_baseline, test_loader)
    recovery_baseline = baseline_acc / teacher_acc * 100
    print(f"  Baseline NCA: {baseline_acc*100:.2f}% (recovery: {recovery_baseline:.1f}%)")

    # Step 3: Vector Field distillation
    print(f"\n[Step 3] Vector Field distillation (Flow Matching)...")
    student_vf = StudentNCA().to(DEVICE)
    distill_vector_field(student_vf, teacher, train_loader, epochs=15)
    vf_acc = evaluate(student_vf, test_loader)
    recovery_vf = vf_acc / teacher_acc * 100
    print(f"  VF-NCA: {vf_acc*100:.2f}% (recovery: {recovery_vf:.1f}%)")

    # Step 4: Train NCA from scratch (no distillation)
    print(f"\n[Step 4] NCA trained from scratch (no teacher)...")
    student_scratch = StudentNCA().to(DEVICE)
    opt = torch.optim.Adam(student_scratch.parameters(), lr=1e-3)
    for epoch in range(15):
        student_scratch.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(student_scratch(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    scratch_acc = evaluate(student_scratch, test_loader)
    print(f"  Scratch NCA: {scratch_acc*100:.2f}%")

    # Summary
    elapsed = time.time() - t0
    improvement = vf_acc - baseline_acc
    print(f"\n{'='*70}")
    print(f"Phase 142 Complete ({elapsed:.0f}s)")
    print(f"  Teacher (8-layer CNN):       {teacher_acc*100:.2f}% ({n_teacher:,} params)")
    print(f"  NCA from scratch:            {scratch_acc*100:.2f}% ({n_student:,} params)")
    print(f"  Output-only distillation:    {baseline_acc*100:.2f}% (recovery: {recovery_baseline:.1f}%)")
    print(f"  Vector Field distillation:   {vf_acc*100:.2f}% (recovery: {recovery_vf:.1f}%)")
    print(f"  VF improvement over output:  {improvement*100:+.2f}%")
    print(f"{'='*70}")

    results = {
        'teacher_acc': teacher_acc, 'teacher_params': n_teacher,
        'baseline_acc': baseline_acc, 'recovery_baseline': recovery_baseline,
        'vf_acc': vf_acc, 'recovery_vf': recovery_vf,
        'scratch_acc': scratch_acc, 'student_params': n_student,
        'improvement': improvement
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase142_vector_field_distillation.json"), 'w',
              encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 142: Vector Field Distillation',
            'timestamp': datetime.now().isoformat(),
            **results, 'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: Accuracy comparison
        names = ['Teacher\nCNN (8L)', 'NCA\nScratch', 'Output\nDistill', 'Vector\nField']
        accs = [teacher_acc*100, scratch_acc*100, baseline_acc*100, vf_acc*100]
        colors = ['#e74c3c', '#95a5a6', '#f39c12', '#2ecc71']
        bars = axes[0].bar(range(4), accs, color=colors, alpha=0.85, edgecolor='black')
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                        f'{acc:.1f}%', ha='center', fontweight='bold')
        axes[0].set_xticks(range(4)); axes[0].set_xticklabels(names)
        axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Distillation Comparison', fontweight='bold')

        # Panel 2: Recovery rates
        methods = ['Output\nDistillation', 'Vector Field\nDistillation']
        recoveries = [recovery_baseline, recovery_vf]
        bars = axes[1].bar(range(2), recoveries, color=['#f39c12', '#2ecc71'],
                          alpha=0.85, edgecolor='black')
        axes[1].axhline(y=100, color='red', linestyle='--', label='100% (teacher)')
        for bar, rec in zip(bars, recoveries):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{rec:.1f}%', ha='center', fontweight='bold')
        axes[1].set_xticks(range(2)); axes[1].set_xticklabels(methods)
        axes[1].set_ylabel('Recovery Rate (%)'); axes[1].set_title('Knowledge Recovery', fontweight='bold')
        axes[1].legend()

        # Panel 3: Conceptual diagram
        axes[2].set_xlim(0, 10); axes[2].set_ylim(0, 8)
        axes[2].set_title('Vector Field Distillation', fontweight='bold')
        # Teacher (vertical stack)
        for i in range(DEPTH):
            y = 0.5 + i * 0.85
            axes[2].add_patch(plt.Rectangle((0.5, y), 3, 0.7, facecolor='#e74c3c', alpha=0.3+i*0.08))
            if i < DEPTH-1:
                axes[2].annotate('', xy=(2, y), xytext=(2, y+0.7),
                               arrowprops=dict(arrowstyle='->', color='red', lw=1))
        axes[2].text(2, 7.7, 'Teacher CNN\n(spatial depth)', ha='center', fontsize=9)
        # Student (single block looping)
        axes[2].add_patch(plt.Rectangle((6, 3), 3, 2, facecolor='#2ecc71', alpha=0.5))
        axes[2].annotate('', xy=(9.2, 4), xytext=(9.2, 4),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=1.5',
                                       color='green', lw=2))
        axes[2].text(7.5, 4, f'NCA\n(T={DEPTH} steps)', ha='center', fontsize=9)
        # Arrow between
        axes[2].annotate('Δ_l ≈ Δ_t', xy=(6, 4), xytext=(3.5, 4),
                        arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                        fontsize=10, color='purple', fontweight='bold')
        axes[2].axis('off')

        plt.suptitle('Phase 142: Vector Field Distillation — Matching the Flow of Thought',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'phase142_vector_field_distillation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    return results


if __name__ == '__main__':
    main()
