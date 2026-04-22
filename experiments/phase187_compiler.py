"""
Phase 187: Target-to-Action Compiler - NCA Prediction to ARC Actions

Test offline: Given an ARC task, use Foundation NCA to predict target
output grid, then compile the diff into a sequence of ARC actions.

This validates the compile-then-paint paradigm before deploying to Kaggle.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026
EMB_DIM = 64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import (
    FoundationSystem, load_arc_training, prepare_arc_meta_dataset
)


def nca_predict_target(model, item, n_steps=8, n_candidates=5, noise_std=0.2):
    """Use NCA to predict target output grid.
    
    Returns: predicted grid (H, W) of color indices 0-9
    """
    di = [d.to(DEVICE) for d in item['demo_inputs']]
    do = [d.to(DEVICE) for d in item['demo_outputs']]
    oh, ow = item['out_h'], item['out_w']

    with torch.no_grad():
        seed_emb = model.task_encoder(di, do)
        ti = item['test_input'].unsqueeze(0).to(DEVICE)

        best_margin = -float('inf')
        best_pred = None

        for i in range(n_candidates):
            emb = seed_emb if i == 0 else seed_emb + torch.randn_like(seed_emb) * noise_std
            logits = model.latent_nca(ti, emb, n_steps=n_steps)
            raw = logits[0, :10, :oh, :ow]

            # Margin-based selection (P186)
            sorted_l, _ = raw.sort(dim=0, descending=True)
            margin = (sorted_l[0] - sorted_l[1]).mean().item()

            if margin > best_margin:
                best_margin = margin
                best_pred = raw.argmax(dim=0)  # (H, W) color indices

    return best_pred.cpu().numpy()


def compile_diff_to_actions(current_grid, target_grid):
    """Compile grid diff into action sequence.
    
    For each pixel where current != target:
    -> (y, x, target_color)
    
    Returns list of (y, x, color) tuples.
    """
    h, w = target_grid.shape
    actions = []
    for y in range(h):
        for x in range(w):
            if current_grid[y, x] != target_grid[y, x]:
                actions.append((y, x, int(target_grid[y, x])))
    return actions


def evaluate_compiler(model, test_tasks):
    """Evaluate how accurately NCA predicts targets and how efficient the action plan is."""
    results = {
        'pa': [], 'em': [], 'n_actions': [], 'grid_sizes': [],
        'correct_actions': [], 'wrong_actions': []
    }

    for item in test_tasks:
        oh, ow = item['out_h'], item['out_w']
        gt = item['test_output'][:10].argmax(dim=0).numpy()[:oh, :ow]

        # NCA prediction
        predicted = nca_predict_target(model, item)

        # Accuracy
        pa = (predicted == gt).mean()
        em = float((predicted == gt).all())
        results['pa'].append(pa)
        results['em'].append(em)
        results['grid_sizes'].append(oh * ow)

        # Compile diff from blank grid (all zeros) to predicted
        blank = np.zeros((oh, ow), dtype=np.int32)
        actions = compile_diff_to_actions(blank, predicted)
        results['n_actions'].append(len(actions))

        # How many of these actions would also be needed for the true target?
        true_actions = compile_diff_to_actions(blank, gt)
        true_set = set((y, x, c) for y, x, c in true_actions)
        pred_set = set((y, x, c) for y, x, c in actions)
        correct = len(true_set & pred_set)
        wrong = len(pred_set - true_set)
        results['correct_actions'].append(correct)
        results['wrong_actions'].append(wrong)

    return results


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 187: Target-to-Action Compiler")
    print(f"  NCA predicts target grid -> compile to action sequence")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Loading Foundation Model...")
    model = FoundationSystem(embed_dim=64, hidden_ch=64, latent_ch=32).to(DEVICE)
    ckpt_path = os.path.join(RESULTS_DIR, "phase123_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print("\n[Step 2] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    test_tasks = all_tasks[:50]

    print("\n[Step 3] Evaluating NCA prediction + action compilation...")
    results = evaluate_compiler(model, test_tasks)

    avg_pa = np.mean(results['pa'])
    avg_em = np.mean(results['em'])
    avg_actions = np.mean(results['n_actions'])
    avg_correct = np.mean(results['correct_actions'])
    avg_wrong = np.mean(results['wrong_actions'])
    action_precision = avg_correct / max(1, avg_correct + avg_wrong)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 187 Complete ({elapsed:.0f}s)")
    print(f"  NCA Prediction:   PA={avg_pa*100:.2f}%, EM={avg_em*100:.1f}%")
    print(f"  Actions per task: {avg_actions:.1f} (avg grid: {np.mean(results['grid_sizes']):.0f}px)")
    print(f"  Action precision: {action_precision*100:.1f}% "
          f"(correct={avg_correct:.1f}, wrong={avg_wrong:.1f})")
    print(f"  -> If agent paints NCA prediction: {action_precision*100:.1f}% of clicks are correct")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase187_compiler.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 187: Target-to-Action Compiler',
            'timestamp': datetime.now().isoformat(),
            'pa': avg_pa, 'em': avg_em,
            'avg_actions': avg_actions,
            'action_precision': action_precision,
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        # PA distribution
        axes[0].hist(np.array(results['pa'])*100, bins=20, color='#2ecc71', alpha=0.85, edgecolor='black')
        axes[0].axvline(avg_pa*100, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('PA (%)'); axes[0].set_ylabel('Count')
        axes[0].set_title(f'NCA Prediction Accuracy (mean={avg_pa*100:.1f}%)', fontweight='bold')
        # Actions vs grid size
        axes[1].scatter(results['grid_sizes'], results['n_actions'], alpha=0.6, color='#3498db')
        axes[1].set_xlabel('Grid Size (pixels)'); axes[1].set_ylabel('N Actions')
        axes[1].set_title('Actions Needed', fontweight='bold')
        # Correct vs wrong actions
        axes[2].scatter(results['correct_actions'], results['wrong_actions'],
                       alpha=0.6, color='#e74c3c')
        axes[2].set_xlabel('Correct Actions'); axes[2].set_ylabel('Wrong Actions')
        axes[2].set_title(f'Action Precision ({action_precision*100:.0f}%)', fontweight='bold')
        axes[2].plot([0, max(results['correct_actions'])],
                    [0, max(results['correct_actions'])], 'k--', alpha=0.3)
        fig.suptitle('Phase 187: NCA Target-to-Action Compiler', fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase187_compiler.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return {'pa': avg_pa, 'action_precision': action_precision}

if __name__ == '__main__':
    main()
