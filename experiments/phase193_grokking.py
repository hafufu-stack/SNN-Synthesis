"""
Phase 193: Leviathan Grokking - Sudden Understanding via Overtraining

Train a large NCA (C=256) on few ARC tasks for thousands of epochs.
Initially: memorizes training, poor test. After long training:
Grokking = sudden jump in test accuracy as model discovers rules.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 2026

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase123_foundation import load_arc_training, prepare_arc_meta_dataset
from phase191_generalization import ScalableNCA


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 193: Leviathan Grokking - Sudden Understanding")
    print(f"  Long overtraining to trigger phase transition")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    print("\n[Step 1] Loading ARC data...")
    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)

    # Few training tasks, separate test tasks
    train_tasks = all_tasks[:30]   # small set to encourage memorization
    test_tasks = all_tasks[30:80]  # larger test set

    # Test multiple model sizes
    configs = [
        {'C': 64, 'label': 'Medium', 'max_epochs': 2000},
        {'C': 256, 'label': 'Large', 'max_epochs': 2000},
    ]

    all_results = {}

    for cfg in configs:
        C = cfg['C']
        max_epochs = cfg['max_epochs']
        label = cfg['label']

        print(f"\n[{label} C={C}] Training for {max_epochs} epochs...", flush=True)
        torch.manual_seed(SEED)
        model = ScalableNCA(11, C, n_steps=5, embed_dim=32).to(DEVICE)
        n_params = model.count_params()
        print(f"  Params: {n_params:,}")

        # Weight decay helps Grokking (regularization forces generalization)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=200)

        history = []

        for epoch in range(max_epochs):
            model.train()
            train_loss, train_pa = 0, 0

            for item in train_tasks:
                do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.encode_task(do_t)
                ti = item['test_input'].unsqueeze(0).to(DEVICE)
                gt = item['test_output'][:10].argmax(dim=0).unsqueeze(0).to(DEVICE)
                oh, ow = item['out_h'], item['out_w']

                logits = model(ti, emb)
                loss = F.cross_entropy(logits[:, :10, :oh, :ow], gt[:, :oh, :ow])
                opt.zero_grad(); loss.backward(); opt.step()
                train_loss += loss.item()

                with torch.no_grad():
                    pred = logits[0, :10, :oh, :ow].argmax(dim=0)
                    train_pa += (pred == gt[0, :oh, :ow]).float().mean().item()

            scheduler.step()
            avg_train_loss = train_loss / len(train_tasks)
            avg_train_pa = train_pa / len(train_tasks)

            # Evaluate every 50 epochs
            if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == max_epochs - 1:
                model.eval()
                test_pa, test_em = 0, 0
                with torch.no_grad():
                    for item in test_tasks:
                        do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                        emb = model.encode_task(do_t)
                        ti = item['test_input'].unsqueeze(0).to(DEVICE)
                        gt = item['test_output'][:10].argmax(dim=0).to(DEVICE)
                        oh, ow = item['out_h'], item['out_w']
                        logits = model(ti, emb)
                        pred = logits[0, :10, :oh, :ow].argmax(dim=0)
                        pa = (pred == gt[:oh, :ow]).float().mean().item()
                        em = float((pred == gt[:oh, :ow]).all().item())
                        test_pa += pa; test_em += em

                avg_test_pa = test_pa / len(test_tasks)
                avg_test_em = test_em / len(test_tasks)

                history.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_pa': avg_train_pa,
                    'test_pa': avg_test_pa,
                    'test_em': avg_test_em,
                })

                if (epoch + 1) % 200 == 0 or epoch < 5 or epoch == max_epochs - 1:
                    print(f"    Epoch {epoch+1:5d}: TrainPA={avg_train_pa*100:.1f}%, "
                          f"TestPA={avg_test_pa*100:.1f}%, TestEM={avg_test_em*100:.1f}%, "
                          f"Loss={avg_train_loss:.4f}")

                # Detect Grokking: sudden jump in test PA
                if len(history) >= 3:
                    recent_jump = history[-1]['test_pa'] - history[-3]['test_pa']
                    if recent_jump > 0.1:  # >10pp jump
                        print(f"    *** GROKKING DETECTED! +{recent_jump*100:.1f}pp jump! ***")

        all_results[C] = {
            'params': n_params, 'label': label,
            'history': history,
            'final_train_pa': history[-1]['train_pa'],
            'final_test_pa': history[-1]['test_pa'],
        }

        del model, opt
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 193 Complete ({elapsed:.0f}s)")
    for C, r in all_results.items():
        print(f"  {r['label']} (C={C}): TrainPA={r['final_train_pa']*100:.1f}%, "
              f"TestPA={r['final_test_pa']*100:.1f}%")
        # Check for Grokking signature
        test_pas = [h['test_pa'] for h in r['history']]
        if len(test_pas) > 5:
            max_jump = max(test_pas[i] - test_pas[i-1]
                          for i in range(1, len(test_pas)))
            print(f"    Max single-step TestPA jump: {max_jump*100:.1f}pp")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase193_grokking.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Phase 193: Leviathan Grokking',
            'timestamp': datetime.now().isoformat(),
            'results': {str(C): {
                'params': r['params'], 'label': r['label'],
                'final_train_pa': r['final_train_pa'],
                'final_test_pa': r['final_test_pa'],
                'history': r['history']
            } for C, r in all_results.items()},
            'elapsed_seconds': elapsed
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors_plot = {'64': '#3498db', '256': '#e74c3c'}

        for C, r in all_results.items():
            h = r['history']
            epochs = [x['epoch'] for x in h]
            train_pa = [x['train_pa']*100 for x in h]
            test_pa = [x['test_pa']*100 for x in h]
            col = colors_plot.get(str(C), '#2ecc71')

            axes[0].plot(epochs, train_pa, '--', color=col, alpha=0.5,
                        label=f'C={C} Train')
            axes[0].plot(epochs, test_pa, '-', color=col, linewidth=2,
                        label=f'C={C} Test')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('PA (%)')
        axes[0].set_title('Train vs Test PA', fontweight='bold')
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

        # Gap (train - test)
        for C, r in all_results.items():
            h = r['history']
            epochs = [x['epoch'] for x in h]
            gap = [(x['train_pa'] - x['test_pa'])*100 for x in h]
            col = colors_plot.get(str(C), '#2ecc71')
            axes[1].plot(epochs, gap, '-', color=col, linewidth=2, label=f'C={C}')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Train - Test PA (pp)')
        axes[1].set_title('Generalization Gap', fontweight='bold')
        axes[1].axhline(0, color='black', linewidth=0.5)
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        # Test PA derivative (to detect Grokking)
        for C, r in all_results.items():
            h = r['history']
            if len(h) > 1:
                epochs = [x['epoch'] for x in h[1:]]
                dpa = [(h[i]['test_pa'] - h[i-1]['test_pa'])*100
                       for i in range(1, len(h))]
                col = colors_plot.get(str(C), '#2ecc71')
                axes[2].plot(epochs, dpa, '-', color=col, linewidth=1.5, label=f'C={C}')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Test PA Change (pp)')
        axes[2].set_title('Grokking Detector (dPA/dt)', fontweight='bold')
        axes[2].axhline(0, color='black', linewidth=0.5)
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.suptitle('Phase 193: Leviathan Grokking (Sudden Understanding)',
                     fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.86, bottom=0.12, left=0.06, right=0.98, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase193_grokking.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return all_results

if __name__ == '__main__':
    main()
