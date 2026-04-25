"""
Phase 221: Task-Conditioned Critic - Causal Verification Model

Unlike P205's AE (only sees output images), this critic sees:
- Input image
- Output candidate
- Task Embedding (from demo pairs)

And learns: "Is this (Input, Output) pair consistent with this Task?"

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
from phase199_gated import GatedHybridNCA


class TaskConditionedCritic(nn.Module):
    """Critic that judges (Input, Output) pairs conditioned on Task Embedding."""
    def __init__(self, n_colors=11, embed_dim=32):
        super().__init__()
        # Encode concatenated (Input, Output) pair
        self.pair_encoder = nn.Sequential(
            nn.Conv2d(n_colors * 2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
        )
        # Fuse with task embedding and produce score
        self.scorer = nn.Sequential(
            nn.Linear(32 + embed_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
        # Task encoder (from demo I/O pairs)
        self.task_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )

    def encode_task(self, demo_outputs):
        embs = []
        for do in demo_outputs:
            embs.append(self.task_encoder(do.unsqueeze(0)))
        return torch.stack(embs).mean(dim=0)

    def forward(self, inp_oh, out_oh, task_emb):
        """Score how well (inp, out) matches the task.
        inp_oh, out_oh: (B, n_colors, H, W)
        task_emb: (B, embed_dim)
        Returns: score in [0, 1]
        """
        # Pad to same spatial size
        maxH = max(inp_oh.shape[2], out_oh.shape[2])
        maxW = max(inp_oh.shape[3], out_oh.shape[3])
        inp_p = F.pad(inp_oh, (0, maxW - inp_oh.shape[3], 0, maxH - inp_oh.shape[2]))
        out_p = F.pad(out_oh, (0, maxW - out_oh.shape[3], 0, maxH - out_oh.shape[2]))
        pair = torch.cat([inp_p, out_p], dim=1)
        pair_feat = self.pair_encoder(pair)
        combined = torch.cat([pair_feat, task_emb], dim=1)
        return self.scorer(combined)


def train_critic(critic, train_tasks, n_epochs=150, lr=1e-3):
    """Contrastive training: correct pairs -> 1.0, wrong pairs -> 0.0."""
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    for epoch in range(n_epochs):
        random.shuffle(train_tasks)
        epoch_loss = 0
        n_batches = 0
        for i, item in enumerate(train_tasks[:100]):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]

            di_list = item.get('demo_inputs', [])
            if not di_list:
                continue

            # Positive + Negative for each demo pair
            for di, do in zip(di_list, do_t):
                opt.zero_grad()
                # Recompute task_emb in each forward to keep graph fresh
                task_emb = critic.encode_task(do_t)

                di_t = di.unsqueeze(0).to(DEVICE)
                do_t_single = do.unsqueeze(0)

                score_pos = critic(di_t[:, :11, :, :], do_t_single[:, :11, :, :], task_emb)
                loss_pos = F.binary_cross_entropy(score_pos, torch.ones_like(score_pos))

                # Negative: wrong output (from different task)
                neg_idx = (i + random.randint(1, len(train_tasks)-1)) % len(train_tasks)
                neg_item = train_tasks[neg_idx]
                neg_out = neg_item['demo_outputs'][0].unsqueeze(0).to(DEVICE)
                score_neg = critic(di_t[:, :11, :, :], neg_out[:, :11, :, :], task_emb)
                loss_neg = F.binary_cross_entropy(score_neg, torch.zeros_like(score_neg))

                loss = (loss_pos + loss_neg) / 2
                loss.backward(); opt.step()
                epoch_loss += loss.item()
                n_batches += 1

        if (epoch + 1) % 50 == 0:
            avg = epoch_loss / max(1, n_batches)
            print(f"    Critic Ep{epoch+1}: Loss={avg:.4f}")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 221: Task-Conditioned Critic")
    print(f"  Causal verification: (Input, Output, Task) -> Score")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    has_di = 'demo_inputs' in train[0] if train else False

    if not has_di:
        print("  WARNING: No demo_inputs. Causal Critic needs I/O pairs!")

    # Train NCA
    print(f"\n[Training GatedHybridNCA]")
    torch.manual_seed(SEED)
    model = GatedHybridNCA(11, 64, 32, 10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        model.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  NCA Params: {model.count_params():,}")

    # Train Critic
    print(f"\n[Training Task-Conditioned Critic]")
    critic = TaskConditionedCritic(11, 32).to(DEVICE)
    cparams = sum(p.numel() for p in critic.parameters())
    print(f"  Critic Params: {cparams:,}")
    train_critic(critic, train, n_epochs=150)
    critic.eval()

    # Evaluate: use critic to select from N candidates
    N = 100
    noise_scale = 0.3
    model.eval()

    greedy_pa, greedy_em = 0, 0
    oracle_pa, oracle_em = 0, 0
    causal_pa, causal_em = 0, 0
    margin_pa, margin_em = 0, 0

    print(f"\n[Evaluating Task-Conditioned Critic as Selector (N={N})]")
    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            c_emb = critic.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            candidates = []
            for trial in range(N):
                out = model(ti, emb)
                logits = out[0] if isinstance(out, tuple) else out
                if trial > 0:
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                    logits = logits + noise_scale * gumbel
                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                pa = (pred == gt[:oh, :ow]).float().mean().item()
                em = float((pred == gt[:oh, :ow]).all().item())

                # Critic score
                cand_oh = F.one_hot(pred.long(), 11).permute(2, 0, 1).float().unsqueeze(0)
                if has_di:
                    score = critic(ti[:, :11, :, :], cand_oh, c_emb).item()
                else:
                    score = 0.0

                probs = F.softmax(logits[0, :, :oh, :ow], dim=0)
                top2 = probs.topk(2, dim=0).values
                margin = (top2[0] - top2[1]).mean().item()

                candidates.append((pa, em, score, margin))

            greedy_pa += candidates[0][0]; greedy_em += candidates[0][1]
            best = max(candidates, key=lambda c: c[0])
            oracle_pa += best[0]; oracle_em += best[1]
            cau_best = max(candidates, key=lambda c: c[2])
            causal_pa += cau_best[0]; causal_em += cau_best[1]
            mar_best = max(candidates, key=lambda c: c[3])
            margin_pa += mar_best[0]; margin_em += mar_best[1]

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    results = {
        'greedy': {'pa': greedy_pa/n_test, 'em': greedy_em/n_test},
        'oracle': {'pa': oracle_pa/n_test, 'em': oracle_em/n_test},
        'causal': {'pa': causal_pa/n_test, 'em': causal_em/n_test},
        'margin': {'pa': margin_pa/n_test, 'em': margin_em/n_test},
    }

    print(f"\n{'='*70}")
    print(f"  TASK-CONDITIONED CRITIC (N={N}):")
    for k, r in results.items():
        print(f"  {k:10s}: PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    vgap = (oracle_pa - causal_pa) / n_test * 100
    print(f"  V-Gap (Causal): {vgap:.1f}pp")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model, critic; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase221_causal_critic.json"), 'w', encoding='utf-8') as f:
        json.dump({'N': N, 'results': results, 'vgap_causal': vgap,
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = list(results.keys())
        pa_vals = [results[k]['pa']*100 for k in labels]
        em_vals = [results[k]['em']*100 for k in labels]
        colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#3498db']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('%'); ax.set_title('Phase 221: Task-Conditioned Critic', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase221_causal_critic.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
