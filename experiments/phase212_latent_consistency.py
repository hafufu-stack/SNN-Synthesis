"""
Phase 212: Latent Consistency Critic - Zero-Shot Semantic Verification

Select the best candidate from N stochastic trials by measuring
how consistent each candidate's implied task embedding is with
the true task embedding from demo examples.

Strategy:
  1. Get E_demo from demo I/O pairs via Foundation Encoder
  2. For each candidate output, pair it with test_input and encode
     to get E_test_i
  3. Pick candidate with highest cosine_sim(E_demo, E_test_i)

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
from phase206_causal_volume import ParametricNCA


class PairEncoder(nn.Module):
    """Encode an (input, output) pair into a task embedding vector."""
    def __init__(self, n_colors=11, embed_dim=32):
        super().__init__()
        # Takes concatenation of input and output one-hot grids
        self.net = nn.Sequential(
            nn.Conv2d(n_colors * 2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, inp_oh, out_oh):
        """inp_oh, out_oh: (B, n_colors, H, W)"""
        # Pad to same spatial size
        maxH = max(inp_oh.shape[2], out_oh.shape[2])
        maxW = max(inp_oh.shape[3], out_oh.shape[3])
        inp_p = F.pad(inp_oh, (0, maxW - inp_oh.shape[3], 0, maxH - inp_oh.shape[2]))
        out_p = F.pad(out_oh, (0, maxW - out_oh.shape[3], 0, maxH - out_oh.shape[2]))
        x = torch.cat([inp_p, out_p], dim=1)
        return self.net(x)


def train_pair_encoder(encoder, train_tasks, n_epochs=100, lr=1e-3):
    """Train PairEncoder: demo pairs from same task should cluster together."""
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    for epoch in range(n_epochs):
        random.shuffle(train_tasks)
        epoch_loss = 0
        for item in train_tasks[:100]:
            di_list = item.get('demo_inputs', [])
            do_list = item['demo_outputs']
            if len(di_list) < 2 or len(do_list) < 2:
                continue
            # Encode two demo pairs, pull embeddings together (contrastive)
            emb1 = encoder(di_list[0].unsqueeze(0).to(DEVICE),
                          do_list[0].unsqueeze(0).to(DEVICE))
            emb2 = encoder(di_list[1].unsqueeze(0).to(DEVICE),
                          do_list[1].unsqueeze(0).to(DEVICE))
            # Positive: same task -> maximize cosine similarity
            cos_pos = F.cosine_similarity(emb1, emb2).mean()
            loss = 1 - cos_pos
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"    PairEnc Ep{epoch+1}: Loss={epoch_loss/100:.4f}")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 212: Latent Consistency Critic")
    print(f"  Cosine similarity between demo and candidate task embeddings")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    # Train NCA (K=1, C=64, T=1)
    print(f"\n[Training NCA (K=1, C=64, T=1)]")
    torch.manual_seed(SEED)
    model = ParametricNCA(11, 64, 1, 1, 32).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        model.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            logits = model(ti, emb)
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"  NCA Params: {model.count_params():,}")

    # Train PairEncoder
    print(f"\n[Training PairEncoder]")
    has_di = 'demo_inputs' in train[0] if train else False
    pair_enc = PairEncoder(11, 32).to(DEVICE)
    if has_di:
        train_pair_encoder(pair_enc, train, n_epochs=100)
    else:
        print("  No demo_inputs, using output-only encoder as fallback")
        # Fallback: just use output encoder (same as model's demo_encoder)
    pair_enc.eval()

    # Generate N=100 candidates per task and evaluate selectors
    N = 100
    noise_scale = 0.3
    print(f"\n[Generating {N} candidates & evaluating selectors]")

    model.eval()
    greedy_pa, greedy_em = 0, 0
    oracle_pa, oracle_em = 0, 0
    cosine_pa, cosine_em = 0, 0
    margin_pa, margin_em = 0, 0

    with torch.no_grad():
        for tidx, item in enumerate(test):
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']

            # Get demo task embedding via PairEncoder
            if has_di and len(item['demo_inputs']) > 0:
                demo_embs = []
                for di, do in zip(item['demo_inputs'], item['demo_outputs']):
                    de = pair_enc(di.unsqueeze(0).to(DEVICE), do.unsqueeze(0).to(DEVICE))
                    demo_embs.append(de)
                e_demo = torch.stack(demo_embs).mean(dim=0)  # (1, embed_dim)
            else:
                # Fallback: use model's own task encoder
                e_demo = emb.unsqueeze(0) if emb.dim() == 1 else emb

            candidates = []
            for trial in range(N):
                logits = model(ti, emb)
                if trial > 0:
                    gumbel = -torch.log(-torch.log(torch.rand_like(logits)+1e-8)+1e-8)
                    logits = logits + noise_scale * gumbel

                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                pa = (pred == gt[:oh, :ow]).float().mean().item()
                em = (pred == gt[:oh, :ow]).all().item()

                # Cosine consistency: encode (test_input, candidate_output)
                cand_oh = F.one_hot(pred.long(), 11).permute(2, 0, 1).float()
                if has_di:
                    e_test = pair_enc(ti[:, :11, :oh, :ow], cand_oh.unsqueeze(0))
                    cos_sim = F.cosine_similarity(
                        e_demo.view(1, -1), e_test.view(1, -1)
                    ).item()
                else:
                    # Fallback: use model's demo encoder on candidate
                    e_test = model.demo_encoder(cand_oh.unsqueeze(0).to(DEVICE))
                    cos_sim = F.cosine_similarity(
                        emb.view(1, -1), e_test.view(1, -1)
                    ).item()

                # Margin
                probs = F.softmax(logits[0, :, :oh, :ow], dim=0)
                top2 = probs.topk(2, dim=0).values
                margin = (top2[0] - top2[1]).mean().item()

                candidates.append((pa, em, cos_sim, margin))

            # Greedy
            greedy_pa += candidates[0][0]; greedy_em += candidates[0][1]
            # Oracle
            best = max(candidates, key=lambda c: c[0])
            oracle_pa += best[0]; oracle_em += best[1]
            # Cosine Consistency (highest cosine sim)
            cos_best = max(candidates, key=lambda c: c[2])
            cosine_pa += cos_best[0]; cosine_em += cos_best[1]
            # Margin
            mar_best = max(candidates, key=lambda c: c[3])
            margin_pa += mar_best[0]; margin_em += mar_best[1]

            if (tidx + 1) % 10 == 0:
                print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    results = {
        'greedy': {'pa': greedy_pa/n_test, 'em': greedy_em/n_test},
        'oracle': {'pa': oracle_pa/n_test, 'em': oracle_em/n_test},
        'cosine': {'pa': cosine_pa/n_test, 'em': cosine_em/n_test},
        'margin': {'pa': margin_pa/n_test, 'em': margin_em/n_test},
    }

    vgap_cos = (oracle_pa - cosine_pa) / n_test * 100
    vgap_mar = (oracle_pa - margin_pa) / n_test * 100

    print(f"\n{'='*70}")
    print(f"  LATENT CONSISTENCY CRITIC (N={N}):")
    print(f"  Greedy:  PA={results['greedy']['pa']*100:.1f}%, EM={results['greedy']['em']*100:.1f}%")
    print(f"  Oracle:  PA={results['oracle']['pa']*100:.1f}%, EM={results['oracle']['em']*100:.1f}%")
    print(f"  Cosine:  PA={results['cosine']['pa']*100:.1f}%, EM={results['cosine']['em']*100:.1f}%")
    print(f"  Margin:  PA={results['margin']['pa']*100:.1f}%, EM={results['margin']['em']*100:.1f}%")
    print(f"  V-Gap Cosine: {vgap_cos:.1f}pp  |  V-Gap Margin: {vgap_mar:.1f}pp")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model, pair_enc; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase212_latent_consistency.json"), 'w', encoding='utf-8') as f:
        json.dump({'N': N, 'results': results,
                   'vgap_cosine': vgap_cos, 'vgap_margin': vgap_mar,
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ['Greedy', 'Oracle', 'Cosine\nConsist.', 'Margin']
        pa_vals = [results[k]['pa']*100 for k in ['greedy','oracle','cosine','margin']]
        em_vals = [results[k]['em']*100 for k in ['greedy','oracle','cosine','margin']]
        colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#3498db']
        x = np.arange(4); w = 0.35
        axes[0].bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        axes[0].bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
        axes[0].set_ylabel('%'); axes[0].set_title('Selection Comparison', fontweight='bold')
        axes[0].legend()

        gaps = [0, 0, vgap_cos, vgap_mar]
        axes[1].bar(labels, gaps, color=colors, alpha=0.85)
        axes[1].set_ylabel('Gap from Oracle (pp)')
        axes[1].set_title('Verification Gap', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        fig.suptitle('Phase 212: Latent Consistency Critic', fontsize=13, fontweight='bold')
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.96, wspace=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase212_latent_consistency.png'), dpi=150)
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return results


if __name__ == '__main__':
    main()
