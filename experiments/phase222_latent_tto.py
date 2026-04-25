"""
Phase 222: Latent TTO - Test-Time Optimization in Embedding Space

Instead of optimizing pixels/logits (catastrophic in discrete space),
optimize the Task Embedding itself using the Task-Conditioned Critic.

"Don't repaint the picture. Reinterpret the rule."

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
from phase221_causal_critic import TaskConditionedCritic, train_critic


def latent_tto(model, critic, item, n_steps=30, lr=0.01):
    """Optimize task embedding to maximize causal critic score.

    Freeze everything except the task embedding vector.
    """
    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
    di_list = item.get('demo_inputs', [])
    if not di_list:
        return None

    # Get initial embedding
    with torch.no_grad():
        emb = model.encode_task(do_t)
        c_emb = critic.encode_task(do_t)

    # Make embedding optimizable
    opt_emb = emb.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_emb], lr=lr)

    for step in range(n_steps):
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

        # For each demo pair, NCA should produce correct output
        for di, do in zip(di_list, do_t):
            di_t = di.unsqueeze(0).to(DEVICE)
            do_gt = do[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            doh, dow = do_gt.shape[1], do_gt.shape[2]

            out = model(di_t, opt_emb)
            logits = out[0] if isinstance(out, tuple) else out

            # CE loss on demos
            ce = F.cross_entropy(logits[:, :, :doh, :dow], do_gt)

            # Critic score (maximize -> minimize negative)
            soft_out = F.softmax(logits[:, :, :doh, :dow], dim=1)
            score = critic(di_t[:, :11, :, :], soft_out, c_emb)
            critic_loss = 1.0 - score.mean()

            total_loss = total_loss + ce + 0.1 * critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return opt_emb.detach()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 222: Latent TTO")
    print(f"  Optimize Task Embedding (not pixels!) at test time")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    has_di = 'demo_inputs' in train[0] if train else False

    # Train GatedHybrid
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
    train_critic(critic, train, n_epochs=150)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False

    # Evaluate
    tto_configs = [10, 30, 50]
    model.eval()

    base_pa, base_em = 0, 0
    tto_results = {n: {'pa': 0, 'em': 0} for n in tto_configs}
    # Also test: embedding-only TTO (no critic, just CE on demos)
    ce_tto_results = {n: {'pa': 0, 'em': 0} for n in tto_configs}

    print(f"\n[Evaluating Latent TTO]")
    for tidx, item in enumerate(test):
        do_t = [d.to(DEVICE) for d in item['demo_outputs']]
        ti = item['test_input'].unsqueeze(0).to(DEVICE)
        gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
        oh, ow = item['out_h'], item['out_w']

        # Baseline
        with torch.no_grad():
            emb = model.encode_task(do_t)
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits[0, :, :oh, :ow].argmax(dim=0)
            base_pa += (pred == gt[:oh, :ow]).float().mean().item()
            base_em += float((pred == gt[:oh, :ow]).all().item())

        # CE-only TTO (optimize embedding using just CE on demos)
        for n_steps in tto_configs:
            if has_di:
                opt_emb = emb.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([opt_emb], lr=0.01)
                di_list = item.get('demo_inputs', [])
                for step in range(n_steps):
                    total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
                    for di, do in zip(di_list, do_t):
                        di_t = di.unsqueeze(0).to(DEVICE)
                        do_gt = do[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
                        doh, dow = do_gt.shape[1], do_gt.shape[2]
                        out2 = model(di_t, opt_emb)
                        lg = out2[0] if isinstance(out2, tuple) else out2
                        total_loss = total_loss + F.cross_entropy(lg[:, :, :doh, :dow], do_gt)
                    optimizer.zero_grad(); total_loss.backward(); optimizer.step()

                with torch.no_grad():
                    out3 = model(ti, opt_emb)
                    lg3 = out3[0] if isinstance(out3, tuple) else out3
                    pred3 = lg3[0, :, :oh, :ow].argmax(dim=0)
                    ce_tto_results[n_steps]['pa'] += (pred3 == gt[:oh, :ow]).float().mean().item()
                    ce_tto_results[n_steps]['em'] += float((pred3 == gt[:oh, :ow]).all().item())
            else:
                ce_tto_results[n_steps]['pa'] += (pred == gt[:oh, :ow]).float().mean().item()
                ce_tto_results[n_steps]['em'] += float((pred == gt[:oh, :ow]).all().item())

        # Full TTO (CE + Critic)
        for n_steps in tto_configs:
            if has_di:
                opt_emb2 = latent_tto(model, critic, item, n_steps=n_steps, lr=0.01)
                if opt_emb2 is not None:
                    with torch.no_grad():
                        out4 = model(ti, opt_emb2)
                        lg4 = out4[0] if isinstance(out4, tuple) else out4
                        pred4 = lg4[0, :, :oh, :ow].argmax(dim=0)
                        tto_results[n_steps]['pa'] += (pred4 == gt[:oh, :ow]).float().mean().item()
                        tto_results[n_steps]['em'] += float((pred4 == gt[:oh, :ow]).all().item())
                else:
                    tto_results[n_steps]['pa'] += (pred == gt[:oh, :ow]).float().mean().item()
                    tto_results[n_steps]['em'] += float((pred == gt[:oh, :ow]).all().item())
            else:
                tto_results[n_steps]['pa'] += (pred == gt[:oh, :ow]).float().mean().item()
                tto_results[n_steps]['em'] += float((pred == gt[:oh, :ow]).all().item())

        if (tidx + 1) % 10 == 0:
            print(f"    Task {tidx+1}/{len(test)}")

    n_test = len(test)
    base_pa /= n_test; base_em /= n_test
    for n in tto_configs:
        tto_results[n]['pa'] /= n_test; tto_results[n]['em'] /= n_test
        ce_tto_results[n]['pa'] /= n_test; ce_tto_results[n]['em'] /= n_test

    print(f"\n{'='*70}")
    print(f"  LATENT TTO:")
    print(f"  Base:        PA={base_pa*100:.1f}%, EM={base_em*100:.1f}%")
    for n in tto_configs:
        r = ce_tto_results[n]
        print(f"  CE-TTO({n:3d}): PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    for n in tto_configs:
        r = tto_results[n]
        print(f"  Full-TTO({n:2d}): PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model, critic; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase222_latent_tto.json"), 'w', encoding='utf-8') as f:
        json.dump({'base': {'pa': base_pa, 'em': base_em},
                   'ce_tto': {str(k): v for k, v in ce_tto_results.items()},
                   'full_tto': {str(k): v for k, v in tto_results.items()},
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        labels = ['Base'] + [f'CE({n})' for n in tto_configs] + [f'Full({n})' for n in tto_configs]
        pa_vals = [base_pa*100] + [ce_tto_results[n]['pa']*100 for n in tto_configs] + \
                  [tto_results[n]['pa']*100 for n in tto_configs]
        em_vals = [base_em*100] + [ce_tto_results[n]['em']*100 for n in tto_configs] + \
                  [tto_results[n]['em']*100 for n in tto_configs]
        colors = ['#95a5a6'] + ['#3498db']*3 + ['#e74c3c']*3
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('%'); ax.set_title('Phase 222: Latent TTO', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase222_latent_tto.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'base_pa': base_pa, 'ce_tto': ce_tto_results, 'full_tto': tto_results}


if __name__ == '__main__':
    main()
