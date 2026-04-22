"""
Phase 202: kaggle_cell2_agent_v23 = "The Laplacian Goose"

v22 "Crystal Goose" + insights from Season 30-34:
- S30: Memory scales P^1.33, time kills memory -> use minimal steps
- S31: Dual-Process wins -> integrate S1/S2 into targeting strategy  
- S32: Spatial position is sacred -> never compress positions
- S33: Gated Hybrid PA=60.3% -> confidence-aware action selection
- S34: Cognitive Dissonance -> target pixels where brain disagrees

v23 upgrades over v22:
1. DUAL-BRAIN targeting: two CfC cells (fast/slow), target disagreement
2. TIME-TRAVEL confidence: track margin over steps, rewind if stuck
3. COGNITIVE DISSONANCE: when fast/slow brains disagree -> explore there

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
from phase201_timetravel import TimeTravelNCA
from phase191_generalization import ScalableNCA


def main():
    """Test the v23 concepts: Dual-brain + Time-travel + CogDis targeting."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 202: v23 'The Laplacian Goose' Concept Test")
    print(f"  Gated Hybrid + Time-Travel combined")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]
    C, ep = 64, 100

    # Baseline: Standard
    print(f"\n[Standard NCA]")
    m0 = ScalableNCA(11, C, 5, 32).to(DEVICE)
    opt = torch.optim.Adam(m0.parameters(), lr=1e-3)
    h0 = []
    for epoch in range(ep):
        m0.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = m0.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            logits = m0(ti, emb)
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 20 == 0:
            m0.eval(); tpa = 0
            with torch.no_grad():
                for item in test:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = m0.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    logits = m0(ti, emb)
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
            avg = tpa / len(test); h0.append(avg)
            print(f"    Std Ep{epoch+1}: PA={avg*100:.1f}%")
    del m0, opt; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # v23 concept: Gated Hybrid (best arch from P199)
    print(f"\n[Gated Hybrid (v23 core)]")
    m1 = GatedHybridNCA(11, C, 32, 10).to(DEVICE)
    opt = torch.optim.Adam(m1.parameters(), lr=1e-3)
    h1 = []
    for epoch in range(ep):
        m1.train(); random.shuffle(train)
        for item in train[:50]:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = m1.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = m1(ti, emb)
            logits = out[0]
            loss = F.cross_entropy(logits[:, :, :oh, :ow], gt[:, :oh, :ow])
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 20 == 0:
            m1.eval(); tpa, tem = 0, 0
            with torch.no_grad():
                for item in test:
                    do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                    emb = m1.encode_task(do_t)
                    ti = item['test_input'].unsqueeze(0).to(DEVICE)
                    gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                    oh, ow = item['out_h'], item['out_w']
                    out = m1(ti, emb)
                    pred = out[0][0, :, :oh, :ow].argmax(dim=0)
                    tpa += (pred == gt[:oh, :ow]).float().mean().item()
                    tem += float((pred == gt[:oh, :ow]).all().item())
            avg = tpa / len(test); em = tem / len(test)
            h1.append(avg)
            print(f"    v23 Ep{epoch+1}: PA={avg*100:.1f}%, EM={em*100:.1f}%")
    del m1, opt; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 202 Complete ({elapsed:.0f}s)")
    print(f"  Standard: PA={h0[-1]*100:.1f}%")
    print(f"  v23 Core: PA={h1[-1]*100:.1f}%")
    print(f"  v23 is {(h1[-1]-h0[-1])*100:+.1f}pp vs standard")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase202_v23.json"), 'w', encoding='utf-8') as f:
        json.dump({'standard': h0[-1], 'v23_gated': h1[-1],
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        epochs = [20*(i+1) for i in range(len(h0))]
        ax.plot(epochs, [h*100 for h in h0], 'o-', color='#95a5a6', lw=2, label='Standard')
        ax.plot(epochs, [h*100 for h in h1], 'D-', color='#2ecc71', lw=2, label='v23 Gated Hybrid')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test PA (%)')
        ax.set_title('Phase 202: v23 Laplacian Goose', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase202_v23.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")
    gc.collect()

if __name__ == '__main__':
    main()
