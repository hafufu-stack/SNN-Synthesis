"""
Phase 218: Gated TTT-LoRA - Test-Time Training on the Strongest Architecture

Apply TTT-LoRA to GatedHybridNCA (PA=60.3%, our best model).
LoRA adapters on System 2's 3x3 Conv layers, fine-tuned per task.

Author: Hiroto Funasaki
"""
import os, json, time, gc, random, sys, copy
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


class LoRALayer(nn.Module):
    """Low-Rank Adaptation wrapper for Conv2d."""
    def __init__(self, base_conv, rank=8):
        super().__init__()
        self.base_conv = base_conv
        for p in self.base_conv.parameters():
            p.requires_grad = False
        in_ch = base_conv.in_channels
        out_ch = base_conv.out_channels
        self.lora_A = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.lora_B = nn.Conv2d(rank, out_ch, 1, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = 0.1

    def forward(self, x):
        return self.base_conv(x) + self.lora_B(self.lora_A(x)) * self.scale


def add_lora_to_gated(model, rank=8):
    """Add LoRA to System 2's 3x3 Conv layers."""
    # S2 encoder has 2 Conv2d with 3x3 kernels
    old_s2_enc = model.s2_encoder
    new_layers = []
    for layer in old_s2_enc:
        if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] == 3:
            new_layers.append(LoRALayer(layer, rank))
        else:
            new_layers.append(layer)
    model.s2_encoder = nn.Sequential(*new_layers)

    # S2 update has Conv2d with 3x3
    old_s2_upd = model.s2_update
    new_layers2 = []
    for layer in old_s2_upd:
        if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] == 3:
            new_layers2.append(LoRALayer(layer, rank))
        else:
            new_layers2.append(layer)
    model.s2_update = nn.Sequential(*new_layers2)

    # S2 decoder has Conv2d with 3x3
    old_dec = model.s2_decoder
    new_layers3 = []
    for layer in old_dec:
        if isinstance(layer, nn.Conv2d) and layer.kernel_size[0] == 3:
            new_layers3.append(LoRALayer(layer, rank))
        else:
            new_layers3.append(layer)
    model.s2_decoder = nn.Sequential(*new_layers3)

    return model


def get_lora_params(model):
    """Get only LoRA parameters."""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALayer):
            params.extend(m.lora_A.parameters())
            params.extend(m.lora_B.parameters())
    return params


def reset_lora(model):
    """Reset LoRA weights for next task."""
    for m in model.modules():
        if isinstance(m, LoRALayer):
            nn.init.normal_(m.lora_A.weight, std=0.01)
            nn.init.zeros_(m.lora_B.weight)


def ttt_lora(model, item, n_steps=30, lr=0.01):
    """Test-Time Training: fine-tune LoRA on demo pairs."""
    reset_lora(model)
    lora_params = get_lora_params(model)
    if not lora_params:
        return

    di_list = item.get('demo_inputs', [])
    do_list = item['demo_outputs']
    if not di_list:
        return

    optimizer = torch.optim.Adam(lora_params, lr=lr)
    for step in range(n_steps):
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for di, do in zip(di_list, do_list):
            di_t = di.unsqueeze(0).to(DEVICE)
            do_gt = do[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            doh, dow = do_gt.shape[1], do_gt.shape[2]
            do_all = [d.to(DEVICE) for d in do_list]
            emb = model.encode_task(do_all)
            out = model(di_t, emb)
            logits = out[0] if isinstance(out, tuple) else out
            total_loss = total_loss + F.cross_entropy(logits[:, :, :doh, :dow], do_gt)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 218: Gated TTT-LoRA")
    print(f"  TTT on GatedHybridNCA (our strongest architecture)")
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
    print(f"  Params: {model.count_params():,}")

    # Evaluate without TTT
    model.eval()
    base_pa, base_em = 0, 0
    with torch.no_grad():
        for item in test:
            do_t = [d.to(DEVICE) for d in item['demo_outputs']]
            emb = model.encode_task(do_t)
            ti = item['test_input'].unsqueeze(0).to(DEVICE)
            gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
            oh, ow = item['out_h'], item['out_w']
            out = model(ti, emb)
            logits = out[0] if isinstance(out, tuple) else out
            pred = logits[0, :, :oh, :ow].argmax(dim=0)
            base_pa += (pred == gt[:oh, :ow]).float().mean().item()
            base_em += float((pred == gt[:oh, :ow]).all().item())
    base_pa /= len(test); base_em /= len(test)
    print(f"  No TTT: PA={base_pa*100:.1f}%, EM={base_em*100:.1f}%")

    # Add LoRA and evaluate with TTT
    model = add_lora_to_gated(model, rank=8)
    model = model.to(DEVICE)
    lora_count = sum(p.numel() for p in get_lora_params(model))
    print(f"  LoRA params added: {lora_count:,}")

    ttt_configs = [10, 30, 50]
    ttt_results = {}
    for n_steps in ttt_configs:
        print(f"\n  [TTT({n_steps} steps)]")
        ttt_pa, ttt_em = 0, 0
        for item in test:
            reset_lora(model)
            if has_di:
                model.train()
                ttt_lora(model, item, n_steps=n_steps, lr=0.01)
            model.eval()
            with torch.no_grad():
                do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.encode_task(do_t)
                ti = item['test_input'].unsqueeze(0).to(DEVICE)
                gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                oh, ow = item['out_h'], item['out_w']
                out = model(ti, emb)
                logits = out[0] if isinstance(out, tuple) else out
                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                ttt_pa += (pred == gt[:oh, :ow]).float().mean().item()
                ttt_em += float((pred == gt[:oh, :ow]).all().item())
        ttt_pa /= len(test); ttt_em /= len(test)
        ttt_results[n_steps] = {'pa': ttt_pa, 'em': ttt_em}
        print(f"    PA={ttt_pa*100:.1f}%, EM={ttt_em*100:.1f}%")

    print(f"\n{'='*70}")
    print(f"  GATED TTT-LoRA:")
    print(f"  Base:     PA={base_pa*100:.1f}%, EM={base_em*100:.1f}%")
    for n, r in ttt_results.items():
        print(f"  TTT({n:3d}): PA={r['pa']*100:.1f}%, EM={r['em']*100:.1f}%")
    print(f"{'='*70}")

    elapsed = time.time() - t0
    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase218_gated_ttt.json"), 'w', encoding='utf-8') as f:
        json.dump({'base': {'pa': base_pa, 'em': base_em},
                   'ttt': {str(k): v for k, v in ttt_results.items()},
                   'lora_params': lora_count,
                   'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
                   }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = ['Base'] + [f'TTT({n})' for n in ttt_configs]
        pa_vals = [base_pa*100] + [ttt_results[n]['pa']*100 for n in ttt_configs]
        em_vals = [base_em*100] + [ttt_results[n]['em']*100 for n in ttt_configs]
        colors = ['#95a5a6', '#3498db', '#2980b9', '#1a5276']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('%'); ax.set_title('Phase 218: Gated TTT-LoRA', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase218_gated_ttt.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'base_pa': base_pa, 'ttt': ttt_results}


if __name__ == '__main__':
    main()
