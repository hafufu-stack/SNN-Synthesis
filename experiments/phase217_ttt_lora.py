"""
Phase 217: Full Test-Time Training (TTT-LoRA)

Add LoRA adapters to NCA and fine-tune them per-task at test time
using demo pairs, creating task-specific synapses on the fly.

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


class LoRAConv2d(nn.Module):
    """Conv2d with Low-Rank Adaptation."""
    def __init__(self, base_conv, rank=4):
        super().__init__()
        self.base_conv = base_conv
        # Freeze base weights
        for p in self.base_conv.parameters():
            p.requires_grad = False
        out_ch = base_conv.out_channels
        in_ch = base_conv.in_channels
        self.lora_down = nn.Conv2d(in_ch, rank, 1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_ch, 1, bias=False)
        nn.init.zeros_(self.lora_up.weight)
        self.scale = 0.1

    def forward(self, x):
        base_out = self.base_conv(x)
        lora_out = self.lora_up(self.lora_down(x)) * self.scale
        return base_out + lora_out


class LoRANCA(nn.Module):
    """NCA with LoRA adapters for test-time training."""
    def __init__(self, n_colors=11, hidden_ch=64, embed_dim=32, lora_rank=4):
        super().__init__()
        self.demo_encoder = nn.Sequential(
            nn.Conv2d(n_colors, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim)
        )
        # Base convolutions (will be frozen during TTT)
        base_enc1 = nn.Conv2d(n_colors + embed_dim, hidden_ch, 1)
        base_enc2 = nn.Conv2d(hidden_ch, hidden_ch, 1)
        base_upd1 = nn.Conv2d(hidden_ch, hidden_ch, 1)
        base_upd2 = nn.Conv2d(hidden_ch, hidden_ch, 1)
        base_dec1 = nn.Conv2d(hidden_ch, hidden_ch, 1)
        base_dec2 = nn.Conv2d(hidden_ch, n_colors, 1)

        # Wrap with LoRA
        self.enc1 = LoRAConv2d(base_enc1, lora_rank)
        self.enc2 = LoRAConv2d(base_enc2, lora_rank)
        self.upd1 = LoRAConv2d(base_upd1, lora_rank)
        self.upd2 = LoRAConv2d(base_upd2, lora_rank)
        self.dec1 = LoRAConv2d(base_dec1, lora_rank)
        self.dec2 = base_dec2  # Final layer: no LoRA needed

        self.relu = nn.ReLU()

    def encode_task(self, demo_outputs):
        embeddings = []
        for do in demo_outputs:
            emb = self.demo_encoder(do.unsqueeze(0))
            embeddings.append(emb)
        return torch.stack(embeddings).mean(dim=0)

    def forward(self, x, task_emb):
        B, _, H, W = x.shape
        te = task_emb.view(B, -1, 1, 1).expand(B, -1, H, W)
        inp = torch.cat([x, te], dim=1)
        state = self.relu(self.enc1(inp))
        state = self.relu(self.enc2(state))
        delta = self.relu(self.upd1(state))
        delta = self.upd2(delta)
        state = state + delta
        out = self.relu(self.dec1(state))
        return self.dec2(out)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_lora_params(self):
        """Get only LoRA parameters (for TTT optimization)."""
        params = []
        for module in self.modules():
            if isinstance(module, LoRAConv2d):
                params.extend(module.lora_down.parameters())
                params.extend(module.lora_up.parameters())
        return params

    def reset_lora(self):
        """Reset LoRA weights to zero (for next task)."""
        for module in self.modules():
            if isinstance(module, LoRAConv2d):
                nn.init.normal_(module.lora_down.weight, std=0.01)
                nn.init.zeros_(module.lora_up.weight)


def ttt_lora(model, item, n_steps=30, lr=0.01):
    """Test-Time Training: fine-tune LoRA on demo pairs."""
    model.reset_lora()
    lora_params = model.get_lora_params()
    if not lora_params:
        return

    optimizer = torch.optim.Adam(lora_params, lr=lr)
    di_list = item.get('demo_inputs', [])
    do_list = item['demo_outputs']

    if not di_list:
        return

    for step in range(n_steps):
        total_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for di, do in zip(di_list, do_list):
            di_t = di.unsqueeze(0).to(DEVICE)
            do_gt = do[:11].argmax(dim=0).unsqueeze(0).to(DEVICE)
            doh, dow = do_gt.shape[1], do_gt.shape[2]

            do_all = [d.to(DEVICE) for d in do_list]
            emb = model.encode_task(do_all)
            logits = model(di_t, emb)
            total_loss = total_loss + F.cross_entropy(logits[:, :, :doh, :dow], do_gt)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    t0 = time.time()
    print("=" * 70)
    print("Phase 217: Full Test-Time Training (TTT-LoRA)")
    print(f"  Fine-tune LoRA adapters per-task at test time")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    arc_data = load_arc_training()
    all_tasks = prepare_arc_meta_dataset(arc_data, max_tasks=300)
    random.shuffle(all_tasks)
    train, test = all_tasks[:200], all_tasks[200:250]

    has_di = 'demo_inputs' in train[0] if train else False

    # Train LoRA-NCA
    print(f"\n[Training LoRA-NCA]")
    for rank in [4, 8]:
        print(f"\n  --- LoRA rank={rank} ---")
        torch.manual_seed(SEED)
        model = LoRANCA(11, 64, 32, lora_rank=rank).to(DEVICE)
        print(f"  Total Params: {model.count_params():,}")
        lora_count = sum(p.numel() for p in model.get_lora_params())
        print(f"  LoRA Params:  {lora_count:,}")

        # Train all parameters (base + lora)
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

        # Evaluate WITHOUT TTT
        model.eval()
        base_pa, base_em = 0, 0
        with torch.no_grad():
            for item in test:
                model.reset_lora()
                do_t = [d.to(DEVICE) for d in item['demo_outputs']]
                emb = model.encode_task(do_t)
                ti = item['test_input'].unsqueeze(0).to(DEVICE)
                gt = item['test_output'][:11].argmax(dim=0).to(DEVICE)
                oh, ow = item['out_h'], item['out_w']
                logits = model(ti, emb)
                pred = logits[0, :, :oh, :ow].argmax(dim=0)
                base_pa += (pred == gt[:oh, :ow]).float().mean().item()
                base_em += float((pred == gt[:oh, :ow]).all().item())
        base_pa /= len(test); base_em /= len(test)
        print(f"  No TTT: PA={base_pa*100:.1f}%, EM={base_em*100:.1f}%")

        # Evaluate WITH TTT-LoRA
        ttt_configs = [10, 30, 50]
        ttt_results = {}
        for n_steps in ttt_configs:
            print(f"  TTT({n_steps} steps)...")
            ttt_pa, ttt_em = 0, 0
            for item in test:
                model.reset_lora()
                # TTT: fine-tune LoRA
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
                    logits = model(ti, emb)
                    pred = logits[0, :, :oh, :ow].argmax(dim=0)
                    ttt_pa += (pred == gt[:oh, :ow]).float().mean().item()
                    ttt_em += float((pred == gt[:oh, :ow]).all().item())
            ttt_pa /= len(test); ttt_em /= len(test)
            ttt_results[n_steps] = {'pa': ttt_pa, 'em': ttt_em}
            print(f"    PA={ttt_pa*100:.1f}%, EM={ttt_em*100:.1f}%")

        del model; gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Phase 217 Complete ({elapsed:.0f}s)")
    print(f"{'='*70}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phase217_ttt_lora.json"), 'w', encoding='utf-8') as f:
        json.dump({
            'base': {'pa': base_pa, 'em': base_em},
            'ttt': {str(k): v for k, v in ttt_results.items()},
            'elapsed': elapsed, 'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        labels = ['No TTT'] + [f'TTT({n})' for n in ttt_configs]
        pa_vals = [base_pa*100] + [ttt_results[n]['pa']*100 for n in ttt_configs]
        em_vals = [base_em*100] + [ttt_results[n]['em']*100 for n in ttt_configs]
        colors = ['#95a5a6', '#3498db', '#2980b9', '#1a5276']
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x-w/2, pa_vals, w, color=colors, alpha=0.85, label='PA')
        ax.bar(x+w/2, em_vals, w, color=colors, alpha=0.4, hatch='//', label='EM')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('%'); ax.set_title('Phase 217: TTT-LoRA', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, 'phase217_ttt_lora.png'), dpi=150, bbox_inches='tight')
        plt.close(); print("  Figure saved!")
    except Exception as e:
        print(f"  Figure error: {e}")

    gc.collect()
    return {'base_pa': base_pa, 'ttt': ttt_results}


if __name__ == '__main__':
    main()
