"""
Phase 71: Resonance Window Quantification

Phase 64 found SR-Quantization requires 'intermediate baseline competence'.
Phase 71 precisely characterizes the resonance window by continuously
sweeping task difficulty and plotting the Delta(4-bit vs FP16) curve.

Setup: Arithmetic with increasing difficulty (1-digit to 4-digit, +/-/*)
  - Sweep: 10 difficulty levels (easy -> hard)
  - N=50 per difficulty level
  - K=1 and K=11 for both FP16 and 4-bit
  - x-axis: FP16 K=1 baseline accuracy
  - y-axis: Delta accuracy (4-bit - FP16)

Expected: Inverted-U curve. SR effect strongest at ~20-80% baseline.

Model: Qwen-1.5B
Resolves: Future Work #1 (intermediate competence characterization)

Author: Hiroto Funasaki
"""
import os, json, gc, time, random, re
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DEVICE = "cuda"
MAX_NEW_TOKENS = 256
SEED = 2026
N_PER_LEVEL = 50


class NoiseHook:
    def __init__(self):
        self.sigma = 0.0
        self.handle = None

    def register(self, model, layer_frac=0.5):
        layers = self._get_layers(model)
        idx = max(0, int(len(layers) * layer_frac))
        hook_obj = self
        def hook_fn(module, args):
            hs = args[0]
            if hook_obj.sigma <= 0: return args
            noise = torch.randn_like(hs) * hook_obj.sigma
            return (hs + noise,) + args[1:]
        self.handle = layers[idx].register_forward_pre_hook(hook_fn)
        return idx

    def _get_layers(self, model):
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel): model = model.base_model.model
        except ImportError: pass
        for attr_path in ['model.layers', 'transformer.h', 'model.decoder.layers']:
            obj = model
            try:
                for attr in attr_path.split('.'): obj = getattr(obj, attr)
                return obj
            except AttributeError: continue
        raise ValueError("Could not find transformer layers")

    def setup(self, sigma): self.sigma = sigma
    def remove(self):
        if self.handle: self.handle.remove()


DIVERSE_SIGMAS = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


def solve(model, tokenizer, prompt, hook, sigma, do_sample=False, temperature=0.7):
    hook.setup(sigma)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                      max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample and sigma > 0,
            temperature=temperature if (do_sample and sigma > 0) else 1.0,
            top_p=0.9 if (do_sample and sigma > 0) else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()


def extract_number(text):
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    return int(numbers[0]) if numbers else None


DIFFICULTY_LEVELS = [
    # (label, digit_range, ops, description)
    ("1d+1d",   (1, 9),     ['+'],       "1-digit addition"),
    ("2d+1d",   (10, 99),   ['+'],       "2-digit + 1-digit addition"),
    ("2d+2d",   (10, 99),   ['+', '-'],  "2-digit add/sub"),
    ("2d*1d",   (10, 99),   ['*'],       "2-digit x 1-digit multiply"),
    ("3d+2d",   (100, 999), ['+', '-'],  "3-digit add/sub"),
    ("2d*2d",   (10, 99),   ['*'],       "2-digit x 2-digit multiply"),
    ("3d+3d",   (100, 999), ['+', '-'],  "3-digit add/sub"),
    ("3d*1d",   (100, 999), ['*'],       "3-digit x 1-digit multiply"),
    ("3d*2d",   (100, 999), ['*'],       "3-digit x 2-digit multiply"),
    ("4d+4d",   (1000, 9999), ['+', '-', '*'], "4-digit mixed ops"),
]


def generate_problems_at_level(level_idx, n, seed=42):
    label, (lo, hi), ops, desc = DIFFICULTY_LEVELS[level_idx]
    rng = random.Random(seed + level_idx * 1000)
    problems = []
    for _ in range(n):
        a = rng.randint(lo, hi)
        if ops == ['*'] and lo >= 100:
            b = rng.randint(1, 9) if level_idx <= 7 else rng.randint(10, 99)
        else:
            b = rng.randint(lo if len(ops) > 1 else 1, hi if lo < 100 else min(hi, 99))
        op = rng.choice(ops)
        if op == '+': answer = a + b
        elif op == '-': answer = a - b
        else: answer = a * b
        problems.append({
            'prompt': f"Calculate: {a} {op} {b} = ? Give only the final number.",
            'answer': answer,
        })
    return problems, label, desc


def evaluate_nbs(model, tokenizer, hook, problems, K):
    correct = 0
    for prob in problems:
        found = False
        for i in range(K):
            sigma = DIVERSE_SIGMAS[i] if K > 1 else 0.0
            resp = solve(model, tokenizer, prob['prompt'], hook, sigma,
                        do_sample=(K > 1), temperature=0.7)
            pred = extract_number(resp)
            if pred is not None and pred == prob['answer']:
                found = True; break
        if found: correct += 1
    return correct / len(problems) if problems else 0


def load_model(model_id, quant_4bit):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if quant_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                     device_map="auto", torch_dtype=torch.float16,
                                                     local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                     torch_dtype=torch.float16, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase71_resonance_window.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 71: Resonance Window',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 71: Resonance Window Quantification")
    print("  At what baseline competence does SR-Quantization work?")
    print("=" * 70)

    MODEL_ID = "Qwen/Qwen2.5-1.5B"
    all_results = []

    for quant_label, use_4bit in [("FP16", False), ("4-bit", True)]:
        print(f"\n  Loading {MODEL_ID} ({quant_label})...")
        model, tokenizer = load_model(MODEL_ID, quant_4bit=use_4bit)

        hook = NoiseHook()
        layer_idx = hook.register(model, layer_frac=0.5)
        print(f"  Hook at layer {layer_idx}")

        for level_idx in range(len(DIFFICULTY_LEVELS)):
            problems, label, desc = generate_problems_at_level(level_idx, N_PER_LEVEL)
            print(f"\n  [{quant_label}] Level {level_idx}: {label} ({desc})")

            for K in [1, 11]:
                t0 = time.time()
                acc = evaluate_nbs(model, tokenizer, hook, problems, K)
                elapsed = time.time() - t0
                print(f"    K={K:2d}: {acc*100:.1f}% ({elapsed:.0f}s)")

                all_results.append({
                    'quant': quant_label,
                    'level': level_idx,
                    'level_label': label,
                    'description': desc,
                    'K': K,
                    'accuracy': acc,
                    'time_sec': elapsed,
                })
            _save(all_results)

        hook.remove()
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

    # Summary: compute deltas
    print(f"\n{'='*70}")
    print("RESONANCE WINDOW ANALYSIS")
    print(f"{'='*70}")
    print(f"  {'Level':<10s} {'FP16 K=1':>10s} {'4bit K=1':>10s} {'Delta K=1':>10s} | "
          f"{'FP16 K=11':>10s} {'4bit K=11':>10s} {'Delta K=11':>10s}")

    for level_idx in range(len(DIFFICULTY_LEVELS)):
        fp16_k1 = next((r['accuracy'] for r in all_results
                        if r['level']==level_idx and r['quant']=='FP16' and r['K']==1), 0)
        bit4_k1 = next((r['accuracy'] for r in all_results
                        if r['level']==level_idx and r['quant']=='4-bit' and r['K']==1), 0)
        fp16_k11 = next((r['accuracy'] for r in all_results
                         if r['level']==level_idx and r['quant']=='FP16' and r['K']==11), 0)
        bit4_k11 = next((r['accuracy'] for r in all_results
                         if r['level']==level_idx and r['quant']=='4-bit' and r['K']==11), 0)
        delta_k1 = (bit4_k1 - fp16_k1) * 100
        delta_k11 = (bit4_k11 - fp16_k11) * 100
        label = DIFFICULTY_LEVELS[level_idx][0]
        print(f"  {label:<10s} {fp16_k1*100:>9.1f}% {bit4_k1*100:>9.1f}% {delta_k1:>+9.1f}pp | "
              f"{fp16_k11*100:>9.1f}% {bit4_k11*100:>9.1f}% {delta_k11:>+9.1f}pp")

    _generate_figure(all_results)
    print("\nPhase 71 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        n_levels = len(DIFFICULTY_LEVELS)

        for ax, K in [(ax1, 1), (ax2, 11)]:
            fp16_accs = []
            deltas = []
            labels = []
            for level_idx in range(n_levels):
                fp16 = next((r['accuracy'] for r in results
                            if r['level']==level_idx and r['quant']=='FP16' and r['K']==K), 0)
                bit4 = next((r['accuracy'] for r in results
                            if r['level']==level_idx and r['quant']=='4-bit' and r['K']==K), 0)
                fp16_accs.append(fp16 * 100)
                deltas.append((bit4 - fp16) * 100)
                labels.append(DIFFICULTY_LEVELS[level_idx][0])

            # Scatter plot: x=FP16 baseline, y=delta
            scatter = ax.scatter(fp16_accs, deltas, c=range(n_levels),
                                cmap='viridis', s=100, zorder=5, edgecolors='white', linewidth=1.5)
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (fp16_accs[i]+1, deltas[i]+0.5), fontsize=8)

            # Fit polynomial to find the window
            if len(set(fp16_accs)) > 2:
                z = np.polyfit(fp16_accs, deltas, 2)
                p = np.poly1d(z)
                xs = np.linspace(0, 100, 100)
                ax.plot(xs, p(xs), '--', color='#EF4444', alpha=0.7, label=f'Quadratic fit')

            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax.set_xlabel('FP16 Baseline Accuracy (%)', fontsize=12)
            ax.set_ylabel('Delta (4-bit - FP16) (pp)', fontsize=12)
            ax.set_title(f'K={K}', fontsize=13, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if len(set(fp16_accs)) > 2:
                ax.legend()

        fig.suptitle('Phase 71: Resonance Window\nWhen does SR-Quantization work?',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase71_resonance_window.png"),
                   bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
