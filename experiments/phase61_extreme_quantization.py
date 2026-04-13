"""
Phase 61: Extreme SR-Quantization - Can noise resurrect a destroyed model?

Tests if σ-diverse NBS can recover intelligence from aggressively quantized models.
If a 4-bit 1.5B model + K=51 NBS can beat FP16 7B at K=1, then
"noise + time can fully compensate for spatial (parameter) destruction."

Experiment matrix:
  - Qwen-1.5B at {FP16, 8-bit, 4-bit} × K={1, 11, 21, 51}
  - Mistral-7B at 4-bit × K=1 (target baseline to beat)
  - Task: Math problems (consistent with Phase 59)

Author: Hiroto Funasaki
"""
import os, json, gc, time, random, re, sys
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"
FIGURES_DIR = r"c:\Users\kyjan\研究\snn-synthesis\figures"
DEVICE = "cuda"
MAX_NEW_TOKENS = 256
SEED = 2026

# σ values for diverse NBS (K=51 max)
DIVERSE_SIGMAS = [
    0.0, 0.001, 0.003, 0.005, 0.008,
    0.01, 0.015, 0.02, 0.03, 0.04,
    0.05, 0.06, 0.07, 0.08, 0.1,
    0.12, 0.15, 0.18, 0.2, 0.22,
    0.25, 0.28, 0.3, 0.33, 0.35,
    0.38, 0.4, 0.42, 0.45, 0.48,
    0.5, 0.52, 0.55, 0.58, 0.6,
    0.62, 0.65, 0.68, 0.7, 0.72,
    0.75, 0.78, 0.8, 0.82, 0.85,
    0.88, 0.9, 0.92, 0.95, 1.0, 1.1,
]


def generate_math_problems(n=50, seed=42):
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(['+', '-', '*'])
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:
            answer = a * b
        problems.append({
            'prompt': f"Calculate: {a} {op} {b} = ? Give only the final number.",
            'answer': answer,
        })
    return problems


def extract_number(text):
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    return int(numbers[0]) if numbers else None


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
            if hook_obj.sigma <= 0:
                return args
            noise = torch.randn_like(hs) * hook_obj.sigma
            return (hs + noise,) + args[1:]
        self.handle = layers[idx].register_forward_pre_hook(hook_fn)
        return idx

    def _get_layers(self, model):
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                model = model.base_model.model
        except ImportError:
            pass
        for attr_path in ['model.layers', 'transformer.h', 'model.decoder.layers']:
            obj = model
            try:
                for attr in attr_path.split('.'):
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise ValueError("Could not find transformer layers in model")

    def setup(self, sigma):
        self.sigma = sigma

    def remove(self):
        if self.handle:
            self.handle.remove()


def solve_problem(model, tokenizer, prompt, hook, sigma,
                  max_new_tokens=MAX_NEW_TOKENS):
    hook.setup(sigma)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                      max_length=512).to(DEVICE)
    temperature = max(0.1, min(2.0, 0.3 + sigma * 2.0))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(sigma > 0),
            temperature=temperature if sigma > 0 else 1.0,
            top_p=0.9 if sigma > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()
    return response


def evaluate_sigma_diverse_nbs(model, tokenizer, hook, problems, K, label=""):
    """Sigma-diverse NBS: try K beams with diverse sigma values."""
    sigmas = DIVERSE_SIGMAS[:K]
    correct = 0
    for prob in problems:
        found = False
        for sigma in sigmas:
            response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
            pred = extract_number(response)
            if pred is not None and pred == prob['answer']:
                found = True
                break
        if found:
            correct += 1
    rate = correct / len(problems) if problems else 0
    if label:
        print(f"    {label}: {correct}/{len(problems)} = {rate*100:.1f}%")
    return rate


def load_model(model_id, quant_mode="fp16"):
    """Load model with specified quantization."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    if quant_mode == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16,
            local_files_only=True)
    elif quant_mode == "8bit":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16,
            local_files_only=True)
    else:  # fp16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16,
            local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 61: Extreme SR-Quantization")
    print("  Can noise + NBS resurrect a destroyed (quantized) model?")
    print("  '4-bit 1.5B + K=51 NBS vs FP16 7B at K=1'")
    print("=" * 70)

    problems = generate_math_problems(50, seed=42)
    print(f"  Test set: {len(problems)} math problems\n")

    all_results = {}
    K_VALUES = [1, 11, 21, 51]

    # ============================================================
    # Part 1: Mistral-7B baseline (4-bit, K=1)
    # ============================================================
    print("=" * 70)
    print("  BASELINE: Mistral-7B (4-bit) K=1")
    print("=" * 70)

    try:
        t0 = time.time()
        model, tokenizer = load_model(
            "mistralai/Mistral-7B-Instruct-v0.3", "4bit")
        print(f"  Loaded in {time.time()-t0:.1f}s")

        hook = NoiseHook()
        layer_idx = hook.register(model, layer_frac=0.5)
        print(f"  Hook at layer {layer_idx}")

        baseline_7b = evaluate_sigma_diverse_nbs(
            model, tokenizer, hook, problems, K=1,
            label="Mistral-7B 4bit K=1")
        all_results['Mistral-7B_4bit_K1'] = {
            'accuracy': baseline_7b, 'model': 'Mistral-7B',
            'quant': '4bit', 'K': 1}

        # Also test Mistral-7B at K=11 for reference
        nbs_7b = evaluate_sigma_diverse_nbs(
            model, tokenizer, hook, problems, K=11,
            label="Mistral-7B 4bit K=11")
        all_results['Mistral-7B_4bit_K11'] = {
            'accuracy': nbs_7b, 'model': 'Mistral-7B',
            'quant': '4bit', 'K': 11}

        hook.remove()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        baseline_7b = 0.42  # fallback from Phase 59

    # Save incremental
    _save(all_results, problems)

    # ============================================================
    # Part 2: Qwen-1.5B at multiple quantization levels × K values
    # ============================================================
    QUANT_MODES = ["fp16", "8bit", "4bit"]
    model_id = "Qwen/Qwen2.5-1.5B"

    for quant in QUANT_MODES:
        print(f"\n{'='*70}")
        print(f"  Qwen-1.5B ({quant}) x K={K_VALUES}")
        print(f"{'='*70}")

        try:
            t0 = time.time()
            model, tokenizer = load_model(model_id, quant)
            print(f"  Loaded in {time.time()-t0:.1f}s")

            hook = NoiseHook()
            layer_idx = hook.register(model, layer_frac=0.5)
            print(f"  Hook at layer {layer_idx}")

            for K in K_VALUES:
                t0 = time.time()
                acc = evaluate_sigma_diverse_nbs(
                    model, tokenizer, hook, problems, K=K,
                    label=f"Qwen-1.5B {quant} K={K}")
                elapsed = time.time() - t0

                key = f"Qwen-1.5B_{quant}_K{K}"
                all_results[key] = {
                    'accuracy': acc,
                    'model': 'Qwen-1.5B',
                    'quant': quant,
                    'K': K,
                    'time_sec': elapsed,
                    'vs_7b': f"{(acc - baseline_7b)*100:+.1f}pp",
                    'beats_7b': acc > baseline_7b,
                }
                # Save after each K
                _save(all_results, problems)

            hook.remove()
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR with {quant}: {e}")
            import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

    # ============================================================
    # Part 3: Llama-1B at 4-bit with high K (extreme small model)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  EXTREME: Llama-1B (4-bit) x K={K_VALUES}")
    print(f"{'='*70}")

    try:
        t0 = time.time()
        model, tokenizer = load_model("meta-llama/Llama-3.2-1B", "4bit")
        print(f"  Loaded in {time.time()-t0:.1f}s")

        hook = NoiseHook()
        layer_idx = hook.register(model, layer_frac=0.5)
        print(f"  Hook at layer {layer_idx}")

        for K in K_VALUES:
            t0 = time.time()
            acc = evaluate_sigma_diverse_nbs(
                model, tokenizer, hook, problems, K=K,
                label=f"Llama-1B 4bit K={K}")
            elapsed = time.time() - t0

            key = f"Llama-1B_4bit_K{K}"
            all_results[key] = {
                'accuracy': acc,
                'model': 'Llama-1B',
                'quant': '4bit',
                'K': K,
                'time_sec': elapsed,
                'vs_7b': f"{(acc - baseline_7b)*100:+.1f}pp",
                'beats_7b': acc > baseline_7b,
            }
            _save(all_results, problems)

        hook.remove()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    # ============================================================
    # Summary & Figures
    # ============================================================
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Extreme SR-Quantization")
    print(f"{'='*70}")
    print(f"  Mistral-7B 4-bit K=1 baseline: {baseline_7b*100:.1f}%")
    print()
    print(f"  {'Config':>30s} | {'Acc':>6s} | {'vs 7B':>8s} | Verdict")
    print(f"  {'-'*65}")

    for key in sorted(all_results.keys()):
        r = all_results[key]
        if 'accuracy' in r:
            acc = r['accuracy']
            vs = r.get('vs_7b', 'N/A')
            verdict = "WIN!" if r.get('beats_7b', False) else ""
            print(f"  {key:>30s} | {acc*100:5.1f}% | {vs:>8s} | {verdict}")

    # Generate figure
    _generate_figure(all_results, baseline_7b)
    _save(all_results, problems)
    print("\nPhase 61 complete!")


def _save(all_results, problems):
    save_path = os.path.join(RESULTS_DIR, "phase61_extreme_quantization.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 61: Extreme SR-Quantization',
            'timestamp': datetime.now().isoformat(),
            'n_problems': len(problems),
            'results': all_results,
        }, f, indent=2, default=str)


def _generate_figure(all_results, baseline_7b):
    """Generate publication-quality figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        matplotlib.rcParams['font.size'] = 11

        K_VALUES = [1, 11, 21, 51]
        colors = {'fp16': '#2563EB', '8bit': '#16A34A', '4bit': '#EA580C'}
        markers = {'fp16': 'o', '8bit': 's', '4bit': '^'}

        # Left: Qwen-1.5B accuracy vs K at different quantization levels
        for quant in ['fp16', '8bit', '4bit']:
            accs = []
            ks = []
            for K in K_VALUES:
                key = f"Qwen-1.5B_{quant}_K{K}"
                if key in all_results and 'accuracy' in all_results[key]:
                    accs.append(all_results[key]['accuracy'] * 100)
                    ks.append(K)
            if accs:
                ax1.plot(ks, accs, f'{markers[quant]}-', color=colors[quant],
                        linewidth=2.5, markersize=9, label=f'Qwen-1.5B {quant}', zorder=5)

        # Llama-1B 4-bit
        l1_accs = []
        l1_ks = []
        for K in K_VALUES:
            key = f"Llama-1B_4bit_K{K}"
            if key in all_results and 'accuracy' in all_results[key]:
                l1_accs.append(all_results[key]['accuracy'] * 100)
                l1_ks.append(K)
        if l1_accs:
            ax1.plot(l1_ks, l1_accs, 'D--', color='#9333EA', linewidth=2,
                    markersize=8, label='Llama-1B 4bit', zorder=5)

        # Mistral-7B baseline
        ax1.axhline(y=baseline_7b * 100, color='#DC2626', linestyle='--',
                   linewidth=2, label='Mistral-7B 4bit K=1', zorder=3)
        ax1.fill_between([0.8, 55], baseline_7b*100, alpha=0.05, color='#DC2626')

        ax1.set_xlabel('Beam Count K', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('(a) Extreme Quantization + NBS Recovery', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='lower right')
        ax1.set_xscale('log', base=2)
        ax1.grid(alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Right: K needed to beat 7B at each quantization level
        quants = []
        k_needed = []
        q_colors = []
        for quant in ['fp16', '8bit', '4bit']:
            needed = None
            for K in K_VALUES:
                key = f"Qwen-1.5B_{quant}_K{K}"
                if key in all_results and 'accuracy' in all_results[key]:
                    if all_results[key]['accuracy'] > baseline_7b:
                        needed = K
                        break
            quants.append(f"Qwen-1.5B\n{quant}")
            k_needed.append(needed if needed else 999)
            q_colors.append(colors[quant])

        # Llama-1B
        needed_1b = None
        for K in K_VALUES:
            key = f"Llama-1B_4bit_K{K}"
            if key in all_results and 'accuracy' in all_results[key]:
                if all_results[key]['accuracy'] > baseline_7b:
                    needed_1b = K
                    break
        quants.append("Llama-1B\n4bit")
        k_needed.append(needed_1b if needed_1b else 999)
        q_colors.append('#9333EA')

        bars = ax2.bar(range(len(quants)), k_needed, color=q_colors,
                      edgecolor='white', linewidth=0.5, alpha=0.85)

        for i, (k, c) in enumerate(zip(k_needed, q_colors)):
            label = f"K={k}" if k < 999 else "Never"
            ax2.text(i, k + 1, label, ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color=c)

        ax2.set_xticks(range(len(quants)))
        ax2.set_xticklabels(quants, fontsize=10)
        ax2.set_ylabel('Min K to Beat Mistral-7B', fontsize=12)
        ax2.set_title('(b) "Resurrection Threshold":\nK Needed to Beat 7B', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase61_extreme_quantization.png")
        fig.savefig(fig_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved: {fig_path}")

    except Exception as e:
        print(f"  Figure generation error: {e}")


if __name__ == '__main__':
    main()
