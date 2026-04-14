"""
Phase 67: σ-T Decoupling — Hook Noise vs Temperature Noise

Tests whether stochastic resonance comes from:
  A) Hook noise (weight-space perturbation) alone
  B) Temperature diversity (sampling noise) alone
  C) Both together (current implementation)
  D) Neither (greedy baseline)

If temperature alone replicates the NBS effect, hooks are unnecessary
and SR can be applied via any LLM API (no model access needed).

Experiment: Qwen-1.5B 4-bit × K=11 × 4 conditions × N=30

Author: Hiroto Funasaki
"""
import os, json, gc, time, random, re, sys
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
N_PROBLEMS = 30
K = 11

DIVERSE_SIGMAS = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
DIVERSE_TEMPS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0]


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
        raise ValueError("Could not find transformer layers")

    def setup(self, sigma):
        self.sigma = sigma

    def remove(self):
        if self.handle:
            self.handle.remove()


def solve_with_params(model, tokenizer, prompt, hook, sigma, temperature, do_sample):
    """Solve with explicit control of hook sigma and sampling temperature."""
    hook.setup(sigma)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                      max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()
    return response


def extract_number(text):
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    return int(numbers[0]) if numbers else None


def generate_math_problems(n=30, seed=42):
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


def evaluate_condition(model, tokenizer, hook, problems, condition, label=""):
    """Evaluate a specific noise condition."""
    correct = 0
    for prob in problems:
        found = False
        for beam_idx in range(K):
            if condition == 'greedy':
                # Condition D: No noise at all (greedy decoding, no hook)
                sigma = 0.0
                temp = 1.0
                do_sample = False
            elif condition == 'temp_only':
                # Condition B: Temperature diversity only, no hook noise
                sigma = 0.0
                temp = DIVERSE_TEMPS[beam_idx]
                do_sample = True
            elif condition == 'hook_only':
                # Condition C: Hook noise only, fixed low temperature
                sigma = DIVERSE_SIGMAS[beam_idx]
                temp = 0.7
                do_sample = (sigma > 0)
            elif condition == 'both':
                # Condition A: Both (current Phase 61 implementation)
                sigma = DIVERSE_SIGMAS[beam_idx]
                temp = max(0.1, min(2.0, 0.3 + sigma * 2.0))
                do_sample = (sigma > 0)
            else:
                raise ValueError(f"Unknown condition: {condition}")

            response = solve_with_params(model, tokenizer, prob['prompt'],
                                        hook, sigma, temp, do_sample)
            pred = extract_number(response)
            if pred is not None and pred == prob['answer']:
                found = True
                break

            # For greedy: only 1 beam (all beams produce same output)
            if condition == 'greedy':
                break

        if found:
            correct += 1

    rate = correct / len(problems) if problems else 0
    if label:
        print(f"    {label}: {correct}/{len(problems)} = {rate*100:.1f}%")
    return rate


def _save(results):
    path = os.path.join(RESULTS_DIR, "phase67_sigma_t_decoupling.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 67: σ-T Decoupling',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 67: σ-T Decoupling")
    print("  Is stochastic resonance from hook noise, temperature, or both?")
    print("=" * 70)

    problems = generate_math_problems(N_PROBLEMS, seed=42)
    print(f"  Test set: {len(problems)} math problems, K={K}\n")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    MODEL_ID = "Qwen/Qwen2.5-1.5B"
    all_results = {}

    # Load model once (4-bit to match Phase 61)
    print("  Loading Qwen-1.5B (4-bit)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16,
        local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    hook = NoiseHook()
    layer_idx = hook.register(model, layer_frac=0.5)
    print(f"  Hook at layer {layer_idx}\n")

    CONDITIONS = [
        ('greedy',    'D: Greedy (no noise, K=1)'),
        ('temp_only', 'B: Temperature-diverse only (K=11)'),
        ('hook_only', 'C: Hook σ-diverse only (K=11)'),
        ('both',      'A: Both σ+T coupled (K=11, current)'),
    ]

    for cond_key, cond_label in CONDITIONS:
        print(f"  --- {cond_label} ---")
        t0 = time.time()
        acc = evaluate_condition(model, tokenizer, hook, problems,
                               cond_key, label=cond_label)
        elapsed = time.time() - t0

        all_results[cond_key] = {
            'accuracy': acc,
            'label': cond_label,
            'time_sec': elapsed,
            'K': 1 if cond_key == 'greedy' else K,
        }
        _save(all_results)
        print()

    hook.remove()
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: σ-T Decoupling")
    print(f"{'='*70}")
    for cond_key, cond_label in CONDITIONS:
        r = all_results.get(cond_key, {})
        print(f"  {cond_label:<45s}: {r.get('accuracy', 0)*100:.1f}%")

    greedy = all_results.get('greedy', {}).get('accuracy', 0)
    temp = all_results.get('temp_only', {}).get('accuracy', 0)
    hook_only = all_results.get('hook_only', {}).get('accuracy', 0)
    both = all_results.get('both', {}).get('accuracy', 0)

    print(f"\n  Temperature-only gain over greedy: {(temp-greedy)*100:+.1f}pp")
    print(f"  Hook-only gain over greedy:        {(hook_only-greedy)*100:+.1f}pp")
    print(f"  Both gain over greedy:             {(both-greedy)*100:+.1f}pp")

    if temp > hook_only + 0.05:
        print("\n  >>> TEMPERATURE IS THE DOMINANT NOISE SOURCE!")
        print("  >>> SR can be applied via any LLM API (no hook needed)")
    elif hook_only > temp + 0.05:
        print("\n  >>> HOOK NOISE IS THE DOMINANT SOURCE!")
        print("  >>> Weight-space perturbation is essential (SNN-Genesis confirmed)")
    else:
        print("\n  >>> BOTH SOURCES CONTRIBUTE COMPARABLY")
        print("  >>> Maximum SR requires both noise channels")

    _generate_figure(all_results)
    print("\nPhase 67 complete!")


def _generate_figure(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        conditions = ['greedy', 'temp_only', 'hook_only', 'both']
        labels = ['D: Greedy\n(no noise)', 'B: Temperature\ndiverse only',
                 'C: Hook σ\ndiverse only', 'A: Both\nσ + T coupled']
        colors = ['#9CA3AF', '#F59E0B', '#3B82F6', '#10B981']

        accs = [results.get(c, {}).get('accuracy', 0) * 100 for c in conditions]

        bars = ax.bar(range(len(conditions)), accs, color=colors,
                     edgecolor='white', linewidth=1.5, width=0.65)

        for i, (a, c) in enumerate(zip(accs, colors)):
            ax.text(i, a + 1, f'{a:.1f}%', ha='center', va='bottom',
                   fontsize=14, fontweight='bold', color=c)

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=13)
        ax.set_title('Phase 67: σ-T Decoupling\n'
                    'Where Does Stochastic Resonance Come From?',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(accs) + 15)

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "phase67_sigma_t_decoupling.png")
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved: {path}")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
