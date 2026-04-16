"""
Phase 70: Sequential Noise Coupling

Phase 67 showed simultaneous hook+temp causes destructive interference.
Phase 70 tests whether SEQUENTIAL application avoids saturation:
  A) Hook-first: Generate K candidates with sigma-diverse hook -> select best via temp-diverse resampling
  B) Interleaved: Split K=12 beams into 6 hook-only + 6 temp-only, ensemble
  C) Baseline: Hook-only K=11 (Phase 67 best: 90%)

Hypothesis: Sequential noise avoids resonance channel saturation.
If interleaved > 90%, we can combine noise sources constructively.

Model: Qwen-1.5B 4-bit, K=11-12, N=50 arithmetic
Resolves: Future Work #10 (optimal noise source combination)

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
N_PROBLEMS = 50

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


def solve_with_params(model, tokenizer, prompt, hook, sigma, temperature, do_sample):
    hook.setup(sigma)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                      max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                            skip_special_tokens=True).strip()


def extract_number(text):
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    return int(numbers[0]) if numbers else None


def generate_math_problems(n, seed=42):
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(['+', '-', '*'])
        if op == '+': answer = a + b
        elif op == '-': answer = a - b
        else: answer = a * b
        problems.append({
            'prompt': f"Calculate: {a} {op} {b} = ? Give only the final number.",
            'answer': answer,
        })
    return problems


def evaluate_hook_only(model, tokenizer, hook, problems, K=11, label=""):
    """Baseline: sigma-diverse hook, fixed T=0.7."""
    correct = 0
    for prob in problems:
        found = False
        for i in range(K):
            sigma = DIVERSE_SIGMAS[i]
            resp = solve_with_params(model, tokenizer, prob['prompt'],
                                     hook, sigma, 0.7, sigma > 0)
            pred = extract_number(resp)
            if pred is not None and pred == prob['answer']:
                found = True; break
        if found: correct += 1
    rate = correct / len(problems)
    print(f"    {label}: {correct}/{len(problems)} = {rate*100:.1f}%")
    return rate


def evaluate_interleaved(model, tokenizer, hook, problems, label=""):
    """Split K=12 into 6 hook-only + 6 temp-only beams, ensemble."""
    correct = 0
    hook_sigmas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    temp_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]

    for prob in problems:
        found = False
        # 6 hook-only beams
        for sigma in hook_sigmas:
            resp = solve_with_params(model, tokenizer, prob['prompt'],
                                     hook, sigma, 0.7, sigma > 0)
            pred = extract_number(resp)
            if pred is not None and pred == prob['answer']:
                found = True; break

        # 6 temp-only beams (if hook didn't find it)
        if not found:
            for t in temp_values:
                resp = solve_with_params(model, tokenizer, prob['prompt'],
                                         hook, 0.0, t, True)
                pred = extract_number(resp)
                if pred is not None and pred == prob['answer']:
                    found = True; break

        if found: correct += 1
    rate = correct / len(problems)
    print(f"    {label}: {correct}/{len(problems)} = {rate*100:.1f}%")
    return rate


def evaluate_sequential(model, tokenizer, hook, problems, label=""):
    """Hook-first sequential: generate K=6 with hook, resample best 3 with temp-diverse."""
    correct = 0
    hook_sigmas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    temp_values = [0.3, 1.0, 2.0]

    for prob in problems:
        found = False
        # Phase 1: Generate 6 candidates with hook-diverse
        candidates = []
        for sigma in hook_sigmas:
            resp = solve_with_params(model, tokenizer, prob['prompt'],
                                     hook, sigma, 0.7, sigma > 0)
            pred = extract_number(resp)
            candidates.append((resp, pred))
            if pred is not None and pred == prob['answer']:
                found = True; break

        # Phase 2: If not found, do temp-diverse resampling (5 more beams)
        if not found:
            for t in temp_values:
                resp = solve_with_params(model, tokenizer, prob['prompt'],
                                         hook, 0.0, t, True)
                pred = extract_number(resp)
                if pred is not None and pred == prob['answer']:
                    found = True; break

        # Phase 3: 2 more with medium hook + medium temp
        if not found:
            for sigma, t in [(0.05, 1.0), (0.1, 1.5)]:
                resp = solve_with_params(model, tokenizer, prob['prompt'],
                                         hook, sigma, t, True)
                pred = extract_number(resp)
                if pred is not None and pred == prob['answer']:
                    found = True; break

        if found: correct += 1
    rate = correct / len(problems)
    print(f"    {label}: {correct}/{len(problems)} = {rate*100:.1f}%")
    return rate


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase70_sequential_noise.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 70: Sequential Noise Coupling',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 70: Sequential Noise Coupling")
    print("  Can sequential application avoid destructive interference?")
    print("=" * 70)

    problems = generate_math_problems(N_PROBLEMS, seed=42)
    print(f"  Test set: {len(problems)} math problems\n")

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    MODEL_ID = "Qwen/Qwen2.5-1.5B"
    print(f"  Loading {MODEL_ID} (4-bit)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.float16,
                                                 local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    hook = NoiseHook()
    layer_idx = hook.register(model, layer_frac=0.5)
    print(f"  Hook at layer {layer_idx}\n")

    all_results = {}

    # A) Hook-only baseline (K=11)
    print("  --- A: Hook-only baseline (K=11) ---")
    t0 = time.time()
    acc_hook = evaluate_hook_only(model, tokenizer, hook, problems, K=11, label="Hook-only K=11")
    all_results['hook_only'] = {'accuracy': acc_hook, 'K': 11, 'time_sec': time.time()-t0}
    _save(all_results)

    # B) Interleaved: 6 hook + 6 temp (K=12 total)
    print("\n  --- B: Interleaved 6 hook + 6 temp (K=12) ---")
    t0 = time.time()
    acc_interleaved = evaluate_interleaved(model, tokenizer, hook, problems,
                                           label="Interleaved 6+6")
    all_results['interleaved'] = {'accuracy': acc_interleaved, 'K': 12, 'time_sec': time.time()-t0}
    _save(all_results)

    # C) Sequential: 6 hook -> 3 temp -> 2 mixed (K=11 total)
    print("\n  --- C: Sequential hook->temp->mixed (K=11) ---")
    t0 = time.time()
    acc_sequential = evaluate_sequential(model, tokenizer, hook, problems,
                                          label="Sequential 6+3+2")
    all_results['sequential'] = {'accuracy': acc_sequential, 'K': 11, 'time_sec': time.time()-t0}
    _save(all_results)

    hook.remove()
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY")
    print(f"{'='*70}")
    for key, r in all_results.items():
        print(f"  {key:20s}: {r['accuracy']*100:.1f}% (K={r['K']})")

    best = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    if best[0] != 'hook_only' and best[1]['accuracy'] > acc_hook:
        print(f"\n  >>> Sequential/Interleaved BEATS hook-only! ({best[0]}: {best[1]['accuracy']*100:.1f}%)")
    else:
        print(f"\n  >>> Hook-only remains best. Noise sources are fundamentally non-combinable.")

    _generate_figure(all_results)
    print("\nPhase 70 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        configs = ['hook_only', 'interleaved', 'sequential']
        labels = ['Hook-only\n(K=11)', 'Interleaved\n6 hook + 6 temp\n(K=12)', 'Sequential\nhook->temp->mix\n(K=11)']
        colors = ['#3B82F6', '#8B5CF6', '#EC4899']
        accs = [results[c]['accuracy'] * 100 for c in configs]

        bars = ax.bar(range(len(configs)), accs, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
        for i, a in enumerate(accs):
            ax.text(i, a + 1, f'{a:.1f}%', ha='center', fontsize=13, fontweight='bold', color=colors[i])

        ax.set_xticks(range(len(configs))); ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=13)
        ax.set_title('Phase 70: Sequential Noise Coupling\nCan we avoid destructive interference?',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(accs) + 15)

        # Add Phase 67 reference line
        ax.axhline(y=83.3, color='#10B981', linestyle='--', alpha=0.7, label='Phase 67: Both=83.3%')
        ax.legend(fontsize=10)

        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase70_sequential_noise.png"), bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
