"""
Phase 59: Stochastic Resonance Quantization (SR-Q)
Can noise + beam search recover intelligence lost to extreme compression?

Hypothesis: "Lost parameters (spatial resolution) can be compensated
by SNN noise + NBS (temporal resolution)."

Experiment design:
  1. Load models at multiple sizes: Llama-3.2-1B, Qwen2.5-1.5B, phi-2 (2.7B),
     Llama-3.2-3B, Mistral-7B
  2. For each model, test:
     - K=1 (no beam search, no noise) → baseline
     - K=1 with optimal σ → noise-only boost
     - K=11 σ-diverse NBS → full SR treatment
  3. Key question: Can small_model + σ-diverse NBS match or beat
     large_model at K=1?

Author: Hiroto Funasaki
"""
import os, json, gc, time, random, re, sys
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"
DEVICE = "cuda"
MAX_NEW_TOKENS = 256
SEED = 2026

# Models from smallest to largest (all locally cached)
MODELS = [
    ("meta-llama/Llama-3.2-1B", "Llama-1B"),
    ("Qwen/Qwen2.5-1.5B", "Qwen-1.5B"),
    ("microsoft/phi-2", "phi-2 (2.7B)"),
    ("meta-llama/Llama-3.2-3B", "Llama-3B"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B"),
]


# ==============================================================
# GSM8K: Simple math problems for evaluation
# ==============================================================
def generate_math_problems(n=50, seed=42):
    """Generate math problems with known answers."""
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
    """Extract the first number from model output."""
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    return int(numbers[0]) if numbers else None


# ==============================================================
# Noise Hook
# ==============================================================
class NoiseHook:
    def __init__(self):
        self.sigma = 0.0
        self.handle = None

    def register(self, model, layer_frac=0.5):
        """Register noise hook at ~50% depth of the model layers."""
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
        """Get transformer layers from various model architectures."""
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                model = model.base_model.model
        except ImportError:
            pass

        # Try common layer access patterns
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


# ==============================================================
# Generate with noise
# ==============================================================
def solve_problem(model, tokenizer, prompt, hook, sigma,
                  max_new_tokens=MAX_NEW_TOKENS):
    """Run inference with given noise sigma."""
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


# ==============================================================
# Evaluation modes
# ==============================================================
def evaluate_k1(model, tokenizer, hook, problems, sigma=0.0, label=""):
    """K=1 evaluation: single attempt per problem."""
    correct = 0
    for prob in problems:
        response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
        pred = extract_number(response)
        if pred is not None and pred == prob['answer']:
            correct += 1
    rate = correct / len(problems) if problems else 0
    if label:
        print(f"    {label}: {correct}/{len(problems)} = {rate*100:.1f}%")
    return rate


def evaluate_sigma_diverse_nbs(model, tokenizer, hook, problems,
                                sigmas, K=11, label=""):
    """σ-diverse NBS: try multiple σ values per problem, first correct wins."""
    correct = 0
    for prob in problems:
        found = False
        for sigma in sigmas[:K]:
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


# ==============================================================
# Main
# ==============================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 59: Stochastic Resonance Quantization (SR-Q)")
    print("  Can noise + compute recover intelligence lost to compression?")
    print("  'Lost params (space) compensated by noise + NBS (time)'")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    problems = generate_math_problems(50, seed=42)
    print(f"  Test set: {len(problems)} math problems")

    # σ values for diverse NBS
    diverse_sigmas = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    all_results = {}

    for model_id, model_label in MODELS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_label} ({model_id})")
        print(f"{'='*70}")

        try:
            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

            # Load in fp16 (or 4-bit for 7B+ models)
            if "7B" in model_label or "14B" in model_label:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=bnb,
                    device_map="auto", torch_dtype=torch.float16,
                    local_files_only=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, device_map="auto", torch_dtype=torch.float16,
                    local_files_only=True)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()

            load_time = time.time() - t0
            print(f"  Loaded in {load_time:.1f}s")

            # Register noise hook
            hook = NoiseHook()
            layer_idx = hook.register(model, layer_frac=0.5)
            print(f"  Hook at layer {layer_idx}")

            model_results = {}

            # Test 1: K=1, σ=0 (pure baseline)
            print(f"\n  --- K=1, σ=0 (baseline) ---")
            t0 = time.time()
            baseline = evaluate_k1(model, tokenizer, hook, problems,
                                  sigma=0.0, label="Baseline")
            model_results['k1_baseline'] = {
                'accuracy': baseline,
                'time_sec': time.time() - t0,
            }

            # Test 2: K=1 with various fixed σ
            print(f"\n  --- K=1, fixed σ sweep ---")
            best_sigma, best_acc = 0.0, baseline
            for sigma in [0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.3]:
                acc = evaluate_k1(model, tokenizer, hook, problems,
                                 sigma=sigma, label=f"σ={sigma}")
                if acc > best_acc:
                    best_acc = acc
                    best_sigma = sigma
                model_results[f'k1_sigma_{sigma}'] = {'accuracy': acc}

            model_results['k1_best_fixed'] = {
                'sigma': best_sigma,
                'accuracy': best_acc,
            }

            # Test 3: σ-diverse NBS (K=11)
            print(f"\n  --- σ-Diverse NBS (K=11) ---")
            t0 = time.time()
            diverse_acc = evaluate_sigma_diverse_nbs(
                model, tokenizer, hook, problems,
                diverse_sigmas, K=11, label="σ-Diverse K=11")
            model_results['sigma_diverse_k11'] = {
                'accuracy': diverse_acc,
                'time_sec': time.time() - t0,
            }

            all_results[model_label] = model_results

            # Cleanup
            hook.remove()
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR loading {model_label}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_label] = {'error': str(e)}
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # ==============================================================
    # Grand Summary
    # ==============================================================
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Accuracy by Model × Method")
    print(f"{'='*70}")
    print(f"{'Model':>20s} | {'K=1 σ=0':>10s} {'K=1 best-σ':>12s} {'K=11 σ-Div':>12s}")
    print("-" * 60)

    model_k1_baselines = {}
    model_diverse = {}

    for model_id, model_label in MODELS:
        if model_label not in all_results or 'error' in all_results[model_label]:
            print(f"{model_label:>20s} | {'ERROR':>10s}")
            continue
        r = all_results[model_label]
        b = r['k1_baseline']['accuracy']
        bf = r.get('k1_best_fixed', {}).get('accuracy', b)
        d = r.get('sigma_diverse_k11', {}).get('accuracy', 0)
        model_k1_baselines[model_label] = b
        model_diverse[model_label] = d
        print(f"{model_label:>20s} | {b*100:>8.1f}%  {bf*100:>10.1f}%  {d*100:>10.1f}%")

    # Key question: Can small + NBS beat large at K=1?
    print(f"\n{'='*70}")
    print("KEY QUESTION: Can Small + sigma-Diverse NBS >= Large at K=1?")
    print(f"{'='*70}")

    if len(model_k1_baselines) >= 2 and len(model_diverse) >= 2:
        labels = [l for _, l in MODELS if l in model_diverse]
        if len(labels) >= 2:
            largest_label = labels[-1]
            largest_baseline = model_k1_baselines.get(largest_label, 0)
            print(f"  Largest model baseline: {largest_label} K=1 = {largest_baseline*100:.1f}%")
            for label in labels[:-1]:
                div = model_diverse.get(label, 0)
                diff = div - largest_baseline
                verdict = "[WIN]" if diff > 0 else ("[TIE]" if diff == 0 else "[LOSE]")
                print(f"  {label:>15s} + NBS = {div*100:.1f}% ({diff*100:+.1f}pp) -> {verdict}")

    # Save results
    save_path = os.path.join(RESULTS_DIR, "phase59_sr_quantization.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 59: Stochastic Resonance Quantization',
            'timestamp': datetime.now().isoformat(),
            'n_problems': len(problems),
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
