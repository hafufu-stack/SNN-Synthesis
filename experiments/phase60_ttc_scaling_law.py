"""
Phase 60: LLM Test-Time Compute Scaling Law (Lightweight)
Does sigma-diverse NBS accuracy scale toward 100% as K increases?

LIGHTWEIGHT VERSION:
  - 30 math problems (not 50)
  - 15 hanoi problems (not 20)
  - K values: 1, 3, 5, 11, 21 (removed K=51)
  - No combined round (Math and Hanoi only)

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
MAX_NEW_TOKENS = 200  # reduced from 256
SEED = 2026
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


# ==============================================================
# Task Generators
# ==============================================================
def generate_math_problems(n=30, seed=42):
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        ptype = rng.randint(0, 2)
        if ptype == 0:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            op = rng.choice(['+', '-', '*'])
            answer = eval(f"{a}{op}{b}")
            problems.append({
                'prompt': f"Calculate: {a} {op} {b} = ? Give only the final number.",
                'answer': answer,
                'type': 'arithmetic',
            })
        elif ptype == 1:
            a, b, c = rng.randint(5, 30), rng.randint(5, 30), rng.randint(2, 10)
            answer = (a + b) * c
            problems.append({
                'prompt': f"What is ({a} + {b}) x {c}? Give only the final number.",
                'answer': answer,
                'type': 'two-step',
            })
        else:
            items = rng.randint(3, 15)
            price = rng.randint(2, 20)
            discount = rng.randint(1, items-1)
            answer = (items - discount) * price
            problems.append({
                'prompt': f"I bought {items} items at ${price} each, but returned {discount}. "
                          f"How much did I pay in total? Give only the final number.",
                'answer': answer,
                'type': 'word-problem',
            })
    return problems


def generate_hanoi_problems(n=15, seed=42):
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        n_disks = rng.randint(2, 4)
        source = rng.choice(['A', 'B', 'C'])
        target = rng.choice([p for p in ['A', 'B', 'C'] if p != source])
        expected_moves = 2 ** n_disks - 1
        problems.append({
            'prompt': f"Move {n_disks} disks from peg {source} to peg {target} in Tower of Hanoi. "
                      f"How many moves does the optimal solution need? Give only the number.",
            'answer': expected_moves,
            'type': 'hanoi',
        })
    return problems


def extract_number(text):
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


# ==============================================================
# Inference
# ==============================================================
def solve_problem(model, tokenizer, prompt, hook, sigma):
    hook.setup(sigma)
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                      max_length=512).to(DEVICE)
    temperature = max(0.1, min(2.0, 0.3 + sigma * 2.0))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(sigma > 0),
            temperature=temperature if sigma > 0 else 1.0,
            top_p=0.9 if sigma > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()
    return response


# ==============================================================
# sigma-Diverse NBS
# ==============================================================
def get_sigma_schedule(K):
    if K == 1:
        return [0.0]
    base_sigmas = [0.0, 0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05,
                   0.07, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35,
                   0.4, 0.45, 0.5, 0.6, 0.7]
    if K <= len(base_sigmas):
        indices = np.linspace(0, len(base_sigmas)-1, K).astype(int)
        return [base_sigmas[i] for i in indices]
    return list(np.linspace(0.0, 1.0, K))


def evaluate_nbs(model, tokenizer, hook, problems, K, label=""):
    """sigma-diverse NBS: try K different sigma per problem, first correct wins."""
    sigmas = get_sigma_schedule(K)
    correct = 0
    for i, prob in enumerate(problems):
        found = False
        for sigma in sigmas:
            response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
            pred = extract_number(response)
            if pred is not None and pred == prob['answer']:
                found = True
                break
        if found:
            correct += 1
        # Progress
        if (i+1) % 10 == 0:
            print(f"      [{label}] {i+1}/{len(problems)} done, {correct} correct")
    rate = correct / len(problems) if problems else 0
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
    print("Phase 60: LLM Test-Time Compute Scaling Law (Lightweight)")
    print("  Does sigma-diverse NBS accuracy scale to 100% with increasing K?")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    math_problems = generate_math_problems(30, seed=42)
    hanoi_problems = generate_hanoi_problems(15, seed=42)
    print(f"  Test set: {len(math_problems)} math + {len(hanoi_problems)} hanoi")

    # Load model
    print(f"\n  Loading {MODEL_NAME} (4-bit)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16,
        local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    hook = NoiseHook()
    layer_idx = hook.register(model, layer_frac=0.5)
    print(f"  Hook at layer {layer_idx}")

    # K values (capped at 21)
    K_values = [1, 3, 5, 11, 21]

    all_results = {}

    # --- Math ---
    print(f"\n{'='*70}")
    print("  TASK: Math Problems")
    print(f"{'='*70}")
    math_results = {}
    for K in K_values:
        sigmas = get_sigma_schedule(K)
        print(f"\n  K = {K} ({len(sigmas)} sigma values)")
        t0 = time.time()
        acc = evaluate_nbs(model, tokenizer, hook, math_problems, K,
                          label=f"Math K={K}")
        elapsed = time.time() - t0
        math_results[f"K={K}"] = {
            'accuracy': acc,
            'time_sec': elapsed,
            'time_per_problem_sec': elapsed / len(math_problems),
        }
        print(f"      Time: {elapsed:.1f}s ({elapsed/len(math_problems):.1f}s/prob)")

        # Save intermediate results after each K
        save_path = os.path.join(RESULTS_DIR, "phase60_ttc_scaling_law.json")
        with open(save_path, 'w') as f:
            json.dump({
                'experiment': 'Phase 60: LLM Test-Time Compute Scaling Law',
                'timestamp': datetime.now().isoformat(),
                'model': MODEL_NAME,
                'status': 'in_progress',
                'K_values': K_values,
                'math_results': math_results,
            }, f, indent=2, default=str)

    all_results['math'] = math_results

    # --- Hanoi ---
    print(f"\n{'='*70}")
    print("  TASK: Tower of Hanoi")
    print(f"{'='*70}")
    hanoi_results = {}
    for K in K_values:
        print(f"\n  K = {K}")
        t0 = time.time()
        acc = evaluate_nbs(model, tokenizer, hook, hanoi_problems, K,
                          label=f"Hanoi K={K}")
        elapsed = time.time() - t0
        hanoi_results[f"K={K}"] = {
            'accuracy': acc,
            'time_sec': elapsed,
        }

    all_results['hanoi'] = hanoi_results

    # ==============================================================
    # Scaling Law Summary
    # ==============================================================
    print(f"\n{'='*70}")
    print("SCALING LAW: Accuracy vs K (Test-Time Compute)")
    print(f"{'='*70}")
    print(f"{'K':>5s} | {'Math':>8s} {'Hanoi':>8s} | {'Time/prob':>10s}")
    print("-" * 45)
    for K in K_values:
        m = math_results[f"K={K}"]['accuracy']
        h = hanoi_results[f"K={K}"]['accuracy']
        t = math_results[f"K={K}"].get('time_per_problem_sec', 0)
        print(f"{K:>5d} | {m*100:>6.1f}%  {h*100:>6.1f}%  | {t:>8.1f}s")

    # Asymptotic analysis
    print(f"\n{'='*70}")
    print("ASYMPTOTIC ANALYSIS:")
    math_rates = [math_results[f"K={K}"]['accuracy'] for K in K_values]
    hanoi_rates = [hanoi_results[f"K={K}"]['accuracy'] for K in K_values]

    for name, rates in [("Math", math_rates), ("Hanoi", hanoi_rates)]:
        if rates[-1] >= 0.99:
            verdict = "-> CONVERGES to ~100%"
        elif len(rates) >= 3:
            last3 = rates[-1] - rates[-3]
            if last3 < 0.01:
                verdict = f"-> PLATEAUS at {rates[-1]*100:.1f}%"
            else:
                verdict = f"-> STILL GROWING (+{last3*100:.1f}pp)"
        else:
            verdict = "-> insufficient data"
        print(f"  {name}: {' -> '.join(f'{r*100:.1f}%' for r in rates)} {verdict}")

    # Compute efficiency
    print(f"\n{'='*70}")
    print("COMPUTE EFFICIENCY:")
    k1_acc = math_results["K=1"]['accuracy']
    for K in K_values[1:]:
        acc = math_results[f"K={K}"]['accuracy']
        gain = acc - k1_acc
        print(f"  K={K:>3d}: +{gain*100:.1f}pp accuracy, {K}x compute -> "
              f"{gain*100/K:.2f}pp per beam")

    # Cleanup
    hook.remove()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Final save
    save_path = os.path.join(RESULTS_DIR, "phase60_ttc_scaling_law.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 60: LLM Test-Time Compute Scaling Law',
            'timestamp': datetime.now().isoformat(),
            'model': MODEL_NAME,
            'status': 'complete',
            'K_values': K_values,
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
