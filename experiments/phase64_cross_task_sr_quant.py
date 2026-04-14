"""
Phase 64: Cross-Task SR-Quantization — Universal Triad Validation

Tests if quantization noise acts as stochastic resonance on GSM8K (math)
and TruthfulQA (factual QA), extending Phase 61's Hanoi-only finding.

If 4-bit K=1 > FP16 K=1 on math/knowledge tasks too, the space-time-precision
triad is a universal law across task domains.

Experiment matrix:
  - Qwen-1.5B at {FP16, 4-bit} × K={1, 11} × {GSM8K, TruthfulQA, Math}
  - N=50 per task

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
N_PROBLEMS = 50

DIVERSE_SIGMAS_11 = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


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


def extract_number(text):
    numbers = re.findall(r'-?\d+', text.replace(',', ''))
    return int(numbers[0]) if numbers else None


def extract_gsm8k_answer(text):
    match = re.search(r'####\s*(-?\d[\d,]*)', text)
    if match:
        return int(match.group(1).replace(',', ''))
    return None


def evaluate_nbs(model, tokenizer, hook, problems, K, label=""):
    """Sigma-diverse NBS: K beams with diverse sigma values."""
    sigmas = DIVERSE_SIGMAS_11[:K]
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


def load_gsm8k(n=50):
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=f'test[:{n}]')
    problems = []
    for item in ds:
        gold = extract_gsm8k_answer(item['answer'])
        if gold is not None:
            problems.append({
                'prompt': f"Solve this math problem step by step, then give the final answer after ####.\n\n{item['question']}",
                'answer': gold,
            })
    return problems


def generate_truthfulqa_mc(n=50):
    """Generate TruthfulQA-style MC questions (factual knowledge)."""
    questions = [
        {"prompt": "What is the capital of France? Answer with just the city name.", "answer_str": "paris"},
        {"prompt": "What is the largest planet in our solar system? Answer with just the planet name.", "answer_str": "jupiter"},
        {"prompt": "Who wrote Romeo and Juliet? Answer with just the author's name.", "answer_str": "shakespeare"},
        {"prompt": "What is the chemical symbol for gold? Answer with just the symbol.", "answer_str": "au"},
        {"prompt": "How many continents are there? Answer with just the number.", "answer_str": "7"},
        {"prompt": "What is the speed of light in m/s approximately? Answer with just the number.", "answer_str": "300000000"},
        {"prompt": "What year did World War II end? Answer with just the year.", "answer_str": "1945"},
        {"prompt": "What is the atomic number of carbon? Answer with just the number.", "answer_str": "6"},
        {"prompt": "Who painted the Mona Lisa? Answer with just the name.", "answer_str": "vinci"},
        {"prompt": "What is the square root of 144? Answer with just the number.", "answer_str": "12"},
        {"prompt": "What is the boiling point of water in Celsius? Answer with just the number.", "answer_str": "100"},
        {"prompt": "How many bones are in the adult human body? Answer with just the number.", "answer_str": "206"},
        {"prompt": "What is the chemical formula for water? Answer with just the formula.", "answer_str": "h2o"},
        {"prompt": "Who discovered penicillin? Answer with just the last name.", "answer_str": "fleming"},
        {"prompt": "What is the largest ocean on Earth? Answer with just the name.", "answer_str": "pacific"},
        {"prompt": "How many sides does a hexagon have? Answer with just the number.", "answer_str": "6"},
        {"prompt": "What is the currency of Japan? Answer with just the name.", "answer_str": "yen"},
        {"prompt": "Who developed the theory of relativity? Answer with just the last name.", "answer_str": "einstein"},
        {"prompt": "What is 17 * 23? Answer with just the number.", "answer_str": "391"},
        {"prompt": "What planet is known as the Red Planet? Answer with just the name.", "answer_str": "mars"},
        {"prompt": "What is the freezing point of water in Fahrenheit? Answer with just the number.", "answer_str": "32"},
        {"prompt": "How many days are in a leap year? Answer with just the number.", "answer_str": "366"},
        {"prompt": "What is the capital of Japan? Answer with just the city name.", "answer_str": "tokyo"},
        {"prompt": "What element has atomic number 1? Answer with just the name.", "answer_str": "hydrogen"},
        {"prompt": "How many minutes are in an hour? Answer with just the number.", "answer_str": "60"},
        {"prompt": "What is the tallest mountain in the world? Answer with just the name.", "answer_str": "everest"},
        {"prompt": "Who invented the telephone? Answer with just the last name.", "answer_str": "bell"},
        {"prompt": "What is 15 squared? Answer with just the number.", "answer_str": "225"},
        {"prompt": "What is the chemical symbol for sodium? Answer with just the symbol.", "answer_str": "na"},
        {"prompt": "How many planets are in our solar system? Answer with just the number.", "answer_str": "8"},
        {"prompt": "What year was the Declaration of Independence signed? Answer with just the year.", "answer_str": "1776"},
        {"prompt": "What is the smallest prime number? Answer with just the number.", "answer_str": "2"},
        {"prompt": "What is the capital of Australia? Answer with just the city name.", "answer_str": "canberra"},
        {"prompt": "How many chromosomes do humans have? Answer with just the number.", "answer_str": "46"},
        {"prompt": "What is the pH of pure water? Answer with just the number.", "answer_str": "7"},
        {"prompt": "Who wrote the Odyssey? Answer with just the name.", "answer_str": "homer"},
        {"prompt": "What is the largest mammal? Answer with just the name.", "answer_str": "whale"},
        {"prompt": "How many seconds are in a minute? Answer with just the number.", "answer_str": "60"},
        {"prompt": "What gas do plants absorb from the atmosphere? Answer with just the formula or name.", "answer_str": "co2"},
        {"prompt": "What is the speed of sound approximately in m/s? Answer with just the number.", "answer_str": "343"},
        {"prompt": "What is 2 to the power of 10? Answer with just the number.", "answer_str": "1024"},
        {"prompt": "What is the capital of Germany? Answer with just the city name.", "answer_str": "berlin"},
        {"prompt": "Who discovered gravity? Answer with just the last name.", "answer_str": "newton"},
        {"prompt": "How many legs does a spider have? Answer with just the number.", "answer_str": "8"},
        {"prompt": "What is the chemical symbol for iron? Answer with just the symbol.", "answer_str": "fe"},
        {"prompt": "What continent is Brazil in? Answer with just the continent name.", "answer_str": "america"},
        {"prompt": "What is absolute zero in Celsius? Answer with just the number.", "answer_str": "-273"},
        {"prompt": "How many teeth does an adult human typically have? Answer with just the number.", "answer_str": "32"},
        {"prompt": "What is the largest desert in the world? Answer with just the name.", "answer_str": "sahara"},
        {"prompt": "What is 13 * 17? Answer with just the number.", "answer_str": "221"},
    ]
    return questions[:n]


def evaluate_factual(model, tokenizer, hook, questions, K, label=""):
    """Evaluate factual QA with NBS. Check if answer_str appears in response."""
    sigmas = DIVERSE_SIGMAS_11[:K]
    correct = 0
    for q in questions:
        found = False
        for sigma in sigmas:
            response = solve_problem(model, tokenizer, q['prompt'], hook, sigma)
            if q['answer_str'].lower() in response.lower():
                found = True
                break
        if found:
            correct += 1
    rate = correct / len(questions) if questions else 0
    if label:
        print(f"    {label}: {correct}/{len(questions)} = {rate*100:.1f}%")
    return rate


def load_model(model_id, quant_mode="fp16"):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if quant_mode == "4bit":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4")
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
    return model, tokenizer


def _save(results):
    path = os.path.join(RESULTS_DIR, "phase64_cross_task_sr_quant.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 64: Cross-Task SR-Quantization',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)
    print(f"  Saved: {path}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 64: Cross-Task SR-Quantization")
    print("  Does quantization noise = SR generalize beyond Hanoi?")
    print("  Tasks: Math (arithmetic), GSM8K (word problems), Factual QA")
    print("=" * 70)

    # Prepare all task datasets
    math_problems = generate_math_problems(N_PROBLEMS, seed=42)
    gsm8k_problems = load_gsm8k(N_PROBLEMS)
    factual_questions = generate_truthfulqa_mc(N_PROBLEMS)

    print(f"  Math: {len(math_problems)}, GSM8K: {len(gsm8k_problems)}, "
          f"Factual: {len(factual_questions)}")

    all_results = {}
    MODEL_ID = "Qwen/Qwen2.5-1.5B"

    for quant in ["fp16", "4bit"]:
        print(f"\n{'='*70}")
        print(f"  Qwen-1.5B ({quant})")
        print(f"{'='*70}")

        try:
            t0 = time.time()
            model, tokenizer = load_model(MODEL_ID, quant)
            print(f"  Loaded in {time.time()-t0:.1f}s")

            hook = NoiseHook()
            layer_idx = hook.register(model, layer_frac=0.5)
            print(f"  Hook at layer {layer_idx}")

            for K in [1, 11]:
                print(f"\n  --- K={K} ---")

                # Task 1: Arithmetic
                t1 = time.time()
                math_acc = evaluate_nbs(model, tokenizer, hook, math_problems,
                                       K=K, label=f"Math {quant} K={K}")
                all_results[f'math_{quant}_K{K}'] = {
                    'accuracy': math_acc, 'task': 'math',
                    'quant': quant, 'K': K, 'time': time.time()-t1}

                # Task 2: GSM8K
                t1 = time.time()
                gsm_acc = evaluate_nbs(model, tokenizer, hook, gsm8k_problems,
                                      K=K, label=f"GSM8K {quant} K={K}")
                all_results[f'gsm8k_{quant}_K{K}'] = {
                    'accuracy': gsm_acc, 'task': 'gsm8k',
                    'quant': quant, 'K': K, 'time': time.time()-t1}

                # Task 3: Factual QA
                t1 = time.time()
                fact_acc = evaluate_factual(model, tokenizer, hook,
                                          factual_questions, K=K,
                                          label=f"Factual {quant} K={K}")
                all_results[f'factual_{quant}_K{K}'] = {
                    'accuracy': fact_acc, 'task': 'factual',
                    'quant': quant, 'K': K, 'time': time.time()-t1}

                _save(all_results)

            hook.remove()
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Cross-Task SR-Quantization")
    print(f"{'='*70}")
    print(f"  {'Config':<25s} | {'Math':>6s} | {'GSM8K':>6s} | {'Factual':>8s}")
    print(f"  {'-'*55}")

    for quant in ["fp16", "4bit"]:
        for K in [1, 11]:
            m = all_results.get(f'math_{quant}_K{K}', {}).get('accuracy', 0)
            g = all_results.get(f'gsm8k_{quant}_K{K}', {}).get('accuracy', 0)
            f_ = all_results.get(f'factual_{quant}_K{K}', {}).get('accuracy', 0)
            print(f"  {quant} K={K:<3d}              | {m*100:5.1f}% | {g*100:5.1f}% | {f_*100:6.1f}%")

    # Figure
    _generate_figure(all_results)
    print("\nPhase 64 complete!")


def _generate_figure(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        tasks = [('math', 'Arithmetic'), ('gsm8k', 'GSM8K'), ('factual', 'Factual QA')]
        colors = {'fp16': '#2563EB', '4bit': '#EA580C'}

        for ax, (task_key, task_name) in zip(axes, tasks):
            for quant in ['fp16', '4bit']:
                accs = []
                ks = []
                for K in [1, 11]:
                    key = f'{task_key}_{quant}_K{K}'
                    if key in results:
                        accs.append(results[key]['accuracy'] * 100)
                        ks.append(K)
                if accs:
                    marker = 'o' if quant == 'fp16' else '^'
                    ax.plot(ks, accs, f'{marker}-', color=colors[quant],
                           linewidth=2.5, markersize=10,
                           label=f'Qwen-1.5B {quant}')
                    for k, a in zip(ks, accs):
                        ax.annotate(f'{a:.1f}%', (k, a), textcoords="offset points",
                                   xytext=(0, 10), ha='center', fontsize=9,
                                   fontweight='bold', color=colors[quant])

            ax.set_xlabel('Beam Count K', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title(f'{task_name}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xticks([1, 11])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.suptitle('Phase 64: Cross-Task SR-Quantization\n'
                    '4-bit quantization noise as stochastic resonance across task domains',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "phase64_cross_task_sr_quant.png")
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved: {path}")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
