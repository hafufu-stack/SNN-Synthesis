"""
Phase 68: API-Only NBS (Temperature-Diverse Beam Search Without Hook Access)

Validates that temperature diversity alone provides consistent SR gains
across multiple tasks and models, simulating API-only deployment where
hidden state access is impossible.

Models: Qwen-1.5B 4-bit, Mistral-7B 4-bit
Tasks: Arithmetic (N=100), Modified Hanoi (N=50)
Conditions: T in {0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0}, K=11

If temperature diversity alone consistently provides +20pp, NBS is
universally deployable without model access (Future Work #9).

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
N_MATH = 100
N_HANOI = 50
K = 11

DIVERSE_TEMPS = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]


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


def solve(model, tokenizer, prompt, temperature=1.0, do_sample=True):
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


def generate_hanoi_problems(n, seed=42):
    """Simple Modified Hanoi: move disc A->C via B."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        discs = rng.randint(2, 4)
        problems.append({
            'prompt': (f"Solve the Tower of Hanoi puzzle with {discs} discs. "
                      f"Move all discs from peg A to peg C using peg B as auxiliary. "
                      f"Rules: only one disc at a time, never place larger on smaller. "
                      f"List the moves as 'Move disc X from Y to Z'. Give the optimal solution."),
            'n_discs': discs,
            'min_moves': 2**discs - 1,
        })
    return problems


def check_hanoi(response, n_discs):
    """Check if response contains the correct number of moves."""
    moves = re.findall(r'[Mm]ove\s+disc', response)
    min_moves = 2**n_discs - 1
    # Accept if number of moves matches optimal
    return len(moves) == min_moves


def evaluate_api_only(model, tokenizer, problems, task_type, label=""):
    """Evaluate temperature-diverse NBS (no hook) vs greedy."""
    results = {'greedy': 0, 'temp_diverse': 0}
    
    for prob in problems:
        # Greedy: K=1, no noise
        resp = solve(model, tokenizer, prob['prompt'], do_sample=False)
        if task_type == 'math':
            pred = extract_number(resp)
            if pred is not None and pred == prob['answer']:
                results['greedy'] += 1
        else:
            if check_hanoi(resp, prob['n_discs']):
                results['greedy'] += 1

        # Temperature-diverse: K=11
        found = False
        for t in DIVERSE_TEMPS[:K]:
            resp = solve(model, tokenizer, prob['prompt'], temperature=t, do_sample=True)
            if task_type == 'math':
                pred = extract_number(resp)
                if pred is not None and pred == prob['answer']:
                    found = True; break
            else:
                if check_hanoi(resp, prob['n_discs']):
                    found = True; break
        if found:
            results['temp_diverse'] += 1

    n = len(problems)
    greedy_acc = results['greedy'] / n
    diverse_acc = results['temp_diverse'] / n
    delta = (diverse_acc - greedy_acc) * 100
    
    print(f"    {label}")
    print(f"      Greedy:        {results['greedy']}/{n} = {greedy_acc*100:.1f}%")
    print(f"      T-Diverse K=11: {results['temp_diverse']}/{n} = {diverse_acc*100:.1f}% ({delta:+.1f}pp)")
    
    return {'greedy': greedy_acc, 'temp_diverse': diverse_acc, 'delta_pp': delta}


def load_model(model_id, quant_4bit=True):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if quant_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto",
                                                     torch_dtype=torch.float16, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                     torch_dtype=torch.float16, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _save(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "phase68_api_only_nbs.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 68: API-Only NBS',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 68: API-Only NBS (Temperature-Diverse Beam Search)")
    print("  Can temperature diversity alone provide consistent SR gains?")
    print("=" * 70)

    math_probs = generate_math_problems(N_MATH, seed=42)
    hanoi_probs = generate_hanoi_problems(N_HANOI, seed=42)
    all_results = {}

    MODELS = [
        ("Qwen/Qwen2.5-1.5B", "Qwen-1.5B (4-bit)", True),
        ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B (4-bit)", True),
    ]

    for model_id, model_label, use_4bit in MODELS:
        print(f"\n  Loading {model_label}...")
        model, tokenizer = load_model(model_id, quant_4bit=use_4bit)

        # Arithmetic
        print(f"\n  --- {model_label} x Arithmetic (N={N_MATH}) ---")
        math_res = evaluate_api_only(model, tokenizer, math_probs, 'math', label=f"{model_label} / Math")

        # Hanoi
        print(f"\n  --- {model_label} x Hanoi (N={N_HANOI}) ---")
        hanoi_res = evaluate_api_only(model, tokenizer, hanoi_probs, 'hanoi', label=f"{model_label} / Hanoi")

        all_results[model_label] = {'math': math_res, 'hanoi': hanoi_res}
        _save(all_results)

        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: API-Only NBS")
    print(f"{'='*70}")
    for model_label, res in all_results.items():
        for task in ['math', 'hanoi']:
            r = res.get(task, {})
            print(f"  {model_label:30s} {task:8s}: greedy={r.get('greedy',0)*100:.1f}%  "
                  f"T-diverse={r.get('temp_diverse',0)*100:.1f}%  delta={r.get('delta_pp',0):+.1f}pp")

    _generate_figure(all_results)
    print("\nPhase 68 complete!")


def _generate_figure(results):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        models = list(results.keys())
        tasks = ['math', 'hanoi']
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, task in zip(axes, tasks):
            x = np.arange(len(models))
            greedy = [results[m][task]['greedy']*100 for m in models]
            diverse = [results[m][task]['temp_diverse']*100 for m in models]

            bars1 = ax.bar(x - 0.18, greedy, 0.35, label='Greedy (K=1)', color='#9CA3AF')
            bars2 = ax.bar(x + 0.18, diverse, 0.35, label='T-Diverse (K=11)', color='#F59E0B')

            for b in bars1:
                ax.text(b.get_x() + b.get_width()/2, b.get_height()+1, f'{b.get_height():.1f}%',
                       ha='center', fontsize=9, fontweight='bold')
            for b in bars2:
                ax.text(b.get_x() + b.get_width()/2, b.get_height()+1, f'{b.get_height():.1f}%',
                       ha='center', fontsize=9, fontweight='bold')

            ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'API-Only NBS: {task.capitalize()}', fontweight='bold')
            ax.legend(); ax.grid(axis='y', alpha=0.3)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        fig.suptitle('Phase 68: Temperature-Diverse NBS (No Hook Access)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURES_DIR, "phase68_api_only_nbs.png"), bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
