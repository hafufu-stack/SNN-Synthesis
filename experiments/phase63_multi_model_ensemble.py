"""
Phase 63: Multi-Model Beam Ensemble NBS

Test if mixing beams from different model architectures (Mistral + Qwen)
provides more diversity than single-model NBS.

Experiment: K=11 beams allocated as:
  - Mistral-7B x 11 (single model)
  - Qwen-7B x 11 (single model -- using Qwen-1.5B as proxy)
  - Mistral-7B x 6 + Qwen-1.5B x 5 (mixed ensemble)
  - Llama-1B x 4 + Qwen-1.5B x 4 + Llama-3B x 3 (triple mix)

Task: Math problems, N=30

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
N_PROBLEMS = 30

DIVERSE_SIGMAS = [
    0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5,
]


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


def run_beams_for_problem(model, tokenizer, hook, prompt, answer, sigmas):
    """Run K beams for a single problem. Return True if any beam solves it."""
    for sigma in sigmas:
        response = solve_problem(model, tokenizer, prompt, hook, sigma)
        pred = extract_number(response)
        if pred is not None and pred == answer:
            return True
    return False


def load_model(model_id, quant_4bit=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    if quant_4bit:
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
    return model, tokenizer


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 63: Multi-Model Beam Ensemble NBS")
    print("  Does mixing model architectures in NBS beams add diversity?")
    print("=" * 70)

    problems = generate_math_problems(N_PROBLEMS, seed=42)
    print(f"  Test set: {len(problems)} math problems\n")

    all_results = {}

    # ============================================================
    # Config 1: Mistral-7B x 11
    # ============================================================
    print("=" * 70)
    print("  Config 1: Mistral-7B x 11 beams")
    print("=" * 70)

    try:
        model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.3", quant_4bit=True)
        hook = NoiseHook()
        hook.register(model, layer_frac=0.5)

        correct = 0
        for prob in problems:
            if run_beams_for_problem(model, tokenizer, hook, prob['prompt'],
                                     prob['answer'], DIVERSE_SIGMAS[:11]):
                correct += 1
        acc1 = correct / len(problems)
        all_results['mistral_x11'] = {'accuracy': acc1, 'config': 'Mistral-7B x 11'}
        print(f"  Result: {correct}/{len(problems)} = {acc1*100:.1f}%")

        # Store per-problem results for this model
        mistral_results = {}
        for i, prob in enumerate(problems):
            for sigma in DIVERSE_SIGMAS[:11]:
                response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
                pred = extract_number(response)
                if pred is not None and pred == prob['answer']:
                    mistral_results[i] = True
                    break
            else:
                mistral_results[i] = False

        hook.remove()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
        mistral_results = {}

    _save(all_results, problems)

    # ============================================================
    # Config 2: Qwen-1.5B x 11
    # ============================================================
    print(f"\n{'='*70}")
    print("  Config 2: Qwen-1.5B x 11 beams")
    print(f"{'='*70}")

    try:
        model, tokenizer = load_model("Qwen/Qwen2.5-1.5B", quant_4bit=False)
        hook = NoiseHook()
        hook.register(model, layer_frac=0.5)

        correct = 0
        qwen_results = {}
        for i, prob in enumerate(problems):
            found = False
            for sigma in DIVERSE_SIGMAS[:11]:
                response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
                pred = extract_number(response)
                if pred is not None and pred == prob['answer']:
                    found = True
                    break
            qwen_results[i] = found
            if found:
                correct += 1
        acc2 = correct / len(problems)
        all_results['qwen_x11'] = {'accuracy': acc2, 'config': 'Qwen-1.5B x 11'}
        print(f"  Result: {correct}/{len(problems)} = {acc2*100:.1f}%")

        hook.remove()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
        qwen_results = {}

    _save(all_results, problems)

    # ============================================================
    # Config 3: Mistral-7B x 6 + Qwen-1.5B x 5 (sequential loading)
    # ============================================================
    print(f"\n{'='*70}")
    print("  Config 3: Mistral-7B x 6 + Qwen-1.5B x 5 (Mixed Ensemble)")
    print(f"{'='*70}")

    # Phase A: Run 6 Mistral beams per problem, record which are unsolved
    try:
        model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.3", quant_4bit=True)
        hook = NoiseHook()
        hook.register(model, layer_frac=0.5)

        mix_solved = {}
        for i, prob in enumerate(problems):
            found = False
            for sigma in DIVERSE_SIGMAS[:6]:  # 6 beams from Mistral
                response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
                pred = extract_number(response)
                if pred is not None and pred == prob['answer']:
                    found = True
                    break
            mix_solved[i] = found

        hook.remove()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  After Mistral x6: {sum(mix_solved.values())}/{len(problems)} solved")

    except Exception as e:
        print(f"  ERROR in Mistral phase: {e}")
        mix_solved = {i: False for i in range(len(problems))}

    # Phase B: Run 5 Qwen beams on UNSOLVED problems only
    try:
        model, tokenizer = load_model("Qwen/Qwen2.5-1.5B", quant_4bit=False)
        hook = NoiseHook()
        hook.register(model, layer_frac=0.5)

        for i, prob in enumerate(problems):
            if mix_solved[i]:
                continue  # already solved by Mistral
            for sigma in DIVERSE_SIGMAS[6:11]:  # 5 beams from Qwen (different sigmas)
                response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
                pred = extract_number(response)
                if pred is not None and pred == prob['answer']:
                    mix_solved[i] = True
                    break

        hook.remove()
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR in Qwen phase: {e}")

    acc3 = sum(mix_solved.values()) / len(problems)
    all_results['mixed_mistral6_qwen5'] = {
        'accuracy': acc3, 'config': 'Mistral-7B x6 + Qwen-1.5B x5'}
    print(f"  Final: {sum(mix_solved.values())}/{len(problems)} = {acc3*100:.1f}%")

    _save(all_results, problems)

    # ============================================================
    # Config 4: Llama-1B x 4 + Qwen-1.5B x 4 + Llama-3B x 3 (Triple Mix)
    # ============================================================
    print(f"\n{'='*70}")
    print("  Config 4: Llama-1B x4 + Qwen-1.5B x4 + Llama-3B x3 (Triple Mix)")
    print(f"{'='*70}")

    triple_solved = {i: False for i in range(len(problems))}

    model_configs = [
        ("meta-llama/Llama-3.2-1B", False, DIVERSE_SIGMAS[:4], "Llama-1B x4"),
        ("Qwen/Qwen2.5-1.5B", False, DIVERSE_SIGMAS[4:8], "Qwen-1.5B x4"),
        ("meta-llama/Llama-3.2-3B", False, DIVERSE_SIGMAS[8:11], "Llama-3B x3"),
    ]

    for model_id, use_4bit, sigmas, config_label in model_configs:
        try:
            model, tokenizer = load_model(model_id, quant_4bit=use_4bit)
            hook = NoiseHook()
            hook.register(model, layer_frac=0.5)

            for i, prob in enumerate(problems):
                if triple_solved[i]:
                    continue
                for sigma in sigmas:
                    response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
                    pred = extract_number(response)
                    if pred is not None and pred == prob['answer']:
                        triple_solved[i] = True
                        break

            hook.remove()
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  After {config_label}: {sum(triple_solved.values())}/{len(problems)} solved")

        except Exception as e:
            print(f"  ERROR with {config_label}: {e}")

    acc4 = sum(triple_solved.values()) / len(problems)
    all_results['triple_mix'] = {
        'accuracy': acc4, 'config': 'Llama-1B x4 + Qwen-1.5B x4 + Llama-3B x3'}
    print(f"  Final: {sum(triple_solved.values())}/{len(problems)} = {acc4*100:.1f}%")

    _save(all_results, problems)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Multi-Model Ensemble NBS")
    print(f"{'='*70}")
    for key in ['mistral_x11', 'qwen_x11', 'mixed_mistral6_qwen5', 'triple_mix']:
        if key in all_results:
            r = all_results[key]
            print(f"  {r['config']:>40s}: {r['accuracy']*100:5.1f}%")

    # Diversity analysis
    if mistral_results and qwen_results:
        only_mistral = sum(1 for i in range(len(problems))
                          if mistral_results.get(i, False) and not qwen_results.get(i, False))
        only_qwen = sum(1 for i in range(len(problems))
                       if qwen_results.get(i, False) and not mistral_results.get(i, False))
        both = sum(1 for i in range(len(problems))
                  if mistral_results.get(i, False) and qwen_results.get(i, False))
        neither = sum(1 for i in range(len(problems))
                     if not mistral_results.get(i, False) and not qwen_results.get(i, False))
        print(f"\n  Diversity Analysis:")
        print(f"    Only Mistral solves: {only_mistral}")
        print(f"    Only Qwen solves:    {only_qwen}")
        print(f"    Both solve:          {both}")
        print(f"    Neither solves:      {neither}")
        print(f"    Orthogonality = {only_mistral + only_qwen}/{only_mistral + only_qwen + both}")
        all_results['diversity'] = {
            'only_mistral': only_mistral, 'only_qwen': only_qwen,
            'both': both, 'neither': neither,
        }

    _generate_figure(all_results)
    _save(all_results, problems)
    print("\nPhase 63 complete!")


def _save(all_results, problems):
    save_path = os.path.join(RESULTS_DIR, "phase63_multi_model_ensemble.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 63: Multi-Model Beam Ensemble NBS',
            'timestamp': datetime.now().isoformat(),
            'n_problems': len(problems),
            'results': all_results,
        }, f, indent=2, default=str)


def _generate_figure(all_results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: bar chart of all 4 configs
        configs = ['mistral_x11', 'qwen_x11', 'mixed_mistral6_qwen5', 'triple_mix']
        labels = ['Mistral-7B\nx11', 'Qwen-1.5B\nx11', 'Mistral x6\n+ Qwen x5', 'Llama-1B x4\n+ Qwen x4\n+ Llama-3B x3']
        colors = ['#2563EB', '#DC2626', '#16A34A', '#9333EA']

        accs = []
        for c in configs:
            if c in all_results and 'accuracy' in all_results[c]:
                accs.append(all_results[c]['accuracy'] * 100)
            else:
                accs.append(0)

        bars = ax1.bar(range(len(labels)), accs, color=colors,
                      edgecolor='white', linewidth=0.5, alpha=0.85)

        for i, acc in enumerate(accs):
            ax1.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color=colors[i])

        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('(a) K=11 NBS: Single vs Mixed Model', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, max(accs) * 1.2 if accs else 100)
        ax1.grid(axis='y', alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Right: Venn-style diversity diagram (simplified as stacked bar)
        if 'diversity' in all_results:
            d = all_results['diversity']
            categories = ['Only\nMistral', 'Both', 'Only\nQwen', 'Neither']
            values = [d['only_mistral'], d['both'], d['only_qwen'], d['neither']]
            div_colors = ['#2563EB', '#16A34A', '#DC2626', '#6B7280']

            ax2.bar(range(len(categories)), values, color=div_colors,
                   edgecolor='white', linewidth=0.5, alpha=0.85)
            for i, v in enumerate(values):
                ax2.text(i, v + 0.3, str(v), ha='center', va='bottom',
                        fontsize=14, fontweight='bold', color=div_colors[i])
            ax2.set_xticks(range(len(categories)))
            ax2.set_xticklabels(categories, fontsize=10)
            ax2.set_ylabel('Problem Count', fontsize=12)
            ax2.set_title('(b) Model Diversity Analysis\n(Mistral vs Qwen K=11)',
                         fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase63_multi_model_ensemble.png")
        fig.savefig(fig_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved: {fig_path}")

    except Exception as e:
        print(f"  Figure generation error: {e}")


if __name__ == '__main__':
    main()
