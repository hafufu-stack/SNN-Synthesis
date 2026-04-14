"""
Phase 66: Heterogeneous Swarm Pareto -- Ensemble Ratio Law

Extends Phase 63 by sweeping the Mistral:Qwen ratio.
K=11 total beams, varying allocation:
  10:1, 8:3, 6:5 (v8 result), 3:8, 1:10

Tests: "Is there a golden ratio for mixing strong+weak models?"

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

DIVERSE_SIGMAS = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]


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


def load_model(model_id, quant_4bit=True):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if quant_4bit:
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


def collect_beam_results(model, tokenizer, hook, problems, n_beams):
    """Run n_beams sigma-diverse beams for each problem.
    Returns: list of sets -- per_problem_solved[i] = True if any beam solved problem i.
    Also returns per_beam: list of dicts {problem_idx: bool}
    """
    sigmas = DIVERSE_SIGMAS[:n_beams]
    # per_problem[i] = set of beam indices that solved problem i
    per_problem = [set() for _ in range(len(problems))]

    for beam_idx, sigma in enumerate(sigmas):
        for prob_idx, prob in enumerate(problems):
            response = solve_problem(model, tokenizer, prob['prompt'], hook, sigma)
            pred = extract_number(response)
            if pred is not None and pred == prob['answer']:
                per_problem[prob_idx].add(beam_idx)

    return per_problem


def _save(results):
    path = os.path.join(RESULTS_DIR, "phase66_ensemble_ratio.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 66: Heterogeneous Swarm Pareto',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 66: Heterogeneous Swarm Pareto")
    print("  What is the optimal Mistral:Qwen mixing ratio?")
    print("=" * 70)

    problems = generate_math_problems(N_PROBLEMS, seed=42)
    print(f"  Test set: {len(problems)} math problems\n")

    # Ratio configs: (mistral_beams, qwen_beams)
    RATIOS = [(10, 1), (8, 3), (6, 5), (3, 8), (1, 10)]
    all_results = {}

    # Step 1: Run all 11 beams with Mistral-7B
    print("=" * 70)
    print("  Step 1: Mistral-7B -- collecting all 11 beam results")
    print("=" * 70)
    try:
        model, tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.3", quant_4bit=True)
        hook = NoiseHook()
        hook.register(model, layer_frac=0.5)

        mistral_per_problem = collect_beam_results(model, tokenizer, hook, problems, 11)
        # Also compute Mistral-only accuracy
        m_correct = sum(1 for s in mistral_per_problem if len(s) > 0)
        all_results['mistral_x11'] = {
            'accuracy': m_correct / len(problems),
            'config': 'Mistral-7B x 11'}
        print(f"  Mistral x11: {m_correct}/{len(problems)} = {m_correct/len(problems)*100:.1f}%")

        hook.remove()
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        mistral_per_problem = [set() for _ in range(len(problems))]

    _save(all_results)

    # Step 2: Run all 11 beams with Qwen-1.5B
    print(f"\n{'='*70}")
    print("  Step 2: Qwen-1.5B -- collecting all 11 beam results")
    print(f"{'='*70}")
    try:
        model, tokenizer = load_model("Qwen/Qwen2.5-1.5B", quant_4bit=True)
        hook = NoiseHook()
        hook.register(model, layer_frac=0.5)

        qwen_per_problem = collect_beam_results(model, tokenizer, hook, problems, 11)
        q_correct = sum(1 for s in qwen_per_problem if len(s) > 0)
        all_results['qwen_x11'] = {
            'accuracy': q_correct / len(problems),
            'config': 'Qwen-1.5B x 11'}
        print(f"  Qwen x11: {q_correct}/{len(problems)} = {q_correct/len(problems)*100:.1f}%")

        hook.remove()
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
        qwen_per_problem = [set() for _ in range(len(problems))]

    _save(all_results)

    # Step 3: Compute mixed ensemble accuracy for each ratio
    print(f"\n{'='*70}")
    print("  Step 3: Computing mixed ensemble results")
    print(f"{'='*70}")

    for n_mistral, n_qwen in RATIOS:
        # Mistral uses first n_mistral beams, Qwen uses first n_qwen beams
        correct = 0
        for prob_idx in range(len(problems)):
            # Check if any of the allocated Mistral beams solved it
            mistral_solved = any(b < n_mistral for b in mistral_per_problem[prob_idx])
            # Check if any of the allocated Qwen beams solved it
            qwen_solved = any(b < n_qwen for b in qwen_per_problem[prob_idx])
            if mistral_solved or qwen_solved:
                correct += 1

        acc = correct / len(problems)
        key = f"mix_{n_mistral}m_{n_qwen}q"
        all_results[key] = {
            'accuracy': acc,
            'config': f'Mistral×{n_mistral} + Qwen×{n_qwen}',
            'n_mistral': n_mistral,
            'n_qwen': n_qwen,
        }
        print(f"  Mistral×{n_mistral} + Qwen×{n_qwen}: "
              f"{correct}/{len(problems)} = {acc*100:.1f}%")

    _save(all_results)

    # Summary
    print(f"\n{'='*70}")
    print("GRAND SUMMARY: Ensemble Ratio Law")
    print(f"{'='*70}")
    print(f"  {'Config':<30s} | {'Accuracy':>8s}")
    print(f"  {'-'*42}")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"  {r['config']:<30s} | {r['accuracy']*100:6.1f}%")

    _generate_figure(all_results, RATIOS)
    print("\nPhase 66 complete!")


def _generate_figure(results, ratios):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Mixed ensemble curve
        qwen_fracs = []
        accs = []
        for n_m, n_q in ratios:
            key = f"mix_{n_m}m_{n_q}q"
            if key in results:
                qwen_fracs.append(n_q / 11 * 100)
                accs.append(results[key]['accuracy'] * 100)

        ax.plot(qwen_fracs, accs, 'o-', color='#8B5CF6', linewidth=3,
               markersize=12, label='Mixed Ensemble', zorder=5)

        for qf, a, (n_m, n_q) in zip(qwen_fracs, accs, ratios):
            ax.annotate(f'{a:.1f}%\n({n_m}:{n_q})', (qf, a),
                       textcoords="offset points", xytext=(0, 15),
                       ha='center', fontsize=9, fontweight='bold')

        # Horizontal lines for single-model baselines
        m11 = results.get('mistral_x11', {}).get('accuracy', 0) * 100
        q11 = results.get('qwen_x11', {}).get('accuracy', 0) * 100
        ax.axhline(y=m11, color='#EF4444', linestyle='--', linewidth=2,
                  label=f'Mistral×11 ({m11:.1f}%)')
        ax.axhline(y=q11, color='#3B82F6', linestyle='--', linewidth=2,
                  label=f'Qwen×11 ({q11:.1f}%)')

        ax.set_xlabel('Qwen Beam Fraction (%)', fontsize=13)
        ax.set_ylabel('Accuracy (%)', fontsize=13)
        ax.set_title('Phase 66: Heterogeneous Swarm Pareto\n'
                    'Optimal Mistral:Qwen Mixing Ratio (K=11)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "phase66_ensemble_ratio.png")
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved: {path}")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
