"""
Phase 65: Weight Pruning + NBS Resurrection

Tests how much spatial (parameter) destruction NBS can compensate for.
Instead of 1-bit quantization (which destroys models completely),
uses progressive weight pruning at 50%, 70%, 90% to find the
"resurrection threshold" — the point where NBS can no longer recover.

Experiment matrix:
  - Qwen-1.5B FP16 at pruning {0%, 50%, 70%, 90%} × K={1, 11, 21}
  - Task: Math problems, N=30

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

DIVERSE_SIGMAS = [
    0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
]


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


def prune_model(model, prune_fraction):
    """Prune the smallest weights globally by setting them to zero.
    Uses numpy with sampling to avoid torch.quantile size limits."""
    if prune_fraction <= 0:
        return 0

    all_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() >= 2:
            all_params.append(param)

    if not all_params:
        print("    WARNING: No prunable parameters found")
        return 0

    # Sample magnitudes to estimate threshold (avoids OOM and size limits)
    sample_size = 1_000_000  # 1M samples is enough for accurate percentile
    samples = []
    total_elements = sum(p.numel() for p in all_params)
    for param in all_params:
        # Sample proportionally from each parameter
        n_sample = max(1, int(sample_size * param.numel() / total_elements))
        flat = param.data.abs().flatten().cpu().float()
        if len(flat) > n_sample:
            indices = torch.randperm(len(flat))[:n_sample]
            samples.append(flat[indices])
        else:
            samples.append(flat)

    all_samples = torch.cat(samples).numpy()
    threshold = float(np.percentile(all_samples, prune_fraction * 100))
    del all_samples, samples

    total_pruned = 0
    total_params_count = 0
    for param in all_params:
        mask = param.data.abs() <= threshold
        param.data[mask] = 0.0
        total_pruned += mask.sum().item()
        total_params_count += param.numel()

    actual_sparsity = total_pruned / total_params_count if total_params_count > 0 else 0
    print(f"    Pruned {total_pruned:,}/{total_params_count:,} = {actual_sparsity*100:.1f}% sparsity")
    gc.collect()
    return actual_sparsity


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


def evaluate_nbs(model, tokenizer, hook, problems, K, label=""):
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


def _save(results):
    path = os.path.join(RESULTS_DIR, "phase65_weight_pruning.json")
    with open(path, 'w') as f:
        json.dump({
            'experiment': 'Phase 65: Weight Pruning + NBS Resurrection',
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2, default=str)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 65: Weight Pruning + NBS Resurrection")
    print("  How much parameter destruction can NBS compensate for?")
    print("=" * 70)

    problems = generate_math_problems(N_PROBLEMS, seed=42)
    print(f"  Test set: {len(problems)} math problems\n")

    all_results = {}
    MODEL_ID = "Qwen/Qwen2.5-1.5B"
    PRUNE_LEVELS = [0.0, 0.5, 0.7, 0.9]
    K_VALUES = [1, 11, 21]

    for prune_frac in PRUNE_LEVELS:
        print(f"\n{'='*70}")
        print(f"  Qwen-1.5B FP16, pruning={prune_frac*100:.0f}%")
        print(f"{'='*70}")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, device_map="auto", torch_dtype=torch.float16,
                local_files_only=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            print(f"  Loaded in {time.time()-t0:.1f}s")

            # Apply pruning
            actual_sparsity = prune_model(model, prune_frac)

            hook = NoiseHook()
            layer_idx = hook.register(model, layer_frac=0.5)
            print(f"  Hook at layer {layer_idx}")

            for K in K_VALUES:
                t1 = time.time()
                acc = evaluate_nbs(model, tokenizer, hook, problems, K=K,
                                  label=f"Prune={prune_frac*100:.0f}% K={K}")
                elapsed = time.time() - t1

                key = f"prune{int(prune_frac*100)}_K{K}"
                all_results[key] = {
                    'accuracy': acc,
                    'prune_fraction': prune_frac,
                    'actual_sparsity': actual_sparsity,
                    'K': K,
                    'time_sec': elapsed,
                }
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
    print("GRAND SUMMARY: Weight Pruning + NBS")
    print(f"{'='*70}")
    baseline = all_results.get('prune0_K1', {}).get('accuracy', 0)
    print(f"  Baseline (0% pruning, K=1): {baseline*100:.1f}%\n")
    print(f"  {'Pruning':<10s} | {'K=1':>6s} | {'K=11':>6s} | {'K=21':>6s}")
    print(f"  {'-'*40}")
    for p in [0, 50, 70, 90]:
        k1 = all_results.get(f'prune{p}_K1', {}).get('accuracy', 0)
        k11 = all_results.get(f'prune{p}_K11', {}).get('accuracy', 0)
        k21 = all_results.get(f'prune{p}_K21', {}).get('accuracy', 0)
        print(f"  {p}%{'':<8s} | {k1*100:5.1f}% | {k11*100:5.1f}% | {k21*100:5.1f}%")

    _generate_figure(all_results, baseline)
    print("\nPhase 65 complete!")


def _generate_figure(results, baseline):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        prune_levels = [0, 50, 70, 90]
        K_VALUES = [1, 11, 21]
        colors = {1: '#EF4444', 11: '#3B82F6', 21: '#10B981'}

        # Left: Accuracy vs K at different pruning levels
        for p in prune_levels:
            accs = []
            ks = []
            for K in K_VALUES:
                key = f'prune{p}_K{K}'
                if key in results:
                    accs.append(results[key]['accuracy'] * 100)
                    ks.append(K)
            if accs:
                ax1.plot(ks, accs, 'o-', linewidth=2.5, markersize=9,
                        label=f'{p}% pruned')

        ax1.set_xlabel('Beam Count K', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('(a) NBS Recovery at Different Pruning Levels',
                      fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Right: Accuracy vs Pruning at different K values
        for K in K_VALUES:
            accs = []
            prunes = []
            for p in prune_levels:
                key = f'prune{p}_K{K}'
                if key in results:
                    accs.append(results[key]['accuracy'] * 100)
                    prunes.append(p)
            if accs:
                ax2.plot(prunes, accs, 'o-', color=colors[K],
                        linewidth=2.5, markersize=9, label=f'K={K}')

        ax2.axhline(y=baseline*100, color='gray', linestyle=':', linewidth=1.5,
                   label='Baseline (0% K=1)')
        ax2.set_xlabel('Weight Pruning (%)', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('(b) Resurrection Threshold:\nHow Much Destruction Can NBS Fix?',
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        fig.suptitle('Phase 65: Weight Pruning + NBS Resurrection',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "phase65_weight_pruning.png")
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved: {path}")
    except Exception as e:
        print(f"  Figure error: {e}")


if __name__ == '__main__':
    main()
