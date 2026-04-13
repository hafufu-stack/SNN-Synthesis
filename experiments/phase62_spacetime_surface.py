"""
Phase 62: Space-Time Equivalence Surface

Generate a 3D surface mapping:
  X = Model parameters (spatial resolution)
  Y = Beam count K (temporal resolution)  
  Z = Accuracy

From this surface, derive the "exchange rate":
  "1 Billion parameters = K=? beams of NBS"

Experiment: 4 models x 5 K values x N=30 math problems

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

# Models (smallest to largest) - all locally cached
MODELS = [
    ("meta-llama/Llama-3.2-1B", "Llama-1B", 1.0),
    ("Qwen/Qwen2.5-1.5B", "Qwen-1.5B", 1.5),
    ("meta-llama/Llama-3.2-3B", "Llama-3B", 3.0),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B", 7.0),
]

K_VALUES = [1, 5, 11, 21, 51]
N_PROBLEMS = 30

# sigma values for diverse NBS (K=51 max)
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


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Phase 62: Space-Time Equivalence Surface")
    print("  Mapping the 3D surface: Params(B) x K(beams) x Accuracy(%)")
    print("  Deriving: '1 Billion params = K=? beams'")
    print("=" * 70)

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    problems = generate_math_problems(N_PROBLEMS, seed=42)
    print(f"  Test set: {len(problems)} math problems\n")

    # Results matrix: model_label -> K -> accuracy
    surface_data = {}
    all_results = {}

    for model_id, model_label, param_b in MODELS:
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_label} ({param_b}B) - {model_id}")
        print(f"{'='*70}")

        try:
            t0 = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

            # Use 4-bit for all models (fair comparison under compute constraint)
            if param_b >= 3.0:
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

            hook = NoiseHook()
            layer_idx = hook.register(model, layer_frac=0.5)
            print(f"  Hook at layer {layer_idx}")

            model_surface = {}

            for K in K_VALUES:
                t0 = time.time()
                acc = evaluate_nbs(
                    model, tokenizer, hook, problems, K=K,
                    label=f"{model_label} K={K}")
                elapsed = time.time() - t0

                model_surface[K] = acc
                all_results[f"{model_label}_K{K}"] = {
                    'accuracy': acc,
                    'model': model_label,
                    'params_B': param_b,
                    'K': K,
                    'time_sec': elapsed,
                }

                # Save incrementally
                _save(all_results, surface_data, problems)

            surface_data[model_label] = {
                'params_B': param_b,
                'K_accuracy': model_surface,
            }
            _save(all_results, surface_data, problems)

            hook.remove()
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR loading {model_label}: {e}")
            import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            continue

    # ============================================================
    # Analysis: Derive exchange rate
    # ============================================================
    print(f"\n{'='*70}")
    print("SPACE-TIME EXCHANGE RATE ANALYSIS")
    print(f"{'='*70}")

    # For each model pair, find K that equalizes them
    exchange_rates = []
    labels_sorted = [l for _, l, _ in MODELS if l in surface_data]

    for i in range(len(labels_sorted)):
        for j in range(i+1, len(labels_sorted)):
            small = labels_sorted[i]
            large = labels_sorted[j]
            small_p = surface_data[small]['params_B']
            large_p = surface_data[large]['params_B']
            large_k1 = surface_data[large]['K_accuracy'].get(1, 0)

            # Find smallest K where small model beats large@K=1
            beat_k = None
            for K in K_VALUES:
                small_acc = surface_data[small]['K_accuracy'].get(K, 0)
                if small_acc >= large_k1 and large_k1 > 0:
                    beat_k = K
                    break

            param_diff = large_p - small_p
            print(f"  {small} ({small_p}B) vs {large} ({large_p}B, K=1={large_k1*100:.1f}%):")
            if beat_k:
                print(f"    -> {small} beats {large}@K=1 at K={beat_k}")
                print(f"    -> {param_diff:.1f}B params = K={beat_k} beams")
                rate = param_diff / beat_k if beat_k > 0 else 0
                print(f"    -> Exchange rate: 1B = K={beat_k/param_diff:.1f} beams")
                exchange_rates.append({
                    'small': small, 'large': large,
                    'param_diff_B': param_diff,
                    'K_to_beat': beat_k,
                    'rate_B_per_beam': rate,
                })
            else:
                print(f"    -> {small} NEVER beats {large}@K=1 (K up to {max(K_VALUES)})")

    all_results['exchange_rates'] = exchange_rates

    # ============================================================
    # Generate 3D Surface Figure
    # ============================================================
    _generate_3d_surface(surface_data)
    _save(all_results, surface_data, problems)
    print("\nPhase 62 complete!")


def _save(all_results, surface_data, problems):
    save_path = os.path.join(RESULTS_DIR, "phase62_spacetime_surface.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 62: Space-Time Equivalence Surface',
            'timestamp': datetime.now().isoformat(),
            'n_problems': len(problems),
            'surface_data': surface_data,
            'results': all_results,
        }, f, indent=2, default=str)


def _generate_3d_surface(surface_data):
    """Generate 3D surface and 2D contour figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(16, 6))

        # Left: 3D surface
        ax1 = fig.add_subplot(121, projection='3d')

        params_list = []
        K_list = []
        acc_list = []
        colors_map = {
            'Llama-1B': '#9333EA', 'Qwen-1.5B': '#DC2626',
            'Llama-3B': '#EA580C', 'Mistral-7B': '#2563EB',
        }

        for label, data in surface_data.items():
            p = data['params_B']
            for K, acc in data['K_accuracy'].items():
                params_list.append(p)
                K_list.append(int(K))
                acc_list.append(acc * 100)
                ax1.scatter(p, int(K), acc * 100,
                          c=colors_map.get(label, 'gray'),
                          s=80, zorder=5, edgecolors='white', linewidth=0.5)

        # Draw lines connecting same model across K
        for label, data in surface_data.items():
            p = data['params_B']
            ks = sorted([int(k) for k in data['K_accuracy'].keys()])
            accs = [data['K_accuracy'][k] * 100 for k in ks]
            ps = [p] * len(ks)
            ax1.plot(ps, ks, accs, '-', color=colors_map.get(label, 'gray'),
                    linewidth=2, label=label)

        ax1.set_xlabel('Parameters (B)', fontsize=10)
        ax1.set_ylabel('Beam Count K', fontsize=10)
        ax1.set_zlabel('Accuracy (%)', fontsize=10)
        ax1.set_title('Space-Time Equivalence\nSurface', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='upper left')

        # Right: 2D iso-accuracy contour (heatmap approximation)
        ax2 = fig.add_subplot(122)

        # Create a grid for plotting
        for label, data in surface_data.items():
            p = data['params_B']
            ks = sorted([int(k) for k in data['K_accuracy'].keys()])
            accs = [data['K_accuracy'][k] * 100 for k in ks]
            ax2.plot(ks, accs, 'o-', color=colors_map.get(label, 'gray'),
                    linewidth=2.5, markersize=9, label=f'{label} ({p}B)')

        # Mark exchange rates
        ax2.set_xlabel('Beam Count K', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy vs K by Model Size\n(Iso-accuracy = Exchange Rate)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.set_xscale('log', base=2)
        ax2.grid(alpha=0.3)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, "phase62_spacetime_surface.png")
        fig.savefig(fig_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Figure saved: {fig_path}")

    except Exception as e:
        print(f"  Figure generation error: {e}")
        import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
