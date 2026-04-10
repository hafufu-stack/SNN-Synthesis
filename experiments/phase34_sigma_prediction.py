"""
Phase 34: σ* Prediction Model — Task-Dependent Noise Theory
=============================================================
Establish a quantitative relationship between σ* and task complexity.

Known data points:
  ARC-AGI (grid nav):   σ*=0.20, short discrete actions
  Modified Hanoi:       σ*=0.15, medium discrete reasoning
  GSM8K (math):         σ*=0.01, long chain-of-thought text

New: TruthfulQA MC1 — between Hanoi and GSM8K in complexity
Grid search: σ × K=11, N=100
"""
import torch
import os, json, time, re, gc
os.environ['HF_DATASETS_OFFLINE'] = '1'
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
NOISE_LAYER = 18
SIGMA_VALUES = [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
K = 11
N_QUESTIONS = 100
MAX_NEW_TOKENS = 256
DEVICE = "cuda"

# ============================================================
# Noise Hook
# ============================================================
class NoiseHook:
    def __init__(self):
        self.sigma = 0.0
        self.handle = None

    def setup(self, sigma):
        self.sigma = sigma

    def register(self, model, layer_idx=NOISE_LAYER):
        hook_obj = self
        def hook_fn(module, args):
            hs = args[0]
            if hook_obj.sigma <= 0:
                return args
            noise = torch.randn_like(hs) * hook_obj.sigma
            return (hs + noise,) + args[1:]
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()

# ============================================================
# Load Model
# ============================================================
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"  Loading {MODEL_NAME} (4-bit)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

# ============================================================
# TruthfulQA MC1
# ============================================================
def solve_truthfulqa(model, tokenizer, question, choices, hook, sigma):
    """Multiple-choice: return index of selected answer."""
    choices_text = "\n".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
    messages = [{"role": "user", "content":
        f"Answer the following question by selecting the correct option. "
        f"Reply with ONLY the letter (A, B, C, etc.).\n\n"
        f"Question: {question}\n{choices_text}\n\nAnswer:"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    hook.setup(sigma)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=32,
            do_sample=(sigma > 0), temperature=0.7 if sigma > 0 else 1.0,
            top_p=0.9 if sigma > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Extract letter
    for letter in re.findall(r'[A-Z]', response):
        idx = ord(letter) - ord('A')
        if 0 <= idx < len(choices):
            return idx
    return -1

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 34: σ* Prediction Model")
    print(f"  σ = {SIGMA_VALUES}")
    print(f"  K = {K}, N = {N_QUESTIONS}")
    print("=" * 60)

    # Load TruthfulQA MC1
    from datasets import load_dataset
    ds = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
    # Filter to MC1 (first N questions)
    questions = []
    for item in ds:
        q = item['question']
        choices = item['mc1_targets']['choices']
        labels = item['mc1_targets']['labels']
        correct_idx = labels.index(1) if 1 in labels else 0
        questions.append({'question': q, 'choices': choices, 'correct_idx': correct_idx})
        if len(questions) >= N_QUESTIONS:
            break

    print(f"  Loaded {len(questions)} TruthfulQA MC1 questions")

    model, tokenizer = load_model()
    hook = NoiseHook()
    hook.register(model)

    results = {
        "experiment": "Phase 34: σ* Prediction Model",
        "model": MODEL_NAME,
        "task": "TruthfulQA MC1",
        "sigma_values": SIGMA_VALUES,
        "K": K,
        "n_questions": N_QUESTIONS,
        "start_time": datetime.now().isoformat(),
        "conditions": {},
        "known_sigma_stars": {
            "ARC-AGI (grid nav)": 0.20,
            "Modified Hanoi": 0.15,
            "GSM8K (math)": 0.01
        }
    }

    # Baseline (σ=0, K=1)
    print("\n--- Baseline (σ=0, K=1) ---")
    correct_baseline = 0
    for i, q in enumerate(questions):
        pred = solve_truthfulqa(model, tokenizer, q['question'], q['choices'], hook, 0.0)
        if pred == q['correct_idx']:
            correct_baseline += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{N_QUESTIONS}] {correct_baseline}/{i+1} ({100*correct_baseline/(i+1):.1f}%)")
    baseline_rate = correct_baseline / N_QUESTIONS
    results["baseline"] = {"correct": correct_baseline, "accuracy": baseline_rate}
    print(f"  Baseline: {correct_baseline}/{N_QUESTIONS} = {100*baseline_rate:.1f}%")

    # Grid search: σ × K=11
    for sigma in SIGMA_VALUES:
        label = f"σ={sigma}_K={K}"
        print(f"\n--- {label} ---")
        correct = 0

        for i, q in enumerate(questions):
            any_correct = False
            for k in range(K):
                pred = solve_truthfulqa(model, tokenizer, q['question'], q['choices'], hook, sigma)
                if pred == q['correct_idx']:
                    any_correct = True
                    break
            if any_correct:
                correct += 1
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{N_QUESTIONS}] {label}: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")

        rate = correct / N_QUESTIONS
        delta = (rate - baseline_rate) * 100
        results["conditions"][label] = {
            "sigma": sigma, "K": K,
            "correct": correct, "accuracy": rate,
            "delta_pp": delta
        }
        print(f"  {label}: {correct}/{N_QUESTIONS} = {100*rate:.1f}% (Δ={delta:+.1f}pp)")

        # Save intermediate
        with open(os.path.join(RESULTS_DIR, "phase34_sigma_prediction.json"), 'w') as f:
            json.dump(results, f, indent=2)

    # Find σ* for TruthfulQA
    best_sigma = max(results["conditions"].items(),
                     key=lambda x: x[1]["accuracy"])
    results["sigma_star_truthfulqa"] = best_sigma[1]["sigma"]
    results["best_accuracy"] = best_sigma[1]["accuracy"]
    results["end_time"] = datetime.now().isoformat()

    # Task complexity analysis
    results["task_complexity"] = {
        "ARC-AGI": {"sigma_star": 0.20, "reasoning_steps": 3, "action_space": "discrete_4",
                    "answer_uniqueness": "deterministic"},
        "Modified_Hanoi": {"sigma_star": 0.15, "reasoning_steps": 7, "action_space": "discrete_6",
                           "answer_uniqueness": "deterministic"},
        "TruthfulQA": {"sigma_star": results["sigma_star_truthfulqa"],
                       "reasoning_steps": 2, "action_space": "discrete_MC",
                       "answer_uniqueness": "deterministic"},
        "GSM8K": {"sigma_star": 0.01, "reasoning_steps": 6, "action_space": "continuous_text",
                  "answer_uniqueness": "multiple_paths"}
    }

    # Final save
    out_path = os.path.join(RESULTS_DIR, "phase34_sigma_prediction.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 34 SUMMARY: σ* Prediction Model")
    print("=" * 60)
    print(f"  Baseline: {100*baseline_rate:.1f}%")
    print(f"  σ* for TruthfulQA: {results['sigma_star_truthfulqa']}")
    print(f"  Best accuracy: {100*results['best_accuracy']:.1f}%")
    print(f"\n  σ* landscape:")
    print(f"    ARC-AGI:      σ*=0.20 (discrete, short)")
    print(f"    Hanoi:        σ*=0.15 (discrete, medium)")
    print(f"    TruthfulQA:   σ*={results['sigma_star_truthfulqa']} (MC, medium)")
    print(f"    GSM8K:        σ*=0.01 (text, long chain)")
    print("=" * 60)

    hook.remove()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
