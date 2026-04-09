"""
Phase 31: Multi-Task Noisy Beam Search on GSM8K
Validates architecture-invariant K scaling on math reasoning (not just Hanoi).
"""
import os, json, time, re, gc
import torch
import numpy as np
from datetime import datetime

EXPERIMENT_DIR = r"c:\Users\kyjan\研究\snn-synthesis"
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Config
# ============================================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
NOISE_LAYER = 18  # Same as Phase 29
SIGMA = 0.15      # Optimal from Phase 29
K_VALUES = [1, 3, 5, 7, 11]
N_QUESTIONS = 200  # First 200 GSM8K test questions
MAX_NEW_TOKENS = 512
DEVICE = "cuda"

# ============================================================
# Noise Hook (same as Phase 29)
# ============================================================
class NoiseHook:
    def __init__(self, sigma):
        self.sigma = sigma
        self.handle = None

    def hook_fn(self, module, input, output):
        if self.sigma > 0 and self.training_mode:
            if isinstance(output, tuple):
                h = output[0]
                noise = torch.randn_like(h) * self.sigma
                return (h + noise,) + output[1:]
            else:
                noise = torch.randn_like(output) * self.sigma
                return output + noise
        return output

    def attach(self, model):
        layer = model.model.layers[NOISE_LAYER]
        self.handle = layer.register_forward_hook(self.hook_fn)
        self.training_mode = True

    def detach(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

# ============================================================
# Load Model
# ============================================================
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"Loading {MODEL_NAME} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

# ============================================================
# GSM8K answer extraction
# ============================================================
def extract_gsm8k_answer(text):
    """Extract the final numerical answer from GSM8K format (#### N)."""
    # Ground truth format: "#### 42"
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None

def extract_model_answer(text):
    """Extract numerical answer from model output."""
    # Try #### format first
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Try "the answer is X" pattern
    match = re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    # Try last number in text
    numbers = re.findall(r'(-?[\d,]+\.?\d*)', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return None

def answers_match(pred, gold):
    """Check if predicted answer matches gold answer."""
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 0.01
    except ValueError:
        return pred.strip() == gold.strip()

# ============================================================
# Single inference with noise
# ============================================================
def solve_gsm8k(model, tokenizer, question, noise_hook, sigma):
    """Generate a solution for a GSM8K question with noise injection."""
    messages = [
        {"role": "user", "content": f"Solve this math problem step by step. End your answer with #### followed by the final numerical answer.\n\n{question}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    noise_hook.sigma = sigma
    noise_hook.training_mode = (sigma > 0)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True if sigma > 0 else False,
            temperature=0.7 if sigma > 0 else 1.0,
            top_p=0.9 if sigma > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# ============================================================
# Main Experiment
# ============================================================
def main():
    print("=" * 60)
    print("Phase 31: GSM8K Noisy Beam Search")
    print("=" * 60)

    # Load data
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=f'test[:{N_QUESTIONS}]')
    print(f"Loaded {len(ds)} GSM8K test questions")

    # Load model
    model, tokenizer = load_model()
    noise_hook = NoiseHook(SIGMA)
    noise_hook.attach(model)

    results = {
        "experiment": "Phase 31: GSM8K Multi-Task NBS",
        "model": MODEL_NAME,
        "noise_layer": NOISE_LAYER,
        "sigma": SIGMA,
        "n_questions": N_QUESTIONS,
        "k_values": K_VALUES,
        "start_time": datetime.now().isoformat(),
        "conditions": {}
    }

    # Run baseline (no noise, K=1)
    print(f"\n--- Baseline (σ=0, K=1) ---")
    baseline_correct = 0
    for i, item in enumerate(ds):
        question = item['question']
        gold = extract_gsm8k_answer(item['answer'])
        response = solve_gsm8k(model, tokenizer, question, noise_hook, sigma=0.0)
        pred = extract_model_answer(response)
        if answers_match(pred, gold):
            baseline_correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{N_QUESTIONS}] Baseline accuracy: {baseline_correct}/{i+1} ({100*baseline_correct/(i+1):.1f}%)")

    baseline_rate = baseline_correct / N_QUESTIONS
    results["conditions"]["baseline"] = {
        "sigma": 0.0, "K": 1,
        "correct": baseline_correct, "total": N_QUESTIONS,
        "accuracy": baseline_rate
    }
    print(f"  Baseline: {baseline_correct}/{N_QUESTIONS} = {100*baseline_rate:.1f}%")

    # Run NBS for each K
    for K in K_VALUES:
        print(f"\n--- NBS σ={SIGMA}, K={K} ---")
        nbs_correct = 0

        for i, item in enumerate(ds):
            question = item['question']
            gold = extract_gsm8k_answer(item['answer'])

            # Run K independent noisy inferences
            any_correct = False
            for k in range(K):
                response = solve_gsm8k(model, tokenizer, question, noise_hook, sigma=SIGMA)
                pred = extract_model_answer(response)
                if answers_match(pred, gold):
                    any_correct = True
                    break  # Early stop: at least one beam solved it

            if any_correct:
                nbs_correct += 1

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{N_QUESTIONS}] NBS K={K}: {nbs_correct}/{i+1} ({100*nbs_correct/(i+1):.1f}%)")

        nbs_rate = nbs_correct / N_QUESTIONS
        results["conditions"][f"K={K}"] = {
            "sigma": SIGMA, "K": K,
            "correct": nbs_correct, "total": N_QUESTIONS,
            "accuracy": nbs_rate
        }
        print(f"  K={K}: {nbs_correct}/{N_QUESTIONS} = {100*nbs_rate:.1f}%")

    results["end_time"] = datetime.now().isoformat()

    # Save results
    out_path = os.path.join(RESULTS_DIR, "phase31_gsm8k_nbs.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 31 SUMMARY: GSM8K Noisy Beam Search")
    print("=" * 60)
    print(f"  Baseline (σ=0):  {results['conditions']['baseline']['accuracy']*100:.1f}%")
    for K in K_VALUES:
        c = results['conditions'][f'K={K}']
        print(f"  NBS K={K:2d} σ={SIGMA}: {c['accuracy']*100:.1f}%")
    print("=" * 60)

    return results

if __name__ == "__main__":
    main()
