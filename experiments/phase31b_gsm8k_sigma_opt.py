"""
Phase 31b: GSM8K Noisy Beam Search — σ Optimization
====================================================
Phase 31でσ=0.15が強すぎてbaseline53%→7%に崩壊した。
小さいσでK=11がbaselineを超えるかテスト。

σ = {0.01, 0.03, 0.05} × K={1, 11} × N=200
"""
import os, json, time, re, gc
import torch
import numpy as np
from datetime import datetime

EXPERIMENT_DIR = r"c:\Users\kyjan\研究\snn-synthesis"
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
NOISE_LAYER = 18
SIGMA_VALUES = [0.01, 0.03, 0.05]
K_VALUES = [1, 11]  # Only test endpoints: single vs full beam
N_QUESTIONS = 200
MAX_NEW_TOKENS = 512
DEVICE = "cuda"

# ============================================================
# Noise Hook (Layer 18 pre-hook, same style as Phase 29)
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
# GSM8K answer extraction (same as Phase 31)
# ============================================================
def extract_gsm8k_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None

def extract_model_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    match = re.search(r'[Tt]he (?:final )?answer is[:\s]*\$?(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    numbers = re.findall(r'(-?[\d,]+\.?\d*)', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return None

def answers_match(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 0.01
    except ValueError:
        return pred.strip() == gold.strip()

# ============================================================
# Load Model (same as Phase 29)
# ============================================================
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print(f"Loading {MODEL_NAME} (4-bit)...")
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
# Generate with noise
# ============================================================
def solve_gsm8k(model, tokenizer, question, noise_hook, sigma):
    messages = [{"role": "user", "content":
        f"Solve this math problem step by step. End your answer with #### followed by the final numerical answer.\n\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    noise_hook.setup(sigma)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=(sigma > 0), temperature=0.7 if sigma > 0 else 1.0,
            top_p=0.9 if sigma > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 31b: GSM8K NBS — σ Optimization")
    print(f"σ = {SIGMA_VALUES}, K = {K_VALUES}, N = {N_QUESTIONS}")
    print("=" * 60)

    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=f'test[:{N_QUESTIONS}]')
    print(f"Loaded {len(ds)} questions")

    model, tokenizer = load_model()
    hook = NoiseHook()
    hook.register(model)

    results = {
        "experiment": "Phase 31b: GSM8K NBS σ Optimization",
        "model": MODEL_NAME,
        "sigma_values": SIGMA_VALUES,
        "k_values": K_VALUES,
        "n_questions": N_QUESTIONS,
        "start_time": datetime.now().isoformat(),
        "conditions": {},
        "phase31_baseline": {"sigma": 0.0, "K": 1, "accuracy": 0.53}  # from Phase 31
    }

    for sigma in SIGMA_VALUES:
        for K in K_VALUES:
            label = f"σ={sigma}_K={K}"
            print(f"\n--- {label} ---")
            correct = 0

            for i, item in enumerate(ds):
                question = item['question']
                gold = extract_gsm8k_answer(item['answer'])

                any_correct = False
                for k in range(K):
                    response = solve_gsm8k(model, tokenizer, question, hook, sigma)
                    pred = extract_model_answer(response)
                    if answers_match(pred, gold):
                        any_correct = True
                        break

                if any_correct:
                    correct += 1

                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{N_QUESTIONS}] {label}: {correct}/{i+1} ({100*correct/(i+1):.1f}%)")

            rate = correct / N_QUESTIONS
            results["conditions"][label] = {
                "sigma": sigma, "K": K,
                "correct": correct, "total": N_QUESTIONS,
                "accuracy": rate
            }
            print(f"  {label}: {correct}/{N_QUESTIONS} = {100*rate:.1f}%")

            # Save intermediate
            with open(os.path.join(RESULTS_DIR, "phase31b_gsm8k_sigma_opt.json"), 'w') as f:
                json.dump(results, f, indent=2)

    results["end_time"] = datetime.now().isoformat()
    out_path = os.path.join(RESULTS_DIR, "phase31b_gsm8k_sigma_opt.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 31b SUMMARY")
    print("=" * 60)
    print(f"  Baseline (σ=0, Phase 31): 53.0%")
    for label, c in results["conditions"].items():
        delta = (c['accuracy'] - 0.53) * 100
        print(f"  {label}: {c['accuracy']*100:.1f}% (Δ={delta:+.1f}pp vs baseline)")
    print("=" * 60)

    hook.remove()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
