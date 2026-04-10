"""
Phase 37: Cross-Task Transfer — Does Hanoi ExIt Help Math?
============================================================
Test whether QLoRA weights trained on Hanoi ExIt (Phase 32b, 100% solve)
transfer to GSM8K math reasoning.

If Hanoi-trained model outperforms vanilla on GSM8K, it proves
self-distillation via noise develops general reasoning, not task-specific.
"""
import torch
import os, json, gc, time, random, re
os.environ['HF_DATASETS_OFFLINE'] = '1'
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
NOISE_LAYER = 18
BASE_SIGMA = 0.01  # GSM8K optimal
K = 11
N_QUESTIONS = 200
MAX_NEW_TOKENS = 512
DEVICE = "cuda"

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

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
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                layers = model.base_model.model.model.layers
            else:
                layers = model.model.layers
        except ImportError:
            layers = model.model.layers
        self.handle = layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()

# ============================================================
# GSM8K answer utils (from Phase 31b)
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
# Load Model
# ============================================================
def load_base_model():
    print(f"  Loading base {MODEL_NAME} (4-bit)...")
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

def load_hanoi_exit_model():
    """Load base model + Hanoi ExIt QLoRA weights."""
    model, tokenizer = load_base_model()

    # Look for Hanoi ExIt checkpoint
    lora_paths = [
        os.path.join(DATA_DIR, "hanoi_exit_lora"),
        os.path.join(EXPERIMENT_DIR, "data", "hanoi_exit_lora"),
    ]
    lora_path = None
    for p in lora_paths:
        if os.path.exists(p):
            lora_path = p
            break

    if lora_path is None:
        print("  WARNING: No Hanoi ExIt LoRA weights found, using base model")
        return model, tokenizer, False

    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        print(f"  Loaded Hanoi ExIt LoRA from {lora_path}")
        return model, tokenizer, True
    except Exception as e:
        print(f"  WARNING: Failed to load LoRA: {e}")
        return model, tokenizer, False

# ============================================================
# GSM8K evaluation
# ============================================================
def evaluate_gsm8k(model, tokenizer, hook, dataset, sigma, K_val, label=""):
    correct = 0
    for i, item in enumerate(dataset):
        question = item['question']
        gold = extract_gsm8k_answer(item['answer'])

        any_correct = False
        for k in range(K_val):
            messages = [{"role": "user", "content":
                f"Solve this math problem step by step. End your answer with #### followed by the final numerical answer.\n\n{question}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

            hook.setup(sigma)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=(sigma > 0), temperature=0.7 if sigma > 0 else 1.0,
                    top_p=0.9 if sigma > 0 else 1.0,
                    pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = extract_model_answer(response)
            if answers_match(pred, gold):
                any_correct = True
                break

        if any_correct:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"    [{label}] [{i+1}/{len(dataset)}] {correct}/{i+1} ({100*correct/(i+1):.1f}%)")

    rate = correct / len(dataset)
    print(f"  {label}: {correct}/{len(dataset)} = {100*rate:.1f}%")
    return rate

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 37: Cross-Task Transfer (Hanoi → GSM8K)")
    print("=" * 60)

    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=f'test[:{N_QUESTIONS}]')
    print(f"  Loaded {len(ds)} GSM8K questions")

    results = {
        "experiment": "Phase 37: Cross-Task Transfer",
        "model": MODEL_NAME,
        "start_time": datetime.now().isoformat(),
        "configs": {}
    }

    # ---- Condition 1: Vanilla model ----
    print("\n--- Vanilla Mistral-7B ---")
    model, tokenizer = load_base_model()
    hook = NoiseHook()
    hook.register(model)

    # K=1, σ=0
    rate_vanilla_k1 = evaluate_gsm8k(model, tokenizer, hook, ds, 0.0, 1, "Vanilla K=1")
    results["configs"]["vanilla_K1"] = {"accuracy": rate_vanilla_k1, "sigma": 0, "K": 1}

    # K=11, σ=0.01
    rate_vanilla_k11 = evaluate_gsm8k(model, tokenizer, hook, ds, BASE_SIGMA, K, "Vanilla K=11 σ=0.01")
    results["configs"]["vanilla_K11"] = {"accuracy": rate_vanilla_k11, "sigma": BASE_SIGMA, "K": K}

    hook.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Condition 2: Hanoi ExIt model ----
    print("\n--- Hanoi ExIt Mistral-7B ---")
    model, tokenizer, lora_loaded = load_hanoi_exit_model()
    hook = NoiseHook()
    hook.register(model)

    results["hanoi_lora_loaded"] = lora_loaded

    # K=1, σ=0
    rate_exit_k1 = evaluate_gsm8k(model, tokenizer, hook, ds, 0.0, 1, "Hanoi-ExIt K=1")
    results["configs"]["hanoi_exit_K1"] = {"accuracy": rate_exit_k1, "sigma": 0, "K": 1}

    # K=11, σ=0.01
    rate_exit_k11 = evaluate_gsm8k(model, tokenizer, hook, ds, BASE_SIGMA, K, "Hanoi-ExIt K=11 σ=0.01")
    results["configs"]["hanoi_exit_K11"] = {"accuracy": rate_exit_k11, "sigma": BASE_SIGMA, "K": K}

    results["end_time"] = datetime.now().isoformat()

    # Analysis
    results["transfer_analysis"] = {
        "k1_delta_pp": (rate_exit_k1 - rate_vanilla_k1) * 100,
        "k11_delta_pp": (rate_exit_k11 - rate_vanilla_k11) * 100,
        "conclusion": "positive_transfer" if rate_exit_k1 > rate_vanilla_k1 else
                      "negative_transfer" if rate_exit_k1 < rate_vanilla_k1 else "neutral"
    }

    out_path = os.path.join(RESULTS_DIR, "phase37_cross_task_transfer.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 37 SUMMARY: Cross-Task Transfer")
    print("=" * 60)
    print(f"  Vanilla K=1:     {100*rate_vanilla_k1:.1f}%")
    print(f"  Hanoi-ExIt K=1:  {100*rate_exit_k1:.1f}%  (Δ={results['transfer_analysis']['k1_delta_pp']:+.1f}pp)")
    print(f"  Vanilla K=11:    {100*rate_vanilla_k11:.1f}%")
    print(f"  Hanoi-ExIt K=11: {100*rate_exit_k11:.1f}%  (Δ={results['transfer_analysis']['k11_delta_pp']:+.1f}pp)")
    print(f"  Conclusion: {results['transfer_analysis']['conclusion']}")
    print("=" * 60)

    hook.remove()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
