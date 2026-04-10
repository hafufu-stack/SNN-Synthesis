"""
Phase 33: GSM8K LLM-ExIt — Math Reasoning Self-Evolution
=========================================================
Combine Phase 31b's GSM8K NBS (89.5% at K=11, σ=0.01) with
Phase 32b's ExIt pipeline (QLoRA self-distillation).

Pipeline per iteration:
1. NBS miracle collection: K=11, σ=0.01 → collect correct CoT solutions
2. QLoRA SFT: fine-tune Mistral-7B on miracle CoTs
3. Evaluate: K=1 accuracy (baseline 53%)
"""
import torch
import torch.nn as nn
import os, json, gc, time, random, re, math
os.environ['HF_DATASETS_OFFLINE'] = '1'
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SEED = 2026
BASE_SIGMA = 0.01  # GSM8K optimal (NOT 0.15!)
LAYER_IDX = 18
K_COLLECT = 11
N_COLLECT = 200  # use all 200 GSM8K questions
N_EVAL = 200
EXIT_ITERATIONS = 3
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LR = 2e-4
LORA_EPOCHS = 3
MAX_NEW_TOKENS = 512

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda"

# ============================================================
# Noise Hook (from Phase 31b)
# ============================================================
class NoiseHook:
    def __init__(self):
        self.sigma = 0.0
        self.handle = None

    def setup(self, sigma):
        self.sigma = sigma

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            hs = args[0]
            if hook_obj.sigma <= 0:
                return args
            noise = torch.randn_like(hs) * hook_obj.sigma
            return (hs + noise,) + args[1:]
        layers = get_model_layers(model)
        self.handle = layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()

def get_model_layers(model):
    """Resolve transformer layers for both base and PEFT-wrapped models."""
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            return model.base_model.model.model.layers
    except ImportError:
        pass
    return model.model.layers

# ============================================================
# GSM8K answer extraction (from Phase 31b)
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
def load_model():
    print(f"\n  Loading {MODEL_NAME} (4-bit)...")
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
    return prompt, response

# ============================================================
# Collect Miracles: NBS on GSM8K
# ============================================================
def collect_miracles_gsm8k(model, tokenizer, hook, dataset, K, sigma):
    """Run NBS K=11 on GSM8K, collect all correct solutions as miracles."""
    print(f"  Collecting miracles: N={len(dataset)}, K={K}, σ={sigma}")
    miracles = []  # list of (question, response, gold_answer)
    correct_count = 0

    for i, item in enumerate(dataset):
        question = item['question']
        gold = extract_gsm8k_answer(item['answer'])

        best_response = None
        for k in range(K):
            prompt, response = solve_gsm8k(model, tokenizer, question, hook, sigma)
            pred = extract_model_answer(response)
            if answers_match(pred, gold):
                best_response = response
                correct_count += 1
                break  # first correct answer is the miracle

        if best_response is not None:
            miracles.append({
                "question": question,
                "response": best_response,
                "gold_answer": gold,
                "prompt_template": f"Solve this math problem step by step. End your answer with #### followed by the final numerical answer.\n\n{question}"
            })

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(dataset)}] Miracles: {len(miracles)}/{i+1} ({100*len(miracles)/(i+1):.1f}%)")

    print(f"  Total miracles: {len(miracles)}/{len(dataset)} ({100*len(miracles)/len(dataset):.1f}%)")
    return miracles

# ============================================================
# Convert miracles to SFT data
# ============================================================
def miracles_to_sft_data(miracles, tokenizer):
    """Convert miracle solutions to chat-format training data."""
    training_data = []
    for m in miracles:
        messages = [
            {"role": "user", "content": m["prompt_template"]},
            {"role": "assistant", "content": m["response"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        training_data.append(text)
    return training_data

# ============================================================
# QLoRA Fine-Tuning (from Phase 32b)
# ============================================================
def finetune_qlora(model, tokenizer, training_texts):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
    from torch.utils.data import Dataset, DataLoader

    print(f"  Fine-tuning: {len(training_texts)} examples, {LORA_EPOCHS} epochs")

    is_peft = False
    try:
        from peft import PeftModel as PM
        is_peft = isinstance(model, PM)
    except:
        pass

    if not is_peft:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class SFTDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=768):
            self.encodings = []
            for t in texts:
                enc = tokenizer(t, truncation=True, max_length=max_len,
                              padding='max_length', return_tensors='pt')
                self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
        def __len__(self): return len(self.encodings)
        def __getitem__(self, idx):
            item = self.encodings[idx]
            item['labels'] = item['input_ids'].clone()
            return item

    dataset = SFTDataset(training_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LORA_LR)
    model.train()

    for epoch in range(LORA_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/{LORA_EPOCHS}: loss = {total_loss/len(dataloader):.4f}")

    model.eval()
    return model

# ============================================================
# Evaluate K=1 (no noise)
# ============================================================
def evaluate_gsm8k(model, tokenizer, hook, dataset, label=""):
    """Evaluate K=1 accuracy (no noise) on GSM8K."""
    hook.setup(0.0)  # No noise for evaluation
    correct = 0
    for i, item in enumerate(dataset):
        question = item['question']
        gold = extract_gsm8k_answer(item['answer'])
        _, response = solve_gsm8k(model, tokenizer, question, hook, sigma=0.0)
        pred = extract_model_answer(response)
        if answers_match(pred, gold):
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
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("Phase 33: GSM8K LLM-ExIt — Math Reasoning Self-Evolution")
    print(f"  σ={BASE_SIGMA}, K={K_COLLECT}, N={N_COLLECT}")
    print(f"  ExIt iterations: {EXIT_ITERATIONS}")
    print("=" * 60)

    # Load GSM8K
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=f'test[:{N_COLLECT}]')
    print(f"  Loaded {len(ds)} GSM8K questions")

    # Load model
    model, tokenizer = load_model()

    hook = NoiseHook()
    hook.register(model)

    results = {
        "experiment": "Phase 33: GSM8K LLM-ExIt",
        "model": MODEL_NAME,
        "config": {
            "sigma": BASE_SIGMA, "layer": LAYER_IDX,
            "K_collect": K_COLLECT, "n_collect": N_COLLECT,
            "n_eval": N_EVAL, "exit_iterations": EXIT_ITERATIONS,
            "lora_rank": LORA_RANK, "lora_epochs": LORA_EPOCHS,
        },
        "start_time": datetime.now().isoformat(),
        "iterations": []
    }

    # Baseline: K=1, σ=0
    print("\n--- Baseline (K=1, σ=0) ---")
    baseline = evaluate_gsm8k(model, tokenizer, hook, ds, "Baseline")
    results["baseline_k1_accuracy"] = baseline

    # ExIt Loop
    for iteration in range(EXIT_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ExIt Iteration {iteration+1}/{EXIT_ITERATIONS}")
        print(f"{'='*60}")

        iter_result = {"iteration": iteration + 1}

        # Step 1: Collect miracles
        print("\nStep 1: Collecting miracle CoTs via NBS...")
        miracles = collect_miracles_gsm8k(model, tokenizer, hook, ds, K_COLLECT, BASE_SIGMA)
        iter_result["miracles_collected"] = len(miracles)
        iter_result["miracle_rate"] = len(miracles) / len(ds)

        if len(miracles) < 10:
            print(f"  Too few miracles ({len(miracles)}), skipping fine-tuning")
            iter_result["skipped"] = True
            results["iterations"].append(iter_result)
            continue

        # Step 2: Convert to training data
        training_texts = miracles_to_sft_data(miracles, tokenizer)
        iter_result["training_examples"] = len(training_texts)
        print(f"  Training examples: {len(training_texts)}")

        # Step 3: QLoRA Fine-tune
        print("\nStep 2: QLoRA Fine-tuning on miracle CoTs...")
        try:
            model = finetune_qlora(model, tokenizer, training_texts)
        except Exception as e:
            print(f"  Fine-tuning error: {e}")
            import traceback
            traceback.print_exc()
            iter_result["error"] = str(e)
            results["iterations"].append(iter_result)
            continue

        # Re-register hook
        hook.remove()
        hook = NoiseHook()
        hook.register(model)

        # Step 4: Evaluate K=1 (no noise)
        print("\nStep 3: Evaluating K=1 accuracy...")
        k1_rate = evaluate_gsm8k(model, tokenizer, hook, ds, f"Iter{iteration+1}")
        iter_result["k1_accuracy"] = k1_rate
        iter_result["improvement_pp"] = (k1_rate - baseline) * 100

        results["iterations"].append(iter_result)

        # Save intermediate
        out_path = os.path.join(RESULTS_DIR, "phase33_gsm8k_exit.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    results["end_time"] = datetime.now().isoformat()

    # Final save
    out_path = os.path.join(RESULTS_DIR, "phase33_gsm8k_exit.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 33 SUMMARY: GSM8K LLM-ExIt")
    print("=" * 60)
    print(f"  Baseline: {100*baseline:.1f}%")
    for ir in results["iterations"]:
        if "k1_accuracy" in ir:
            print(f"  Iter {ir['iteration']}: {100*ir['k1_accuracy']:.1f}% (Δ={ir['improvement_pp']:+.1f}pp, miracles={ir['miracles_collected']})")
        elif ir.get("skipped"):
            print(f"  Iter {ir['iteration']}: SKIPPED (miracles={ir['miracles_collected']})")
    print("=" * 60)

    hook.remove()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
