"""
Phase 32: LLM-ExIt (Expert Iteration on Mistral-7B)
Self-evolution: collect miracle trajectories via NBS, fine-tune with QLoRA, measure improvement.
"""
import os, json, time, re, gc, copy
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
NOISE_LAYER = 18
SIGMA = 0.15
K_COLLECT = 11       # K for trajectory collection
N_COLLECT = 50       # Games per collection round
N_EVAL = 50          # Games for evaluation
MAX_MOVES = 15       # Max moves per Hanoi game
EXIT_ITERATIONS = 3  # Number of ExIt loops
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LR = 2e-4
LORA_EPOCHS = 3
DEVICE = "cuda"

# ============================================================
# Modified Hanoi (same as Phase 29)
# ============================================================
class HanoiGame:
    def __init__(self, n_disks=3):
        self.n_disks = n_disks
        self.pegs = [list(range(n_disks, 0, -1)), [], []]

    def reset(self):
        self.pegs = [list(range(self.n_disks, 0, -1)), [], []]
        return self.get_state()

    def get_state(self):
        return f"Peg A: {self.pegs[0]}, Peg B: {self.pegs[1]}, Peg C: {self.pegs[2]}"

    def move(self, src, dst):
        peg_map = {'A': 0, 'B': 1, 'C': 2}
        s, d = peg_map.get(src.upper()), peg_map.get(dst.upper())
        if s is None or d is None:
            return False, "Invalid peg"
        if not self.pegs[s]:
            return False, "Source peg empty"
        if self.pegs[d] and self.pegs[d][-1] < self.pegs[s][-1]:
            return False, "Cannot place larger disk on smaller"
        disk = self.pegs[s].pop()
        self.pegs[d].append(disk)
        return True, f"Moved disk {disk} from {src} to {dst}"

    def is_solved(self):
        return self.pegs[2] == list(range(self.n_disks, 0, -1))

# ============================================================
# Noise Hook
# ============================================================
class NoiseHook:
    def __init__(self, sigma):
        self.sigma = sigma
        self.handle = None
        self.active = True

    def hook_fn(self, module, input, output):
        if self.sigma > 0 and self.active:
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

    def detach(self):
        if self.handle:
            self.handle.remove()

# ============================================================
# Play one Hanoi game with LLM
# ============================================================
def play_hanoi(model, tokenizer, noise_hook, sigma, record_trajectory=False):
    game = HanoiGame(3)
    state = game.reset()
    trajectory = []
    noise_hook.sigma = sigma
    noise_hook.active = (sigma > 0)

    for step in range(MAX_MOVES):
        prompt_text = f"""You are solving the Tower of Hanoi puzzle with 3 disks.
Move all disks from Peg A to Peg C. Rules: Only move one disk at a time. Never place a larger disk on a smaller one.

Current state: {state}
What is your next move? Reply with exactly: Move disk from X to Y"""

        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True if sigma > 0 else False,
                temperature=0.7 if sigma > 0 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Parse move
        match = re.search(r'[Mm]ove disk from ([ABC]) to ([ABC])', response)
        if not match:
            match = re.search(r'([ABC])\s*(?:to|→)\s*([ABC])', response)
        if not match:
            break

        src, dst = match.group(1), match.group(2)
        success, msg = game.move(src, dst)

        if record_trajectory and success:
            trajectory.append({
                "state": state,
                "action": f"Move disk from {src} to {dst}",
                "response": response
            })

        if not success:
            break

        state = game.get_state()
        if game.is_solved():
            if record_trajectory:
                return True, trajectory
            return True, []

    if record_trajectory:
        return False, trajectory
    return False, []

# ============================================================
# Collect miracle trajectories via NBS
# ============================================================
def collect_miracles(model, tokenizer, noise_hook, n_games, K):
    print(f"  Collecting miracles: N={n_games}, K={K}, σ={SIGMA}")
    miracles = []
    total_solved = 0

    for i in range(n_games):
        for k in range(K):
            solved, traj = play_hanoi(model, tokenizer, noise_hook, SIGMA, record_trajectory=True)
            if solved:
                miracles.append(traj)
                total_solved += 1
                break

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{n_games}] Miracles: {total_solved}/{i+1}")

    print(f"  Total miracles: {total_solved}/{n_games} ({100*total_solved/n_games:.0f}%)")
    return miracles

# ============================================================
# Convert trajectories to SFT training data
# ============================================================
def trajectories_to_sft_data(miracles, tokenizer):
    """Convert miracle trajectories to prompt-completion pairs for SFT."""
    training_data = []
    for traj in miracles:
        for step in traj:
            messages = [
                {"role": "user", "content": f"You are solving the Tower of Hanoi puzzle with 3 disks.\nMove all disks from Peg A to Peg C. Rules: Only move one disk at a time. Never place a larger disk on a smaller one.\n\nCurrent state: {step['state']}\nWhat is your next move? Reply with exactly: Move disk from X to Y"},
                {"role": "assistant", "content": step['action']}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            training_data.append(text)
    return training_data

# ============================================================
# QLoRA Fine-tuning
# ============================================================
def finetune_qlora(model, tokenizer, training_texts):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from torch.utils.data import Dataset, DataLoader

    print(f"  Fine-tuning with {len(training_texts)} examples, {LORA_EPOCHS} epochs")

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    class SFTDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=512):
            self.encodings = []
            for t in texts:
                enc = tokenizer(t, truncation=True, max_length=max_len,
                              padding='max_length', return_tensors='pt')
                self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            item = self.encodings[idx]
            item['labels'] = item['input_ids'].clone()
            return item

    dataset = SFTDataset(training_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LORA_LR)
    model.train()

    for epoch in range(LORA_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"    Epoch {epoch+1}/{LORA_EPOCHS}: loss = {avg_loss:.4f}")

    model.eval()
    return model

# ============================================================
# Evaluate
# ============================================================
def evaluate(model, tokenizer, noise_hook, n_games, label=""):
    """Evaluate solve rate at K=1 with no noise (pure model ability)."""
    solved = 0
    noise_hook.active = False
    noise_hook.sigma = 0.0

    for i in range(n_games):
        s, _ = play_hanoi(model, tokenizer, noise_hook, 0.0)
        if s:
            solved += 1
        if (i + 1) % 10 == 0:
            print(f"    [{label}] [{i+1}/{n_games}] Solve rate: {solved}/{i+1} ({100*solved/(i+1):.1f}%)")

    rate = solved / n_games
    print(f"  {label} Final: {solved}/{n_games} = {100*rate:.1f}%")
    return rate

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Phase 32: LLM-ExIt (Mistral-7B Self-Evolution)")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading base model...")
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

    noise_hook = NoiseHook(SIGMA)
    noise_hook.attach(model)

    results = {
        "experiment": "Phase 32: LLM-ExIt",
        "model": MODEL_NAME,
        "config": {
            "noise_layer": NOISE_LAYER, "sigma": SIGMA,
            "K_collect": K_COLLECT, "n_collect": N_COLLECT,
            "n_eval": N_EVAL, "exit_iterations": EXIT_ITERATIONS,
            "lora_rank": LORA_RANK, "lora_epochs": LORA_EPOCHS,
        },
        "start_time": datetime.now().isoformat(),
        "iterations": []
    }

    # Baseline evaluation
    print("\n--- Baseline Evaluation (K=1, σ=0) ---")
    baseline_rate = evaluate(model, tokenizer, noise_hook, N_EVAL, "Baseline")
    results["baseline_solve_rate"] = baseline_rate

    # ExIt Loop
    for iteration in range(EXIT_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ExIt Iteration {iteration + 1}/{EXIT_ITERATIONS}")
        print(f"{'='*60}")

        iter_result = {"iteration": iteration + 1}

        # Step 1: Collect miracle trajectories
        print("\nStep 1: Collecting miracle trajectories...")
        miracles = collect_miracles(model, tokenizer, noise_hook, N_COLLECT, K_COLLECT)
        iter_result["miracles_collected"] = len(miracles)

        if len(miracles) < 5:
            print(f"  Too few miracles ({len(miracles)}), skipping fine-tuning")
            iter_result["skipped"] = True
            results["iterations"].append(iter_result)
            continue

        # Step 2: Convert to SFT data
        training_texts = trajectories_to_sft_data(miracles, tokenizer)
        iter_result["training_examples"] = len(training_texts)
        print(f"  Training examples: {len(training_texts)}")

        # Step 3: QLoRA fine-tuning
        print("\nStep 2: QLoRA Fine-tuning...")
        try:
            model = finetune_qlora(model, tokenizer, training_texts)
        except Exception as e:
            print(f"  Fine-tuning failed: {e}")
            iter_result["finetune_error"] = str(e)
            results["iterations"].append(iter_result)
            continue

        # Step 4: Evaluate improved model
        print("\nStep 3: Evaluating...")
        noise_hook.detach()
        noise_hook = NoiseHook(SIGMA)
        noise_hook.attach(model)

        k1_rate = evaluate(model, tokenizer, noise_hook, N_EVAL, f"Iter{iteration+1} K=1")
        iter_result["k1_solve_rate"] = k1_rate
        iter_result["improvement_pp"] = (k1_rate - baseline_rate) * 100

        results["iterations"].append(iter_result)

        # Save intermediate
        with open(os.path.join(RESULTS_DIR, "phase32_llm_exit.json"), 'w') as f:
            json.dump(results, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    results["end_time"] = datetime.now().isoformat()

    # Final save
    out_path = os.path.join(RESULTS_DIR, "phase32_llm_exit.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 32 SUMMARY: LLM-ExIt")
    print("=" * 60)
    print(f"  Baseline (K=1, σ=0): {100*baseline_rate:.1f}%")
    for ir in results["iterations"]:
        if "k1_solve_rate" in ir:
            print(f"  Iter {ir['iteration']}: K=1 = {100*ir['k1_solve_rate']:.1f}% (Δ={ir['improvement_pp']:+.1f}pp)")
    print("=" * 60)

    return results

if __name__ == "__main__":
    main()
