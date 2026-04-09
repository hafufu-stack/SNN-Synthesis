"""
Phase 32b: LLM-ExIt using Phase 29's proven implementation
===========================================================
Phase 32 failed because it used a different Hanoi implementation.
This version reuses Phase 29's exact HanoiEnv, prompts, noise hook, and game loop.
"""
import torch
import torch.nn as nn
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SEED = 2026
MAX_STEPS = 50
BASE_SIGMA = 0.15
LAYER_IDX = 18
K_COLLECT = 11
N_COLLECT = 50
N_EVAL = 50
EXIT_ITERATIONS = 3
LORA_RANK = 8
LORA_ALPHA = 16
LORA_LR = 2e-4
LORA_EPOCHS = 3

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===================================================
#  HANOI ENVIRONMENT (exact copy from Phase 29)
# ===================================================
class HanoiEnv:
    def __init__(self, n_disks=3, modified=True):
        self.n_disks = n_disks
        self.modified = modified
        self.reset()

    def reset(self):
        self.pegs = {"A": list(range(self.n_disks, 0, -1)), "B": [], "C": []}
        self.moves = []
        self.illegal_count = 0
        self.total_attempts = 0
        self.self_corrections = 0
        self._prev_illegal = False

    def is_solved(self):
        return len(self.pegs["C"]) == self.n_disks

    def legal_moves(self):
        result = []
        for f in "ABC":
            for t in "ABC":
                if f != t and self.pegs[f]:
                    disk = self.pegs[f][-1]
                    if not self.pegs[t] or \
                       (self.modified and disk > self.pegs[t][-1]) or \
                       (not self.modified and disk < self.pegs[t][-1]):
                        result.append(f"{f}->{t}")
        return result

    def try_move(self, from_p, to_p):
        self.total_attempts += 1
        from_p, to_p = from_p.upper(), to_p.upper()
        if from_p not in "ABC" or to_p not in "ABC" or from_p == to_p:
            self.illegal_count += 1
            self._prev_illegal = True
            return False, "Invalid peg"
        if not self.pegs[from_p]:
            self.illegal_count += 1
            self._prev_illegal = True
            return False, f"{from_p} is empty"
        disk = self.pegs[from_p][-1]
        if self.pegs[to_p]:
            top = self.pegs[to_p][-1]
            if self.modified and disk <= top:
                self.illegal_count += 1
                self._prev_illegal = True
                return False, "Illegal"
            if not self.modified and disk >= top:
                self.illegal_count += 1
                self._prev_illegal = True
                return False, "Illegal"
        if self._prev_illegal:
            self.self_corrections += 1
        self._prev_illegal = False
        self.pegs[from_p].pop()
        self.pegs[to_p].append(disk)
        self.moves.append(f"{from_p}->{to_p} (disk {disk})")
        return True, f"Moved disk {disk}: {from_p}->{to_p}"

    def state_str(self):
        return f"A:{self.pegs['A']} B:{self.pegs['B']} C:{self.pegs['C']}"

    def stats(self):
        return {"solved": self.is_solved(), "legal_moves": len(self.moves),
                "illegal_moves": self.illegal_count, "self_corrections": self.self_corrections}

# ===================================================
#  PROMPT & PARSER (exact copy from Phase 29)
# ===================================================
def build_chat_prompt(tokenizer, env, error=None):
    rules = "MODIFIED RULES: You can ONLY place a LARGER disk onto a SMALLER disk. The opposite of standard."
    system = (
        f"You are solving Tower of Hanoi with {env.n_disks} disks. "
        f"{rules} "
        f"Goal: move ALL disks from A to C. "
        f"Respond with EXACTLY one move in format: Move: X->Y (e.g. Move: A->C). "
        f"You may add a brief Think: line before it."
    )
    msg = f"State: {env.state_str()}\n"
    legal = env.legal_moves()
    msg += f"Legal moves: {', '.join(legal)}\n"
    if env.moves:
        recent = env.moves[-3:]
        msg += f"Your last moves: {'; '.join(recent)}\n"
    if error:
        msg += f"ERROR: {error}. Pick from legal moves above.\n"
    msg += "Your move:"
    messages = [{"role": "user", "content": system + "\n\n" + msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_move(response):
    patterns = [
        r'Move:\s*([A-Ca-c])\s*->\s*([A-Ca-c])',
        r'move\s+(?:disk\s+\d+\s+)?(?:from\s+)?([A-Ca-c])\s+to\s+([A-Ca-c])',
        r'([A-Ca-c])\s*->\s*([A-Ca-c])',
    ]
    for p in patterns:
        m = re.search(p, response, re.IGNORECASE)
        if m:
            return m.group(1).upper(), m.group(2).upper()
    return None

# ===================================================
#  MODEL + GENERATION (exact copy from Phase 29)
# ===================================================
def load_model():
    print(f"\n Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        local_files_only=True)
    model.eval()
    return model, tok

def gen(model, tok, prompt, temperature=0.5, max_tokens=80):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_p=0.9,
            repetition_penalty=1.2, pad_token_id=tok.pad_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    inp = tok.decode(inputs['input_ids'][0], skip_special_tokens=True)
    return full[len(inp):].strip()

# ===================================================
#  NOISE HOOK (from Phase 29, fixed for PEFT models)
# ===================================================
def get_model_layers(model):
    """Resolve transformer layers for both base and PEFT-wrapped models."""
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            # PEFT wrapped: model.base_model.model.model.layers
            return model.base_model.model.model.layers
    except ImportError:
        pass
    # Base model: model.model.layers
    return model.model.layers

class NoiseHook:
    def __init__(self):
        self.sigma = 0.0
        self.diff_offset = None
        self.handle = None

    def setup(self, sigma, diff_unit=None, device='cuda'):
        self.sigma = sigma
        if diff_unit is not None:
            self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        else:
            self.diff_offset = None

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            hs = args[0]
            if hook_obj.sigma <= 0:
                return args
            if hook_obj.diff_offset is not None:
                d = hs.shape[-1]
                det_scale = hook_obj.sigma * math.sqrt(d) * 0.5
                det_noise = hook_obj.diff_offset * det_scale
                if hs.dim() == 3:
                    det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
                else:
                    det_noise = det_noise.unsqueeze(0).expand_as(hs)
                stoch_noise = torch.randn_like(hs) * (hook_obj.sigma * 0.5)
                return (hs + det_noise + stoch_noise,) + args[1:]
            else:
                noise = torch.randn_like(hs) * hook_obj.sigma
                return (hs + noise,) + args[1:]
        layers = get_model_layers(model)
        self.handle = layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()

# ===================================================
#  PLAY GAME (exact copy from Phase 29)
# ===================================================
def play_game(model, tok, hook, record_trajectory=False):
    env = HanoiEnv(n_disks=3, modified=True)
    error = None
    consec_fail = 0
    trajectory = []
    for step in range(MAX_STEPS):
        prompt = build_chat_prompt(tok, env, error)
        resp = gen(model, tok, prompt)
        move = parse_move(resp)
        if move is None:
            env.illegal_count += 1
            env.total_attempts += 1
            env._prev_illegal = True
            error = "Parse fail. Use Move: X->Y"
            consec_fail += 1
            if consec_fail >= 10:
                break
            continue
        ok, msg = env.try_move(move[0], move[1])
        if ok:
            if record_trajectory:
                trajectory.append({
                    "state": env.state_str(),
                    "prompt": prompt,
                    "response": resp,
                    "move": f"{move[0]}->{move[1]}"
                })
            error = None
            consec_fail = 0
            if env.is_solved():
                break
        else:
            error = msg
            consec_fail += 1
            if consec_fail >= 10:
                break
    stats = env.stats()
    stats["steps_taken"] = step + 1
    return stats, trajectory

# ===================================================
#  COLLECT MIRACLES VIA NBS
# ===================================================
def collect_miracles(model, tok, hook, diff_unit, device, n_games, K):
    print(f"  Collecting miracles: N={n_games}, K={K}, σ={BASE_SIGMA}")
    miracles = []
    for i in range(n_games):
        for k in range(K):
            hook.setup(sigma=BASE_SIGMA, diff_unit=diff_unit, device=device)
            stats, traj = play_game(model, tok, hook, record_trajectory=True)
            if stats["solved"]:
                miracles.append(traj)
                break
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{n_games}] Miracles: {len(miracles)}/{i+1}")
    print(f"  Total miracles: {len(miracles)}/{n_games} ({100*len(miracles)/n_games:.0f}%)")
    return miracles

# ===================================================
#  CONVERT TO SFT DATA
# ===================================================
def trajectories_to_sft_data(miracles, tokenizer):
    training_data = []
    for traj in miracles:
        for step in traj:
            messages = [
                {"role": "user", "content": step["prompt"].split("[/INST]")[0].split("[INST]")[-1].strip() if "[INST]" in step["prompt"] else step["prompt"]},
                {"role": "assistant", "content": step["response"]}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            training_data.append(text)
    return training_data

# ===================================================
#  QLORA FINE-TUNING
# ===================================================
def finetune_qlora(model, tokenizer, training_texts):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
    from torch.utils.data import Dataset, DataLoader

    print(f"  Fine-tuning: {len(training_texts)} examples, {LORA_EPOCHS} epochs")

    # Only wrap with PEFT on first iteration; on subsequent ones, model is already PEFT
    is_peft = isinstance(model, PeftModel)
    if not is_peft:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    class SFTDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=512):
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

# ===================================================
#  EVALUATE
# ===================================================
def evaluate(model, tok, hook, n_games, label=""):
    hook.setup(sigma=0.0)
    solved = 0
    for i in range(n_games):
        stats, _ = play_game(model, tok, hook)
        if stats["solved"]:
            solved += 1
        if (i + 1) % 10 == 0:
            print(f"    [{label}] [{i+1}/{n_games}] {solved}/{i+1} ({100*solved/(i+1):.1f}%)")
    rate = solved / n_games
    print(f"  {label}: {solved}/{n_games} = {100*rate:.1f}%")
    return rate

# ===================================================
#  MAIN
# ===================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("Phase 32b: LLM-ExIt (Phase 29 implementation)")
    print("=" * 60)

    # Load Aha! direction
    diff_data = np.load(os.path.join(DATA_DIR, "diff_pca.npz"))
    diff_unit = diff_data["diff_unit"]
    print(f"Diff unit loaded: shape={diff_unit.shape}")

    model, tok = load_model()
    device = next(model.parameters()).device

    hook = NoiseHook()
    hook.register(model)

    results = {
        "experiment": "Phase 32b: LLM-ExIt",
        "model": MODEL_NAME,
        "config": {
            "sigma": BASE_SIGMA, "layer": LAYER_IDX,
            "K_collect": K_COLLECT, "n_collect": N_COLLECT,
            "n_eval": N_EVAL, "exit_iterations": EXIT_ITERATIONS,
            "lora_rank": LORA_RANK, "lora_epochs": LORA_EPOCHS,
        },
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iterations": []
    }

    # Baseline
    print("\n--- Baseline (K=1, σ=0) ---")
    baseline = evaluate(model, tok, hook, N_EVAL, "Baseline")
    results["baseline_solve_rate"] = baseline

    # ExIt Loop
    for iteration in range(EXIT_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ExIt Iteration {iteration+1}/{EXIT_ITERATIONS}")
        print(f"{'='*60}")

        iter_result = {"iteration": iteration + 1}

        # Collect miracles
        print("\nStep 1: Collecting miracle trajectories...")
        miracles = collect_miracles(model, tok, hook, diff_unit, device, N_COLLECT, K_COLLECT)
        iter_result["miracles_collected"] = len(miracles)

        if len(miracles) < 5:
            print(f"  Too few miracles ({len(miracles)}), skipping fine-tuning")
            iter_result["skipped"] = True
            results["iterations"].append(iter_result)
            continue

        # Convert to training data
        training_texts = trajectories_to_sft_data(miracles, tok)
        iter_result["training_examples"] = len(training_texts)
        print(f"  Training examples: {len(training_texts)}")

        # Fine-tune
        print("\nStep 2: QLoRA Fine-tuning...")
        try:
            model = finetune_qlora(model, tok, training_texts)
        except Exception as e:
            print(f"  Fine-tuning error: {e}")
            iter_result["error"] = str(e)
            results["iterations"].append(iter_result)
            continue

        # Re-register hook on new model
        hook.remove()
        hook = NoiseHook()
        hook.register(model)

        # Evaluate
        print("\nStep 3: Evaluating...")
        k1_rate = evaluate(model, tok, hook, N_EVAL, f"Iter{iteration+1}")
        iter_result["k1_solve_rate"] = k1_rate
        iter_result["improvement_pp"] = (k1_rate - baseline) * 100

        results["iterations"].append(iter_result)

        # Save intermediate
        with open(os.path.join(RESULTS_DIR, "phase32b_llm_exit.json"), 'w') as f:
            json.dump(results, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    out_path = os.path.join(RESULTS_DIR, "phase32b_llm_exit.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 32b SUMMARY: LLM-ExIt")
    print("=" * 60)
    print(f"  Baseline: {100*baseline:.1f}%")
    for ir in results["iterations"]:
        if "k1_solve_rate" in ir:
            print(f"  Iter {ir['iteration']}: {100*ir['k1_solve_rate']:.1f}% (Δ={ir['improvement_pp']:+.1f}pp)")
        elif ir.get("skipped"):
            print(f"  Iter {ir['iteration']}: SKIPPED (miracles={ir['miracles_collected']})")
    print("=" * 60)

    hook.remove()
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
