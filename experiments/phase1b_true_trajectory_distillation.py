"""
Phase 1b: True Latent Trajectory Distillation (Mistral-7B)
============================================================

Killer Experiment #1: Breaks the ceiling effect from Phase 1 by using
Mistral-7B instead of Qwen-0.5B. The original Phase 1 failed because
Qwen-0.5B's base capability was too low — the model couldn't solve
Modified Hanoi even with SFT, so trajectory alignment had nothing to amplify.

Key improvements over Phase 1:
  - Mistral-7B (7B params, much higher base capability)
  - Native 4096-dim matching trajectory template (no projection artifacts)
  - Stronger trajectory penalty (λ=0.2 vs λ=0.1)
  - More SFT examples (50 vs 30)
  - 4th condition: SFT+Trajectory + runtime Aha! injection (synergy test)

Architecture:
  - Mistral-7B-Instruct-v0.3 + LoRA (rank=16, q_proj + v_proj)
  - SFT on BFS-optimal Modified Hanoi solutions
  - Loss = CE + λ × (1 - cos_sim(hidden[L18], trajectory_template))
  - L18 target layer (native Aha! layer from SNN-Genesis)

Conditions (N=100 each, Modified Hanoi):
  1. baseline:                Original Mistral-7B (no SFT)
  2. sft_only:                Standard SFT (CE loss only)
  3. sft_trajectory:          SFT + trajectory alignment penalty
  4. sft_trajectory_inject:   SFT+trajectory model + runtime Aha! injection

Mistral-7B-Instruct-v0.3
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, math, copy
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from scipy.stats import fisher_exact

# === Config ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 4096
TARGET_LAYER = 18
BASE_SIGMA = 0.15
N_TEST = 100
N_SFT_EXAMPLES = 50
N_SFT_EPOCHS = 3
LAMBDA_TRAJ = 0.2          # Stronger than Phase 1's 0.1
LORA_RANK = 16

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  HANOI ENVIRONMENT
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
            self.illegal_count += 1; self._prev_illegal = True
            return False, "Invalid peg"
        if not self.pegs[from_p]:
            self.illegal_count += 1; self._prev_illegal = True
            return False, f"{from_p} is empty"
        disk = self.pegs[from_p][-1]
        if self.pegs[to_p]:
            top = self.pegs[to_p][-1]
            if self.modified and disk <= top:
                self.illegal_count += 1; self._prev_illegal = True
                return False, "Illegal"
            if not self.modified and disk >= top:
                self.illegal_count += 1; self._prev_illegal = True
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
#  PROMPT & PARSER
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


def build_sft_pair(tokenizer, env, correct_move):
    """Build a prompt-response pair for SFT training."""
    prompt = build_chat_prompt(tokenizer, env)
    response = f"Think: Following modified rules. Move: {correct_move}"
    return prompt, response


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
#  MODEL LOADING
# ===================================================

def load_model(add_lora=False):
    print(f"\n Loading {MODEL_NAME} (LoRA={add_lora})...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        local_files_only=True)

    n_layers = len(model.model.layers)
    print(f"  Done: {n_layers} layers, hidden_dim={model.config.hidden_size}")

    if add_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

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
#  SFT DATA GENERATION (via BFS solver)
# ===================================================

def solve_modified_hanoi_bfs():
    """BFS solver for 3-disk modified Hanoi → optimal solution steps."""
    from collections import deque
    initial = (tuple(range(3, 0, -1)), (), ())

    def get_legal_moves(state):
        pegs = list(state)
        moves = []
        for f in range(3):
            for t in range(3):
                if f != t and pegs[f]:
                    disk = pegs[f][-1]
                    if not pegs[t] or disk > pegs[t][-1]:
                        moves.append((f, t))
        return moves

    def apply_move(state, f, t):
        pegs = [list(p) for p in state]
        disk = pegs[f].pop()
        pegs[t].append(disk)
        return tuple(tuple(p) for p in pegs)

    goal = ((), (), (1, 2, 3))
    queue = deque([(initial, [])])
    visited = {initial}

    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        for f, t in get_legal_moves(state):
            new_state = apply_move(state, f, t)
            if new_state not in visited:
                visited.add(new_state)
                peg_names = "ABC"
                queue.append((new_state, path + [(peg_names[f], peg_names[t])]))
    return []


def generate_sft_dataset(tok, n_examples=N_SFT_EXAMPLES):
    """Generate SFT training pairs from BFS-optimal solutions with variations."""
    solution_path = solve_modified_hanoi_bfs()
    print(f"  BFS solution: {len(solution_path)} moves")

    dataset = []
    for ex_idx in range(n_examples):
        env = HanoiEnv(n_disks=3, modified=True)
        n_play = random.randint(0, len(solution_path) - 1)
        for i in range(n_play):
            f, t = solution_path[i]
            env.try_move(f, t)

        if n_play < len(solution_path):
            f, t = solution_path[n_play]
            correct_move = f"{f}->{t}"
            prompt, response = build_sft_pair(tok, env, correct_move)
            dataset.append({"prompt": prompt, "response": response, "full": prompt + response})

    print(f"  Generated {len(dataset)} SFT examples")
    return dataset


# ===================================================
#  HIDDEN STATE CAPTURE HOOK
# ===================================================

class HiddenStateCaptureHook:
    def __init__(self):
        self.captured = []
        self.handle = None

    def register(self, model, layer_idx=TARGET_LAYER):
        hook_obj = self
        def hook_fn(module, args, output):
            hs = output[0]
            if hs.dim() == 3:
                state = hs[0, -1, :].detach()
            else:
                state = hs[-1, :].detach()
            hook_obj.captured.append(state)
        self.handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    def reset(self):
        self.captured = []

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  RUNTIME AHA! INJECTION HOOK
# ===================================================

class AhaInjectionHook:
    """Runtime Aha! noise injection with Flash Annealing."""
    def __init__(self):
        self.active = False
        self.diff_offset = None
        self.sigma = BASE_SIGMA
        self.handle = None
        self.legal_move_count = 0

    def setup(self, diff_unit, device):
        self.active = True
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)

    def setup_off(self):
        self.active = False

    def reset_game(self):
        self.legal_move_count = 0

    def _get_sigma(self):
        if self.legal_move_count < 10:
            return self.sigma * (1.0 - self.legal_move_count / 10.0)
        return 0.0

    def register(self, model, layer_idx=TARGET_LAYER):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active:
                return args
            sigma = hook_obj._get_sigma()
            if sigma <= 0:
                return args
            hs = args[0]
            d = hs.shape[-1]
            offset = hook_obj.diff_offset
            det_scale = sigma * math.sqrt(d) * 0.5
            det_noise = offset * det_scale
            if hs.dim() == 3:
                det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
            else:
                det_noise = det_noise.unsqueeze(0).expand_as(hs)
            stoch_noise = torch.randn_like(hs) * (sigma * 0.5)
            return (hs + det_noise + stoch_noise,) + args[1:]
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  SFT TRAINING WITH TRAJECTORY PENALTY
# ===================================================

def train_sft(model, tok, dataset, trajectory_template, use_trajectory_penalty=False):
    """Train with LoRA SFT, optionally with trajectory alignment penalty."""
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=0.01
    )

    # Setup hidden state capture if using trajectory penalty
    capture_hook = None
    if use_trajectory_penalty and trajectory_template is not None:
        capture_hook = HiddenStateCaptureHook()
        # Access base model layers through PEFT wrapper
        base_model = model.base_model.model if hasattr(model, 'base_model') else model
        capture_hook.register(base_model, TARGET_LAYER)
        # Native 4096-dim — no projection needed
        traj_proj = torch.tensor(trajectory_template, dtype=torch.float32).to(model.device)
        traj_proj = traj_proj / (traj_proj.norm() + 1e-8)

    loss_history = []
    for epoch in range(N_SFT_EPOCHS):
        random.shuffle(dataset)
        epoch_loss = 0
        epoch_traj_loss = 0

        for i, item in enumerate(dataset):
            optimizer.zero_grad()

            inputs = tok(item["full"], return_tensors="pt", truncation=True,
                        max_length=512, padding=True).to(model.device)

            if capture_hook:
                capture_hook.reset()

            outputs = model(**inputs, labels=inputs["input_ids"])
            ce_loss = outputs.loss

            traj_loss = torch.tensor(0.0, device=model.device)
            if capture_hook and capture_hook.captured:
                avg_hidden = torch.stack(capture_hook.captured).mean(dim=0).float()
                avg_hidden = avg_hidden / (avg_hidden.norm() + 1e-8)
                traj_loss = 1.0 - torch.cosine_similarity(
                    avg_hidden.unsqueeze(0), traj_proj.unsqueeze(0)
                ).mean()

            total_loss = ce_loss + LAMBDA_TRAJ * traj_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += ce_loss.item()
            epoch_traj_loss += traj_loss.item()

        avg_ce = epoch_loss / len(dataset)
        avg_traj = epoch_traj_loss / len(dataset)
        loss_history.append({"epoch": epoch+1, "ce_loss": round(avg_ce, 4),
                            "traj_loss": round(avg_traj, 4)})
        mode = "SFT+Traj" if use_trajectory_penalty else "SFT-only"
        print(f"    [{mode}] Epoch {epoch+1}/{N_SFT_EPOCHS}: CE={avg_ce:.4f}, Traj={avg_traj:.4f}")

    if capture_hook:
        capture_hook.remove()

    model.eval()
    return loss_history


# ===================================================
#  TEST GAME FUNCTIONS
# ===================================================

def play_game(model, tok, inject_hook=None):
    env = HanoiEnv(n_disks=3, modified=True)
    if inject_hook:
        inject_hook.reset_game()
    error = None
    consec_fail = 0
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
            if inject_hook:
                inject_hook.legal_move_count += 1
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
    return stats


def evaluate_condition(model, tok, condition_name, n_test=N_TEST, inject_hook=None):
    """Run N games and return condition results."""
    print(f"\n  Condition: {condition_name}")
    games = []
    for trial in range(n_test):
        stats = play_game(model, tok, inject_hook)
        games.append(stats)
        if (trial + 1) % 50 == 0:
            sr = sum(1 for g in games if g["solved"]) / len(games) * 100
            print(f"    [{trial+1}/{n_test}] {sr:.1f}%")
    solved = sum(1 for g in games if g["solved"])
    print(f"    {condition_name}: {solved}/{n_test} = {solved/n_test*100:.1f}%")
    return {
        "condition": condition_name,
        "solve_rate": round(solved / n_test, 4),
        "n_solved": solved,
        "n_total": n_test,
        "games": games,
    }


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 1b: True Latent Trajectory Distillation (Mistral-7B)\n"
                 "LoRA SFT + trajectory alignment on 7B-parameter model",
                 fontsize=12, fontweight="bold")

    # Panel 1: Solve rates (4 conditions)
    ax = axes[0]
    test = all_results.get("test_results", [])
    names = [t["condition"] for t in test]
    rates = [t["solve_rate"] * 100 for t in test]
    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#E91E63"]
    bars = ax.bar(range(len(test)), rates, color=colors[:len(test)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("4-Way Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Training loss curves
    ax = axes[1]
    for label, key, color in [("SFT-only", "sft_only_training", "#2196F3"),
                               ("SFT+Trajectory", "sft_trajectory_training", "#FF9800")]:
        hist = all_results.get(key, [])
        if hist:
            epochs = [h["epoch"] for h in hist]
            ce = [h["ce_loss"] for h in hist]
            ax.plot(epochs, ce, color=color, linewidth=2, label=label, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(FIGURES_DIR, "phase1b_true_trajectory_distillation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path}")
    return path


# ===================================================
#  MAIN
# ===================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"\n{'='*80}")
    print(f"  Phase 1b: True Latent Trajectory Distillation (Mistral-7B)")
    print(f"  4 conditions x N={N_TEST}")
    print(f"  λ_traj={LAMBDA_TRAJ}, SFT examples={N_SFT_EXAMPLES}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load trajectory template (native 4096-dim from SNN-Genesis Phase 109)
    traj_path = os.path.join(DATA_DIR, "trajectory_template.npz")
    if os.path.exists(traj_path):
        traj_data = np.load(traj_path)
        trajectory_template = traj_data["trajectory_unit"]
        print(f"  Loaded trajectory template: shape={trajectory_template.shape} (native 4096-dim)")
    else:
        print(f"  WARNING: Trajectory template not found, using random")
        trajectory_template = np.random.randn(HIDDEN_DIM).astype(np.float32)
        trajectory_template /= np.linalg.norm(trajectory_template)

    # Load diff_unit for runtime injection (condition 4)
    diff_pca_path = os.path.join(DATA_DIR, "diff_pca.npz")
    if os.path.exists(diff_pca_path):
        diff_data = np.load(diff_pca_path)
        diff_unit = diff_data["diff_unit"]  # Native 4096-dim
        print(f"  Loaded diff_unit: shape={diff_unit.shape}")
    else:
        print(f"  WARNING: diff_pca.npz not found, using random diff_unit")
        diff_unit = np.random.randn(HIDDEN_DIM).astype(np.float32)
        diff_unit /= np.linalg.norm(diff_unit)

    all_results = {
        "experiment": "Phase 1b: True Latent Trajectory Distillation",
        "model": MODEL_SHORT,
        "target_layer": TARGET_LAYER,
        "lambda_traj": LAMBDA_TRAJ,
        "n_test": N_TEST,
        "n_sft_examples": N_SFT_EXAMPLES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase1b_log.json")

    # === Condition 1: Baseline (no SFT) ===
    print(f"\n  === Condition 1: baseline (original Mistral-7B) ===")
    model, tok = load_model(add_lora=False)
    model.eval()

    test_results = []
    result = evaluate_condition(model, tok, "baseline", N_TEST)
    baseline_solved = result["n_solved"]
    test_results.append(result)

    all_results["test_results"] = test_results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Generate SFT dataset ===
    print(f"\n  === Generating SFT dataset ===")
    tmp_tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)
    if tmp_tok.pad_token is None:
        tmp_tok.pad_token = tmp_tok.eos_token
    sft_dataset = generate_sft_dataset(tmp_tok)
    del tmp_tok

    # === Condition 2: SFT-only ===
    print(f"\n  === Condition 2: sft_only ===")
    model, tok = load_model(add_lora=True)
    sft_only_history = train_sft(model, tok, sft_dataset, trajectory_template,
                                  use_trajectory_penalty=False)
    all_results["sft_only_training"] = sft_only_history

    result = evaluate_condition(model, tok, "sft_only", N_TEST)
    sft_solved = result["n_solved"]
    test_results.append(result)

    all_results["test_results"] = test_results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Condition 3: SFT + Trajectory penalty ===
    print(f"\n  === Condition 3: sft_trajectory ===")
    model, tok = load_model(add_lora=True)
    sft_traj_history = train_sft(model, tok, sft_dataset, trajectory_template,
                                  use_trajectory_penalty=True)
    all_results["sft_trajectory_training"] = sft_traj_history

    result = evaluate_condition(model, tok, "sft_trajectory", N_TEST)
    traj_solved = result["n_solved"]
    test_results.append(result)

    all_results["test_results"] = test_results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # === Condition 4: SFT+Trajectory + runtime Aha! injection ===
    # Keep the same SFT+trajectory model, add runtime injection hook
    print(f"\n  === Condition 4: sft_trajectory + runtime injection ===")
    inject_hook = AhaInjectionHook()
    # Access base model through PEFT wrapper for hook registration
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    inject_hook.register(base_model, TARGET_LAYER)
    inject_hook.setup(diff_unit, next(model.parameters()).device)

    result = evaluate_condition(model, tok, "sft_trajectory_inject", N_TEST, inject_hook)
    test_results.append(result)

    inject_hook.remove()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    all_results["test_results"] = test_results

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl = test_results[0]
    for tr in test_results:
        delta = tr["solve_rate"] - bl["solve_rate"]
        print(f"    {tr['condition']:25s}: {tr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Fisher exact tests
    for i in range(1, len(test_results)):
        tr = test_results[i]
        if tr["n_solved"] != baseline_solved:
            alt = "greater" if tr["n_solved"] > baseline_solved else "less"
            table = [[tr["n_solved"], N_TEST - tr["n_solved"]],
                     [baseline_solved, N_TEST - baseline_solved]]
            _, pval = fisher_exact(table, alternative=alt)
            key = f"fisher_p_{tr['condition']}_vs_baseline"
            all_results[key] = round(pval, 6)
            print(f"    Fisher p ({tr['condition']} vs baseline): {pval:.6f}")

    # Verdict
    best = max(test_results, key=lambda t: t["solve_rate"])
    traj_rate = test_results[2]["solve_rate"] if len(test_results) > 2 else 0
    sft_rate = test_results[1]["solve_rate"] if len(test_results) > 1 else 0
    bl_rate = bl["solve_rate"]

    if traj_rate > sft_rate + 0.03 and traj_rate > bl_rate + 0.05:
        verdict = "TRAJECTORY_DISTILLATION_BREAKTHROUGH"
    elif traj_rate > bl_rate + 0.05:
        verdict = "TRAJECTORY_HELPS_BUT_NOT_BEYOND_SFT"
    elif sft_rate > bl_rate + 0.05:
        verdict = "SFT_HELPS_TRAJECTORY_NO_EXTRA"
    else:
        verdict = "CEILING_EFFECT_PERSISTS"

    # Check synergy (condition 4)
    if len(test_results) > 3:
        synergy_rate = test_results[3]["solve_rate"]
        if synergy_rate > max(traj_rate, sft_rate) + 0.03:
            verdict += "_WITH_SYNERGY"
            print(f"  SYNERGY: Runtime injection + SFT+Trajectory = {synergy_rate*100:.1f}%")

    all_results["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    # Compare with Phase 1 Qwen results
    phase1_path = os.path.join(RESULTS_DIR, "phase1_log.json")
    if os.path.exists(phase1_path):
        with open(phase1_path) as f:
            p1_data = json.load(f)
        p1_test = p1_data.get("test_results", [])
        if p1_test:
            p1_traj = next((t for t in p1_test if t["condition"] == "sft_trajectory"), None)
            if p1_traj:
                all_results["phase1_comparison"] = {
                    "qwen_0.5B_traj_rate": p1_traj["solve_rate"],
                    "mistral_7B_traj_rate": traj_rate,
                    "improvement_pp": round(traj_rate - p1_traj["solve_rate"], 4),
                }
                print(f"  vs Phase 1: Qwen={p1_traj['solve_rate']*100:.1f}% → "
                      f"Mistral={traj_rate*100:.1f}%")

    fig_path = visualize(all_results)
    all_results["figure"] = fig_path

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")
    print(f"  Total elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")

    return all_results, elapsed


if __name__ == "__main__":
    main()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n Phase 1b complete.")
