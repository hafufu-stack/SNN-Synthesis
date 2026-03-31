"""
Phase 1: Latent Trajectory Distillation
=========================================

Distill the Trajectory Template (from Phase 109) into Qwen-0.5B's weights
via LoRA SFT with a trajectory-alignment penalty.

Architecture:
  - Qwen2.5-0.5B-Instruct + LoRA (rank=16)
  - SFT on Modified Hanoi correct solutions (self-generated)
  - Custom Loss = CE(sft) + λ × MSE(hidden_state[L14], trajectory_template)
  - L14 of 24-layer Qwen ≈ L18 of 32-layer Mistral (proportional depth)

Conditions (N=50 each, Modified Hanoi):
  1. baseline:         Original Qwen-0.5B (no SFT)
  2. sft_only:         Standard SFT (CE loss only)
  3. sft_trajectory:   SFT + trajectory alignment penalty

Qwen2.5-0.5B-Instruct
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
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SHORT = "Qwen-0.5B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 896       # Qwen-0.5B hidden dim
TARGET_LAYER = 14      # ≈ L18/32 * 24
BASE_SIGMA = 0.15
N_TEST = 50
N_SFT_EXAMPLES = 30    # SFT training examples
N_SFT_EPOCHS = 3
LAMBDA_TRAJ = 0.1      # Trajectory alignment penalty weight
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
    print(f"\n Loading {MODEL_NAME}...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        local_files_only=True, trust_remote_code=True)

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

    def get_state_key(state):
        return state

    def get_legal_moves(state):
        pegs = list(state)
        moves = []
        for f in range(3):
            for t in range(3):
                if f != t and pegs[f]:
                    disk = pegs[f][-1]
                    if not pegs[t] or disk > pegs[t][-1]:  # Modified: larger onto smaller
                        moves.append((f, t))
        return moves

    def apply_move(state, f, t):
        pegs = [list(p) for p in state]
        disk = pegs[f].pop()
        pegs[t].append(disk)
        return tuple(tuple(p) for p in pegs)

    goal = ((), (), (1, 2, 3))  # Modified: largest on top, smallest at bottom
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
        # Play through random prefix of the solution
        n_play = random.randint(0, len(solution_path) - 1)
        for i in range(n_play):
            f, t = solution_path[i]
            env.try_move(f, t)

        # Next correct move
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
        # Project trajectory template to Qwen's hidden dim if needed
        if trajectory_template.shape[0] != HIDDEN_DIM:
            # Linear interpolation from Mistral 4096 to Qwen 896
            indices = np.linspace(0, len(trajectory_template)-1, HIDDEN_DIM).astype(int)
            traj_proj = torch.tensor(trajectory_template[indices], dtype=torch.float32).to(model.device)
        else:
            traj_proj = torch.tensor(trajectory_template, dtype=torch.float32).to(model.device)
        traj_proj = traj_proj / (traj_proj.norm() + 1e-8)

    loss_history = []
    for epoch in range(N_SFT_EPOCHS):
        random.shuffle(dataset)
        epoch_loss = 0
        epoch_traj_loss = 0

        for i, item in enumerate(dataset):
            optimizer.zero_grad()

            # Tokenize full sequence
            inputs = tok(item["full"], return_tensors="pt", truncation=True,
                        max_length=512, padding=True).to(model.device)

            if capture_hook:
                capture_hook.reset()

            # Forward pass with labels for CE loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            ce_loss = outputs.loss

            # Trajectory alignment penalty
            traj_loss = torch.tensor(0.0, device=model.device)
            if capture_hook and capture_hook.captured:
                # Average hidden states from this forward pass
                avg_hidden = torch.stack(capture_hook.captured).mean(dim=0).float()
                avg_hidden = avg_hidden / (avg_hidden.norm() + 1e-8)
                # MSE with trajectory template (want alignment → minimize distance)
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
#  TEST GAME FUNCTION
# ===================================================

def play_game(model, tok):
    env = HanoiEnv(n_disks=3, modified=True)
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


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 1: Latent Trajectory Distillation\n"
                 "SFT with trajectory-alignment penalty on Qwen-0.5B",
                 fontsize=12, fontweight="bold")

    # Panel 1: Solve rates
    ax = axes[0]
    test = all_results.get("test_results", [])
    names = [t["condition"] for t in test]
    rates = [t["solve_rate"] * 100 for t in test]
    colors = ["#9E9E9E", "#2196F3", "#FF9800"]
    bars = ax.bar(range(len(test)), rates, color=colors[:len(test)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("3-Way Comparison", fontweight="bold")
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

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase1_trajectory_distillation.png")
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
    print(f"  Phase 1: Latent Trajectory Distillation")
    print(f"  3 conditions x N={N_TEST}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load trajectory template (Mistral 4096-dim, from SNN-Genesis Phase 109)
    traj_path = os.path.join(DATA_DIR, "trajectory_template.npz")
    if os.path.exists(traj_path):
        traj_data = np.load(traj_path)
        trajectory_template = traj_data["trajectory_unit"]
        print(f"  Loaded trajectory template: shape={trajectory_template.shape}")
    else:
        print(f"  WARNING: Trajectory template not found, using random")
        trajectory_template = np.random.randn(4096).astype(np.float32)
        trajectory_template /= np.linalg.norm(trajectory_template)

    all_results = {
        "experiment": "Phase 1: Latent Trajectory Distillation",
        "model": MODEL_SHORT,
        "target_layer": TARGET_LAYER,
        "lambda_traj": LAMBDA_TRAJ,
        "n_test": N_TEST,
        "n_sft_examples": N_SFT_EXAMPLES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase1_log.json")

    # === Condition 1: Baseline (no SFT) ===
    print(f"\n  === Condition 1: baseline (original Qwen-0.5B) ===")
    model, tok = load_model(add_lora=False)
    model.eval()

    games = []
    for trial in range(N_TEST):
        stats = play_game(model, tok)
        games.append(stats)
        if (trial + 1) % 25 == 0:
            sr = sum(1 for g in games if g["solved"]) / len(games) * 100
            print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")
    baseline_solved = sum(1 for g in games if g["solved"])
    print(f"    Baseline: {baseline_solved}/{N_TEST} = {baseline_solved/N_TEST*100:.1f}%")

    test_results = [{
        "condition": "baseline",
        "solve_rate": round(baseline_solved / N_TEST, 4),
        "n_solved": baseline_solved,
        "n_total": N_TEST,
        "games": games,
    }]

    # Save intermediate
    all_results["test_results"] = test_results
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Clean up baseline model
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === Generate SFT dataset ===
    print(f"\n  === Generating SFT dataset ===")
    tmp_tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True,
                                            local_files_only=True, trust_remote_code=True)
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

    games = []
    for trial in range(N_TEST):
        stats = play_game(model, tok)
        games.append(stats)
        if (trial + 1) % 25 == 0:
            sr = sum(1 for g in games if g["solved"]) / len(games) * 100
            print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")
    sft_solved = sum(1 for g in games if g["solved"])
    print(f"    SFT-only: {sft_solved}/{N_TEST} = {sft_solved/N_TEST*100:.1f}%")

    test_results.append({
        "condition": "sft_only",
        "solve_rate": round(sft_solved / N_TEST, 4),
        "n_solved": sft_solved,
        "n_total": N_TEST,
        "games": games,
    })

    # Save intermediate
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

    games = []
    for trial in range(N_TEST):
        stats = play_game(model, tok)
        games.append(stats)
        if (trial + 1) % 25 == 0:
            sr = sum(1 for g in games if g["solved"]) / len(games) * 100
            print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")
    traj_solved = sum(1 for g in games if g["solved"])
    print(f"    SFT+Trajectory: {traj_solved}/{N_TEST} = {traj_solved/N_TEST*100:.1f}%")

    test_results.append({
        "condition": "sft_trajectory",
        "solve_rate": round(traj_solved / N_TEST, 4),
        "n_solved": traj_solved,
        "n_total": N_TEST,
        "games": games,
    })

    all_results["test_results"] = test_results

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl = test_results[0]
    for tr in test_results:
        delta = tr["solve_rate"] - bl["solve_rate"]
        print(f"    {tr['condition']:20s}: {tr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Fisher exact: trajectory vs baseline
    if traj_solved > baseline_solved:
        table = [[traj_solved, N_TEST - traj_solved],
                 [baseline_solved, N_TEST - baseline_solved]]
        _, pval = fisher_exact(table, alternative='greater')
        all_results["fisher_p_traj_vs_baseline"] = round(pval, 6)
        print(f"    Fisher p (traj vs baseline): {pval:.6f}")

    # Verdict
    best = max(test_results, key=lambda t: t["solve_rate"])
    if best["condition"] == "sft_trajectory" and best["solve_rate"] > bl["solve_rate"] + 0.05:
        verdict = "TRAJECTORY_DISTILLATION_WORKS"
    elif best["condition"] == "sft_only" and best["solve_rate"] > bl["solve_rate"] + 0.05:
        verdict = "SFT_HELPS_BUT_TRAJECTORY_NO_EXTRA_BENEFIT"
    else:
        verdict = "DISTILLATION_NEEDS_REFINEMENT"
    all_results["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

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
    print(f"\n Phase 1 complete.")
