"""
Phase 2: Dual-Layer Cognitive Orchestration
=============================================

Simultaneous steering at two different layers:
  - L16: IID random noise (Prior destruction — break fixed beliefs)
  - L18: Trajectory Template injection (Aha! creation — guide to success)

Architecture:
  - Mistral-7B-Instruct-v0.3
  - Two separate pre-hooks: one on L16, one on L18
  - Each with independent σ control

Conditions (N=100 each, Modified Hanoi):
  1. baseline:           No injection
  2. L16_only:           IID noise at L16 (σ=0.15)
  3. L18_only:           Trajectory template at L18 (σ=0.15)
  4. dual_simultaneous:  Both L16 + L18
  5. aha_noise_L18:      Phase 108 control (Aha!+noise at L18 only)

Mistral-7B-Instruct-v0.3
"""

import torch
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import fisher_exact

# === Config ===
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 4096
BASE_SIGMA = 0.15
LAYER_L16 = 16
LAYER_L18 = 18
N_TEST = 100

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
#  MODEL + GENERATION
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
    print(f"  Done: {len(model.model.layers)} layers")
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
#  DUAL-LAYER HOOK SYSTEM
# ===================================================

class DualLayerHook:
    """Two independent hooks for L16 (destruction) and L18 (creation)."""
    def __init__(self):
        self.l16_active = False
        self.l18_active = False
        self.sigma_l16 = BASE_SIGMA
        self.sigma_l18 = BASE_SIGMA
        self.diff_offset = None      # Aha! direction (for L18)
        self.traj_offset = None      # Trajectory template (for L18)
        self.handle_l16 = None
        self.handle_l18 = None
        self.legal_move_count = 0

    def setup(self, mode, diff_unit, traj_unit, device):
        """
        Modes:
          off            — no injection
          L16_only       — IID noise at L16
          L18_only       — trajectory template at L18
          dual           — both L16 + L18
          aha_noise_L18  — Aha!+noise at L18 (Phase 108 control)
        """
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        self.traj_offset = torch.tensor(traj_unit, dtype=torch.float16, device=device)
        self.l16_active = mode in ("L16_only", "dual")
        self.l18_active = mode in ("L18_only", "dual", "aha_noise_L18")
        self.mode = mode

    def reset_game(self):
        self.legal_move_count = 0

    def _get_sigma(self, base_sigma):
        """Flash Annealing schedule."""
        if self.legal_move_count < 10:
            return base_sigma * (1.0 - self.legal_move_count / 10.0)
        return 0.0

    def register(self, model):
        hook_obj = self

        # L16 hook: IID random noise (prior destruction)
        def hook_l16(module, args):
            if not hook_obj.l16_active:
                return args
            sigma = hook_obj._get_sigma(hook_obj.sigma_l16)
            if sigma <= 0:
                return args
            hs = args[0]
            # Pure IID noise (no directional component — destroy, don't guide)
            noise = torch.randn_like(hs) * sigma
            return (hs + noise,) + args[1:]

        # L18 hook: directed trajectory injection (Aha! creation)
        def hook_l18(module, args):
            if not hook_obj.l18_active:
                return args
            sigma = hook_obj._get_sigma(hook_obj.sigma_l18)
            if sigma <= 0:
                return args
            hs = args[0]
            d = hs.shape[-1]

            if hook_obj.mode == "aha_noise_L18":
                # Phase 108 control: Aha! direction + stochastic noise
                offset = hook_obj.diff_offset
            else:
                # Trajectory template direction
                offset = hook_obj.traj_offset

            det_scale = sigma * math.sqrt(d) * 0.5
            det_noise = offset * det_scale
            if hs.dim() == 3:
                det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
            else:
                det_noise = det_noise.unsqueeze(0).expand_as(hs)
            stoch_noise = torch.randn_like(hs) * (sigma * 0.5)
            return (hs + det_noise + stoch_noise,) + args[1:]

        self.handle_l16 = model.model.layers[LAYER_L16].register_forward_pre_hook(hook_l16)
        self.handle_l18 = model.model.layers[LAYER_L18].register_forward_pre_hook(hook_l18)

    def remove(self):
        if self.handle_l16:
            self.handle_l16.remove()
            self.handle_l16 = None
        if self.handle_l18:
            self.handle_l18.remove()
            self.handle_l18 = None


# ===================================================
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, hook, use_annealing=True):
    env = HanoiEnv(n_disks=3, modified=True)
    hook.reset_game()
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
            hook.legal_move_count += 1
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Phase 2: Dual-Layer Cognitive Orchestration\n"
                 "L16 (Prior Destruction) + L18 (Aha! Creation)",
                 fontsize=12, fontweight="bold")

    # Panel 1: Solve rates
    ax = axes[0]
    test = all_results.get("test_results", [])
    names = [t["condition"] for t in test]
    rates = [t["solve_rate"] * 100 for t in test]
    colors = ["#9E9E9E", "#E91E63", "#4CAF50", "#FF9800", "#2196F3"]
    bars = ax.bar(range(len(test)), rates, color=colors[:len(test)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("5-Way Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Illegal move rates
    ax = axes[1]
    avg_illegals = []
    for t in test:
        ill = [g["illegal_moves"] for g in t["games"]]
        avg_illegals.append(np.mean(ill))
    bars2 = ax.bar(range(len(test)), avg_illegals, color=colors[:len(test)], alpha=0.6,
                   edgecolor="white", linewidth=2)
    for bar, val in zip(bars2, avg_illegals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Avg Illegal Moves")
    ax.set_title("Error Rate by Condition", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase2_n100_dual_layer_orchestration.png")
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
    print(f"  Phase 2: Dual-Layer Cognitive Orchestration")
    print(f"  5 conditions x N={N_TEST}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load artifacts (from SNN-Genesis, bundled in data/)
    diff_pca_path = os.path.join(DATA_DIR, "diff_pca.npz")
    traj_path = os.path.join(DATA_DIR, "trajectory_template.npz")

    diff_data = np.load(diff_pca_path)
    diff_unit = diff_data["diff_unit"]

    traj_data = np.load(traj_path)
    traj_unit = traj_data["trajectory_unit"]

    print(f"  Diff unit: shape={diff_unit.shape}")
    print(f"  Trajectory unit: shape={traj_unit.shape}")

    model, tok = load_model()
    device = next(model.parameters()).device

    hook = DualLayerHook()
    hook.register(model)

    all_results = {
        "experiment": "Phase 2: Dual-Layer Cognitive Orchestration",
        "model": MODEL_SHORT,
        "layers": {"destruction": LAYER_L16, "creation": LAYER_L18},
        "sigma": BASE_SIGMA,
        "n_test": N_TEST,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase2_n100_log.json")

    # === Test 5 conditions ===
    conditions = [
        {"name": "baseline",          "mode": "off"},
        {"name": "L16_only",          "mode": "L16_only"},
        {"name": "L18_only",          "mode": "L18_only"},
        {"name": "dual_simultaneous", "mode": "dual"},
        {"name": "aha_noise_L18",     "mode": "aha_noise_L18"},
    ]

    test_results = []
    for cfg in conditions:
        print(f"\n  === Condition: {cfg['name']} ===")
        hook.setup(cfg["mode"], diff_unit, traj_unit, device)

        games = []
        for trial in range(N_TEST):
            stats = play_game(model, tok, hook)
            games.append(stats)
            if (trial + 1) % 25 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")

        solved = sum(1 for g in games if g["solved"])
        test_results.append({
            "condition": cfg["name"],
            "solve_rate": round(solved / N_TEST, 4),
            "n_solved": solved,
            "n_total": N_TEST,
            "games": games,
        })
        print(f"    Result: {solved}/{N_TEST} = {solved/N_TEST*100:.1f}%")

        # Save intermediate
        all_results["test_results"] = test_results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl = test_results[0]
    for tr in test_results:
        delta = tr["solve_rate"] - bl["solve_rate"]
        print(f"    {tr['condition']:25s}: {tr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Fisher exact tests
    for i in range(1, len(test_results)):
        tr = test_results[i]
        if tr["n_solved"] != bl["n_solved"]:
            alt = "greater" if tr["n_solved"] > bl["n_solved"] else "less"
            table = [[tr["n_solved"], N_TEST - tr["n_solved"]],
                     [bl["n_solved"], N_TEST - bl["n_solved"]]]
            _, pval = fisher_exact(table, alternative=alt)
            key = f"fisher_p_{tr['condition']}_vs_baseline"
            all_results[key] = round(pval, 6)
            print(f"    Fisher p ({tr['condition']} vs baseline): {pval:.6f}")

    # Check if dual > max(L16_only, L18_only)
    dual_rate = test_results[3]["solve_rate"]
    l16_rate = test_results[1]["solve_rate"]
    l18_rate = test_results[2]["solve_rate"]
    single_best = max(l16_rate, l18_rate)

    if dual_rate > single_best + 0.03:
        verdict = "SYNERGY_CONFIRMED"
        print(f"\n  VERDICT: {verdict} — Dual-layer > best single-layer!")
    elif dual_rate >= single_best - 0.02:
        verdict = "ADDITIVE_NOT_SYNERGISTIC"
        print(f"\n  VERDICT: {verdict} — Dual-layer matches but doesn't exceed single-layer")
    else:
        verdict = "INTERFERENCE_DETECTED"
        print(f"\n  VERDICT: {verdict} — Dual injection hurts performance")

    all_results["verdict"] = verdict
    all_results["analysis"] = {
        "L16_rate": l16_rate,
        "L18_rate": l18_rate,
        "dual_rate": dual_rate,
        "single_best": single_best,
        "synergy_delta": round(dual_rate - single_best, 4),
    }

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
    print(f"\n Phase 2 complete.")
