"""
Phase 7: Reasoning Phase Portrait
====================================

Map the complete reasoning state space during Modified Hanoi problem-solving.
Capture hidden-state trajectories at every game step, build phase portraits,
and identify attractors, divergence points, and dynamical structures.

Architecture:
  - Qwen2.5-0.5B-Instruct (24 layers, hidden_dim=896)
  - Layer 14 capture (proportional to Mistral L18/32)
  - Modified Hanoi 3-disk

Conditions (N=30 each):
  1. baseline:          No injection
  2. trajectory:        Trajectory Template injection at L14
  3. anti_trajectory:   Reversed trajectory injection at L14

Analysis:
  - PCA phase portrait (2D projection of all hidden states)
  - Divergence analysis: when do success/fail trajectories separate?
  - Velocity profile: how fast do states change between steps?
  - Attractor analysis: do solved games converge to a common state?
  - Injection effect: how does steering reshape the state space?

Qwen2.5-0.5B-Instruct
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.decomposition import PCA

# === Config ===
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SHORT = "Qwen-0.5B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 896       # Qwen-0.5B hidden dim
TARGET_LAYER = 14      # proportional to Mistral L18/32
BASE_SIGMA = 0.15
N_GAMES = 30           # Games per condition (dense capture → moderate N)

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
#  MODEL LOADING
# ===================================================

def load_model():
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
    model.eval()

    n_layers = len(model.model.layers)
    print(f"  Done: {n_layers} layers, hidden_dim={model.config.hidden_size}")
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
#  TRAJECTORY CAPTURE HOOK
# ===================================================

class TrajectoryCaptureHook:
    """Capture one hidden state per game step (last forward pass in generate()).

    During model.generate(), the hook fires once per generated token.
    We only keep the last-captured state from each generate() call as the
    representative state for that game step.
    """

    def __init__(self):
        self.current_state = None   # overwritten on every forward pass
        self.game_trajectory = []   # one state per game step
        self.all_trajectories = []  # completed game trajectories
        self.handle = None

    def register(self, model, layer_idx=TARGET_LAYER):
        hook_obj = self
        def hook_fn(module, args, output):
            hs = output[0]
            if hs.dim() == 3:
                state = hs[0, -1, :].detach().cpu().float().numpy()
            else:
                state = hs[-1, :].detach().cpu().float().numpy()
            hook_obj.current_state = state
        self.handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    def step_start(self):
        """Call before each gen() to reset per-step capture."""
        self.current_state = None

    def step_end(self):
        """Call after each gen() to record the step's representative state."""
        if self.current_state is not None:
            self.game_trajectory.append(self.current_state.copy())

    def new_game(self):
        """Call at the start of a new game."""
        self.game_trajectory = []

    def finish_game(self, solved, steps):
        """Call when game ends. Returns trajectory dict and stores it."""
        traj = np.array(self.game_trajectory) if self.game_trajectory else np.zeros((1, HIDDEN_DIM))
        result = {"states": traj, "solved": solved, "steps": steps}
        self.all_trajectories.append(result)
        self.game_trajectory = []
        return result

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  INJECTION HOOK
# ===================================================

class InjectionHook:
    """Inject trajectory vector at target layer with Flash Annealing."""

    def __init__(self):
        self.active = False
        self.direction = None
        self.sigma = BASE_SIGMA
        self.legal_move_count = 0
        self.handle = None

    def setup(self, direction, device):
        self.active = True
        self.direction = torch.tensor(direction, dtype=torch.float16, device=device)

    def setup_off(self):
        self.active = False

    def reset_game(self):
        self.legal_move_count = 0

    def register(self, model, layer_idx=TARGET_LAYER):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active:
                return args
            # Flash Annealing schedule
            if hook_obj.legal_move_count < 10:
                sigma = hook_obj.sigma * (1.0 - hook_obj.legal_move_count / 10.0)
            else:
                sigma = 0.0
            if sigma <= 0:
                return args
            hs = args[0]
            d = hs.shape[-1]
            det_scale = sigma * math.sqrt(d) * 0.5
            det_noise = hook_obj.direction * det_scale
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
#  GAME FUNCTION (with trajectory capture)
# ===================================================

def play_game_with_capture(model, tok, capture_hook, inject_hook):
    """Play one Modified Hanoi game, capturing hidden states at every step."""
    env = HanoiEnv(n_disks=3, modified=True)
    inject_hook.reset_game()
    capture_hook.new_game()
    error = None
    consec_fail = 0

    for step in range(MAX_STEPS):
        capture_hook.step_start()
        prompt = build_chat_prompt(tok, env, error)
        resp = gen(model, tok, prompt)
        capture_hook.step_end()

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
    capture_hook.finish_game(env.is_solved(), step + 1)
    return stats


# ===================================================
#  ANALYSIS FUNCTIONS
# ===================================================

def compute_divergence_profile(trajectories):
    """Step-by-step divergence between success and failure trajectories."""
    success_trajs = [t for t in trajectories if t["solved"]]
    failure_trajs = [t for t in trajectories if not t["solved"]]

    if not success_trajs or not failure_trajs:
        return None

    max_steps = max(t["states"].shape[0] for t in trajectories)
    divergence = []

    for step in range(min(max_steps, MAX_STEPS)):
        success_states = [t["states"][step] for t in success_trajs if step < t["states"].shape[0]]
        failure_states = [t["states"][step] for t in failure_trajs if step < t["states"].shape[0]]

        if success_states and failure_states:
            s_mean = np.mean(success_states, axis=0)
            f_mean = np.mean(failure_states, axis=0)
            cos_sim = np.dot(s_mean, f_mean) / (np.linalg.norm(s_mean) * np.linalg.norm(f_mean) + 1e-8)
            l2_dist = np.linalg.norm(s_mean - f_mean)
            divergence.append({
                "step": step,
                "cosine_similarity": round(float(cos_sim), 6),
                "l2_distance": round(float(l2_dist), 4),
                "n_success": len(success_states),
                "n_failure": len(failure_states),
            })

    return divergence


def compute_velocity_profile(trajectories):
    """State-space velocity: L2 distance between consecutive hidden states."""
    profiles = {"success": [], "failure": []}

    for traj in trajectories:
        states = traj["states"]
        if states.shape[0] < 2:
            continue
        velocities = [float(np.linalg.norm(states[i] - states[i - 1])) for i in range(1, states.shape[0])]
        key = "success" if traj["solved"] else "failure"
        profiles[key].append(velocities)

    result = {}
    for key in ["success", "failure"]:
        if not profiles[key]:
            continue
        max_len = max(len(v) for v in profiles[key])
        avg_vel = []
        for step in range(max_len):
            vals = [v[step] for v in profiles[key] if step < len(v)]
            avg_vel.append(round(float(np.mean(vals)), 4))
        result[key] = avg_vel

    return result


def compute_attractor_analysis(trajectories):
    """Do solved games converge to a common final state (attractor)?"""
    final_solved = [t["states"][-1] for t in trajectories if t["solved"]]
    final_failed = [t["states"][-1] for t in trajectories if not t["solved"]]

    result = {}
    for label, finals in [("solved", final_solved), ("failed", final_failed)]:
        if len(finals) < 2:
            continue
        arr = np.array(finals)
        result[f"{label}_centroid_variance"] = round(float(np.mean(np.var(arr, axis=0))), 6)
        # Pairwise cosine similarity
        sims = []
        for i in range(len(finals)):
            for j in range(i + 1, len(finals)):
                s = np.dot(finals[i], finals[j]) / (np.linalg.norm(finals[i]) * np.linalg.norm(finals[j]) + 1e-8)
                sims.append(s)
        result[f"{label}_pairwise_cosine_mean"] = round(float(np.mean(sims)), 6)
        result[f"n_{label}"] = len(finals)

    return result


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results, all_condition_trajectories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    condition_names = list(all_condition_trajectories.keys())

    # ---- Collect ALL states for a single global PCA ----
    all_states = []
    meta = []   # (cond_idx, traj_idx, step_idx, solved)

    for ci, cname in enumerate(condition_names):
        for ti, traj in enumerate(all_condition_trajectories[cname]):
            for si in range(traj["states"].shape[0]):
                all_states.append(traj["states"][si])
                meta.append((ci, ti, si, traj["solved"]))

    all_states = np.array(all_states)

    pca = PCA(n_components=2, random_state=SEED)
    states_2d = pca.fit_transform(all_states)
    explained = pca.explained_variance_ratio_

    all_results["pca"] = {
        "explained_variance": [round(float(v), 6) for v in explained],
        "n_total_states": len(all_states),
    }

    # Build per-condition 2D trajectories
    idx = 0
    traj_2d = {}
    for ci, cname in enumerate(condition_names):
        traj_2d[cname] = []
        for ti, traj in enumerate(all_condition_trajectories[cname]):
            n_pts = traj["states"].shape[0]
            xy = states_2d[idx:idx + n_pts]
            traj_2d[cname].append({"xy": xy, "solved": traj["solved"]})
            idx += n_pts

    # ==== Figure 1: Phase Portrait (one panel per condition) ====
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle("Phase 7 · Reasoning Phase Portrait\n"
                 "Hidden-state trajectories through PCA space during Modified Hanoi",
                 fontsize=13, fontweight="bold")

    cond_colors_ok = {"baseline": "#4CAF50", "trajectory": "#2196F3", "anti_trajectory": "#FF9800"}
    cond_colors_ng = {"baseline": "#E53935", "trajectory": "#E53935", "anti_trajectory": "#E53935"}

    for ci, cname in enumerate(condition_names):
        ax = axes[ci]
        trajs = traj_2d[cname]

        for t in trajs:
            color = cond_colors_ok.get(cname, "#4CAF50") if t["solved"] else cond_colors_ng.get(cname, "#E53935")
            alpha = 0.7 if t["solved"] else 0.3
            lw = 1.5 if t["solved"] else 0.8
            ax.plot(t["xy"][:, 0], t["xy"][:, 1], color=color, alpha=alpha, linewidth=lw)
            # Start (circle) and end (star) markers
            ax.scatter(t["xy"][0, 0], t["xy"][0, 1], color=color, s=30, marker="o", alpha=alpha, zorder=5)
            ax.scatter(t["xy"][-1, 0], t["xy"][-1, 1], color=color, s=60, marker="*", alpha=alpha, zorder=5)

        n_ok = sum(1 for t in trajs if t["solved"])
        ax.set_title(f"{cname.replace('_', ' ').title()}\n"
                     f"{n_ok}/{len(trajs)} solved ({n_ok / len(trajs) * 100:.0f}%)",
                     fontweight="bold", fontsize=10)
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path1 = os.path.join(FIGURES_DIR, "phase7_phase_portrait.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {path1}")

    # ==== Figure 2: Four-panel analysis ====
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Phase 7 · Reasoning Dynamics Analysis", fontsize=13, fontweight="bold")

    cc = {"baseline": "#9E9E9E", "trajectory": "#2196F3", "anti_trajectory": "#FF9800"}

    # --- Panel 1: Divergence profile ---
    ax = axes[0, 0]
    for cname in condition_names:
        div = all_results.get("divergence", {}).get(cname)
        if div:
            steps = [d["step"] for d in div]
            dists = [d["l2_distance"] for d in div]
            ax.plot(steps, dists, color=cc.get(cname, "#000"), linewidth=2,
                    label=cname.replace("_", " ").title())
    ax.set_xlabel("Game Step")
    ax.set_ylabel("L2 Distance (success vs failure mean)")
    ax.set_title("Trajectory Divergence Profile", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Panel 2: Velocity profile ---
    ax = axes[0, 1]
    for cname in condition_names:
        vel = all_results.get("velocity", {}).get(cname, {})
        for outcome, ls in [("success", "-"), ("failure", "--")]:
            if outcome in vel:
                ax.plot(vel[outcome], color=cc.get(cname, "#000"), linestyle=ls, linewidth=1.5,
                        label=f"{cname.replace('_', ' ')} ({outcome})")
    ax.set_xlabel("Step")
    ax.set_ylabel("State-Space Velocity (L2)")
    ax.set_title("State-Space Velocity Profile", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # --- Panel 3: Attractor convergence ---
    ax = axes[1, 0]
    attractor_data = all_results.get("attractor", {})
    if attractor_data:
        x_labels = []
        solved_sims = []
        failed_sims = []
        for cname in condition_names:
            att = attractor_data.get(cname, {})
            x_labels.append(cname.replace("_", "\n"))
            solved_sims.append(att.get("solved_pairwise_cosine_mean", 0))
            failed_sims.append(att.get("failed_pairwise_cosine_mean", 0))
        x = np.arange(len(x_labels))
        w = 0.35
        ax.bar(x - w / 2, solved_sims, w, label="Solved", color="#4CAF50", alpha=0.85)
        ax.bar(x + w / 2, failed_sims, w, label="Failed", color="#E53935", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("Pairwise Cosine Similarity")
        ax.set_title("Final-State Convergence (Attractor Strength)", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # --- Panel 4: Solve rates ---
    ax = axes[1, 1]
    test = all_results.get("test_results", [])
    if test:
        names = [t["condition"].replace("_", "\n") for t in test]
        rates = [t["solve_rate"] * 100 for t in test]
        cs = [cc.get(t["condition"], "#000") for t in test]
        bars = ax.bar(range(len(test)), rates, color=cs, alpha=0.85, edgecolor="white", linewidth=2)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(test)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Solve Rate (%)")
        ax.set_title("3-Way Comparison", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    for row in axes:
        for a in row:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

    plt.tight_layout()
    path2 = os.path.join(FIGURES_DIR, "phase7_dynamics_analysis.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {path2}")

    return path1, path2


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
    print(f"  Phase 7: Reasoning Phase Portrait")
    print(f"  3 conditions × N={N_GAMES}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load trajectory template (Mistral 4096-dim → Qwen 896-dim)
    traj_path = os.path.join(DATA_DIR, "trajectory_template.npz")
    if os.path.exists(traj_path):
        traj_data = np.load(traj_path)
        traj_unit_full = traj_data["trajectory_unit"]   # (4096,) Mistral dim
        # Linear interpolation to Qwen's 896-dim
        indices = np.linspace(0, len(traj_unit_full) - 1, HIDDEN_DIM).astype(int)
        traj_unit = traj_unit_full[indices].astype(np.float32)
        traj_unit = traj_unit / (np.linalg.norm(traj_unit) + 1e-8)
        print(f"  Trajectory template loaded and projected to {HIDDEN_DIM}-dim")
    else:
        print(f"  WARNING: No trajectory template, using random direction")
        traj_unit = np.random.randn(HIDDEN_DIM).astype(np.float32)
        traj_unit /= np.linalg.norm(traj_unit)

    model, tok = load_model()
    device = next(model.parameters()).device

    # Setup hooks
    capture_hook = TrajectoryCaptureHook()
    capture_hook.register(model, TARGET_LAYER)

    inject_hook = InjectionHook()
    inject_hook.register(model, TARGET_LAYER)

    conditions = [
        {"name": "baseline",        "direction": None},
        {"name": "trajectory",      "direction": traj_unit},
        {"name": "anti_trajectory",  "direction": -traj_unit},
    ]

    all_results = {
        "experiment": "Phase 7: Reasoning Phase Portrait",
        "model": MODEL_SHORT,
        "target_layer": TARGET_LAYER,
        "sigma": BASE_SIGMA,
        "n_games": N_GAMES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase7_log.json")

    all_condition_trajectories = {}
    test_results = []

    for cfg in conditions:
        cname = cfg["name"]
        print(f"\n  === Condition: {cname} ===")

        # Setup injection
        if cfg["direction"] is not None:
            inject_hook.setup(cfg["direction"], device)
        else:
            inject_hook.setup_off()

        # Reset trajectory storage
        capture_hook.all_trajectories = []

        games = []
        for trial in range(N_GAMES):
            stats = play_game_with_capture(model, tok, capture_hook, inject_hook)
            games.append(stats)
            if (trial + 1) % 10 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                print(f"    [{trial+1}/{N_GAMES}] {sr:.1f}%")

        solved = sum(1 for g in games if g["solved"])
        print(f"    Result: {solved}/{N_GAMES} = {solved/N_GAMES*100:.1f}%")

        test_results.append({
            "condition": cname,
            "solve_rate": round(solved / N_GAMES, 4),
            "n_solved": solved,
            "n_total": N_GAMES,
        })

        all_condition_trajectories[cname] = list(capture_hook.all_trajectories)

        # Save intermediate
        all_results["test_results"] = test_results
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Remove hooks (model no longer needed for analysis)
    capture_hook.remove()
    inject_hook.remove()

    # === Analysis ===
    print(f"\n  === Analysis ===")

    # Divergence analysis
    divergence = {}
    for cname, trajs in all_condition_trajectories.items():
        div = compute_divergence_profile(trajs)
        if div:
            divergence[cname] = div
            max_div = max(div, key=lambda d: d["l2_distance"])
            print(f"    {cname}: max divergence at step {max_div['step']} (L2={max_div['l2_distance']:.3f})")
    all_results["divergence"] = divergence

    # Velocity profile
    velocity = {}
    for cname, trajs in all_condition_trajectories.items():
        vel = compute_velocity_profile(trajs)
        velocity[cname] = vel
        if "success" in vel and vel["success"]:
            print(f"    {cname}: avg velocity (success) = {np.mean(vel['success']):.3f}")
    all_results["velocity"] = velocity

    # Attractor analysis
    attractor = {}
    for cname, trajs in all_condition_trajectories.items():
        att = compute_attractor_analysis(trajs)
        attractor[cname] = att
        if "solved_pairwise_cosine_mean" in att:
            print(f"    {cname}: solved attractor strength = {att['solved_pairwise_cosine_mean']:.4f}")
    all_results["attractor"] = attractor

    # Verdict
    bl = test_results[0]
    traj_r = test_results[1]
    anti_r = test_results[2]

    if traj_r["solve_rate"] > bl["solve_rate"] + 0.05:
        if anti_r["solve_rate"] < bl["solve_rate"] - 0.03:
            verdict = "BIDIRECTIONAL_CONTROL_CONFIRMED"
        else:
            verdict = "TRAJECTORY_ENHANCEMENT_CONFIRMED"
    elif anti_r["solve_rate"] < bl["solve_rate"] - 0.05:
        verdict = "ANTI_TRAJECTORY_DISRUPTION_CONFIRMED"
    else:
        verdict = "STATE_SPACE_MAPPED_NO_CLEAR_SEPARATION"

    all_results["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")

    # Visualize
    fig1, fig2 = visualize(all_results, all_condition_trajectories)
    all_results["figures"] = [fig1, fig2]

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)

    # Save final results (without large trajectory arrays)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save PCA projection data separately
    pca_save_path = os.path.join(RESULTS_DIR, "phase7_pca_projection.npz")
    save_dict = {}
    for cname, trajs in all_condition_trajectories.items():
        for ti, traj in enumerate(trajs):
            save_dict[f"{cname}_game{ti}"] = traj["states"]
            save_dict[f"{cname}_game{ti}_solved"] = np.array([traj["solved"]])
    np.savez_compressed(pca_save_path, **save_dict)

    print(f"\n  Results saved: {results_path}")
    print(f"  PCA data saved: {pca_save_path}")
    print(f"  Total elapsed: {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")

    return all_results, elapsed


if __name__ == "__main__":
    main()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\n Phase 7 complete.")
