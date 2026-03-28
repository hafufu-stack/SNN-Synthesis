"""
Phase 3: Long-Horizon SNN Meta-Learning
==========================================

Train an SNN controller via PPO over 5000 trials to learn when/how
to inject Aha! noise. LLM weights are frozen — only SNN is trained.

Architecture:
  - Qwen2.5-0.5B-Instruct (frozen, 4-bit)
  - SNN controller: 2-layer LIF (320→64→2)
  - Input: rolling window of 5 hidden-state Δ vectors (PCA → 64-dim)
  - Output: inject/no-inject decision + σ magnitude
  - PPO training: 5000 trials (100 epochs × 50 trials/epoch)
  - Reward: +1 solved, 0 failed, -0.1 per illegal move, -0.01 per step

Evaluation (N=100 × 3):
  1. baseline:      No injection
  2. fixed_aha:     Fixed σ=0.15 with annealing
  3. ppo_snn:       PPO-trained SNN controller

Qwen2.5-0.5B-Instruct, Layer 14
"""

import torch
import torch.nn as nn
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from scipy.stats import fisher_exact

# === Config ===
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_SHORT = "Qwen-0.5B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 896
BASE_SIGMA = 0.15
LAYER_IDX = 14
PCA_DIM = 64
WINDOW_SIZE = 5
N_PPO_EPOCHS = 100
N_TRIALS_PER_EPOCH = 50
N_TEST = 100

GENESIS_RESULTS = r"C:\Users\kyjan\研究\snn-genesis\results"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  SNN CONTROLLER (Policy Network)
# ===================================================

class LIFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, tau=0.9, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.tau = tau
        self.threshold = threshold

    def forward(self, x, membrane=None):
        if membrane is None:
            membrane = torch.zeros(x.size(0), self.fc.out_features, device=x.device)
        current = self.fc(x)
        membrane = self.tau * membrane + current
        spikes = (membrane >= self.threshold).float()
        membrane = membrane * (1 - spikes)
        return spikes, membrane


class SNNPolicyController(nn.Module):
    """SNN that outputs: [inject_prob, sigma_scale]"""
    def __init__(self, input_dim=PCA_DIM * WINDOW_SIZE, hidden_dim=64):
        super().__init__()
        self.lif1 = LIFLayer(input_dim, hidden_dim, tau=0.9)
        self.lif2 = LIFLayer(hidden_dim, 16, tau=0.85)
        # Policy head: inject probability
        self.policy_head = nn.Linear(16, 2)  # [no_inject, inject]
        # Value head for PPO
        self.value_head = nn.Linear(16, 1)

    def forward(self, x):
        s1, _ = self.lif1(x)
        s2, _ = self.lif2(s1)
        return s2

    def get_action_and_value(self, x):
        features = self.forward(x)
        logits = self.policy_head(features)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_head(features).squeeze(-1)
        return action, log_prob, value, dist.entropy()

    def get_value(self, x):
        features = self.forward(x)
        return self.value_head(features).squeeze(-1)

    def evaluate_actions(self, x, actions):
        features = self.forward(x)
        logits = self.policy_head(features)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_head(features).squeeze(-1)
        return log_probs, values, entropy


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
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True,
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
        local_files_only=True, trust_remote_code=True)
    model.eval()
    print(f"  Done: {len(model.model.layers)} layers, hidden_dim={model.config.hidden_size}")
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
#  ADAPTIVE HOOK WITH SNN CONTROL
# ===================================================

class SNNControlledHook:
    def __init__(self):
        self.mode = "off"  # off, fixed, snn
        self.sigma = BASE_SIGMA
        self.diff_offset = None
        self.handle = None
        self.captured_states = []
        self.window_buffer = []
        self.pca_proj = None
        self.snn = None
        self.device = 'cuda'
        self.legal_move_count = 0
        # PPO rollout buffers
        self.step_states = []
        self.step_actions = []
        self.step_log_probs = []
        self.step_rewards = []

    def setup_off(self):
        self.mode = "off"

    def setup_fixed(self, diff_unit, device):
        self.mode = "fixed"
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        self.device = device

    def setup_snn(self, diff_unit, pca_proj, snn, device):
        self.mode = "snn"
        self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        self.pca_proj = pca_proj
        self.snn = snn
        self.device = device

    def reset_game(self):
        self.captured_states = []
        self.window_buffer = []
        self.legal_move_count = 0
        self.step_states = []
        self.step_actions = []
        self.step_log_probs = []
        self.step_rewards = []

    def _get_sigma(self):
        if self.legal_move_count < 10:
            return self.sigma * (1.0 - self.legal_move_count / 10.0)
        return 0.0

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            hs = args[0]

            # Capture states for SNN
            if hook_obj.mode == "snn":
                if hs.dim() == 3:
                    state = hs[0, -1, :].detach().cpu().float().numpy()
                else:
                    state = hs[-1, :].detach().cpu().float().numpy()
                hook_obj.captured_states.append(state)
                if len(hook_obj.captured_states) >= 2:
                    delta = hook_obj.captured_states[-1] - hook_obj.captured_states[-2]
                    hook_obj.window_buffer.append(delta)
                    if len(hook_obj.window_buffer) > WINDOW_SIZE:
                        hook_obj.window_buffer.pop(0)

            if hook_obj.mode == "off":
                return args

            sigma = hook_obj._get_sigma()
            if sigma <= 0:
                return args

            should_inject = True
            if hook_obj.mode == "snn" and hook_obj.snn is not None:
                if len(hook_obj.window_buffer) >= WINDOW_SIZE:
                    window = np.array(hook_obj.window_buffer[-WINDOW_SIZE:])
                    projected = window @ hook_obj.pca_proj.T
                    flat = projected.flatten()
                    x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(hook_obj.device)

                    with torch.no_grad():
                        action, log_prob, value, _ = hook_obj.snn.get_action_and_value(x)
                    should_inject = action.item() == 1

                    # Store for PPO
                    hook_obj.step_states.append(flat.copy())
                    hook_obj.step_actions.append(action.item())
                    hook_obj.step_log_probs.append(log_prob.item())

            if not should_inject:
                return args

            d = hs.shape[-1]
            offset = hook_obj.diff_offset
            if offset is None:
                return args  # diff_offset not yet initialized (e.g. during PCA fitting)
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
#  GAME FUNCTION
# ===================================================

def play_game(model, tok, hook):
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
#  PPO UPDATE
# ===================================================

def ppo_update(snn, optimizer, rollout_buffer, device, clip_eps=0.2, gamma=0.99):
    """Perform PPO update from collected rollout data."""
    if not rollout_buffer:
        return 0.0

    all_states = []
    all_actions = []
    all_old_log_probs = []
    all_rewards = []

    for episode in rollout_buffer:
        states = episode["states"]
        actions = episode["actions"]
        old_log_probs = episode["log_probs"]
        reward = episode["reward"]

        if not states:
            continue

        # Assign reward to all steps (final reward only)
        n_steps = len(states)
        step_rewards = [0.0] * n_steps
        step_rewards[-1] = reward  # Only final step gets the reward

        all_states.extend(states)
        all_actions.extend(actions)
        all_old_log_probs.extend(old_log_probs)
        all_rewards.extend(step_rewards)

    if not all_states:
        return 0.0

    states_t = torch.tensor(np.array(all_states), dtype=torch.float32).to(device)
    actions_t = torch.tensor(all_actions, dtype=torch.long).to(device)
    old_log_probs_t = torch.tensor(all_old_log_probs, dtype=torch.float32).to(device)

    # Compute returns (simple: just the reward, no discounting for single-step)
    returns_t = torch.tensor(all_rewards, dtype=torch.float32).to(device)

    # Normalize returns
    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    # PPO update (3 mini-epochs)
    total_loss = 0
    for _ in range(3):
        new_log_probs, values, entropy = snn.evaluate_actions(states_t, actions_t)
        advantages = returns_t - values.detach()

        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.functional.mse_loss(values, returns_t)
        entropy_loss = -entropy.mean()

        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(snn.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / 3


# ===================================================
#  PCA FITTING
# ===================================================

def fit_pca(model, tok, hook, n_games=30):
    """Collect hidden-state deltas and fit PCA."""
    print(f"  Fitting PCA from {n_games} games...")
    hook.mode = "snn"
    hook.snn = None  # Disable SNN decisions during collection

    all_deltas = []
    for gi in range(n_games):
        hook.reset_game()
        hook.captured_states = []
        hook.window_buffer = []

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
                error = "Parse fail"
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

        all_deltas.extend(hook.window_buffer)

    if len(all_deltas) < PCA_DIM:
        print(f"  WARNING: Only {len(all_deltas)} deltas, using random projection")
        pca_proj = np.random.randn(PCA_DIM, HIDDEN_DIM).astype(np.float32)
        pca_proj /= np.linalg.norm(pca_proj, axis=1, keepdims=True)
        return pca_proj

    from sklearn.decomposition import PCA
    all_deltas = np.array(all_deltas)
    pca = PCA(n_components=PCA_DIM)
    pca.fit(all_deltas)
    print(f"  PCA fitted: {pca.explained_variance_ratio_[:5].sum()*100:.1f}% var in top 5 PCs")
    return pca.components_.astype(np.float32)


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Phase 3: Long-Horizon SNN Meta-Learning (PPO)\n"
                 "5000-trial SNN controller training",
                 fontsize=12, fontweight="bold")

    # Panel 1: PPO learning curve
    ax = axes[0]
    ppo = all_results.get("ppo_training", {})
    if "epoch_solve_rates" in ppo:
        rates = ppo["epoch_solve_rates"]
        ax.plot(range(1, len(rates)+1), [r*100 for r in rates],
                color="#9C27B0", linewidth=2)
    ax.set_xlabel("PPO Epoch")
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("PPO Training Progress", fontweight="bold")
    ax.grid(alpha=0.3)

    # Panel 2: Final test comparison
    ax = axes[1]
    test = all_results.get("test_results", [])
    names = [t["condition"] for t in test]
    rates = [t["solve_rate"] * 100 for t in test]
    colors = ["#9E9E9E", "#2196F3", "#9C27B0"]
    bars = ax.bar(range(len(test)), rates, color=colors[:len(test)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Final Evaluation (N=100)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase3_ppo_snn.png")
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
    print(f"  Phase 3: Long-Horizon SNN Meta-Learning (PPO)")
    print(f"  {N_PPO_EPOCHS} epochs × {N_TRIALS_PER_EPOCH} trials = {N_PPO_EPOCHS*N_TRIALS_PER_EPOCH} total")
    print(f"{'='*80}")

    t0 = time.time()

    # Load diff-PCA from Genesis
    diff_pca_path = os.path.join(GENESIS_RESULTS, "phase91_diff_pca.npz")
    diff_data = np.load(diff_pca_path)
    diff_unit_full = diff_data["diff_unit"]  # shape=(4096,) for Mistral
    # Project to Qwen's 896-dim
    indices = np.linspace(0, len(diff_unit_full)-1, HIDDEN_DIM).astype(int)
    diff_unit = diff_unit_full[indices].astype(np.float32)
    diff_unit /= (np.linalg.norm(diff_unit) + 1e-8)
    print(f"  Diff unit projected: {diff_unit_full.shape} → {diff_unit.shape}")

    model, tok = load_model()
    device = next(model.parameters()).device

    hook = SNNControlledHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 3: Long-Horizon SNN Meta-Learning",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_ppo_epochs": N_PPO_EPOCHS,
        "n_trials_per_epoch": N_TRIALS_PER_EPOCH,
        "n_test": N_TEST,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase3_log.json")

    # === Step 1: Fit PCA ===
    pca_proj = fit_pca(model, tok, hook, n_games=30)

    # === Step 2: PPO Training ===
    print(f"\n  === PPO Training: {N_PPO_EPOCHS} epochs ===")
    snn = SNNPolicyController(input_dim=PCA_DIM * WINDOW_SIZE, hidden_dim=64).to(device)
    snn_optimizer = torch.optim.Adam(snn.parameters(), lr=3e-4)

    hook.setup_snn(diff_unit, pca_proj, snn, device)
    hook.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)

    epoch_solve_rates = []
    epoch_losses = []

    for epoch in range(N_PPO_EPOCHS):
        snn.eval()  # Eval mode for rollout collection
        rollout_buffer = []
        epoch_solved = 0

        for trial in range(N_TRIALS_PER_EPOCH):
            stats = play_game(model, tok, hook)

            # Compute reward
            reward = 0.0
            if stats["solved"]:
                reward = 1.0
                epoch_solved += 1
            reward -= 0.1 * stats["illegal_moves"]
            reward -= 0.01 * stats["steps_taken"]

            # Store rollout
            if hook.step_states:
                rollout_buffer.append({
                    "states": list(hook.step_states),
                    "actions": list(hook.step_actions),
                    "log_probs": list(hook.step_log_probs),
                    "reward": reward,
                })

        # PPO update
        snn.train()
        ppo_loss = ppo_update(snn, snn_optimizer, rollout_buffer, device)
        epoch_losses.append(ppo_loss)

        solve_rate = epoch_solved / N_TRIALS_PER_EPOCH
        epoch_solve_rates.append(solve_rate)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_PPO_EPOCHS}: "
                  f"solve_rate={solve_rate*100:.1f}%, loss={ppo_loss:.4f}")

        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            all_results["ppo_training"] = {
                "epoch_solve_rates": [round(r, 4) for r in epoch_solve_rates],
                "epoch_losses": [round(l, 4) for l in epoch_losses],
                "current_epoch": epoch + 1,
            }
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # Save trained SNN
    snn_save_path = os.path.join(RESULTS_DIR, "phase3_ppo_snn.pt")
    torch.save(snn.state_dict(), snn_save_path)
    pca_save_path = os.path.join(RESULTS_DIR, "phase3_pca_projection.npz")
    np.savez(pca_save_path, pca_components=pca_proj)
    print(f"  SNN saved: {snn_save_path}")

    all_results["ppo_training"] = {
        "epoch_solve_rates": [round(r, 4) for r in epoch_solve_rates],
        "epoch_losses": [round(l, 4) for l in epoch_losses],
        "total_trials": N_PPO_EPOCHS * N_TRIALS_PER_EPOCH,
        "final_solve_rate": round(epoch_solve_rates[-1], 4) if epoch_solve_rates else 0,
    }

    # === Step 3: Evaluation ===
    print(f"\n  === Evaluation: {N_TEST} games × 3 conditions ===")

    test_results = []

    # Condition 1: baseline
    print(f"\n  Condition: baseline")
    hook.setup_off()
    games = []
    for trial in range(N_TEST):
        stats = play_game(model, tok, hook)
        games.append(stats)
        if (trial + 1) % 50 == 0:
            sr = sum(1 for g in games if g["solved"]) / len(games) * 100
            print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")
    solved = sum(1 for g in games if g["solved"])
    test_results.append({"condition": "baseline", "solve_rate": round(solved/N_TEST, 4),
                         "n_solved": solved, "n_total": N_TEST, "games": games})
    print(f"    Baseline: {solved}/{N_TEST} = {solved/N_TEST*100:.1f}%")

    # Condition 2: fixed_aha
    print(f"\n  Condition: fixed_aha")
    hook.setup_fixed(diff_unit, device)
    games = []
    for trial in range(N_TEST):
        stats = play_game(model, tok, hook)
        games.append(stats)
        if (trial + 1) % 50 == 0:
            sr = sum(1 for g in games if g["solved"]) / len(games) * 100
            print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")
    solved = sum(1 for g in games if g["solved"])
    test_results.append({"condition": "fixed_aha", "solve_rate": round(solved/N_TEST, 4),
                         "n_solved": solved, "n_total": N_TEST, "games": games})
    print(f"    Fixed: {solved}/{N_TEST} = {solved/N_TEST*100:.1f}%")

    # Condition 3: ppo_snn
    print(f"\n  Condition: ppo_snn")
    hook.setup_snn(diff_unit, pca_proj, snn, device)
    snn.eval()
    games = []
    for trial in range(N_TEST):
        stats = play_game(model, tok, hook)
        games.append(stats)
        if (trial + 1) % 50 == 0:
            sr = sum(1 for g in games if g["solved"]) / len(games) * 100
            print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")
    solved = sum(1 for g in games if g["solved"])
    test_results.append({"condition": "ppo_snn", "solve_rate": round(solved/N_TEST, 4),
                         "n_solved": solved, "n_total": N_TEST, "games": games})
    print(f"    PPO-SNN: {solved}/{N_TEST} = {solved/N_TEST*100:.1f}%")

    all_results["test_results"] = test_results

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl = test_results[0]
    for tr in test_results:
        delta = tr["solve_rate"] - bl["solve_rate"]
        print(f"    {tr['condition']:15s}: {tr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    ppo_rate = test_results[2]["solve_rate"]
    fixed_rate = test_results[1]["solve_rate"]
    if ppo_rate >= fixed_rate:
        verdict = "PPO_SNN_SUPERIOR"
    elif ppo_rate > bl["solve_rate"] + 0.05:
        verdict = "PPO_SNN_EFFECTIVE"
    else:
        verdict = "PPO_SNN_NEEDS_MORE_TRAINING"
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
    print(f"\n Phase 3 complete.")
