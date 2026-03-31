"""
Phase 6b: Mixture of Aha!s (MoA) — Universal Reasoning Router
================================================================

Killer Experiment #3: Instead of a single intervention direction,
train an SNN router that dynamically selects between multiple
"expert" directions at each reasoning step.

Phase 6 showed single Hanoi vectors don't transfer to arithmetic.
MoA hypothesis: different reasoning phases need different intervention
directions. A learned router can discover these dynamics.

Architecture:
  - Mistral-7B-Instruct-v0.3 (frozen, 4-bit)
  - 3 Expert directions:
    1. diff_unit (Aha! direction from Phase 91)
    2. trajectory_unit (from Phase 109)
    3. orthogonal_unit (Gram-Schmidt complement)
  - SNN Router: LIF(320→64→4) → categorical over {no_inject, e1, e2, e3}
  - PPO training on Modified Hanoi (50 epochs × 30 trials)
  - Evaluation on Hanoi (N=100) + Arithmetic (N=100) cross-task transfer

Conditions (N=100 each):
  1. baseline:           No injection
  2. single_aha:         Only Aha! direction (fixed)
  3. single_trajectory:  Only trajectory direction (fixed)
  4. moa_router:         PPO-trained dynamic router
  5. random_select:      Randomly choose direction each step

Mistral-7B-Instruct-v0.3, Layer 18
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
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_SHORT = "Mistral-7B"
SEED = 2026
MAX_STEPS = 50
HIDDEN_DIM = 4096
BASE_SIGMA = 0.15
LAYER_IDX = 18
PCA_DIM = 64
WINDOW_SIZE = 5
N_PPO_EPOCHS = 50
N_TRIALS_PER_EPOCH = 30
N_TEST = 100
N_MATH_TEST = 100

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  SNN ROUTER (Policy Network)
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


class MoARouter(nn.Module):
    """SNN router that selects among 4 actions:
       0: no_inject, 1: aha, 2: trajectory, 3: orthogonal"""
    def __init__(self, input_dim=PCA_DIM * WINDOW_SIZE, hidden_dim=64, n_experts=3):
        super().__init__()
        self.n_actions = n_experts + 1  # +1 for no_inject
        self.lif1 = LIFLayer(input_dim, hidden_dim, tau=0.9)
        self.lif2 = LIFLayer(hidden_dim, 16, tau=0.85)
        self.policy_head = nn.Linear(16, self.n_actions)
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
#  ARITHMETIC WORD PROBLEM GENERATOR
# ===================================================

def generate_word_problems(n=100, seed=SEED):
    rng = random.Random(seed)
    problems = []
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]
    items = ["apples", "books", "pencils", "cookies", "stickers", "marbles", "cards", "toys"]

    for i in range(n):
        cat = i % 4
        if cat == 0:  # Shopping
            name = rng.choice(names)
            item = rng.choice(items)
            qty1 = rng.randint(3, 12)
            price = rng.randint(2, 8)
            qty2 = rng.randint(1, qty1 - 1)
            answer = (qty1 - qty2) * price
            question = (f"{name} bought {qty1} {item} at ${price} each. "
                       f"Later, {name} returned {qty2} of them for a full refund. "
                       f"How much did {name} spend in total?")
            problems.append({"question": question, "answer": answer, "category": "shopping"})
        elif cat == 1:  # Distance
            name = rng.choice(names)
            d1 = rng.randint(10, 50)
            d2 = rng.randint(10, 50)
            answer = d1 + d2
            question = (f"{name} drove {d1} miles to the store, then drove {d2} miles "
                       f"to the library. How many miles did {name} drive in total?")
            problems.append({"question": question, "answer": answer, "category": "distance"})
        elif cat == 2:  # Time
            name = rng.choice(names)
            hours = rng.randint(4, 10)
            breaks_count = rng.randint(1, 3)
            break_min = rng.randint(10, 30)
            total_break = breaks_count * break_min
            work_min = hours * 60 - total_break
            answer = work_min
            question = (f"{name} worked for {hours} hours but took {breaks_count} breaks "
                       f"of {break_min} minutes each. "
                       f"How many minutes did {name} actually work?")
            problems.append({"question": question, "answer": answer, "category": "time"})
        elif cat == 3:  # Sharing
            name1 = rng.choice(names)
            name2 = rng.choice([n for n in names if n != name1])
            item = rng.choice(items)
            total = rng.randint(12, 40)
            total = total - (total % 2)
            each = total // 2
            give = rng.randint(1, each - 1)
            answer = each + give
            question = (f"{name1} and {name2} split {total} {item} equally. "
                       f"Then {name1} gave {give} {item} to {name2}. "
                       f"How many {item} does {name2} have now?")
            problems.append({"question": question, "answer": answer, "category": "sharing"})
    return problems


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

def build_math_prompt(tokenizer, problem):
    system = ("You are a math tutor. Solve the word problem step by step. "
              "Show your work, then give the final answer as: Answer: <number>")
    msg = f"Problem: {problem['question']}"
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

def extract_answer(response):
    m = re.search(r'Answer:\s*\$?\s*(-?\d+(?:\.\d+)?)', response)
    if m:
        return float(m.group(1))
    m = re.search(r'=\s*\$?\s*(-?\d+(?:\.\d+)?)\s*$', response, re.MULTILINE)
    if m:
        return float(m.group(1))
    nums = re.findall(r'(-?\d+(?:\.\d+)?)', response)
    if nums:
        return float(nums[-1])
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
#  MoA HOOK SYSTEM
# ===================================================

class MoAHook:
    """Mixture of Aha!s hook with dynamic direction selection."""
    def __init__(self):
        self.mode = "off"  # off, single, moa, random_select
        self.sigma = BASE_SIGMA
        self.expert_dirs = []  # list of torch tensors (4096-dim each)
        self.active_expert = 0  # for single mode
        self.handle = None
        self.captured_states = []
        self.window_buffer = []
        self.pca_proj = None
        self.router = None
        self.device = 'cuda'
        self.legal_move_count = 0
        # PPO rollout buffers
        self.step_states = []
        self.step_actions = []
        self.step_log_probs = []
        # Routing statistics
        self.route_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    def setup_off(self):
        self.mode = "off"

    def setup_single(self, expert_idx, expert_dirs, device):
        self.mode = "single"
        self.active_expert = expert_idx
        self.expert_dirs = [torch.tensor(d, dtype=torch.float16, device=device) for d in expert_dirs]
        self.device = device

    def setup_moa(self, expert_dirs, pca_proj, router, device):
        self.mode = "moa"
        self.expert_dirs = [torch.tensor(d, dtype=torch.float16, device=device) for d in expert_dirs]
        self.pca_proj = pca_proj
        self.router = router
        self.device = device

    def setup_random(self, expert_dirs, device):
        self.mode = "random_select"
        self.expert_dirs = [torch.tensor(d, dtype=torch.float16, device=device) for d in expert_dirs]
        self.device = device

    def reset_game(self):
        self.captured_states = []
        self.window_buffer = []
        self.legal_move_count = 0
        self.step_states = []
        self.step_actions = []
        self.step_log_probs = []
        self.route_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    def _get_sigma(self):
        if self.legal_move_count < 10:
            return self.sigma * (1.0 - self.legal_move_count / 10.0)
        return 0.0

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self

        def hook_fn(module, args):
            hs = args[0]

            # Capture hidden states for routing decisions
            if hook_obj.mode in ("moa", "random_select"):
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

            # Determine which direction to use
            chosen_expert = -1  # -1 = no inject

            if hook_obj.mode == "single":
                chosen_expert = hook_obj.active_expert

            elif hook_obj.mode == "moa" and hook_obj.router is not None:
                if len(hook_obj.window_buffer) >= WINDOW_SIZE:
                    window = np.array(hook_obj.window_buffer[-WINDOW_SIZE:])
                    projected = window @ hook_obj.pca_proj.T
                    flat = projected.flatten()
                    x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(hook_obj.device)

                    with torch.no_grad():
                        action, log_prob, value, _ = hook_obj.router.get_action_and_value(x)

                    action_val = action.item()
                    hook_obj.route_counts[action_val] = hook_obj.route_counts.get(action_val, 0) + 1

                    # Store for PPO
                    hook_obj.step_states.append(flat.copy())
                    hook_obj.step_actions.append(action_val)
                    hook_obj.step_log_probs.append(log_prob.item())

                    if action_val == 0:
                        return args  # No inject
                    chosen_expert = action_val - 1  # Map 1,2,3 → expert 0,1,2
                else:
                    return args  # Not enough data for router

            elif hook_obj.mode == "random_select":
                action_val = random.randint(0, 3)
                if action_val == 0:
                    return args
                chosen_expert = action_val - 1

            if chosen_expert < 0 or chosen_expert >= len(hook_obj.expert_dirs):
                return args

            # Inject chosen expert direction
            d = hs.shape[-1]
            offset = hook_obj.expert_dirs[chosen_expert]
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
#  GAME FUNCTIONS
# ===================================================

def play_hanoi(model, tok, hook):
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


def evaluate_math(model, tok, problem, hook=None):
    if hook is not None:
        hook.reset_game()  # Reset state before each math problem
    prompt = build_math_prompt(tok, problem)
    response = gen(model, tok, prompt, temperature=0.3, max_tokens=150)
    extracted = extract_answer(response)
    if extracted is None:
        return False
    return abs(extracted - problem["answer"]) < 0.5


# ===================================================
#  PPO UPDATE
# ===================================================

def ppo_update(router, optimizer, rollout_buffer, device, clip_eps=0.2):
    if not rollout_buffer:
        return 0.0

    all_states, all_actions, all_old_log_probs, all_rewards = [], [], [], []

    for episode in rollout_buffer:
        if not episode["states"]:
            continue
        n_steps = len(episode["states"])
        step_rewards = [0.0] * n_steps
        step_rewards[-1] = episode["reward"]
        all_states.extend(episode["states"])
        all_actions.extend(episode["actions"])
        all_old_log_probs.extend(episode["log_probs"])
        all_rewards.extend(step_rewards)

    if not all_states:
        return 0.0

    states_t = torch.tensor(np.array(all_states), dtype=torch.float32).to(device)
    actions_t = torch.tensor(all_actions, dtype=torch.long).to(device)
    old_log_probs_t = torch.tensor(all_old_log_probs, dtype=torch.float32).to(device)
    returns_t = torch.tensor(all_rewards, dtype=torch.float32).to(device)

    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    total_loss = 0
    for _ in range(3):
        new_log_probs, values, entropy = router.evaluate_actions(states_t, actions_t)
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
        torch.nn.utils.clip_grad_norm_(router.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / 3


# ===================================================
#  PCA FITTING
# ===================================================

def fit_pca(model, tok, hook, n_games=30):
    print(f"  Fitting PCA from {n_games} games (native {HIDDEN_DIM}-dim)...")
    hook.mode = "moa"
    hook.router = None

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
        if (gi + 1) % 10 == 0:
            print(f"    PCA: {gi+1}/{n_games} games, {len(all_deltas)} deltas")

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
#  EXPERT DIRECTION CONSTRUCTION
# ===================================================

def build_expert_directions(diff_unit, traj_unit):
    """Build 3 expert directions: aha, trajectory, orthogonal complement."""
    # e1 = diff_unit (already unit norm)
    e1 = diff_unit.copy()
    e1 /= (np.linalg.norm(e1) + 1e-8)

    # e2 = trajectory_unit (already unit norm)
    e2 = traj_unit.copy()
    e2 /= (np.linalg.norm(e2) + 1e-8)

    # e3 = Gram-Schmidt orthogonal complement
    # Start with a random vector, subtract projections onto e1 and e2
    rng = np.random.RandomState(SEED)
    v = rng.randn(HIDDEN_DIM).astype(np.float32)
    v -= np.dot(v, e1) * e1
    v -= np.dot(v, e2) * e2
    e3 = v / (np.linalg.norm(v) + 1e-8)

    cos12 = np.dot(e1, e2)
    cos13 = np.dot(e1, e3)
    cos23 = np.dot(e2, e3)
    print(f"  Expert directions built:")
    print(f"    cos(aha, traj)={cos12:.4f}, cos(aha, orth)={cos13:.4f}, cos(traj, orth)={cos23:.4f}")

    return [e1, e2, e3]


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Phase 6b: Mixture of Aha!s (MoA)\n"
                 "Dynamic reasoning router with 3 expert directions",
                 fontsize=12, fontweight="bold")

    # Panel 1: PPO training curve
    ax = axes[0, 0]
    ppo = all_results.get("ppo_training", {})
    if "epoch_solve_rates" in ppo:
        rates = ppo["epoch_solve_rates"]
        ax.plot(range(1, len(rates)+1), [r*100 for r in rates],
                color="#9C27B0", linewidth=2)
    ax.set_xlabel("PPO Epoch")
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("MoA Router Training", fontweight="bold")
    ax.grid(alpha=0.3)

    # Panel 2: Hanoi solve rates (5 conditions)
    ax = axes[0, 1]
    hanoi = all_results.get("hanoi_results", [])
    if hanoi:
        names_h = [t["condition"] for t in hanoi]
        rates_h = [t["solve_rate"] * 100 for t in hanoi]
        colors_h = ["#9E9E9E", "#4CAF50", "#2196F3", "#9C27B0", "#FFC107"]
        bars = ax.bar(range(len(hanoi)), rates_h, color=colors_h[:len(hanoi)], alpha=0.85,
                      edgecolor="white", linewidth=2)
        for bar, val in zip(bars, rates_h):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(hanoi)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names_h], fontsize=7)
    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Modified Hanoi (N=100)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Arithmetic accuracy (5 conditions)
    ax = axes[1, 0]
    math_ = all_results.get("math_results", [])
    if math_:
        names_m = [t["condition"] for t in math_]
        rates_m = [t["accuracy"] * 100 for t in math_]
        colors_m = ["#9E9E9E", "#4CAF50", "#2196F3", "#9C27B0", "#FFC107"]
        bars = ax.bar(range(len(math_)), rates_m, color=colors_m[:len(math_)], alpha=0.85,
                      edgecolor="white", linewidth=2)
        for bar, val in zip(bars, rates_m):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(math_)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names_m], fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Arithmetic Transfer (N=100)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Routing distribution (MoA)
    ax = axes[1, 1]
    route_dist = all_results.get("routing_distribution", {})
    if route_dist:
        labels = ["no_inject", "aha", "trajectory", "orthogonal"]
        counts = [route_dist.get(str(i), 0) for i in range(4)]
        total = sum(counts) or 1
        pcts = [c/total*100 for c in counts]
        colors_r = ["#9E9E9E", "#4CAF50", "#2196F3", "#FF9800"]
        ax.bar(labels, pcts, color=colors_r, alpha=0.85, edgecolor="white", linewidth=2)
        for i, val in enumerate(pcts):
            ax.text(i, val + 1, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Routing Frequency (%)")
    ax.set_title("MoA Routing Distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(FIGURES_DIR, "phase6b_mixture_of_ahas.png")
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
    print(f"  Phase 6b: Mixture of Aha!s (MoA)")
    print(f"  3 expert directions, PPO router")
    print(f"  Hanoi N={N_TEST} + Arithmetic N={N_MATH_TEST}")
    print(f"{'='*80}")

    t0 = time.time()

    # Load vectors (native 4096-dim for Mistral)
    diff_data = np.load(os.path.join(DATA_DIR, "diff_pca.npz"))
    diff_unit = diff_data["diff_unit"]
    traj_data = np.load(os.path.join(DATA_DIR, "trajectory_template.npz"))
    traj_unit = traj_data["trajectory_unit"]

    expert_dirs = build_expert_directions(diff_unit, traj_unit)

    # Generate math problems
    math_problems = generate_word_problems(n=N_MATH_TEST, seed=SEED + 42)
    print(f"  Generated {len(math_problems)} arithmetic problems")

    model, tok = load_model()
    device = next(model.parameters()).device

    hook = MoAHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 6b: Mixture of Aha!s (MoA)",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_experts": 3,
        "n_test_hanoi": N_TEST,
        "n_test_math": N_MATH_TEST,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase6b_log.json")

    # === Step 1: Fit PCA ===
    pca_proj = fit_pca(model, tok, hook, n_games=30)

    # === Step 2: PPO Training ===
    print(f"\n  === PPO Training: {N_PPO_EPOCHS} epochs × {N_TRIALS_PER_EPOCH} trials ===")
    router = MoARouter(input_dim=PCA_DIM * WINDOW_SIZE, hidden_dim=64, n_experts=3).to(device)
    router_optimizer = torch.optim.Adam(router.parameters(), lr=3e-4)

    hook.setup_moa(expert_dirs, pca_proj, router, device)

    epoch_solve_rates = []
    epoch_losses = []

    for epoch in range(N_PPO_EPOCHS):
        router.eval()
        rollout_buffer = []
        epoch_solved = 0

        for trial in range(N_TRIALS_PER_EPOCH):
            stats = play_hanoi(model, tok, hook)

            reward = 0.0
            if stats["solved"]:
                reward = 1.0
                epoch_solved += 1
            reward -= 0.05 * min(stats["illegal_moves"], 10)  # Capped penalty
            reward -= 0.005 * stats["steps_taken"]
            reward = max(reward, -1.0)  # Floor clipping

            if hook.step_states:
                rollout_buffer.append({
                    "states": list(hook.step_states),
                    "actions": list(hook.step_actions),
                    "log_probs": list(hook.step_log_probs),
                    "reward": reward,
                })

        router.train()
        ppo_loss = ppo_update(router, router_optimizer, rollout_buffer, device)
        epoch_losses.append(ppo_loss)

        solve_rate = epoch_solved / N_TRIALS_PER_EPOCH
        epoch_solve_rates.append(solve_rate)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{N_PPO_EPOCHS}: "
                  f"solve_rate={solve_rate*100:.1f}%, loss={ppo_loss:.4f}")

        if (epoch + 1) % 25 == 0:
            all_results["ppo_training"] = {
                "epoch_solve_rates": [round(r, 4) for r in epoch_solve_rates],
                "epoch_losses": [round(l, 4) for l in epoch_losses],
            }
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    # Save trained router
    router_save_path = os.path.join(RESULTS_DIR, "phase6b_moa_router.pt")
    torch.save(router.state_dict(), router_save_path)
    pca_save_path = os.path.join(RESULTS_DIR, "phase6b_pca_projection.npz")
    np.savez(pca_save_path, pca_components=pca_proj)

    all_results["ppo_training"] = {
        "epoch_solve_rates": [round(r, 4) for r in epoch_solve_rates],
        "epoch_losses": [round(l, 4) for l in epoch_losses],
        "total_trials": N_PPO_EPOCHS * N_TRIALS_PER_EPOCH,
    }

    # === Step 3: Hanoi Evaluation ===
    print(f"\n  === Hanoi Evaluation: {N_TEST} games × 5 conditions ===")
    hanoi_results = []

    conditions_hanoi = [
        ("baseline", "off"),
        ("single_aha", "single_0"),
        ("single_trajectory", "single_1"),
        ("moa_router", "moa"),
        ("random_select", "random"),
    ]

    for cond_name, cond_mode in conditions_hanoi:
        print(f"\n  Hanoi — {cond_name}")
        if cond_mode == "off":
            hook.setup_off()
        elif cond_mode.startswith("single_"):
            expert_idx = int(cond_mode.split("_")[1])
            hook.setup_single(expert_idx, expert_dirs, device)
        elif cond_mode == "moa":
            hook.setup_moa(expert_dirs, pca_proj, router, device)
            router.eval()
        elif cond_mode == "random":
            hook.setup_random(expert_dirs, device)

        games = []
        for trial in range(N_TEST):
            stats = play_hanoi(model, tok, hook)
            games.append(stats)
            if (trial + 1) % 50 == 0:
                sr = sum(1 for g in games if g["solved"]) / len(games) * 100
                print(f"    [{trial+1}/{N_TEST}] {sr:.1f}%")

        solved = sum(1 for g in games if g["solved"])
        hanoi_results.append({
            "condition": cond_name,
            "solve_rate": round(solved / N_TEST, 4),
            "n_solved": solved,
            "n_total": N_TEST,
            "games": games,
        })

        # Save routing distribution for MoA
        if cond_mode == "moa":
            total_routes = sum(hook.route_counts.values())
            if total_routes > 0:
                all_results["routing_distribution"] = {
                    str(k): v for k, v in hook.route_counts.items()
                }

        print(f"    {cond_name}: {solved}/{N_TEST} = {solved/N_TEST*100:.1f}%")

    all_results["hanoi_results"] = hanoi_results

    # === Step 4: Arithmetic Evaluation ===
    print(f"\n  === Arithmetic Evaluation: {N_MATH_TEST} problems × 5 conditions ===")
    math_results = []

    for cond_name, cond_mode in conditions_hanoi:
        print(f"\n  Math — {cond_name}")
        if cond_mode == "off":
            hook.setup_off()
        elif cond_mode.startswith("single_"):
            expert_idx = int(cond_mode.split("_")[1])
            hook.setup_single(expert_idx, expert_dirs, device)
        elif cond_mode == "moa":
            hook.setup_moa(expert_dirs, pca_proj, router, device)
            router.eval()
        elif cond_mode == "random":
            hook.setup_random(expert_dirs, device)

        correct = 0
        for pi, problem in enumerate(math_problems):
            is_correct = evaluate_math(model, tok, problem, hook=hook)
            if is_correct:
                correct += 1
            if (pi + 1) % 50 == 0:
                print(f"    [{pi+1}/{N_MATH_TEST}] {correct/(pi+1)*100:.1f}%")

        math_results.append({
            "condition": cond_name,
            "accuracy": round(correct / N_MATH_TEST, 4),
            "n_correct": correct,
            "n_total": N_MATH_TEST,
        })
        print(f"    {cond_name}: {correct}/{N_MATH_TEST} = {correct/N_MATH_TEST*100:.1f}%")

    all_results["math_results"] = math_results

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    print(f"\n  Hanoi:")
    bl_hanoi = hanoi_results[0]
    for tr in hanoi_results:
        delta = tr["solve_rate"] - bl_hanoi["solve_rate"]
        print(f"    {tr['condition']:20s}: {tr['solve_rate']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Fisher exact tests — Hanoi
    for i in range(1, len(hanoi_results)):
        tr = hanoi_results[i]
        if tr["n_solved"] != bl_hanoi["n_solved"]:
            alt = "greater" if tr["n_solved"] > bl_hanoi["n_solved"] else "less"
            table = [[tr["n_solved"], N_TEST - tr["n_solved"]],
                     [bl_hanoi["n_solved"], N_TEST - bl_hanoi["n_solved"]]]
            _, pval = fisher_exact(table, alternative=alt)
            key = f"fisher_p_hanoi_{tr['condition']}_vs_baseline"
            all_results[key] = round(pval, 6)
            print(f"    Fisher p ({tr['condition']} vs baseline): {pval:.6f}")

    print(f"\n  Arithmetic:")
    bl_math = math_results[0]
    for tr in math_results:
        delta = tr["accuracy"] - bl_math["accuracy"]
        print(f"    {tr['condition']:20s}: {tr['accuracy']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Fisher exact tests — Arithmetic
    for i in range(1, len(math_results)):
        tr = math_results[i]
        if tr["n_correct"] != bl_math["n_correct"]:
            alt = "greater" if tr["n_correct"] > bl_math["n_correct"] else "less"
            table = [[tr["n_correct"], N_MATH_TEST - tr["n_correct"]],
                     [bl_math["n_correct"], N_MATH_TEST - bl_math["n_correct"]]]
            _, pval = fisher_exact(table, alternative=alt)
            key = f"fisher_p_math_{tr['condition']}_vs_baseline"
            all_results[key] = round(pval, 6)
            print(f"    Fisher p math ({tr['condition']} vs baseline): {pval:.6f}")

    # Verdict
    moa_hanoi = hanoi_results[3]["solve_rate"]
    best_single_hanoi = max(hanoi_results[1]["solve_rate"], hanoi_results[2]["solve_rate"])
    moa_math = math_results[3]["accuracy"]
    best_single_math = max(math_results[1]["accuracy"], math_results[2]["accuracy"])

    if moa_hanoi > best_single_hanoi + 0.03 and moa_math > best_single_math + 0.03:
        verdict = "MOA_UNIVERSALLY_SUPERIOR"
    elif moa_hanoi > best_single_hanoi + 0.03:
        verdict = "MOA_SUPERIOR_ON_TRAINING_TASK"
    elif moa_math > best_single_math + 0.03:
        verdict = "MOA_ENABLES_TRANSFER"
    elif moa_hanoi >= best_single_hanoi - 0.02:
        verdict = "MOA_MATCHES_SINGLE"
    else:
        verdict = "SINGLE_DIRECTION_SUFFICIENT"

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
    print(f"\n Phase 6b complete.")
