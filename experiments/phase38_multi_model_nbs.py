"""
Phase 38: Multi-Model NBS — Architecture Invariance Validation
================================================================
Test NBS on Qwen2.5-7B-Instruct to validate that log-K scaling
is architecture-invariant beyond Mistral-7B.

K={1, 3, 7, 11}, σ=0.15, N=50 per condition
Compare K-scaling curve with Mistral-7B (Phase 29)
"""
import torch
import os, json, gc, time, random, re, math
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SEED = 2026
MAX_STEPS = 50
BASE_SIGMA = 0.15
K_VALUES = [1, 3, 7, 11]
N_GAMES = 50
MAX_NEW_TOKENS = 80

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
DATA_DIR = os.path.join(EXPERIMENT_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda"

# ===================================================
# HANOI ENVIRONMENT (from Phase 32b)
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
        self._prev_illegal = False
        self.pegs[from_p].pop()
        self.pegs[to_p].append(disk)
        self.moves.append(f"{from_p}->{to_p} (disk {disk})")
        return True, f"Moved disk {disk}: {from_p}->{to_p}"

    def state_str(self):
        return f"A:{self.pegs['A']} B:{self.pegs['B']} C:{self.pegs['C']}"

    def stats(self):
        return {"solved": self.is_solved(), "legal_moves": len(self.moves),
                "illegal_moves": self.illegal_count}

# ===================================================
# PROMPT & PARSER
# ===================================================
def build_prompt(tokenizer, env, error=None, model_type="mistral"):
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

    if model_type == "qwen":
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": msg}]
    else:
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
# Model Management
# ===================================================
class ModelRunner:
    def __init__(self, model_name, layer_idx=18):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.model = None
        self.tokenizer = None
        self.hook_handle = None
        self.sigma = 0.0
        self.model_type = "qwen" if "Qwen" in model_name else "mistral"

    def load(self):
        print(f"\n  Loading {self.model_name} (4-bit)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.float16, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        # Register noise hook
        runner = self
        def hook_fn(module, args):
            hs = args[0]
            if runner.sigma <= 0:
                return args
            noise = torch.randn_like(hs) * runner.sigma
            return (hs + noise,) + args[1:]
        self.hook_handle = self.model.model.layers[self.layer_idx].register_forward_pre_hook(hook_fn)

    def generate(self, prompt, temperature=0.5, max_tokens=80):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=temperature, top_p=0.9,
                repetition_penalty=1.2, pad_token_id=self.tokenizer.pad_token_id)
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        inp = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        return full[len(inp):].strip()

    def play_game(self, sigma):
        self.sigma = sigma
        env = HanoiEnv(n_disks=3, modified=True)
        error = None
        consec_fail = 0
        for step in range(MAX_STEPS):
            prompt = build_prompt(self.tokenizer, env, error, self.model_type)
            resp = self.generate(prompt)
            move = parse_move(resp)
            if move is None:
                env.illegal_count += 1; env.total_attempts += 1
                error = "Parse fail. Use Move: X->Y"
                consec_fail += 1
                if consec_fail >= 10: break
                continue
            ok, msg = env.try_move(move[0], move[1])
            if ok:
                error = None; consec_fail = 0
                if env.is_solved(): break
            else:
                error = msg; consec_fail += 1
                if consec_fail >= 10: break
        return env.stats()

    def unload(self):
        if self.hook_handle:
            self.hook_handle.remove()
        del self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# ===================================================
# Main
# ===================================================
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    MODELS = [
        ("Qwen/Qwen2.5-7B-Instruct", 18),
    ]

    print("=" * 60)
    print("Phase 38: Multi-Model NBS — Architecture Invariance")
    print(f"  K = {K_VALUES}, σ = {BASE_SIGMA}, N = {N_GAMES}")
    print("=" * 60)

    results = {
        "experiment": "Phase 38: Multi-Model NBS",
        "start_time": datetime.now().isoformat(),
        "models": {},
        "mistral_reference": {
            "K1": 0.16, "K3": 0.72, "K7": 0.96, "K11": 1.00
        }
    }

    for model_name, layer_idx in MODELS:
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        runner = ModelRunner(model_name, layer_idx)
        runner.load()

        model_results = {}

        for K in K_VALUES:
            label = f"K={K}"
            print(f"\n--- {label}, σ={BASE_SIGMA}, N={N_GAMES} ---")
            solved = 0

            for i in range(N_GAMES):
                for k in range(K):
                    stats = runner.play_game(BASE_SIGMA)
                    if stats["solved"]:
                        solved += 1
                        break
                if (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{N_GAMES}] {solved}/{i+1} ({100*solved/(i+1):.1f}%)")

            rate = solved / N_GAMES
            model_results[label] = {"K": K, "solved": solved, "rate": rate}
            print(f"  {label}: {solved}/{N_GAMES} = {100*rate:.1f}%")

            # Save intermediate
            results["models"][model_name] = model_results
            with open(os.path.join(RESULTS_DIR, "phase38_multi_model_nbs.json"), 'w') as f:
                json.dump(results, f, indent=2)

        runner.unload()

    results["end_time"] = datetime.now().isoformat()

    # Compare with Mistral
    print("\n" + "=" * 60)
    print("PHASE 38 SUMMARY: Multi-Model NBS Comparison")
    print("=" * 60)
    print(f"  {'K':>4s}  {'Mistral-7B':>12s}  {'Qwen-7B':>12s}")
    for K in K_VALUES:
        mistral = results["mistral_reference"].get(f"K{K}", 0)
        for mname, mres in results["models"].items():
            qwen = mres.get(f"K={K}", {}).get("rate", 0)
        print(f"  {K:4d}  {100*mistral:10.1f}%  {100*qwen:10.1f}%")
    print("=" * 60)

    out_path = os.path.join(RESULTS_DIR, "phase38_multi_model_nbs.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
