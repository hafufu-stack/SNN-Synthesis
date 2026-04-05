"""
Phase 29: Mistral-7B Noisy Beam Search — Architecture-Invariant Validation
===========================================================================
THE experiment everyone agreed on: Does Noisy Beam Search scale from 63K CNN to 7B LLM?

Method:
  - Mistral-7B-Instruct-v0.3 (4-bit, Layer 18 noise injection)
  - Modified Hanoi 3-disk (same as Phase 2,5,3b,6b)
  - K independent noisy games in parallel, select best (any solved = beam success)
  - σ=0.15 (Phase 5 optimal)
  - K = 1,3,5,7,11
  - N = 50 per K condition (LLM is heavy: ~10-15s per step × 50 steps × 50 games)
  - Additional baselines: K=1 no-noise, K=1 fixed-aha

Expected: K=1 ~40% → K=11 ~80%+ = same logarithmic K scaling as CNN
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
N_GAMES_PER_K = 50
K_VALUES = [1, 3, 5, 7, 11]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  HANOI ENVIRONMENT (from Phase 3b)
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
#  PROMPT & PARSER (from Phase 3b)
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
#  MODEL + GENERATION (from Phase 3b)
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
#  NOISE INJECTION HOOK
# ===================================================

class NoiseHook:
    """Simple fixed noise injection at Layer 18."""
    def __init__(self):
        self.sigma = 0.0
        self.diff_offset = None
        self.handle = None
        self.inject_count = 0

    def setup(self, sigma, diff_unit=None, device='cuda'):
        self.sigma = sigma
        if diff_unit is not None:
            self.diff_offset = torch.tensor(diff_unit, dtype=torch.float16, device=device)
        else:
            self.diff_offset = None
        self.inject_count = 0

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self

        def hook_fn(module, args):
            hs = args[0]
            if hook_obj.sigma <= 0:
                return args

            hook_obj.inject_count += 1

            if hook_obj.diff_offset is not None:
                # Directional + stochastic noise (Aha! style)
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
                # Pure stochastic noise
                noise = torch.randn_like(hs) * hook_obj.sigma
                return (hs + noise,) + args[1:]

        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  GAME PLAY (single game)
# ===================================================

def play_game(model, tok, hook):
    """Play a single Hanoi game. Returns stats dict."""
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
#  NOISY BEAM SEARCH
# ===================================================

def noisy_beam_search(model, tok, hook, K, sigma, diff_unit, device):
    """Run K independent noisy games. Return True if ANY solved."""
    for k in range(K):
        # Each beam gets fresh noise (different random seed via stochastic generation)
        hook.setup(sigma=sigma, diff_unit=diff_unit, device=device)
        stats = play_game(model, tok, hook)
        if stats["solved"]:
            return True, stats
    return False, stats  # Return last game's stats


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Phase 29: Mistral-7B Noisy Beam Search\n"
                 "Trajectory Ensemble Scales from 63K CNN to 7B LLM",
                 fontsize=13, fontweight="bold")

    # Panel 1: Noisy Beam Search K scaling
    ax = axes[0]
    k_results = all_results.get("k_scaling", {})
    if k_results:
        ks = sorted([int(k) for k in k_results.keys()])
        rates = [k_results[str(k)]["clear_rate"] * 100 for k in ks]

        ax.plot(ks, rates, 'o-', color='#9C27B0', linewidth=2.5, markersize=10,
                label='Mistral-7B (Noisy Beam)')
        ax.fill_between(ks, rates, alpha=0.15, color='#9C27B0')

        for k, r in zip(ks, rates):
            ax.annotate(f'{r:.0f}%', (k, r), textcoords="offset points",
                        xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    # Add CNN reference (from Phase 14, Table 4)
    cnn_ks = [1, 3, 5, 7, 11]
    cnn_rates = [12, 25, 46, 59, 78]  # LS20 L2 Oracle CNN
    ax.plot(cnn_ks, cnn_rates, 's--', color='#FF9800', linewidth=1.5, markersize=6,
            alpha=0.7, label='CNN 63K (LS20 L2)')

    ax.set_xlabel('K (Parallel Beams)', fontsize=12)
    ax.set_ylabel('Clear Rate (%)', fontsize=12)
    ax.set_title('Noisy Beam Search: K Scaling', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Panel 2: Baselines comparison
    ax = axes[1]
    baselines = all_results.get("baselines", {})
    if baselines:
        names = list(baselines.keys())
        rates = [baselines[n]["clear_rate"] * 100 for n in names]
        colors = ['#9E9E9E', '#2196F3', '#4CAF50']
        bars = ax.bar(range(len(names)), rates, color=colors[:len(names)], alpha=0.85,
                      edgecolor='white', linewidth=2)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel('Solve Rate (%)')
    ax.set_title(f'Baselines (N={N_GAMES_PER_K})', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Cross-architecture comparison
    ax = axes[2]
    architectures = {
        'CNN 63K\n(LS20 L2)': {'K1': 12, 'K11': 78},
        'CNN 244K\n(LS20 L2)': {'K1': 12, 'K11': 78},
    }
    # Add LLM results
    if k_results and '1' in k_results and '11' in k_results:
        architectures['Mistral-7B\n(Mod. Hanoi)'] = {
            'K1': k_results['1']['clear_rate'] * 100,
            'K11': k_results['11']['clear_rate'] * 100 if '11' in k_results else 0,
        }

    x = np.arange(len(architectures))
    width = 0.35
    k1_vals = [v['K1'] for v in architectures.values()]
    k11_vals = [v['K11'] for v in architectures.values()]

    ax.bar(x - width / 2, k1_vals, width, label='K=1', color='#9E9E9E', alpha=0.8)
    ax.bar(x + width / 2, k11_vals, width, label='K=11', color='#9C27B0', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(list(architectures.keys()), fontsize=9)
    ax.set_ylabel('Clear/Solve Rate (%)')
    ax.set_title('Architecture Invariance\nK=1 vs K=11', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(FIGURES_DIR, "phase29_llm_noisy_beam.png")
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

    print(f"\n{'=' * 80}")
    print(f"  Phase 29: Mistral-7B Noisy Beam Search")
    print(f"  K = {K_VALUES}, N = {N_GAMES_PER_K} per K")
    print(f"  σ = {BASE_SIGMA}, Layer {LAYER_IDX}")
    print(f"{'=' * 80}")

    t0 = time.time()

    # Load Aha! direction vector
    diff_pca_path = os.path.join(DATA_DIR, "diff_pca.npz")
    diff_data = np.load(diff_pca_path)
    diff_unit = diff_data["diff_unit"]
    print(f"  Diff unit loaded: shape={diff_unit.shape}")

    model, tok = load_model()
    device = next(model.parameters()).device

    hook = NoiseHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 29: Mistral-7B Noisy Beam Search",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_games_per_k": N_GAMES_PER_K,
        "k_values": K_VALUES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase29_llm_noisy_beam.json")

    # === Step 1: Baselines ===
    print(f"\n  === Baselines (N={N_GAMES_PER_K}) ===")

    baselines = {}

    # Baseline 1: No noise
    print(f"\n  Baseline: no_noise")
    hook.setup(sigma=0.0)
    solved_count = 0
    for i in range(N_GAMES_PER_K):
        stats = play_game(model, tok, hook)
        if stats["solved"]:
            solved_count += 1
        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{N_GAMES_PER_K}] {solved_count}/{i + 1} = {solved_count / (i + 1) * 100:.1f}%")
    baselines["no_noise"] = {
        "n_solved": solved_count, "n_total": N_GAMES_PER_K,
        "clear_rate": round(solved_count / N_GAMES_PER_K, 4)
    }
    print(f"    Result: {solved_count}/{N_GAMES_PER_K} = {solved_count / N_GAMES_PER_K * 100:.1f}%")

    # Baseline 2: Fixed Aha! (K=1)
    print(f"\n  Baseline: fixed_aha (σ={BASE_SIGMA})")
    hook.setup(sigma=BASE_SIGMA, diff_unit=diff_unit, device=device)
    solved_count = 0
    for i in range(N_GAMES_PER_K):
        stats = play_game(model, tok, hook)
        if stats["solved"]:
            solved_count += 1
        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{N_GAMES_PER_K}] {solved_count}/{i + 1} = {solved_count / (i + 1) * 100:.1f}%")
    baselines["fixed_aha_K1"] = {
        "n_solved": solved_count, "n_total": N_GAMES_PER_K,
        "clear_rate": round(solved_count / N_GAMES_PER_K, 4)
    }
    print(f"    Result: {solved_count}/{N_GAMES_PER_K} = {solved_count / N_GAMES_PER_K * 100:.1f}%")

    # Baseline 3: Pure noise K=1
    print(f"\n  Baseline: pure_noise (σ={BASE_SIGMA}, no Aha! direction)")
    hook.setup(sigma=BASE_SIGMA, diff_unit=None, device=device)
    solved_count = 0
    for i in range(N_GAMES_PER_K):
        stats = play_game(model, tok, hook)
        if stats["solved"]:
            solved_count += 1
        if (i + 1) % 10 == 0:
            print(f"    [{i + 1}/{N_GAMES_PER_K}] {solved_count}/{i + 1} = {solved_count / (i + 1) * 100:.1f}%")
    baselines["pure_noise_K1"] = {
        "n_solved": solved_count, "n_total": N_GAMES_PER_K,
        "clear_rate": round(solved_count / N_GAMES_PER_K, 4)
    }
    print(f"    Result: {solved_count}/{N_GAMES_PER_K} = {solved_count / N_GAMES_PER_K * 100:.1f}%")

    all_results["baselines"] = baselines

    # Save intermediate
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # === Step 2: Noisy Beam Search K scaling ===
    print(f"\n  === Noisy Beam Search: K = {K_VALUES} ===")

    k_scaling = {}
    for K in K_VALUES:
        print(f"\n  K = {K} (N={N_GAMES_PER_K}):")
        t_k = time.time()
        solved_count = 0
        all_beam_stats = []

        for gi in range(N_GAMES_PER_K):
            beam_solved, last_stats = noisy_beam_search(
                model, tok, hook, K=K, sigma=BASE_SIGMA,
                diff_unit=diff_unit, device=device)
            if beam_solved:
                solved_count += 1
            all_beam_stats.append({"solved": beam_solved, "game_idx": gi})

            if (gi + 1) % 10 == 0:
                elapsed_k = time.time() - t_k
                rate = solved_count / (gi + 1) * 100
                eta = elapsed_k / (gi + 1) * (N_GAMES_PER_K - gi - 1)
                print(f"    [{gi + 1}/{N_GAMES_PER_K}] {solved_count}/{gi + 1} = {rate:.1f}% "
                      f"(elapsed: {elapsed_k / 60:.1f}m, ETA: {eta / 60:.1f}m)")

        clear_rate = solved_count / N_GAMES_PER_K
        elapsed_k = time.time() - t_k

        k_scaling[str(K)] = {
            "K": K, "n_solved": solved_count, "n_total": N_GAMES_PER_K,
            "clear_rate": round(clear_rate, 4),
            "elapsed_s": round(elapsed_k, 1),
        }

        print(f"    K={K}: {solved_count}/{N_GAMES_PER_K} = {clear_rate * 100:.1f}% "
              f"({elapsed_k / 60:.1f} min)")

        # Save after each K
        all_results["k_scaling"] = k_scaling
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    print(f"  Baselines:")
    for name, data in baselines.items():
        print(f"    {name:20s}: {data['clear_rate'] * 100:5.1f}%")

    print(f"\n  Noisy Beam Search K scaling:")
    for k_str, data in sorted(k_scaling.items(), key=lambda x: int(x[0])):
        bar = "█" * int(data['clear_rate'] * 50)
        print(f"    K={data['K']:3d}: {data['clear_rate'] * 100:5.1f}%  {bar}")

    # Fisher exact tests
    bl_solved = baselines["no_noise"]["n_solved"]
    for k_str, data in k_scaling.items():
        if data["n_solved"] != bl_solved:
            table = [[data["n_solved"], N_GAMES_PER_K - data["n_solved"]],
                      [bl_solved, N_GAMES_PER_K - bl_solved]]
            alt = "greater" if data["n_solved"] > bl_solved else "less"
            _, pval = fisher_exact(table, alternative=alt)
            print(f"    Fisher p (K={data['K']} vs baseline): {pval:.6f}")
            all_results[f"fisher_p_K{data['K']}_vs_baseline"] = round(pval, 6)

    # Monotonic check
    rates = [k_scaling[str(k)]["clear_rate"] for k in K_VALUES]
    is_monotonic = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
    all_results["k_monotonic"] = is_monotonic
    print(f"\n  K scaling monotonic: {is_monotonic}")

    # Verdict
    k1_rate = k_scaling.get("1", {}).get("clear_rate", 0)
    k11_rate = k_scaling.get("11", {}).get("clear_rate", 0)

    if k11_rate > k1_rate + 0.2:
        verdict = "NOISY_BEAM_SEARCH_SCALES_TO_LLM"
    elif k11_rate > k1_rate + 0.1:
        verdict = "MODERATE_K_SCALING_ON_LLM"
    elif k11_rate > k1_rate:
        verdict = "WEAK_K_SCALING_ON_LLM"
    else:
        verdict = "NO_K_SCALING_ON_LLM"
    all_results["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")
    print(f"  (K=1: {k1_rate * 100:.1f}% → K=11: {k11_rate * 100:.1f}%, "
          f"Δ={((k11_rate - k1_rate) * 100):+.1f}pp)")

    all_results["total_elapsed_s"] = round(time.time() - t0, 1)

    # Save final
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # === Visualization ===
    visualize(all_results)

    # Cleanup
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print(f"  Phase 29 COMPLETE: Mistral-7B Noisy Beam Search")
    print(f"  Total time: {all_results['total_elapsed_s'] / 3600:.1f} hours")
    print(f"  Verdict: {verdict}")
    print(f"{'=' * 80}")
    print(f"[{time.strftime('%H:%M:%S')}] Phase 29 complete!", flush=True)


if __name__ == '__main__':
    main()
