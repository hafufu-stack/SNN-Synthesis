"""
Phase 6: Cross-Task Trajectory Transfer
==========================================

Test whether the Aha!/Trajectory vectors extracted from Modified Hanoi
transfer to a completely different reasoning task: arithmetic word problems.

If the vectors encode general "reasoning breakthrough" patterns rather than
task-specific knowledge, they should improve performance on unrelated tasks.

Architecture:
  - Qwen2.5-0.5B-Instruct (same as Phase 1, 3)
  - Task: 2-step arithmetic word problems
  - Phase 91 diff_unit projected to 896-dim
  - Phase 109 trajectory_unit projected to 896-dim

Conditions (N=80 each):
  1. baseline:         No injection
  2. hanoi_aha:        Hanoi Aha! vector (diff_unit)
  3. hanoi_trajectory: Hanoi trajectory template
  4. random_direction: Random unit vector (control)
  5. anti_aha:         Reversed Aha! vector

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
HIDDEN_DIM = 896
BASE_SIGMA = 0.15
LAYER_IDX = 14
N_TEST = 80

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================================================
#  ARITHMETIC WORD PROBLEM GENERATOR
# ===================================================

def generate_word_problems(n=100, seed=SEED):
    """
    Generate 2-step arithmetic word problems that require
    sequential reasoning (multi-step, not one-shot).

    Categories:
      - Shopping (buy items, apply discount, calculate change)
      - Distance (two legs of a journey)
      - Time (work hours, breaks)
      - Sharing (divide then redistribute)
    """
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
            cost = qty1 * price
            refund = qty2 * price
            answer = cost - refund
            question = (
                f"{name} bought {qty1} {item} at ${price} each. "
                f"Later, {name} returned {qty2} of them for a full refund. "
                f"How much did {name} spend in total?"
            )
            problems.append({"question": question, "answer": answer,
                            "category": "shopping", "steps": 2})

        elif cat == 1:  # Distance
            name = rng.choice(names)
            d1 = rng.randint(10, 50)
            d2 = rng.randint(10, 50)
            answer = d1 + d2
            question = (
                f"{name} drove {d1} miles to the store, then drove {d2} miles "
                f"to the library. How many miles did {name} drive in total?"
            )
            problems.append({"question": question, "answer": answer,
                            "category": "distance", "steps": 2})

        elif cat == 2:  # Time
            name = rng.choice(names)
            hours = rng.randint(4, 10)
            breaks_count = rng.randint(1, 3)
            break_min = rng.randint(10, 30)
            total_break = breaks_count * break_min
            work_min = hours * 60 - total_break
            answer = work_min
            question = (
                f"{name} worked for {hours} hours but took {breaks_count} breaks "
                f"of {break_min} minutes each. "
                f"How many minutes did {name} actually work?"
            )
            problems.append({"question": question, "answer": answer,
                            "category": "time", "steps": 2})

        elif cat == 3:  # Sharing
            name1 = rng.choice(names)
            name2 = rng.choice([n for n in names if n != name1])
            item = rng.choice(items)
            total = rng.randint(12, 40)
            # Make divisible by 2
            total = total - (total % 2)
            each = total // 2
            give = rng.randint(1, each - 1)
            n1_final = each - give
            n2_final = each + give
            answer = n2_final
            question = (
                f"{name1} and {name2} split {total} {item} equally. "
                f"Then {name1} gave {give} {item} to {name2}. "
                f"How many {item} does {name2} have now?"
            )
            problems.append({"question": question, "answer": answer,
                            "category": "sharing", "steps": 2})

    return problems


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


def gen(model, tok, prompt, temperature=0.3, max_tokens=150):
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
#  INJECTION HOOK
# ===================================================

class DirectionHook:
    def __init__(self):
        self.active = False
        self.direction = None
        self.sigma = BASE_SIGMA
        self.handle = None

    def setup(self, direction, device):
        self.active = True
        self.direction = torch.tensor(direction, dtype=torch.float16, device=device)

    def setup_off(self):
        self.active = False

    def register(self, model, layer_idx=LAYER_IDX):
        hook_obj = self
        def hook_fn(module, args):
            if not hook_obj.active or hook_obj.sigma <= 0:
                return args
            hs = args[0]
            d = hs.shape[-1]
            det_scale = hook_obj.sigma * math.sqrt(d) * 0.5
            det_noise = hook_obj.direction * det_scale
            if hs.dim() == 3:
                det_noise = det_noise.unsqueeze(0).unsqueeze(0).expand_as(hs)
            else:
                det_noise = det_noise.unsqueeze(0).expand_as(hs)
            stoch_noise = torch.randn_like(hs) * (hook_obj.sigma * 0.5)
            return (hs + det_noise + stoch_noise,) + args[1:]
        self.handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


# ===================================================
#  EVALUATION
# ===================================================

def build_math_prompt(tokenizer, problem):
    system = (
        "You are a math tutor. Solve the word problem step by step. "
        "Show your work, then give the final answer as: Answer: <number>"
    )
    msg = f"Problem: {problem['question']}"
    messages = [{"role": "user", "content": system + "\n\n" + msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer(response):
    """Extract numeric answer from model response."""
    # Try "Answer: X" pattern first
    m = re.search(r'Answer:\s*\$?\s*(-?\d+(?:\.\d+)?)', response)
    if m:
        return float(m.group(1))

    # Try "= X" at end
    m = re.search(r'=\s*\$?\s*(-?\d+(?:\.\d+)?)\s*$', response, re.MULTILINE)
    if m:
        return float(m.group(1))

    # Try last number in response
    nums = re.findall(r'(-?\d+(?:\.\d+)?)', response)
    if nums:
        return float(nums[-1])

    return None


def evaluate_problem(model, tok, problem, hook=None):
    """Evaluate a single word problem. Returns (correct, extracted_answer, response)."""
    prompt = build_math_prompt(tok, problem)
    response = gen(model, tok, prompt)
    extracted = extract_answer(response)

    if extracted is None:
        return False, None, response

    # Allow small floating point tolerance
    correct = abs(extracted - problem["answer"]) < 0.5
    return correct, extracted, response


# ===================================================
#  VISUALIZATION
# ===================================================

def visualize(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 6: Cross-Task Trajectory Transfer\n"
                 "Hanoi vectors → Arithmetic Word Problems",
                 fontsize=12, fontweight="bold")

    # Panel 1: Overall accuracy by condition
    ax = axes[0]
    test = all_results.get("test_results", [])
    names = [t["condition"] for t in test]
    rates = [t["accuracy"] * 100 for t in test]
    colors = ["#9E9E9E", "#4CAF50", "#2196F3", "#FFC107", "#F44336"]
    bars = ax.bar(range(len(test)), rates, color=colors[:len(test)], alpha=0.85,
                  edgecolor="white", linewidth=2)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(test)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5-Way Transfer Test", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: By-category breakdown for baseline vs best injected
    ax = axes[1]
    cat_data = all_results.get("category_breakdown", {})
    if cat_data:
        categories = list(next(iter(cat_data.values())).keys())
        x = np.arange(len(categories))
        w = 0.35
        bl_rates = [cat_data.get("baseline", {}).get(c, 0) * 100 for c in categories]

        best_cond = max(test, key=lambda t: t["accuracy"] if t["condition"] != "baseline" else -1)
        best_name = best_cond["condition"]
        best_rates = [cat_data.get(best_name, {}).get(c, 0) * 100 for c in categories]

        ax.bar(x - w/2, bl_rates, w, label="Baseline", color="#9E9E9E", alpha=0.85)
        ax.bar(x + w/2, best_rates, w, label=best_name, color="#4CAF50", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.legend(fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Category Breakdown", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "phase6_cross_task_transfer.png")
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
    print(f"  Phase 6: Cross-Task Trajectory Transfer")
    print(f"  Hanoi vectors → Arithmetic problems")
    print(f"  5 conditions x N={N_TEST}")
    print(f"{'='*80}")

    t0 = time.time()

    # Generate test problems
    problems = generate_word_problems(n=N_TEST, seed=SEED)
    print(f"  Generated {len(problems)} word problems")
    cats = {}
    for p in problems:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"  Categories: {cats}")

    # Load Hanoi vectors (Mistral 4096-dim → Qwen 896-dim, bundled in data/)
    diff_pca_path = os.path.join(DATA_DIR, "diff_pca.npz")
    traj_path = os.path.join(DATA_DIR, "trajectory_template.npz")

    diff_data = np.load(diff_pca_path)
    diff_unit_full = diff_data["diff_unit"]  # (4096,)
    traj_data = np.load(traj_path)
    traj_unit_full = traj_data["trajectory_unit"]  # (4096,)

    # Project to Qwen's 896-dim via equispaced sampling
    indices = np.linspace(0, len(diff_unit_full)-1, HIDDEN_DIM).astype(int)
    diff_unit = diff_unit_full[indices].astype(np.float32)
    diff_unit /= (np.linalg.norm(diff_unit) + 1e-8)

    traj_unit = traj_unit_full[indices].astype(np.float32)
    traj_unit /= (np.linalg.norm(traj_unit) + 1e-8)

    # Random direction (control)
    rng = np.random.RandomState(SEED)
    rand_unit = rng.randn(HIDDEN_DIM).astype(np.float32)
    rand_unit /= (np.linalg.norm(rand_unit) + 1e-8)

    print(f"  Diff unit projected: {diff_unit_full.shape} → {diff_unit.shape}")
    print(f"  Traj unit projected: {traj_unit_full.shape} → {traj_unit.shape}")

    model, tok = load_model()
    device = next(model.parameters()).device

    hook = DirectionHook()
    hook.register(model, LAYER_IDX)

    all_results = {
        "experiment": "Phase 6: Cross-Task Trajectory Transfer",
        "model": MODEL_SHORT,
        "layer": LAYER_IDX,
        "sigma": BASE_SIGMA,
        "n_test": N_TEST,
        "source_task": "Modified Hanoi (3-disk)",
        "target_task": "Arithmetic Word Problems",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.join(RESULTS_DIR, "phase6_log.json")

    # === Test 5 conditions ===
    conditions = [
        {"name": "baseline",         "direction": None},
        {"name": "hanoi_aha",        "direction": diff_unit},
        {"name": "hanoi_trajectory", "direction": traj_unit},
        {"name": "random_direction", "direction": rand_unit},
        {"name": "anti_aha",         "direction": -diff_unit},
    ]

    test_results = []
    category_breakdown = {}

    for cfg in conditions:
        print(f"\n  === Condition: {cfg['name']} ===")
        if cfg["direction"] is None:
            hook.setup_off()
        else:
            hook.setup(cfg["direction"], device)

        correct_count = 0
        cat_correct = {}
        cat_total = {}
        details = []

        for pi, problem in enumerate(problems):
            is_correct, extracted, response = evaluate_problem(model, tok, problem, hook)

            cat = problem["category"]
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if is_correct:
                correct_count += 1
                cat_correct[cat] = cat_correct.get(cat, 0) + 1

            details.append({
                "idx": pi,
                "category": cat,
                "correct": is_correct,
                "expected": problem["answer"],
                "extracted": extracted,
            })

            if (pi + 1) % 40 == 0:
                acc = correct_count / (pi + 1) * 100
                print(f"    [{pi+1}/{N_TEST}] {acc:.1f}%")

        accuracy = correct_count / N_TEST
        print(f"    Final: {correct_count}/{N_TEST} = {accuracy*100:.1f}%")

        # Per-category accuracy
        cat_acc = {}
        for c in cat_total:
            cat_acc[c] = round(cat_correct.get(c, 0) / cat_total[c], 4) if cat_total[c] > 0 else 0
        category_breakdown[cfg["name"]] = cat_acc
        print(f"    By category: {cat_acc}")

        test_results.append({
            "condition": cfg["name"],
            "accuracy": round(accuracy, 4),
            "n_correct": correct_count,
            "n_total": N_TEST,
            "category_accuracy": cat_acc,
            "details": details,
        })

        # Save intermediate
        all_results["test_results"] = test_results
        all_results["category_breakdown"] = category_breakdown
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    hook.remove()

    # === Analysis ===
    print(f"\n  === Results Summary ===")
    bl = test_results[0]
    for tr in test_results:
        delta = tr["accuracy"] - bl["accuracy"]
        print(f"    {tr['condition']:20s}: {tr['accuracy']*100:5.1f}% (Δ={delta*100:+.1f}pp)")

    # Fisher exact: aha vs baseline
    aha_correct = test_results[1]["n_correct"]
    bl_correct = test_results[0]["n_correct"]
    if aha_correct != bl_correct:
        alt = "greater" if aha_correct > bl_correct else "less"
        table = [[aha_correct, N_TEST - aha_correct],
                 [bl_correct, N_TEST - bl_correct]]
        _, pval = fisher_exact(table, alternative=alt)
        all_results["fisher_p_aha_vs_baseline"] = round(pval, 6)
        print(f"    Fisher p (aha vs baseline): {pval:.6f}")

    # Cosine similarity between hanoi vectors and random
    cos_aha_rand = np.dot(diff_unit, rand_unit)
    cos_traj_rand = np.dot(traj_unit, rand_unit)
    cos_aha_traj = np.dot(diff_unit, traj_unit)
    all_results["vector_cosines"] = {
        "aha_vs_random": round(float(cos_aha_rand), 6),
        "traj_vs_random": round(float(cos_traj_rand), 6),
        "aha_vs_trajectory": round(float(cos_aha_traj), 6),
    }
    print(f"    cos(aha, random) = {cos_aha_rand:.4f}")
    print(f"    cos(traj, random) = {cos_traj_rand:.4f}")
    print(f"    cos(aha, traj) = {cos_aha_traj:.4f}")

    # Verdict
    aha_rate = test_results[1]["accuracy"]
    traj_rate = test_results[2]["accuracy"]
    rand_rate = test_results[3]["accuracy"]
    anti_rate = test_results[4]["accuracy"]
    bl_rate = test_results[0]["accuracy"]

    best_hanoi = max(aha_rate, traj_rate)
    if best_hanoi > rand_rate + 0.05 and best_hanoi > bl_rate + 0.03:
        verdict = "CROSS_TASK_TRANSFER_CONFIRMED"
        print(f"\n  VERDICT: {verdict} — Hanoi vectors improve arithmetic!")
    elif best_hanoi > bl_rate + 0.03:
        verdict = "POSSIBLE_TRANSFER"
        print(f"\n  VERDICT: {verdict} — Improvement but not clearly above random")
    elif anti_rate < bl_rate - 0.05:
        verdict = "ANTI_AHA_TRANSFER_ONLY"
        print(f"\n  VERDICT: {verdict} — Reversed vector hurts, but forward doesn't help")
    elif abs(best_hanoi - rand_rate) < 0.03:
        verdict = "GENERAL_NOISE_EFFECT"
        print(f"\n  VERDICT: {verdict} — All directions equally effective/ineffective")
    else:
        verdict = "NO_TRANSFER"
        print(f"\n  VERDICT: {verdict} — Hanoi vectors do not transfer to arithmetic")

    all_results["verdict"] = verdict

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
    print(f"\n Phase 6 complete.")
