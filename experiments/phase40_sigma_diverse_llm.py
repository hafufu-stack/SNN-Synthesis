"""
Phase 40: σ-Diverse NBS on LLMs
Extend Phase 37a's CNN result to LLM scale (Mistral-7B).
Test if σ-diverse NBS eliminates need for σ* tuning on LLMs too.

Author: Hiroto Funasaki
"""
import os, json, gc, time, random, numpy as np, torch
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"

# Modified Hanoi task (from Phase 29)
def generate_hanoi_problems(n=50, seed=42):
    """Generate Modified Hanoi problems with solutions."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        n_disks = rng.randint(2, 4)
        source = rng.choice(['A', 'B', 'C'])
        target = rng.choice([p for p in ['A', 'B', 'C'] if p != source])
        problems.append({
            'n_disks': n_disks,
            'source': source,
            'target': target,
            'prompt': f"Move {n_disks} disks from peg {source} to peg {target} in Tower of Hanoi. "
                      f"Show each move as (disk, from_peg, to_peg). Never place larger disk on smaller."
        })
    return problems


def generate_gsm8k_problems(n=50, seed=42):
    """Generate simple math word problems (GSM8K-style)."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        op = rng.choice(['+', '-', '*'])
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:
            answer = a * b
        problems.append({
            'a': a, 'b': b, 'op': op, 'answer': answer,
            'prompt': f"What is {a} {op} {b}? Give only the number."
        })
    return problems


def run_nbs_on_llm(model, tokenizer, problems, sigma, n_beams=1,
                   max_new_tokens=200, device='cuda'):
    """Run NBS with given sigma on LLM."""
    correct = 0
    for prob in problems:
        prompt = prob['prompt']
        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        # Temperature = 0.3 + sigma * 2.0 (same mapping as Kaggle agent)
        temperature = max(0.1, min(2.0, 0.3 + sigma * 2.0))

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0.01 else None,
                do_sample=temperature > 0.01,
                top_p=0.9 if temperature > 0.01 else None,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                     skip_special_tokens=True).strip()

        if 'answer' in prob:
            # GSM8K: check numeric answer
            try:
                nums = [int(s) for s in response.replace(',', '').split()
                        if s.lstrip('-').isdigit()]
                if nums and nums[-1] == prob['answer']:
                    correct += 1
            except:
                pass
        elif 'n_disks' in prob:
            # Hanoi: check if response contains correct number of moves
            n_moves = response.count('->')  + response.count('→')
            expected = 2 ** prob['n_disks'] - 1
            if n_moves >= expected * 0.8:  # allow some formatting variance
                correct += 1

    return correct / len(problems) if problems else 0


def main():
    print("=" * 60)
    print("Phase 40: σ-Diverse NBS on LLMs (Mistral-7B)")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("\nLoading Mistral-7B (local_files_only=True)...")
    tokenizer = AutoTokenizer.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.3', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        'mistralai/Mistral-7B-Instruct-v0.3',
        local_files_only=True, device_map='auto', torch_dtype=torch.float16)
    model.eval()
    print("Model loaded!")

    # Test tasks
    hanoi_problems = generate_hanoi_problems(30, seed=42)
    gsm_problems = generate_gsm8k_problems(30, seed=42)

    # σ values to test
    sigmas = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    all_results = {}

    # --- Fixed-σ experiments ---
    print(f"\n--- Fixed-σ experiments ---")
    print(f"{'σ':>8s} | {'GSM8K':>8s} {'Hanoi':>8s}")
    print("-" * 30)

    for sigma in sigmas:
        t0 = time.time()
        gsm_acc = run_nbs_on_llm(model, tokenizer, gsm_problems, sigma, device='cuda')
        hanoi_acc = run_nbs_on_llm(model, tokenizer, hanoi_problems, sigma, device='cuda')
        elapsed = time.time() - t0
        print(f"{sigma:>8.3f} | {gsm_acc*100:>7.1f}% {hanoi_acc*100:>7.1f}%  ({elapsed:.0f}s)")

        all_results[f"fixed_sigma_{sigma}"] = {
            'sigma': sigma,
            'gsm8k_acc': gsm_acc,
            'hanoi_acc': hanoi_acc,
        }

    # --- σ-Diverse NBS (K=11) ---
    print(f"\n--- σ-Diverse NBS (best of K=11) ---")
    diverse_sigmas = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    # GSM8K: run all σ values and take best answer per problem
    gsm_diverse_correct = 0
    for prob in gsm_problems:
        any_correct = False
        for sigma in diverse_sigmas:
            acc = run_nbs_on_llm(model, tokenizer, [prob], sigma, device='cuda')
            if acc > 0:
                any_correct = True
                break
        if any_correct:
            gsm_diverse_correct += 1

    # Hanoi: same approach
    hanoi_diverse_correct = 0
    for prob in hanoi_problems:
        any_correct = False
        for sigma in diverse_sigmas:
            acc = run_nbs_on_llm(model, tokenizer, [prob], sigma, device='cuda')
            if acc > 0:
                any_correct = True
                break
        if any_correct:
            hanoi_diverse_correct += 1

    gsm_diverse_rate = gsm_diverse_correct / len(gsm_problems)
    hanoi_diverse_rate = hanoi_diverse_correct / len(hanoi_problems)

    print(f"  σ-Diverse GSM8K: {gsm_diverse_rate*100:.1f}%")
    print(f"  σ-Diverse Hanoi: {hanoi_diverse_rate*100:.1f}%")

    # Find best fixed-σ
    best_gsm = max(all_results.values(), key=lambda x: x.get('gsm8k_acc', 0))
    best_hanoi = max(all_results.values(), key=lambda x: x.get('hanoi_acc', 0))

    all_results['sigma_diverse'] = {
        'gsm8k_acc': gsm_diverse_rate,
        'hanoi_acc': hanoi_diverse_rate,
    }
    all_results['best_fixed'] = {
        'gsm8k_best_sigma': best_gsm['sigma'],
        'gsm8k_best_acc': best_gsm['gsm8k_acc'],
        'hanoi_best_sigma': best_hanoi['sigma'],
        'hanoi_best_acc': best_hanoi['hanoi_acc'],
    }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Best fixed-σ GSM8K:    σ={best_gsm['sigma']:.3f} → {best_gsm['gsm8k_acc']*100:.1f}%")
    print(f"  σ-Diverse GSM8K:       {gsm_diverse_rate*100:.1f}%  "
          f"(gap: {(gsm_diverse_rate - best_gsm['gsm8k_acc'])*100:+.1f}pp)")
    print(f"  Best fixed-σ Hanoi:    σ={best_hanoi['sigma']:.3f} → {best_hanoi['hanoi_acc']*100:.1f}%")
    print(f"  σ-Diverse Hanoi:       {hanoi_diverse_rate*100:.1f}%  "
          f"(gap: {(hanoi_diverse_rate - best_hanoi['hanoi_acc'])*100:+.1f}pp)")

    save_path = os.path.join(RESULTS_DIR, "phase40_sigma_diverse_llm.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 40: σ-Diverse NBS on LLMs',
            'timestamp': datetime.now().isoformat(),
            'model': 'Mistral-7B-Instruct-v0.3',
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")

    # Cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
