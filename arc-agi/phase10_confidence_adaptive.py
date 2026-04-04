"""
Phase 10: Confidence-Adaptive Noise Injection
==============================================
Instead of fixed σ, noise is modulated by the model's own uncertainty.
High entropy (uncertain) → more noise (explore)
Low entropy (confident) → less noise (don't disturb)

Biologically inspired: neurons are more stochastic when signal is weak.
N=100 per condition for Fisher exact test significance.
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # arc-agi/
REPO_DIR = os.path.dirname(SCRIPT_DIR)  # snn-synthesis/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "environment_files", "ls20", "9607627b"))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

from arc_micro_brain import (
    MicroBrainLarge, MicroBrainSmall,
    train, build_oracle_cache, p2g,
    IDX_ACT
)
import arc_agi
from arcengine import GameAction

print(f"[{time.strftime('%H:%M:%S')}] Phase 10: Confidence-Adaptive Noise")
print(f"Results: {RESULTS_DIR}", flush=True)

# Setup
arc = arc_agi.Arcade()
cache, meta, ncol, max_clr = build_oracle_cache()
print(f"[{time.strftime('%H:%M:%S')}] Oracle: {max_clr} levels", flush=True)

# ============================================================
# Noise Strategies
# ============================================================

def compute_entropy(logits):
    """Compute normalized entropy of softmax distribution (0=certain, 1=max uncertain)"""
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    max_entropy = np.log(logits.shape[1])  # log(4) = 1.386
    return (entropy / max_entropy).item()  # normalized to [0, 1]

def static_noise(logits, base_sigma, step, max_steps):
    """Baseline: constant σ"""
    return base_sigma

def entropy_proportional(logits, base_sigma, step, max_steps):
    """σ = base_sigma * entropy (linear scaling)
    Uncertain → full noise, Confident → zero noise"""
    h = compute_entropy(logits)
    return base_sigma * h

def entropy_amplified(logits, base_sigma, step, max_steps):
    """σ = base_sigma * (2 * entropy) capped at 2*base
    Amplifies noise when uncertain, still zero when confident"""
    h = compute_entropy(logits)
    return min(base_sigma * 2 * h, base_sigma * 2)

def entropy_threshold(logits, base_sigma, step, max_steps):
    """Binary: if entropy > 0.5 → inject 1.5*σ, else → 0.5*σ
    Sharp decision boundary"""
    h = compute_entropy(logits)
    return base_sigma * 1.5 if h > 0.5 else base_sigma * 0.5

def confidence_gate(logits, base_sigma, step, max_steps):
    """Only inject noise when max_prob < 0.7 (uncertain)
    Confident decisions pass through unperturbed"""
    probs = F.softmax(logits, dim=1)
    max_prob = probs.max().item()
    if max_prob > 0.7:
        return 0.0  # confident → no noise
    else:
        return base_sigma * (1.0 - max_prob) / 0.3  # scale by uncertainty

def inverse_confidence(logits, base_sigma, step, max_steps):
    """σ = base_sigma * (1 - max_prob)
    Direct mapping: 100% confident → 0 noise, 25% confident → 0.75*σ"""
    probs = F.softmax(logits, dim=1)
    max_prob = probs.max().item()
    return base_sigma * (1 - max_prob) * 4  # scale factor so 50% conf → σ

def entropy_plus_decay(logits, base_sigma, step, max_steps):
    """Combine entropy-adaptive + temporal decay
    Early: entropy-driven exploration. Late: stabilize."""
    h = compute_entropy(logits)
    temporal = max(0, 1 - step / max_steps)
    return base_sigma * h * (0.5 + 0.5 * temporal)

STRATEGIES = {
    "static":               static_noise,
    "entropy_proportional": entropy_proportional,
    "entropy_amplified":    entropy_amplified,
    "entropy_threshold":    entropy_threshold,
    "confidence_gate":      confidence_gate,
    "inverse_confidence":   inverse_confidence,
    "entropy_plus_decay":   entropy_plus_decay,
}

# ============================================================
# Evaluation with adaptive noise
# ============================================================

def run_l2_adaptive(model, strategy_fn, base_sigma, arc_inst, cache, meta, ncol, max_steps=300):
    """Evaluate L2 with confidence-adaptive noise. 
    Noise is computed BEFORE action selection, injected into hidden layer."""
    env = arc_inst.make("ls20")
    game = env._game
    obs = env.step(GameAction.RESET)
    
    # Oracle L1 skip
    for a in cache.get(1, []):
        obs = env.step(a)
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False, 'steps': 0, 'avg_sigma': 0, 'avg_entropy': 0}
    
    expected_lc = 1
    m = meta[1]
    sg, pxo, pyo = m['sg'], m['pxo'], m['pyo']
    gs0, gci0, gri0 = m['gs'], m['gci'], m['gri']
    model.eval()
    
    sigmas_used = []
    entropies = []
    
    for step in range(max_steps):
        pc = np.zeros((1,12,12), dtype=np.float32)
        px, py = game.gudziatsk.x, game.gudziatsk.y
        x, y = p2g(px, py, pxo, pyo)
        if 0<=x<12 and 0<=y<12: pc[0,y,x] = 1.0
        full = np.concatenate([sg, pc], axis=0)
        sv = np.array([game.fwckfzsyc/5.0, game.hiaauhahz/max(1,ncol-1),
            game.cklxociuu/3.0, gs0/5.0, gci0/max(1,ncol-1), gri0/3.0,
            max(0,1-step/200), step/200], dtype=np.float32)
        
        grid_t = torch.from_numpy(full).unsqueeze(0)
        state_t = torch.from_numpy(sv).unsqueeze(0)
        
        with torch.no_grad():
            # Forward pass WITHOUT noise first to get logits for confidence
            clean_logits = model(grid_t, state_t, noise_sigma=0.0)
            
            # Compute adaptive sigma
            adaptive_sigma = strategy_fn(clean_logits, base_sigma, step, max_steps)
            sigmas_used.append(adaptive_sigma)
            entropies.append(compute_entropy(clean_logits))
            
            # Now do the actual forward pass WITH computed noise
            if adaptive_sigma > 0:
                noisy_logits = model(grid_t, state_t, noise_sigma=adaptive_sigma)
            else:
                noisy_logits = clean_logits
            
            ai = noisy_logits.argmax(1).item()
        
        obs = env.step(IDX_ACT[ai])
        if obs.levels_completed > expected_lc or obs.state.value == 'WIN':
            return {
                'cleared': True, 'steps': step+1,
                'avg_sigma': float(np.mean(sigmas_used)),
                'avg_entropy': float(np.mean(entropies))
            }
        if obs.state.value == 'GAME_OVER':
            return {
                'cleared': False, 'steps': step+1,
                'avg_sigma': float(np.mean(sigmas_used)),
                'avg_entropy': float(np.mean(entropies))
            }
    
    return {
        'cleared': False, 'steps': max_steps,
        'avg_sigma': float(np.mean(sigmas_used)),
        'avg_entropy': float(np.mean(entropies))
    }

# ============================================================
# Main experiment: N=100 per condition
# ============================================================

N = 100
BASE_SIGMAS = [0.20, 0.25]  # optimal zone from Phase 8
results = {}

for model_name, model_cls in [("large_244K", MicroBrainLarge), ("small_63K", MicroBrainSmall)]:
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}", flush=True)
    
    model = model_cls()
    acc = train(model)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  Trained: {nparams:,} params, acc={acc:.4f}", flush=True)
    
    results[model_name] = {}
    
    for base_sigma in BASE_SIGMAS:
        results[model_name][f"sigma_{base_sigma:.2f}"] = {}
        print(f"\n  --- Base σ = {base_sigma} ---", flush=True)
        
        for strat_name, strat_fn in STRATEGIES.items():
            t0 = time.time()
            clears = 0
            all_avg_sigma = []
            all_avg_entropy = []
            steps_when_cleared = []
            
            for ep in range(N):
                r = run_l2_adaptive(model, strat_fn, base_sigma, arc, cache, meta, ncol)
                if r['cleared']:
                    clears += 1
                    steps_when_cleared.append(r['steps'])
                all_avg_sigma.append(r['avg_sigma'])
                all_avg_entropy.append(r['avg_entropy'])
            
            rate = clears / N * 100
            elapsed = time.time() - t0
            
            entry = {
                "clears": clears, "n": N, "rate": round(rate, 1),
                "avg_sigma_injected": round(float(np.mean(all_avg_sigma)), 4),
                "avg_entropy": round(float(np.mean(all_avg_entropy)), 4),
                "avg_steps_cleared": round(float(np.mean(steps_when_cleared)), 1) if steps_when_cleared else 0,
            }
            results[model_name][f"sigma_{base_sigma:.2f}"][strat_name] = entry
            
            sig = "★" if rate > 0 else " "
            print(f"    {sig} {strat_name:25s}: {rate:5.1f}% ({clears}/{N})"
                  f" σ_avg={entry['avg_sigma_injected']:.3f}"
                  f" H_avg={entry['avg_entropy']:.3f}"
                  f" [{elapsed:.0f}s]", flush=True)

# Add Fisher exact tests
from scipy import stats as sp_stats

for model_name in results:
    for skey in results[model_name]:
        baseline = results[model_name][skey].get("static", {})
        bl_c = baseline.get("clears", 0)
        bl_n = baseline.get("n", N)
        
        for strat_name, entry in results[model_name][skey].items():
            if strat_name == "static":
                entry["fisher_p"] = None
                continue
            table = [[entry["clears"], entry["n"] - entry["clears"]],
                     [bl_c, bl_n - bl_c]]
            _, p = sp_stats.fisher_exact(table)
            entry["fisher_p"] = round(p, 6)
            
            # Wilson CI
            k, nn = entry["clears"], entry["n"]
            z = 1.96
            p_hat = k / nn
            denom = 1 + z**2 / nn
            center = (p_hat + z**2 / (2*nn)) / denom
            margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*nn)) / nn) / denom
            entry["ci_lower"] = round(max(0, center - margin) * 100, 1)
            entry["ci_upper"] = round(min(1, center + margin) * 100, 1)

# Save
out_path = os.path.join(RESULTS_DIR, "confidence_adaptive_noise.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}", flush=True)

# ============================================================
# Plot
# ============================================================
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(BASE_SIGMAS), figsize=(8*len(BASE_SIGMAS), 7))
if len(BASE_SIGMAS) == 1:
    axes = [axes]

strat_names = list(STRATEGIES.keys())
colors = {
    "static": '#666666',
    "entropy_proportional": '#2196F3',
    "entropy_amplified": '#1976D2',
    "entropy_threshold": '#FF9800',
    "confidence_gate": '#4CAF50',
    "inverse_confidence": '#9C27B0',
    "entropy_plus_decay": '#F44336',
}

for col, base_sigma in enumerate(BASE_SIGMAS):
    ax = axes[col]
    skey = f"sigma_{base_sigma:.2f}"
    
    x = np.arange(len(strat_names))
    width = 0.35
    
    for i, (model_name, offset, alpha) in enumerate([
        ("large_244K", -width/2, 1.0), ("small_63K", width/2, 0.6)
    ]):
        if model_name not in results or skey not in results[model_name]:
            continue
        rates = [results[model_name][skey].get(s, {}).get("rate", 0) for s in strat_names]
        bar_colors = [colors.get(s, '#999') for s in strat_names]
        
        bars = ax.bar(x + offset, rates, width, alpha=alpha,
                      color=bar_colors, edgecolor='white', linewidth=0.5,
                      label=model_name)
        
        # Value labels
        for xi, ri in zip(x + offset, rates):
            if ri > 0:
                ax.text(xi, ri + 0.5, f'{ri:.0f}', ha='center', fontsize=8, fontweight='bold')
    
    # Mark static baseline
    bl_rate = results.get("large_244K", {}).get(skey, {}).get("static", {}).get("rate", 0)
    ax.axhline(y=bl_rate, color='red', linestyle='--', alpha=0.5, label=f'static baseline ({bl_rate}%)')
    
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in strat_names], fontsize=8)
    ax.set_ylabel('L2 Clear Rate (%)', fontsize=11)
    ax.set_title(f'Base σ = {base_sigma}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Phase 10: Confidence-Adaptive Noise (N=100, ARC-AGI-3 L2)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "confidence_adaptive.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
import shutil
shutil.copy(plot_path, os.path.join(FIGURES_DIR, "arc_confidence_adaptive.png"))
print(f"Plot: {plot_path}", flush=True)

# ============================================================
# Summary with Fisher tests
# ============================================================
print(f"\n{'='*60}")
print(f"  FINAL SUMMARY (N={N}, Fisher exact test vs static)")
print(f"{'='*60}")

for model_name in results:
    print(f"\n  {model_name}:")
    for skey in results[model_name]:
        print(f"    {skey}:")
        bl = results[model_name][skey].get("static", {})
        print(f"      {'Strategy':25s} {'Rate':>6s} {'c/n':>7s} {'σ_avg':>6s} {'H_avg':>6s} {'Fisher p':>10s}")
        for strat_name in strat_names:
            e = results[model_name][skey].get(strat_name, {})
            p_str = f"p={e.get('fisher_p', 0):.4f}" if e.get('fisher_p') is not None else "(baseline)"
            sig = "***" if e.get('fisher_p') and e['fisher_p'] < 0.001 else \
                  "**" if e.get('fisher_p') and e['fisher_p'] < 0.01 else \
                  "*" if e.get('fisher_p') and e['fisher_p'] < 0.05 else ""
            print(f"      {strat_name:25s} {e.get('rate',0):5.1f}% "
                  f"{e.get('clears',0):3d}/{e.get('n',0):3d} "
                  f"{e.get('avg_sigma_injected',0):5.3f} "
                  f"{e.get('avg_entropy',0):5.3f} "
                  f"{p_str:>12s} {sig}")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 10 complete!", flush=True)
