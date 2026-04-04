"""
Phase 9: Dynamic σ Scheduling for ARC-AGI-3
=============================================
Tests whether dynamic noise schedules outperform static optimal σ.
Schedules: static, linear_decay, cosine_decay, warmup_decay, step_schedule
Uses Large CNN (244K) only (more stable resonance at N=100).
N=50 per schedule, L2 only.
"""
import sys, os, json, time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # arc-agi/
REPO_DIR = os.path.dirname(SCRIPT_DIR)  # snn-synthesis/
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "environment_files", "ls20", "9607627b"))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")

from arc_micro_brain import (
    MicroBrainLarge, MicroBrainSmall,
    train, build_oracle_cache, 
    ACT_IDX, IDX_ACT, p2g
)
import arc_agi
from arcengine import GameAction
import torch
import torch.nn as nn

print(f"[{time.strftime('%H:%M:%S')}] Phase 9: Dynamic σ Scheduling")
print(f"Results: {RESULTS_DIR}", flush=True)

# Setup
arc = arc_agi.Arcade()
cache, meta, ncol, max_clr = build_oracle_cache()
print(f"[{time.strftime('%H:%M:%S')}] Oracle built: {max_clr} levels", flush=True)

# ============================================================
# Dynamic Noise Schedules
# ============================================================

def static_schedule(step, max_steps, sigma):
    """Constant σ throughout"""
    return sigma

def linear_decay(step, max_steps, sigma):
    """Start high, decay to 0"""
    return sigma * (1 - step / max_steps)

def cosine_decay(step, max_steps, sigma):
    """Cosine annealing from σ to 0"""
    return sigma * 0.5 * (1 + np.cos(np.pi * step / max_steps))

def warmup_decay(step, max_steps, sigma):
    """Grok's suggestion: high→peak→low"""
    # Phase 1 (0-20%): warmup σ_max=1.5*σ (exploration)
    # Phase 2 (20-60%): hold at σ (peak resonance zone)
    # Phase 3 (60-100%): decay to 0.5*σ (stabilization)
    progress = step / max_steps
    if progress < 0.2:
        return sigma * 1.5  # explore
    elif progress < 0.6:
        return sigma  # resonate
    else:
        t = (progress - 0.6) / 0.4
        return sigma * (1.0 - 0.5 * t)  # stabilize down to 0.5*σ

def step_schedule(step, max_steps, sigma):
    """Three discrete phases"""
    progress = step / max_steps
    if progress < 0.33:
        return sigma * 1.5  # high exploration
    elif progress < 0.66:
        return sigma  # resonance
    else:
        return sigma * 0.5  # low stabilization

def inverse_schedule(step, max_steps, sigma):
    """Opposite of warmup: start low, peak in middle, end high"""
    progress = step / max_steps
    if progress < 0.3:
        return sigma * 0.5
    elif progress < 0.7:
        return sigma
    else:
        return sigma * 1.5

def flash_anneal(step, max_steps, sigma):
    """Inspired by Genesis Flash Annealing: high burst then rapid decay"""
    progress = step / max_steps
    if progress < 0.1:
        return sigma * 2.0  # flash
    else:
        return sigma * np.exp(-3 * (progress - 0.1))  # rapid exponential decay

SCHEDULES = {
    "static": static_schedule,
    "linear_decay": linear_decay,
    "cosine_decay": cosine_decay,
    "warmup_decay": warmup_decay,
    "step_schedule": step_schedule,
    "inverse": inverse_schedule,
    "flash_anneal": flash_anneal,
}

# ============================================================
# Evaluation with dynamic σ
# ============================================================

def run_l2_dynamic(model, schedule_fn, base_sigma, arc_inst, cache, meta, ncol, max_steps=300):
    """Evaluate L2 with dynamic noise schedule"""
    env = arc_inst.make("ls20")
    game = env._game
    obs = env.step(GameAction.RESET)
    
    # Oracle L1 skip
    for a in cache.get(1, []):
        obs = env.step(a)
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False, 'steps': 0, 'sigmas': []}
    
    expected_lc = 1  # L2 = index 1
    m = meta[1]
    sg, pxo, pyo = m['sg'], m['pxo'], m['pyo']
    gs0, gci0, gri0 = m['gs'], m['gci'], m['gri']
    model.eval()
    sigma_trace = []
    
    for step in range(max_steps):
        current_sigma = schedule_fn(step, max_steps, base_sigma)
        sigma_trace.append(current_sigma)
        
        pc = np.zeros((1,12,12), dtype=np.float32)
        px, py = game.gudziatsk.x, game.gudziatsk.y
        x, y = p2g(px, py, pxo, pyo)
        if 0<=x<12 and 0<=y<12: pc[0,y,x] = 1.0
        full = np.concatenate([sg, pc], axis=0)
        sv = np.array([game.fwckfzsyc/5.0, game.hiaauhahz/max(1,ncol-1),
            game.cklxociuu/3.0, gs0/5.0, gci0/max(1,ncol-1), gri0/3.0,
            max(0,1-step/200), step/200], dtype=np.float32)
        
        with torch.no_grad():
            ai = model(torch.from_numpy(full).unsqueeze(0),
                       torch.from_numpy(sv).unsqueeze(0),
                       noise_sigma=current_sigma).argmax(1).item()
        obs = env.step(IDX_ACT[ai])
        if obs.levels_completed > expected_lc or obs.state.value == 'WIN':
            return {'cleared': True, 'steps': step+1, 'sigmas': sigma_trace}
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False, 'steps': step+1, 'sigmas': sigma_trace}
    
    return {'cleared': False, 'steps': max_steps, 'sigmas': sigma_trace}

# ============================================================
# Main experiment
# ============================================================

N_EPISODES = 50
BASE_SIGMAS = [0.15, 0.20, 0.25]  # Test around the resonance zone

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
        
        for sched_name, sched_fn in SCHEDULES.items():
            t0 = time.time()
            clears = 0
            total_steps_cleared = []
            
            for ep in range(N_EPISODES):
                r = run_l2_dynamic(model, sched_fn, base_sigma, arc, cache, meta, ncol)
                if r['cleared']:
                    clears += 1
                    total_steps_cleared.append(r['steps'])
            
            rate = clears / N_EPISODES * 100
            avg_steps = float(np.mean(total_steps_cleared)) if total_steps_cleared else 0
            elapsed = time.time() - t0
            
            results[model_name][f"sigma_{base_sigma:.2f}"][sched_name] = {
                "clears": clears, "n": N_EPISODES, "rate": round(rate, 1),
                "avg_steps_when_cleared": round(avg_steps, 1),
            }
            
            sig = "★" if rate > 0 else " "
            print(f"    {sig} {sched_name:20s}: {rate:5.1f}% ({clears}/{N_EPISODES})"
                  f" avg_steps={avg_steps:5.1f} [{elapsed:.0f}s]", flush=True)

# Save results
out_path = os.path.join(RESULTS_DIR, "dynamic_sigma_scheduling.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}", flush=True)

# ============================================================
# Plot
# ============================================================
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(BASE_SIGMAS), figsize=(6*len(BASE_SIGMAS), 6))
if len(BASE_SIGMAS) == 1:
    axes = [axes]

colors = plt.cm.Set2(np.linspace(0, 1, len(SCHEDULES)))

for col, base_sigma in enumerate(BASE_SIGMAS):
    ax = axes[col]
    
    for model_name, marker, alpha in [("large_244K", 'o', 1.0), ("small_63K", 's', 0.5)]:
        if model_name not in results:
            continue
        skey = f"sigma_{base_sigma:.2f}"
        if skey not in results[model_name]:
            continue
        
        sched_names = list(SCHEDULES.keys())
        rates = [results[model_name][skey].get(s, {}).get("rate", 0) for s in sched_names]
        
        x = np.arange(len(sched_names))
        bars = ax.bar(x + (0.2 if model_name == "large_244K" else -0.2), rates, 
                      width=0.35, alpha=alpha,
                      label=model_name, 
                      color=colors[:len(sched_names)])
        
        # Add value labels
        for i, (xi, ri) in enumerate(zip(x + (0.2 if model_name == "large_244K" else -0.2), rates)):
            if ri > 0:
                ax.text(xi, ri + 0.5, f'{ri:.0f}%', ha='center', fontsize=8, fontweight='bold')
    
    ax.set_xticks(range(len(sched_names)))
    ax.set_xticklabels(sched_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('L2 Clear Rate (%)', fontsize=11)
    ax.set_title(f'Base σ = {base_sigma}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(35, ax.get_ylim()[1]))

fig.suptitle('Phase 9: Dynamic σ Scheduling vs Static (ARC-AGI-3 L2, N=50)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

plot_path = os.path.join(RESULTS_DIR, "dynamic_scheduling.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
import shutil
shutil.copy(plot_path, os.path.join(FIGURES_DIR, "arc_dynamic_scheduling.png"))
print(f"Plot saved: {plot_path}", flush=True)

# Schedule visualization
fig2, ax2 = plt.subplots(figsize=(10, 5))
steps = np.arange(300)
for sched_name, sched_fn in SCHEDULES.items():
    sigmas = [sched_fn(s, 300, 0.2) for s in steps]
    ax2.plot(steps, sigmas, lw=2, label=sched_name)
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Effective σ', fontsize=12)
ax2.set_title('Noise Schedule Profiles (base σ=0.2)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, ncol=2)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
sched_viz_path = os.path.join(RESULTS_DIR, "schedule_profiles.png")
plt.savefig(sched_viz_path, dpi=150, bbox_inches='tight')
shutil.copy(sched_viz_path, os.path.join(FIGURES_DIR, "arc_schedule_profiles.png"))

# Summary
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
for model_name in results:
    print(f"\n  {model_name}:")
    for skey in results[model_name]:
        print(f"    {skey}:")
        best_name, best_rate = "", 0
        for sched_name, entry in results[model_name][skey].items():
            if entry["rate"] > best_rate:
                best_rate = entry["rate"]
                best_name = sched_name
        if best_name:
            print(f"      BEST: {best_name} ({best_rate}%)")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 9 complete!", flush=True)
