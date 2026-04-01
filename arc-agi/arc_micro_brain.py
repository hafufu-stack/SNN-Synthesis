"""
ARC-AGI LS20 Micro-Brain v5: Fixed levels_completed Bug
==========================================================
CRITICAL FIX: levels_completed persists across RESET (scorecard-based).
Solution: Create NEW env for each evaluation episode.
Optimization: Cache Oracle actions, create env only once per sigma-level combo.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, io, time, os, json
from collections import defaultdict

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi")

import arc_agi
from arcengine import GameAction
import ls20
from agent_ls20_v14 import StateSpaceSolver

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(SCRIPT_DIR, "data", "arc_oracle_data.pt")
RESULTS = os.path.join(SCRIPT_DIR, "results", "arc_noise_final.json")

ACT_IDX = {GameAction.ACTION1: 0, GameAction.ACTION2: 1,
            GameAction.ACTION3: 2, GameAction.ACTION4: 3}
IDX_ACT = {0: GameAction.ACTION1, 1: GameAction.ACTION2,
            2: GameAction.ACTION3, 3: GameAction.ACTION4}

# ============================================================
#  Models
# ============================================================

class MicroBrainSmall(nn.Module):
    """63K params."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(72, 64), nn.ReLU())
        self.output = nn.Linear(64, 4)
    
    def forward(self, grid, state, noise_sigma=0.0):
        x = self.gap(self.conv(grid)).squeeze(-1).squeeze(-1)
        h = self.head(torch.cat([x, state], dim=1))
        if noise_sigma > 0 and not self.training:
            h = h + torch.randn_like(h) * noise_sigma
        return self.output(h)

class MicroBrainLarge(nn.Module):
    """244K params."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(136, 128), nn.ReLU())
        self.output = nn.Linear(128, 4)
    
    def forward(self, grid, state, noise_sigma=0.0):
        x = self.gap(self.conv(grid)).squeeze(-1).squeeze(-1)
        h = self.head(torch.cat([x, state], dim=1))
        if noise_sigma > 0 and not self.training:
            h = h + torch.randn_like(h) * noise_sigma
        return self.output(h)

# ============================================================
#  Training
# ============================================================

def train(model, epochs=300, lr=1e-3):
    ds = torch.load(DATASET, weights_only=True)
    g, s, a = ds['grid'], ds['state'], ds['action']
    n = len(a)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(epochs):
        p = torch.randperm(n)
        loss = crit(model(g[p], s[p]), a[p])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        acc = (model(g, s).argmax(1) == a).float().mean().item()
    return acc

# ============================================================
#  Oracle Cache Builder
# ============================================================

def p2g(px, py, pxo, pyo):
    return (px - pxo) // 5, (py - pyo) // 5

def build_static_grid(info, pxo, pyo):
    grid = np.zeros((7, 12, 12), dtype=np.float32)
    ch_map = {'shape': 2, 'color': 3, 'rot': 4}
    for gx in range(12):
        for gy in range(12):
            cx, cy = pxo + gx*5, pyo + gy*5
            for wx, wy in info['walls']:
                if cx >= wx and cx < wx+5 and cy >= wy and cy < wy+5:
                    grid[0, gy, gx] = 1.0; break
    for g in info.get('goal_cells', set()):
        x, y = p2g(g[0], g[1], pxo, pyo)
        if 0 <= x < 12 and 0 <= y < 12: grid[1, y, x] = 1.0
    for pos, ct in info.get('changer_cells', {}).items():
        x, y = p2g(pos[0], pos[1], pxo, pyo)
        if 0 <= x < 12 and 0 <= y < 12 and ct in ch_map:
            grid[ch_map[ct], y, x] = 1.0
    for ec in info.get('enemy_cells', set()):
        x, y = p2g(ec[0], ec[1], pxo, pyo)
        if 0 <= x < 12 and 0 <= y < 12: grid[5, y, x] = 1.0
    for tc in info.get('timer_cells', {}).keys():
        x, y = p2g(tc[0], tc[1], pxo, pyo)
        if 0 <= x < 12 and 0 <= y < 12: grid[6, y, x] = 1.0
    return grid

class RecordingSolver(StateSpaceSolver):
    def __init__(self, f, color_order):
        super().__init__(f)
        self.color_order = color_order
        self.level_meta = {}
    def get_level_info(self, level_idx):
        info = super().get_level_info(level_idx)
        pxo, pyo = info['start'][0] % 5, info['start'][1] % 5
        lev = info['level']
        rots = [0, 90, 180, 270]
        gc = lev.get_data("GoalColor"); gr = lev.get_data("GoalRotation")
        gs = lev.get_data("kvynsvxbpi")
        if isinstance(gc, list):
            self.level_meta[level_idx] = {'sg': build_static_grid(info, pxo, pyo),
                'pxo': pxo, 'pyo': pyo, 'gs': gs[0],
                'gci': self.color_order.index(gc[0]), 'gri': rots.index(gr[0])}
        else:
            self.level_meta[level_idx] = {'sg': build_static_grid(info, pxo, pyo),
                'pxo': pxo, 'pyo': pyo, 'gs': gs,
                'gci': self.color_order.index(gc), 'gri': rots.index(gr)}
        return info

def build_oracle_cache():
    """Build Oracle cache with FRESH env (discarded after)."""
    print("Building Oracle cache...")
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    game = env._game
    obs = env.step(GameAction.RESET)
    
    original_step = env.step
    action_log = []
    cur_lv = [0]
    def rec(a):
        if a != GameAction.RESET: action_log.append((cur_lv[0], a))
        return original_step(a)
    env.step = rec
    
    solver = RecordingSolver(io.StringIO(), game.tnkekoeuk)
    ncol = len(solver.color_order)
    max_cleared = 0
    for li in range(7):
        cur_lv[0] = li
        cleared, _ = solver.solve_level(env, game, li, li)
        if not cleared: break
        max_cleared = li + 1
        print(f"  L{li+1}: {sum(1 for l,a in action_log if l==li)} actions")
    
    cache = {i: [a for l, a in action_log if l < i] for i in range(8)}
    print(f"  Cache: {max_cleared} levels, {len(action_log)} actions")
    # DON'T return env — it has stale scorecard. Each eval gets FRESH env.
    return cache, solver.level_meta, ncol, max_cleared

# ============================================================
#  Evaluation: FRESH env per episode (levels_completed = 0)
# ============================================================

def run_cnn(model, arc, target, sigma, cache, meta, ncol, max_steps=300):
    """Fresh env per episode (new scorecard), reuse Arcade instance."""
    env = arc.make("ls20")
    game = env._game
    obs = env.step(GameAction.RESET)
    
    # Oracle skip
    for a in cache.get(target, []):
        obs = env.step(a)
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False, 'steps': 0}
    
    # Verify levels_completed == target
    expected_lc = target
    # CNN plays
    m = meta[target]
    sg, pxo, pyo = m['sg'], m['pxo'], m['pyo']
    gs0, gci0, gri0 = m['gs'], m['gci'], m['gri']
    model.eval()
    
    for step in range(max_steps):
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
                       noise_sigma=sigma).argmax(1).item()
        obs = env.step(IDX_ACT[ai])
        if obs.levels_completed > expected_lc or obs.state.value == 'WIN':
            return {'cleared': True, 'steps': step+1}
        if obs.state.value == 'GAME_OVER':
            return {'cleared': False, 'steps': step+1}
    return {'cleared': False, 'steps': max_steps}

# ============================================================
#  Sweep and Plot
# ============================================================

def sweep(model, arc, sigmas, cache, meta, ncol, N=30, max_lvl=4):
    results = {}
    for sigma in sigmas:
        sr = {}
        for li in range(max_lvl):
            c, sl = 0, []
            for _ in range(N):
                r = run_cnn(model, arc, li, sigma, cache, meta, ncol)
                if r['cleared']: c += 1
                sl.append(r['steps'])
            n = len(sl)
            rate = c/max(1,n)*100
            avg = float(np.mean(sl)) if sl else 0
            sr[f"L{li+1}"] = {'rate': rate, 'avg': avg, 'n': n, 'c': c}
        results[f"{sigma:.2f}"] = sr
    return results

def plot_comparison(all_results, sigmas):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except: return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for col, (name, label, color) in enumerate([
        ("small_63K", "Small CNN (63K)", '#2196F3'),
        ("large_244K", "Large CNN (244K)", '#FF5722')
    ]):
        if name not in all_results: continue
        res = all_results[name]
        
        ax = axes[0, col]
        r1 = [res.get(f"{s:.2f}", {}).get("L1", {}).get('rate', 0) for s in sigmas]
        ax.plot(sigmas, r1, '-o', color=color, lw=2, ms=6)
        ax.set_title(f'{label}\nL1 (learned)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clear Rate (%)'); ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3); ax.set_xlabel('Noise σ')
        
        ax = axes[1, col]
        r2 = [res.get(f"{s:.2f}", {}).get("L2", {}).get('rate', 0) for s in sigmas]
        ax.plot(sigmas, r2, '-o', color=color, lw=2, ms=6)
        ax.fill_between(sigmas, r2, alpha=0.2, color=color)
        ax.set_title(f'{label}\nL2 (stochastic resonance)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clear Rate (%)'); ax.set_ylim(-5, max(max(r2)+10, 45))
        ax.grid(True, alpha=0.3); ax.set_xlabel('Noise σ')
        pi = np.argmax(r2)
        if r2[pi] > 0:
            ax.annotate(f'Peak: {r2[pi]:.0f}%\nσ={sigmas[pi]}',
                xy=(sigmas[pi], r2[pi]), xytext=(sigmas[pi]+0.12, r2[pi]+3),
                fontsize=10, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', color=color))
    
    fig.suptitle('SNN Stochastic Resonance in ARC-AGI: Model Size Effect',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results", "arc_noise_final.png"), dpi=150, bbox_inches='tight')
    
    # L2 overlay
    fig2, ax = plt.subplots(figsize=(10, 6))
    for name, label, color, mk in [
        ("small_63K", "Small CNN (63K)", '#2196F3', 'o'),
        ("large_244K", "Large CNN (244K)", '#FF5722', 's')
    ]:
        if name not in all_results: continue
        res = all_results[name]
        r2 = [res.get(f"{s:.2f}", {}).get("L2", {}).get('rate', 0) for s in sigmas]
        ax.plot(sigmas, r2, f'-{mk}', color=color, label=label, lw=2.5, ms=8)
        ax.fill_between(sigmas, r2, alpha=0.12, color=color)
    ax.set_xlabel('Noise σ', fontsize=14); ax.set_ylabel('L2 Clear Rate (%)', fontsize=14)
    ax.set_title('Stochastic Resonance: Small vs Large CNN on ARC-AGI',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results", "arc_l2_overlay.png"), dpi=150, bbox_inches='tight')
    print("Plots saved.")

# ============================================================
#  Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 60)
    print("  SNN NOISE: FINAL COMPARISON (fresh env per episode)")
    print("=" * 60)
    
    cache, meta, ncol, max_lvl = build_oracle_cache()
    
    sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    N = 30
    all_results = {}
    
    # Single Arcade instance (API key once), fresh env per episode
    print("Creating shared Arcade instance...")
    arc = arc_agi.Arcade()
    
    for name, Cls in [("small_63K", MicroBrainSmall), ("large_244K", MicroBrainLarge)]:
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        model = Cls()
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
        acc = train(model, epochs=300)
        print(f"  Train acc: {acc:.4f}")
        
        print(f"  Sweep (N={N})...")
        results = sweep(model, arc, sigmas, cache, meta, ncol, N, max_lvl)
        all_results[name] = results
        
        for s in sigmas:
            k = f"{s:.2f}"
            parts = [f"L{i+1}:{results[k][f'L{i+1}']['rate']:5.1f}%({results[k][f'L{i+1}']['avg']:.0f}s)"
                     for i in range(max_lvl)]
            print(f"    σ={s:.2f}: {' | '.join(parts)}")
    
    with open(RESULTS, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {RESULTS}")
    
    plot_comparison(all_results, sigmas)
    print(f"\nTotal: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
