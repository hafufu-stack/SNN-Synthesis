"""
Phase 17: Game-Agnostic CNN Pipeline for m0r0
==============================================
Step 1: Collect successful trajectories via random K=21
Step 2: Extract observation features
Step 3: Train CNN (Behavioral Cloning)
Step 4: Test CNN + Noise + Trajectory Ensemble
"""
import sys, os, json, time, random, gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'environment_files')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

import arc_agi
from arcengine import GameAction

ALL_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2,
               GameAction.ACTION3, GameAction.ACTION4]
ACT_IDX = {a: i for i, a in enumerate(ALL_ACTIONS)}

N_WORKERS = min(cpu_count() - 4, 20)

print(f"[{time.strftime('%H:%M:%S')}] Phase 17: m0r0 Game-Agnostic CNN Pipeline")
print(f"  Workers: {N_WORKERS}", flush=True)

# ============================================================
# Step 1: Explore m0r0 observation structure
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 1: Explore m0r0 observations")
print(f"{'='*60}", flush=True)

arcade = arc_agi.Arcade()
env = arcade.make("m0r0")
obs = env.step(GameAction.RESET)

print(f"  Observation type: {type(obs).__name__}")
print(f"  Observation attrs: {[a for a in dir(obs) if not a.startswith('_')]}")
print(f"  state: {obs.state}")
print(f"  levels_completed: {obs.levels_completed}")

if hasattr(obs, 'pixels'):
    px = obs.pixels
    print(f"  pixels type: {type(px).__name__}")
    if hasattr(px, 'shape'):
        print(f"  pixels shape: {px.shape}")
    elif isinstance(px, (list, tuple)):
        print(f"  pixels length: {len(px)}")
        if len(px) > 0 and isinstance(px[0], (list, tuple)):
            print(f"  pixels[0] length: {len(px[0])}")

# Try to get pixel observations after a few actions
game = env._game
print(f"\n  Game class: {type(game).__name__}")
game_attrs = {}
for attr in dir(game):
    if not attr.startswith('_'):
        try:
            val = getattr(game, attr)
            if isinstance(val, (int, float, str, bool)):
                game_attrs[attr] = val
        except:
            pass
print(f"  Game numeric attrs ({len(game_attrs)}):")
for k, v in sorted(game_attrs.items())[:20]:
    print(f"    {k} = {v}")

# Understand observation by stepping
print(f"\n  Stepping through game:")
for i in range(10):
    action = random.choice(ALL_ACTIONS)
    obs = env.step(action)
    px = np.array(obs.pixels) if hasattr(obs, 'pixels') else None
    print(f"    Step {i+1}: action={action.name}, state={obs.state.value}, "
          f"lc={obs.levels_completed}", end="")
    if px is not None:
        print(f", pixels={px.shape if hasattr(px,'shape') else len(px)}", end="")
    print()
    if obs.state.value in ('GAME_OVER', 'WIN'):
        break

# ============================================================
# Step 2: Collect successful trajectories
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 2: Collect successful trajectories (random K=21)")
print(f"{'='*60}", flush=True)

def collect_trajectory(args):
    """Run one trajectory, return (actions, observations, levels_cleared)."""
    game_id, max_steps, seed = args
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import arc_agi
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    
    rng = random.Random(seed)
    arcade = arc_agi.Arcade()
    env = arcade.make(game_id)
    obs = env.step(GameAction.RESET)
    
    actions = []
    pixels_list = []
    
    for step in range(max_steps):
        action = rng.choice(ALL_A)
        
        # Record pixel observation BEFORE taking action
        px = np.array(obs.pixels, dtype=np.uint8) if hasattr(obs, 'pixels') else None
        if px is not None:
            pixels_list.append(px)
        actions.append(action.value)
        
        try:
            obs = env.step(action)
            if obs.state.value in ('GAME_OVER', 'WIN'):
                break
        except:
            break
    
    return {
        'actions': actions,
        'levels_cleared': obs.levels_completed,
        'n_steps': len(actions),
        'pixel_shape': px.shape if px is not None else None,
        'pixels': pixels_list if obs.levels_completed > 0 else None  # Only save if successful
    }

# Collect many trajectories in parallel
N_COLLECT = 500  # Total trajectories
print(f"  Collecting {N_COLLECT} random trajectories...", flush=True)
t0 = time.time()

tasks = [("m0r0", 300, i*7 + 42) for i in range(N_COLLECT)]

success_trajs = []
total_clears = 0

with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    results = list(executor.map(collect_trajectory, tasks))

for r in results:
    if r['levels_cleared'] > 0:
        total_clears += 1
        if r['pixels'] is not None:
            success_trajs.append(r)

elapsed = time.time() - t0
print(f"  Collected {N_COLLECT} trajectories in {elapsed:.0f}s")
print(f"  Successful: {total_clears}/{N_COLLECT} ({total_clears/N_COLLECT*100:.1f}%)")
print(f"  With pixel data: {len(success_trajs)}")
if results[0]['pixel_shape']:
    print(f"  Pixel shape: {results[0]['pixel_shape']}")

# ============================================================
# Step 3: Build training dataset and train CNN
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 3: Train CNN (Behavioral Cloning)")
print(f"{'='*60}", flush=True)

import torch
import torch.nn as nn
import torch.optim as optim

if len(success_trajs) < 5:
    print(f"  Not enough successful trajectories ({len(success_trajs)}). Need >= 5.")
    print(f"  Trying with K=21 best-of-K to get more data...")
    
    # Use best-of-K to get more successes
    K_COLLECT = 21
    N_ROUNDS = 200
    tasks2 = []
    for r in range(N_ROUNDS):
        for k in range(K_COLLECT):
            tasks2.append(("m0r0", 300, r * 1000 + k))
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results2 = list(executor.map(collect_trajectory, tasks2))
    
    # Group by round, keep best
    for r in range(N_ROUNDS):
        round_results = results2[r*K_COLLECT:(r+1)*K_COLLECT]
        best = max(round_results, key=lambda x: x['levels_cleared'])
        if best['levels_cleared'] > 0 and best['pixels'] is not None:
            success_trajs.append(best)
    
    print(f"  After K={K_COLLECT} collection: {len(success_trajs)} successful trajectories")

# Build pixel → action dataset
all_pixels = []
all_actions = []

for traj in success_trajs:
    if traj['pixels'] is None:
        continue
    for px, act in zip(traj['pixels'], traj['actions']):
        all_pixels.append(px)
        all_actions.append(act - 1)  # GameAction values start at 1

if len(all_pixels) == 0:
    print("  ERROR: No pixel data collected. Exiting.")
    sys.exit(1)

print(f"  Training samples: {len(all_pixels)}")
print(f"  Pixel shape: {all_pixels[0].shape}")

# Convert to tensors
X = torch.tensor(np.array(all_pixels), dtype=torch.float32)
if X.dim() == 4 and X.shape[-1] in (3, 4):  # HWC → CHW
    X = X.permute(0, 3, 1, 2)
X = X / 255.0  # Normalize
Y = torch.tensor(all_actions, dtype=torch.long)

print(f"  X shape: {X.shape}")
print(f"  Y distribution: {dict(zip(*np.unique(Y.numpy(), return_counts=True)))}")

# Simple PixelCNN
class PixelBrain(nn.Module):
    """Game-agnostic CNN that works on raw pixels."""
    def __init__(self, in_channels, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.output = nn.Linear(64, n_actions)
    
    def forward(self, x, noise_sigma=0.0):
        feat = self.gap(self.conv(x)).squeeze(-1).squeeze(-1)
        h = self.head(feat)
        if noise_sigma > 0 and not self.training:
            h = h + torch.randn_like(h) * noise_sigma
        return self.output(h)

model = PixelBrain(in_channels=X.shape[1])
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model params: {n_params:,}")

# Train
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

model.train()
for epoch in range(200):
    perm = torch.randperm(len(X))
    loss = criterion(model(X[perm[:min(256, len(X))]], 0.0), Y[perm[:min(256, len(X))]])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(X, 0.0).argmax(1) == Y).float().mean().item()
        model.train()
        print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.3f}")

model.eval()
with torch.no_grad():
    final_acc = (model(X, 0.0).argmax(1) == Y).float().mean().item()
print(f"  Final accuracy: {final_acc:.3f}")

# Save model
model_path = os.path.join(SCRIPT_DIR, "data", "m0r0_pixel_brain.pt")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"  Model saved: {model_path}")

# ============================================================
# Step 4: Evaluate CNN + Noise + Trajectory Ensemble
# ============================================================
print(f"\n{'='*60}")
print(f"  Step 4: CNN + Noise + Trajectory Ensemble on m0r0")
print(f"{'='*60}", flush=True)

def eval_cnn_m0r0(sigma, K, N_eval, model_state, pixel_shape, in_channels):
    """Evaluate CNN with noise and trajectory ensemble."""
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')
    
    import torch, torch.nn as nn, arc_agi, numpy as np
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    
    # Rebuild model
    class PB(nn.Module):
        def __init__(self, ic, na=4):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ic, 32, 5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
            self.output = nn.Linear(64, na)
        def forward(self, x, ns=0.0):
            f = self.gap(self.conv(x)).squeeze(-1).squeeze(-1)
            h = self.head(f)
            if ns > 0 and not self.training:
                h = h + torch.randn_like(h) * ns
            return self.output(h)
    
    m = PB(ic=in_channels)
    m.load_state_dict(model_state)
    m.eval()
    
    arcade = arc_agi.Arcade()
    clears = 0
    
    for ep in range(N_eval):
        best_lc = 0
        for k in range(K):
            env = arcade.make("m0r0")
            obs = env.step(GameAction.RESET)
            lc = 0
            
            for step in range(300):
                px = np.array(obs.pixels, dtype=np.float32)
                px_t = torch.tensor(px).unsqueeze(0)
                if px_t.dim() == 4 and px_t.shape[-1] in (3, 4):
                    px_t = px_t.permute(0, 3, 1, 2)
                px_t = px_t / 255.0
                
                with torch.no_grad():
                    logits = m(px_t, ns=sigma)
                    action_idx = logits.argmax(1).item()
                
                action = ALL_A[action_idx]
                try:
                    obs = env.step(action)
                    if obs.levels_completed > lc:
                        lc = obs.levels_completed
                    if obs.state.value in ('GAME_OVER', 'WIN'):
                        break
                except:
                    break
            
            if lc > best_lc:
                best_lc = lc
        
        if best_lc > 0:
            clears += 1
    
    return clears

# Test configurations
configs = [
    ("baseline (random)", None, 0.0, 1),
    ("CNN only", "cnn", 0.0, 1),
    ("CNN + noise σ=0.05", "cnn", 0.05, 1),
    ("CNN + noise σ=0.10", "cnn", 0.10, 1),
    ("CNN + noise σ=0.20", "cnn", 0.20, 1),
    ("CNN + noise σ=0.05, K=5", "cnn", 0.05, 5),
    ("CNN + noise σ=0.05, K=11", "cnn", 0.05, 11),
    ("Random K=11", None, 0.0, 11),
    ("Random K=21", None, 0.0, 21),
]

N_EVAL = 100
model_state = model.state_dict()
in_ch = X.shape[1]

results_eval = {}
for name, policy, sigma, K in configs:
    t0 = time.time()
    
    if policy is None:
        # Random baseline
        if K == 1:
            clears = sum(1 for r in results[:N_EVAL] if r['levels_cleared'] > 0)
        else:
            # Use pre-computed Phase 16c results
            from phase16c_parallel import worker_trajectory_ensemble
            tasks = [("m0r0", K, 300, 9999 + i) for i in range(N_EVAL)]
            with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
                clears = sum(1 for lc in ex.map(worker_trajectory_ensemble, tasks) if lc > 0)
    else:
        # CNN-based evaluation (sequential for now, not bottleneck)
        clears = eval_cnn_m0r0(sigma, K, N_EVAL, model_state, None, in_ch)
    
    elapsed = time.time() - t0
    rate = clears / N_EVAL * 100
    results_eval[name] = {"clears": clears, "rate": rate, "time_s": round(elapsed, 1)}
    print(f"  {name:35s}: {clears:3d}/100 = {rate:5.1f}%  [{elapsed:.0f}s]", flush=True)

# Save
out_path = os.path.join(RESULTS_DIR, "phase17_m0r0_cnn_ensemble.json")
with open(out_path, "w") as f:
    json.dump(results_eval, f, indent=2)
print(f"\n  Saved: {out_path}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  PHASE 17 SUMMARY: m0r0 CNN + Noisy Beam Search")
print(f"{'='*60}")
for name, data in results_eval.items():
    bar = "█" * int(data['rate'] / 2)
    print(f"  {name:35s}: {data['rate']:5.1f}%  {bar}")

print(f"\n[{time.strftime('%H:%M:%S')}] Phase 17 complete!", flush=True)
