"""
Phase 35: Micro-Transformer for TR87 — Breaking the Learnability Wall
======================================================================
Replace the MLP with a Decision Transformer-style policy that uses
Self-Attention over past (state, action) history. This should enable
learning temporal dependencies that the MLP/CNN cannot capture.

Key hypothesis: TR87 fails Condition 2 because the correct action
depends on multi-step lookahead. Attention over history enables this.
"""
import sys, os, json, time, random, gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'environment_files')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_WORKERS = min(cpu_count() - 4, 16)
HISTORY_LEN = 12  # attend to last 12 steps
N_ACTIONS = 4
HIDDEN_DIM = 128
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.1
TRAIN_EPOCHS = 800
BOOTSTRAP_K = 500
N_BOOTSTRAP = 300
N_EXIT_ITERS = 4
NOISE_SIGMA = 0.1
MAX_STEPS = 200

import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# State extraction (from Phase 24)
# ============================================================
def extract_game_state(game):
    state = []
    for attr in sorted(dir(game)):
        if attr.startswith('_') or attr in ('level_index', 'win_score'):
            continue
        try:
            val = getattr(game, attr)
            if callable(val):
                continue
            if isinstance(val, (int, float, bool)):
                state.append(float(val))
            elif isinstance(val, (list, tuple)):
                for v in val[:4]:
                    if isinstance(v, (int, float, bool)):
                        state.append(float(v))
        except:
            pass
    return np.array(state, dtype=np.float32)


# ============================================================
# Micro-Transformer: Decision Transformer-style policy
# ============================================================
class MicroTransformer(nn.Module):
    """
    Attends to past H steps of (state, action) pairs to predict next action.
    Input: history of (state_dim + n_actions) per timestep, H timesteps
    Output: action logits for next step
    """
    def __init__(self, state_dim, n_actions=4, hidden=128, n_heads=4, n_layers=3,
                 history_len=12, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.history_len = history_len
        self.hidden = hidden

        # Embed (state + action_onehot) -> hidden
        self.input_proj = nn.Linear(state_dim + n_actions, hidden)
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, history_len, hidden) * 0.02)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden*4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_actions)
        )

    def forward(self, states, actions_onehot, noise_sigma=0.0):
        """
        states: (B, H, state_dim)
        actions_onehot: (B, H, n_actions)
        """
        x = torch.cat([states, actions_onehot], dim=-1)  # (B, H, state_dim + n_actions)
        x = self.input_proj(x)  # (B, H, hidden)
        H = x.size(1)
        x = x + self.pos_embed[:, :H, :]
        # Causal mask: each position can only attend to itself and earlier
        mask = torch.triu(torch.ones(H, H, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)

        if noise_sigma > 0:
            x = x + torch.randn_like(x) * noise_sigma

        # Use last timestep's output for prediction
        out = self.output_head(x[:, -1, :])  # (B, n_actions)
        return out

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Worker: Random trajectory collection
# ============================================================
def random_trajectory(args):
    game_id, max_steps, seed = args
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')

    import arc_agi, random
    import numpy as np
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    rng = random.Random(seed)

    try:
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
        obs = env.step(GameAction.RESET)
        game = env._game
    except:
        return None

    states, actions, max_lc = [], [], 0
    for step in range(max_steps):
        try:
            s = extract_game_state(game)
            states.append(s.tolist())
        except:
            states.append([0.0] * 7)
        action = rng.choice(ALL_A)
        actions.append(ALL_A.index(action))
        try:
            obs = env.step(action)
            if obs.levels_completed > max_lc:
                max_lc = obs.levels_completed
            if obs.state.value in ('GAME_OVER', 'WIN'):
                break
        except:
            break

    return {'states': states, 'actions': actions,
            'levels_cleared': max_lc, 'n_steps': len(actions)}


def best_of_k_random(args):
    game_id, K, max_steps, seed = args
    best = None
    for k in range(K):
        result = random_trajectory((game_id, max_steps, seed * 100000 + k))
        if result is None:
            continue
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result
    return best


# ============================================================
# Worker: Trajectory with Transformer model
# ============================================================
def collect_with_transformer(args):
    game_id, K, max_steps, seed, model_path, state_dim, noise_sigma, history_len = args
    os.environ['OPERATION_MODE'] = 'OFFLINE'
    os.environ['ENVIRONMENTS_DIR'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'environment_files')

    import arc_agi, random, torch, torch.nn as nn
    import numpy as np
    from arcengine import GameAction
    ALL_A = [GameAction.ACTION1, GameAction.ACTION2,
             GameAction.ACTION3, GameAction.ACTION4]
    rng = random.Random(seed)
    n_actions = 4

    # Load transformer model
    model = None
    x_mean = x_std = None
    if model_path and os.path.exists(model_path):
        try:
            data = torch.load(model_path, weights_only=False)
            # Rebuild MicroTransformer
            class MT(nn.Module):
                def __init__(self, sd, na, h, nh, nl, hl, do):
                    super().__init__()
                    self.input_proj = nn.Linear(sd + na, h)
                    self.pos_embed = nn.Parameter(torch.randn(1, hl, h)*0.02)
                    enc_layer = nn.TransformerEncoderLayer(d_model=h, nhead=nh,
                        dim_feedforward=h*4, dropout=do, batch_first=True, activation='gelu')
                    self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nl)
                    self.output_head = nn.Sequential(
                        nn.LayerNorm(h), nn.Linear(h, h//2), nn.GELU(),
                        nn.Dropout(do), nn.Linear(h//2, na))
                def forward(self, s, a, ns=0.0):
                    x = torch.cat([s, a], dim=-1)
                    x = self.input_proj(x)
                    H = x.size(1)
                    x = x + self.pos_embed[:, :H, :]
                    mask = torch.triu(torch.ones(H, H, device=x.device), diagonal=1).bool()
                    x = self.transformer(x, mask=mask)
                    if ns > 0: x = x + torch.randn_like(x) * ns
                    return self.output_head(x[:, -1, :])

            cfg = data['config']
            model = MT(cfg['state_dim'], cfg['n_actions'], cfg['hidden'],
                      cfg['n_heads'], cfg['n_layers'], cfg['history_len'], cfg['dropout'])
            model.load_state_dict(data['model'])
            model.eval()
            x_mean = data['x_mean']
            x_std = data['x_std']
        except Exception as e:
            model = None

    best = None
    for k in range(K):
        try:
            arcade = arc_agi.Arcade()
            env = arcade.make(game_id)
            obs = env.step(GameAction.RESET)
            game = env._game
        except:
            continue

        states_hist, actions_hist = [], []
        all_states, all_actions, max_lc = [], [], 0

        for step in range(max_steps):
            try:
                s = extract_game_state(game)
                s_list = s.tolist()
            except:
                s_list = [0.0] * state_dim
            all_states.append(s_list)

            if model is not None and len(states_hist) > 0:
                try:
                    # Build history tensor
                    h_len = min(len(states_hist), history_len)
                    s_tensor = torch.tensor(states_hist[-h_len:], dtype=torch.float32)
                    if x_mean is not None:
                        s_tensor = (s_tensor - x_mean) / x_std
                    # Pad if needed
                    if s_tensor.size(0) < history_len:
                        pad = torch.zeros(history_len - s_tensor.size(0), state_dim)
                        s_tensor = torch.cat([pad, s_tensor], dim=0)

                    a_tensor = torch.zeros(history_len, n_actions)
                    a_list = actions_hist[-h_len:] if h_len > 0 else []
                    offset = history_len - len(a_list)
                    for j, a in enumerate(a_list):
                        a_tensor[offset + j, a] = 1.0

                    with torch.no_grad():
                        logits = model(s_tensor.unsqueeze(0), a_tensor.unsqueeze(0), ns=noise_sigma)
                        probs = torch.softmax(logits / max(0.5, 1.0 - noise_sigma), dim=1)
                        action_idx = torch.multinomial(probs, 1).item()
                    action = ALL_A[action_idx]
                except:
                    action = rng.choice(ALL_A)
            else:
                action = rng.choice(ALL_A)

            action_idx = ALL_A.index(action)
            all_actions.append(action_idx)
            states_hist.append(s_list)
            actions_hist.append(action_idx)

            try:
                obs = env.step(action)
                if obs.levels_completed > max_lc:
                    max_lc = obs.levels_completed
                if obs.state.value in ('GAME_OVER', 'WIN'):
                    break
            except:
                break

        result = {'states': all_states, 'actions': all_actions,
                  'levels_cleared': max_lc, 'n_steps': len(all_actions)}
        if best is None or result['levels_cleared'] > best['levels_cleared']:
            best = result

    return best


# ============================================================
# Main
# ============================================================
def main():
    GAME_ID = "tr87"
    print(f"[{time.strftime('%H:%M:%S')}] Phase 35: Micro-Transformer for {GAME_ID}")
    print(f"  History: {HISTORY_LEN}, Hidden: {HIDDEN_DIM}, Heads: {N_HEADS}, Layers: {N_LAYERS}")
    print(f"  Bootstrap: K={BOOTSTRAP_K}, N={N_BOOTSTRAP}")
    print(f"  Workers: {N_WORKERS}")
    print(f"{'='*60}", flush=True)

    # Step 1: Collect bootstrap miracles
    print(f"\n  [Step 1] Collecting bootstrap miracles (K={BOOTSTRAP_K})...", flush=True)
    tasks = [(GAME_ID, BOOTSTRAP_K, MAX_STEPS, ep) for ep in range(N_BOOTSTRAP)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        bootstrap = list(executor.map(best_of_k_random, tasks))
    miracles = [r for r in bootstrap if r and r['levels_cleared'] > 0]
    print(f"  Bootstrap miracles: {len(miracles)}/{N_BOOTSTRAP} ({100*len(miracles)/N_BOOTSTRAP:.1f}%)")

    if len(miracles) < 10:
        print("  Insufficient miracles, exiting")
        return

    # Determine state dim
    state_dim = len(miracles[0]['states'][0]) if miracles[0]['states'] else 7
    print(f"  State dim: {state_dim}")

    # ExIt Loop
    cumulative_miracles = list(miracles)
    model_path = None
    exit_results = []

    for iteration in range(N_EXIT_ITERS):
        print(f"\n  --- ExIt Iteration {iteration+1}/{N_EXIT_ITERS} ---")
        print(f"  Cumulative miracles: {len(cumulative_miracles)}")

        # Build training data: sequences of (state, action)
        all_state_seqs = []
        all_action_seqs = []
        all_target_actions = []
        for m in cumulative_miracles:
            for t in range(1, len(m['states'])):  # predict action at each timestep
                h_start = max(0, t - HISTORY_LEN)
                state_seq = m['states'][h_start:t]
                action_seq = m['actions'][h_start:t]
                target = m['actions'][t] if t < len(m['actions']) else m['actions'][-1]
                all_state_seqs.append(state_seq)
                all_action_seqs.append(action_seq)
                all_target_actions.append(target)

        print(f"  Training samples: {len(all_state_seqs)}")

        # Pad sequences
        padded_states = []
        padded_actions = []
        for ss, aa in zip(all_state_seqs, all_action_seqs):
            s_arr = np.array(ss, dtype=np.float32)
            pad_len = HISTORY_LEN - len(ss)
            if pad_len > 0:
                s_arr = np.vstack([np.zeros((pad_len, state_dim), dtype=np.float32), s_arr])
            elif pad_len < 0:
                s_arr = s_arr[-HISTORY_LEN:]

            a_onehot = np.zeros((HISTORY_LEN, N_ACTIONS), dtype=np.float32)
            offset = max(0, HISTORY_LEN - len(aa))
            for j, a in enumerate(aa[-HISTORY_LEN:]):
                a_onehot[offset + j, a] = 1.0

            padded_states.append(s_arr)
            padded_actions.append(a_onehot)

        X_states = torch.tensor(np.array(padded_states), dtype=torch.float32)
        X_actions = torch.tensor(np.array(padded_actions), dtype=torch.float32)
        Y = torch.tensor(all_target_actions, dtype=torch.long)

        # Normalize states
        flat = X_states.reshape(-1, state_dim)
        x_mean = flat.mean(0, keepdim=True)
        x_std = flat.std(0, keepdim=True).clamp(min=1e-6)
        X_states = (X_states - x_mean.unsqueeze(0)) / x_std.unsqueeze(0)

        # Build and train Micro-Transformer
        model = MicroTransformer(state_dim, N_ACTIONS, HIDDEN_DIM, N_HEADS,
                                N_LAYERS, HISTORY_LEN, DROPOUT)
        print(f"  Model params: {model.count_params():,}")

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

        model.train()
        best_acc = 0
        for epoch in range(TRAIN_EPOCHS):
            perm = torch.randperm(len(X_states))
            batch_idx = perm[:min(256, len(X_states))]
            logits = model(X_states[batch_idx], X_actions[batch_idx], noise_sigma=0.0)
            loss = criterion(logits, Y[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch + 1) % 200 == 0:
                model.eval()
                with torch.no_grad():
                    acc = (model(X_states, X_actions, 0.0).argmax(1) == Y).float().mean().item()
                model.train()
                best_acc = max(best_acc, acc)
                print(f"    Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.3f}")

        print(f"  Best train accuracy: {best_acc:.3f} (MLP baseline: 0.345)")

        # Save model
        mp = os.path.join(SCRIPT_DIR, "data", f"micro_transformer_tr87_iter{iteration+1}.pt")
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        torch.save({
            'model': model.state_dict(),
            'x_mean': x_mean, 'x_std': x_std,
            'config': {'state_dim': state_dim, 'n_actions': N_ACTIONS,
                       'hidden': HIDDEN_DIM, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                       'history_len': HISTORY_LEN, 'dropout': DROPOUT}
        }, mp)
        model_path = mp

        # Collect more miracles with Transformer
        K_COLLECT = max(20, BOOTSTRAP_K // (2 ** (iteration + 1)))
        N_COLLECT = 200
        print(f"  Collecting with Transformer + K={K_COLLECT}...")
        tasks = [(GAME_ID, K_COLLECT, MAX_STEPS,
                  (iteration+1) * 10000000 + ep, model_path, state_dim,
                  NOISE_SIGMA, HISTORY_LEN)
                 for ep in range(N_COLLECT)]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            new_results = list(executor.map(collect_with_transformer, tasks))

        new_miracles = [r for r in new_results if r and r['levels_cleared'] > 0]
        cumulative_miracles.extend(new_miracles)
        print(f"  New miracles: {len(new_miracles)}/{N_COLLECT} ({100*len(new_miracles)/N_COLLECT:.1f}%)")

        # Evaluate
        N_EVAL = 100
        iter_result = {"iteration": iteration+1, "train_acc": best_acc,
                       "miracles_total": len(cumulative_miracles),
                       "new_miracles": len(new_miracles), "configs": {}}

        for name, mp_eval, K_eval, sigma in [
            ("Random K=11", None, 11, 0.0),
            (f"Transformer K=1", model_path, 1, 0.0),
            (f"Transformer K=11 σ=0.1", model_path, 11, NOISE_SIGMA),
        ]:
            tasks = [(GAME_ID, K_eval, MAX_STEPS,
                      88000000 + iteration*10000 + ep, mp_eval, state_dim,
                      sigma, HISTORY_LEN)
                     for ep in range(N_EVAL)]
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                eval_results = list(executor.map(collect_with_transformer, tasks))
            clears = sum(1 for r in eval_results if r and r['levels_cleared'] > 0)
            rate = clears / N_EVAL * 100
            iter_result["configs"][name] = {"clears": clears, "rate": rate}
            bar = "█" * int(rate / 2)
            print(f"    {name:35s}: {clears:3d}/{N_EVAL} = {rate:5.1f}%  {bar}")

        exit_results.append(iter_result)
        gc.collect()

    # Save results
    output = {
        "experiment": "Phase 35: Micro-Transformer for TR87",
        "game": "tr87",
        "architecture": {"type": "Decision Transformer", "history_len": HISTORY_LEN,
                         "hidden": HIDDEN_DIM, "heads": N_HEADS, "layers": N_LAYERS,
                         "params": model.count_params() if 'model' in dir() else 0},
        "bootstrap": {"K": BOOTSTRAP_K, "N": N_BOOTSTRAP,
                      "miracles": len(miracles), "rate": len(miracles)/N_BOOTSTRAP},
        "exit_results": exit_results,
        "comparison": {"mlp_train_acc": 0.345, "mlp_exit_rate": 0.03}
    }
    out_path = os.path.join(RESULTS_DIR, "phase35_micro_transformer.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PHASE 35 COMPLETE: Micro-Transformer for TR87")
    print(f"  MLP baseline train acc: 34.5%")
    for r in exit_results:
        print(f"  Iter {r['iteration']}: train_acc={r['train_acc']:.3f}, " +
              f"clear={r['configs'].get('Transformer K=11 σ=0.1', {}).get('rate', 0):.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
