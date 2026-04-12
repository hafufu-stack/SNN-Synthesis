"""
Phase 40-DT: Micro-Decision-Transformer
Ultra-small Transformer (100K-300K params) for sequence-based decision making.
Tests whether self-attention over (State, Action) history can break
the learnability wall (Condition 2) that CNNs cannot cross.

Author: Hiroto Funasaki
"""
import os, json, math, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from datetime import datetime

RESULTS_DIR = r"c:\Users\kyjan\研究\snn-synthesis\results"


# ==============================================================
# Synthetic Game with Temporal Dependencies
# ==============================================================
class TemporalGame:
    """Game where optimal action depends on HISTORY of states.
    CNNs (which only see current state) cannot solve this.
    Transformers (which attend to history) should be able to.
    """
    GRID_SIZE = 6
    N_ACTIONS = 4

    def __init__(self, rule_type=0, seed=None):
        self.rng = random.Random(seed)
        self.rule_type = rule_type
        self.reset()

    def reset(self):
        self.steps = 0
        self.max_steps = 30
        self.history = []
        gs = self.GRID_SIZE
        self.grid = np.zeros((gs, gs), dtype=np.float32)
        # Place initial markers
        self.grid[self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)] = 1.0
        self.grid[self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)] = 2.0
        return self.get_state()

    def get_state(self):
        return self.grid.flatten()

    def get_optimal_action(self):
        """The optimal action depends on history."""
        if self.rule_type == 0:
            # Rule: repeat the action from 2 steps ago
            if len(self.history) >= 2:
                return self.history[-2] % self.N_ACTIONS
            return 0
        elif self.rule_type == 1:
            # Rule: alternate between two actions
            return (len(self.history)) % 2
        elif self.rule_type == 2:
            # Rule: follow pattern A,B,C,D,A,B,C,D,...
            return len(self.history) % self.N_ACTIONS
        else:
            # Rule: do the OPPOSITE of last action
            if self.history:
                return (self.history[-1] + 2) % self.N_ACTIONS
            return 0

    def step(self, action):
        self.steps += 1
        optimal = self.get_optimal_action()
        correct = (action == optimal)
        self.history.append(action)

        # Update grid based on correctness
        gs = self.GRID_SIZE
        if correct:
            r, c = self.rng.randint(0, gs-1), self.rng.randint(0, gs-1)
            self.grid[r, c] = min(self.grid[r, c] + 1.0, 9.0)

        done = self.steps >= self.max_steps
        n_correct = sum(1 for i, a in enumerate(self.history)
                        if i >= 2 and a == self.history[i-2] % self.N_ACTIONS) \
                    if self.rule_type == 0 else 0
        return self.get_state(), 1.0 if correct else 0.0, done, correct


# ==============================================================
# Micro-Decision-Transformer
# ==============================================================
class MicroDecisionTransformer(nn.Module):
    """Tiny Transformer for sequential decision making.
    Input: sequence of (state, action) pairs
    Output: next action prediction
    """
    def __init__(self, state_dim, n_actions=4, d_model=64, nhead=4,
                 n_layers=2, context_len=10):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.context_len = context_len
        self.d_model = d_model

        # State and action embeddings
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Embedding(n_actions + 1, d_model)  # +1 for padding
        self.pos_embed = nn.Embedding(context_len, d_model)

        # Combine state + action
        self.combine = nn.Linear(d_model * 2, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.output = nn.Linear(d_model, n_actions)

    def forward(self, states, actions, noise_sigma=0.0):
        """
        states: (batch, seq_len, state_dim)
        actions: (batch, seq_len) — previous actions (padded with n_actions for empty)
        Returns: (batch, n_actions) logits for next action
        """
        bs, seq_len = states.shape[0], states.shape[1]

        # Project states and actions
        s_emb = self.state_proj(states)  # (batch, seq, d_model)
        a_emb = self.action_embed(actions)  # (batch, seq, d_model)

        # Position embeddings
        pos = torch.arange(seq_len, device=states.device)
        p_emb = self.pos_embed(pos).unsqueeze(0)  # (1, seq, d_model)

        # Combine
        combined = self.combine(torch.cat([s_emb, a_emb], dim=-1)) + p_emb

        # Noise injection (SNN-Synthesis)
        if noise_sigma > 0:
            combined = combined + torch.randn_like(combined) * noise_sigma

        # Causal mask (can only attend to past)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(states.device)

        # Transform
        out = self.transformer(combined, mask=mask)

        # Predict from last position
        return self.output(out[:, -1, :])


# ==============================================================
# CNN Baseline
# ==============================================================
class CNNBaseline(nn.Module):
    """CNN that only sees current state (no history)."""
    def __init__(self, state_dim, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================
# Training and Evaluation
# ==============================================================
def generate_training_data(rule_type, n_episodes=500, context_len=10, seed=42):
    """Generate (state_history, action_history, target_action) tuples."""
    random.seed(seed)
    states_all, actions_all, targets_all = [], [], []
    state_dim = TemporalGame.GRID_SIZE ** 2

    for ep in range(n_episodes):
        env = TemporalGame(rule_type=rule_type, seed=seed + ep)

        ep_states, ep_actions = [], []
        state = env.reset()

        for step in range(env.max_steps):
            optimal = env.get_optimal_action()

            # Build context window
            ctx_states = ep_states[-context_len:] + [state]
            ctx_actions = ep_actions[-context_len:] + [4]  # 4 = padding

            # Pad to context_len
            while len(ctx_states) < context_len:
                ctx_states.insert(0, np.zeros(state_dim, dtype=np.float32))
                ctx_actions.insert(0, 4)

            ctx_states = ctx_states[-context_len:]
            ctx_actions = ctx_actions[-context_len:]

            states_all.append(np.array(ctx_states))
            actions_all.append(ctx_actions)
            targets_all.append(optimal)

            ep_states.append(state.copy())
            ep_actions.append(optimal)  # teacher forcing
            state, _, done, _ = env.step(optimal)
            if done:
                break

    return (
        torch.tensor(np.array(states_all), dtype=torch.float32),
        torch.tensor(np.array(actions_all), dtype=torch.long),
        torch.tensor(targets_all, dtype=torch.long),
    )


def train_and_eval(model, X_states, X_actions, y, is_transformer=True,
                   epochs=100, lr=0.001):
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(y)

    # Split: 80% train, 20% test
    split = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx, test_idx = perm[:split], perm[split:]

    best_test_acc = 0
    for epoch in range(epochs):
        model.train()
        batch = train_idx[torch.randperm(len(train_idx))[:min(256, len(train_idx))]]

        if is_transformer:
            logits = model(X_states[batch], X_actions[batch])
        else:
            logits = model(X_states[batch, -1, :])  # CNN: only last state

        loss = F.cross_entropy(logits, y[batch])
        opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                if is_transformer:
                    test_logits = model(X_states[test_idx], X_actions[test_idx])
                else:
                    test_logits = model(X_states[test_idx, -1, :])
                test_acc = (test_logits.argmax(1) == y[test_idx]).float().mean().item()
                best_test_acc = max(best_test_acc, test_acc)

    return best_test_acc


def main():
    print("=" * 60)
    print("Phase 40-DT: Micro-Decision-Transformer")
    print("  Can attention over history break the learnability wall?")
    print("=" * 60)

    torch.manual_seed(42)
    state_dim = TemporalGame.GRID_SIZE ** 2  # 36
    context_len = 10
    all_results = {}

    for rule_type in range(4):
        rule_names = ["Repeat-2-ago", "Alternate", "Cycle-4", "Opposite-last"]
        print(f"\n--- Rule: {rule_names[rule_type]} ---")

        X_states, X_actions, y = generate_training_data(
            rule_type, n_episodes=500, context_len=context_len, seed=42)
        print(f"  Data: {len(y)} samples")

        # CNN baseline (no history)
        cnn = CNNBaseline(state_dim, n_actions=4, hidden=64)
        cnn_n = sum(p.numel() for p in cnn.parameters())
        cnn_acc = train_and_eval(cnn, X_states, X_actions, y,
                                 is_transformer=False, epochs=100)
        print(f"  CNN ({cnn_n:,} params): {cnn_acc*100:.1f}%")

        # Micro-Transformer
        transformer = MicroDecisionTransformer(
            state_dim, n_actions=4, d_model=64, nhead=4,
            n_layers=2, context_len=context_len)
        tf_n = sum(p.numel() for p in transformer.parameters())
        tf_acc = train_and_eval(transformer, X_states, X_actions, y,
                                is_transformer=True, epochs=100)
        print(f"  Transformer ({tf_n:,} params): {tf_acc*100:.1f}%")

        all_results[rule_names[rule_type]] = {
            'rule_type': rule_type,
            'cnn_acc': cnn_acc,
            'cnn_params': cnn_n,
            'transformer_acc': tf_acc,
            'transformer_params': tf_n,
            'improvement': tf_acc - cnn_acc,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Transformer vs CNN on temporal tasks")
    for rule, r in all_results.items():
        winner = "TRANSFORMER" if r['transformer_acc'] > r['cnn_acc'] else "CNN"
        print(f"  {rule:18s}: CNN={r['cnn_acc']*100:.1f}% TF={r['transformer_acc']*100:.1f}% "
              f"[{winner}] ({r['improvement']*100:+.1f}pp)")

    save_path = os.path.join(RESULTS_DIR, "phase40dt_micro_decision_transformer.json")
    with open(save_path, 'w') as f:
        json.dump({
            'experiment': 'Phase 40-DT: Micro-Decision-Transformer',
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    main()
