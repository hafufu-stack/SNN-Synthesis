%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v11 Agent for ARC-AGI-3
# Test-Time ExIt: Self-evolving CNN during gameplay
#
# Innovation: No pre-training needed. Agent learns from its own
# miracles during gameplay, building a policy from scratch.
#
# Paper: https://doi.org/10.5281/zenodo.19343952
# GitHub: https://github.com/hafufu-stack/SNN-Synthesis
# Author: Hiroto Funasaki
# ==============================================================
import hashlib
import logging
import os
import random
import time
from typing import Any, Optional

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# Sigma-Diverse NBS Schedule (Phase 37a)
SIGMA_SCHEDULE = [
    0.0, 0.05, 0.15, 0.30, 0.50, 0.01, 0.10, 0.20, 0.75, 1.00,
]

# Test-Time ExIt config
MIN_MIRACLES_TO_TRAIN = 3    # minimum miracles before training CNN
CNN_HIDDEN = 64              # hidden layer size
CNN_TRAIN_EPOCHS = 30        # quick training epochs
CNN_USE_THRESHOLD = 5        # use CNN after this many miracles


# ==============================================================
# Tiny MLP for Test-Time Learning (no torch dependency)
# ==============================================================
class TinyMLPNumpy:
    """Minimal MLP using only numpy. No PyTorch needed on Kaggle."""
    def __init__(self, input_dim, hidden=64, n_actions=7):
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_actions = n_actions
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.w1 = np.random.randn(input_dim, hidden).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = np.random.randn(hidden, hidden).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.w3 = np.random.randn(hidden, n_actions).astype(np.float32) * scale2
        self.b3 = np.zeros(n_actions, dtype=np.float32)

    def forward(self, x, noise_sigma=0.0):
        """Forward pass with optional noise injection."""
        h = x @ self.w1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        if noise_sigma > 0:
            h = h + np.random.randn(*h.shape).astype(np.float32) * noise_sigma
        h = h @ self.w2 + self.b2
        h = np.maximum(h, 0)  # ReLU
        logits = h @ self.w3 + self.b3
        return logits

    def train_on_data(self, X, y, epochs=30, lr=0.01):
        """Mini-batch SGD training using numpy only."""
        n = len(X)
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            batch = perm[:min(128, n)]
            Xb, yb = X[batch], y[batch]

            # Forward
            h1 = Xb @ self.w1 + self.b1
            a1 = np.maximum(h1, 0)
            h2 = a1 @ self.w2 + self.b2
            a2 = np.maximum(h2, 0)
            logits = a2 @ self.w3 + self.b3

            # Softmax + cross-entropy gradient
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            bs = len(Xb)
            dlogits = probs.copy()
            dlogits[np.arange(bs), yb] -= 1
            dlogits /= bs

            # Backprop through layer 3
            dw3 = a2.T @ dlogits
            db3 = dlogits.sum(axis=0)
            da2 = dlogits @ self.w3.T

            # Backprop through ReLU + layer 2
            da2 = da2 * (h2 > 0)
            dw2 = a1.T @ da2
            db2 = da2.sum(axis=0)
            da1 = da2 @ self.w2.T

            # Backprop through ReLU + layer 1
            da1 = da1 * (h1 > 0)
            dw1 = Xb.T @ da1
            db1 = da1.sum(axis=0)

            # Update
            self.w3 -= lr * dw3
            self.b3 -= lr * db3
            self.w2 -= lr * dw2
            self.b2 -= lr * db2
            self.w1 -= lr * dw1
            self.b1 -= lr * db1


# ==============================================================
# SNN-Synthesis v11: Test-Time ExIt Agent
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v11: Test-Time SNN-ExIt.

    Strategy:
    1. Start with curiosity-guided random exploration
    2. When miracles occur, collect (state, action) pairs
    3. After MIN_MIRACLES, train a lightweight MLP on miracle data
    4. Use MLP + sigma-diverse noise for subsequent attempts
    5. Continuously accumulate miracles and retrain
    """
    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32))

        # State-action visit counts for novelty
        self.sa_visits = {}

        # Miracle memory
        self.miracle_trajectories = []      # list of (states, actions)
        self.current_episode_states = []
        self.current_episode_actions = []

        # Progress tracking
        self.prev_levels = 0
        self.attempt_count = 0

        # Test-Time CNN
        self.cnn_model = None
        self.state_dim = None

        logger.info(f"SNN-Synthesis v11 (Test-Time ExIt) initialized for {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:

        # --- RESET when needed ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.current_episode_states = []
            self.current_episode_actions = []
            self.attempt_count += 1

            # Retrain CNN periodically when we have enough miracles
            if (len(self.miracle_trajectories) >= MIN_MIRACLES_TO_TRAIN and
                self.attempt_count % 5 == 0):
                self._train_cnn()

            return GameAction.RESET

        # --- Check for miracle ---
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            # Save miracle trajectory
            self.miracle_trajectories.append({
                'states': list(self.current_episode_states),
                'actions': list(self.current_episode_actions),
            })
            logger.info(
                f"MIRACLE #{len(self.miracle_trajectories)}! Level {current_levels}. "
                f"Trajectory length: {len(self.current_episode_actions)}"
            )
            self.current_episode_states = []
            self.current_episode_actions = []
            self.prev_levels = current_levels

            # Train CNN if we just hit the threshold
            if len(self.miracle_trajectories) == MIN_MIRACLES_TO_TRAIN:
                self._train_cnn()
                logger.info("CNN trained on first miracles!")

        # --- Extract state features ---
        state_features = self._extract_features(latest_frame)
        self.current_episode_states.append(state_features)
        state_hash = self._grid_hash(latest_frame)

        # --- Choose action ---
        if (self.cnn_model is not None and
            len(self.miracle_trajectories) >= CNN_USE_THRESHOLD):
            # Use CNN with sigma-diverse noise
            action = self._cnn_action(state_features)
        elif self.miracle_trajectories and random.random() < 0.3:
            # Exploit miracle memory
            action = self._exploit_miracle()
        else:
            # Curiosity-guided exploration
            action = self._curiosity_action(state_hash)

        self.current_episode_actions.append(action.name)
        return action

    def _extract_features(self, frame: FrameData) -> list:
        """Extract numeric features from frame for CNN input."""
        features = []
        if frame.frame:
            try:
                arr = np.array(frame.frame[0], dtype=np.float32)
                features.extend(arr.flatten()[:200].tolist())
            except Exception:
                pass

        # Pad to fixed size
        while len(features) < 200:
            features.append(0.0)
        return features[:200]

    def _train_cnn(self):
        """Train/retrain CNN on accumulated miracle data."""
        all_states, all_actions = [], []
        action_names = [a.name for a in GameAction if a is not GameAction.RESET]

        for m in self.miracle_trajectories:
            for s, a_name in zip(m['states'], m['actions']):
                if a_name in action_names:
                    all_states.append(s)
                    all_actions.append(action_names.index(a_name))

        if len(all_states) < 10:
            return

        X = np.array(all_states, dtype=np.float32)
        y = np.array(all_actions, dtype=np.int64)
        self.state_dim = X.shape[1]
        n_actions = len(action_names)

        self.cnn_model = TinyMLPNumpy(self.state_dim, CNN_HIDDEN, n_actions)
        self.cnn_model.train_on_data(X, y, epochs=CNN_TRAIN_EPOCHS, lr=0.01)

        # Quick accuracy check
        logits = self.cnn_model.forward(X)
        preds = logits.argmax(axis=1)
        acc = (preds == y).mean()
        logger.info(f"CNN retrained: {len(all_states)} samples, acc={acc:.3f}")

    def _cnn_action(self, state_features: list) -> GameAction:
        """Use CNN to choose action with sigma-diverse noise."""
        candidates = [a for a in GameAction if a is not GameAction.RESET]

        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx]

        x = np.array(state_features[:self.state_dim], dtype=np.float32).reshape(1, -1)
        logits = self.cnn_model.forward(x, noise_sigma=sigma)

        # Softmax sampling
        logits = logits[0, :len(candidates)]
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        action_idx = np.random.choice(len(candidates), p=probs)
        action = candidates[action_idx]
        self._configure_action(action)
        return action

    def _curiosity_action(self, state_hash: str) -> GameAction:
        """Select action favoring less-visited state-action pairs."""
        candidates = [a for a in GameAction if a is not GameAction.RESET]

        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        scores = []
        for action in candidates:
            sa_key = f"{state_hash}_{action.name}"
            visits = self.sa_visits.get(sa_key, 0)
            novelty = 1.0 / (visits + 1)
            noise = random.gauss(0, max(0.05, sigma))
            scores.append(novelty + noise)

        max_score = max(scores)
        weights = [2.718 ** ((s - max_score) / 0.3) for s in scores]
        total = sum(weights)
        probs = [w / total for w in weights]

        r = random.random()
        cumsum = 0
        selected_idx = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                selected_idx = i
                break

        action = candidates[selected_idx]
        sa_key = f"{state_hash}_{action.name}"
        self.sa_visits[sa_key] = self.sa_visits.get(sa_key, 0) + 1
        self._configure_action(action)
        return action

    def _exploit_miracle(self) -> GameAction:
        """Replay a random action from miracle memory."""
        m = random.choice(self.miracle_trajectories)
        miracle_name = random.choice(m['actions'])
        for action in GameAction:
            if action.name == miracle_name:
                self._configure_action(action)
                return action
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action)
        return action

    def _configure_action(self, action: GameAction) -> None:
        """Configure action with proper data."""
        if action.is_simple():
            action.reasoning = f"SNN-v11-ExIt: attempt={self.attempt_count}, miracles={len(self.miracle_trajectories)}"
        elif action.is_complex():
            x = random.randint(0, 63)
            y = random.randint(0, 63)
            if self.frames and self.frames[-1].frame:
                try:
                    arr = np.array(self.frames[-1].frame[0], dtype=np.int32)
                    h, w = arr.shape
                    if random.random() < 0.5:
                        nonzero = np.argwhere(arr != 0)
                        if len(nonzero) > 0:
                            idx = random.randint(0, len(nonzero) - 1)
                            y, x = int(nonzero[idx][0]), int(nonzero[idx][1])
                    else:
                        x = random.randint(0, min(w - 1, 63))
                        y = random.randint(0, min(h - 1, 63))
                except Exception:
                    pass
            action.set_data({"x": x, "y": y})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"SNN-v11-ExIt: target ({x},{y}), cnn={'yes' if self.cnn_model else 'no'}",
            }

    def _grid_hash(self, frame: FrameData) -> str:
        if not frame.frame:
            return "empty"
        try:
            arr = np.array(frame.frame[0], dtype=np.int32)
            return hashlib.md5(arr.tobytes()).hexdigest()[:12]
        except Exception:
            return "error"


# Alias for Kaggle
MyAgent = StochasticGoose
