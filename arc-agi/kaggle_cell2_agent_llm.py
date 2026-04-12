%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v12 Agent for ARC-AGI-3
# LLM-ExIt + Test-Time SNN-ExIt + RND Curiosity
# 4-tier: LLM -> Learned CNN -> RND Curiosity -> Random
#
# Paper: https://doi.org/10.5281/zenodo.19343952
# GitHub: https://github.com/hafufu-stack/SNN-Synthesis
# Author: Hiroto Funasaki
# ==============================================================
import hashlib
import logging
import os
import random
import re
import time
from typing import Any, Optional

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# Path to GGUF model (uploaded as Kaggle Dataset)
MODEL_PATH = "/kaggle/input/snn-synthesis-models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"

# Sigma-Diverse NBS Schedule (Phase 37a: proven equivalent to best-tuned)
# Each attempt uses a different noise level -> natural selection finds optimal
SIGMA_SCHEDULE = [
    0.0,   # attempt 1: greedy (no noise)
    0.05,  # attempt 2: minimal exploration
    0.15,  # attempt 3: moderate (Hanoi-optimal)
    0.30,  # attempt 4: strong exploration
    0.50,  # attempt 5: wild
    0.01,  # attempt 6: precision (GSM8K-optimal)
    0.10,  # attempt 7: gentle
    0.20,  # attempt 8: moderate-high
    0.75,  # attempt 9: very aggressive
    1.00,  # attempt 10: maximum chaos
]

# Test-Time ExIt config
MIN_MIRACLES_TO_TRAIN = 3    # minimum miracles before training CNN
CNN_HIDDEN = 64              # hidden layer size
CNN_TRAIN_EPOCHS = 30        # quick training epochs
CNN_USE_THRESHOLD = 5        # use CNN after this many miracles


# Map sigma to LLM temperature (higher sigma = higher temperature)
def sigma_to_temperature(sigma):
    """Convert NBS sigma to LLM temperature for inference."""
    return max(0.1, min(2.0, 0.3 + sigma * 2.0))


# ==============================================================
# Tiny MLP for Test-Time Learning (numpy only, no PyTorch)
# ==============================================================
class TinyMLPNumpy:
    """Minimal MLP using only numpy. Learns from miracle trajectories at runtime."""
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
        h = x @ self.w1 + self.b1
        h = np.maximum(h, 0)
        if noise_sigma > 0:
            h = h + np.random.randn(*h.shape).astype(np.float32) * noise_sigma
        h = h @ self.w2 + self.b2
        h = np.maximum(h, 0)
        return h @ self.w3 + self.b3

    def train_on_data(self, X, y, epochs=30, lr=0.01):
        n = len(X)
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            batch = perm[:min(128, n)]
            Xb, yb = X[batch], y[batch]
            h1 = Xb @ self.w1 + self.b1
            a1 = np.maximum(h1, 0)
            h2 = a1 @ self.w2 + self.b2
            a2 = np.maximum(h2, 0)
            logits = a2 @ self.w3 + self.b3
            exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)
            bs = len(Xb)
            dl = probs.copy()
            dl[np.arange(bs), yb] -= 1
            dl /= bs
            dw3 = a2.T @ dl; db3 = dl.sum(0)
            da2 = (dl @ self.w3.T) * (h2 > 0)
            dw2 = a1.T @ da2; db2 = da2.sum(0)
            da1 = (da2 @ self.w2.T) * (h1 > 0)
            dw1 = Xb.T @ da1; db1 = da1.sum(0)
            self.w3 -= lr*dw3; self.b3 -= lr*db3
            self.w2 -= lr*dw2; self.b2 -= lr*db2
            self.w1 -= lr*dw1; self.b1 -= lr*db1


# ==============================================================
# RND Curiosity Module (Phase 39: +61pp on hard games)
# ==============================================================
class RNDCuriosity:
    """Random Network Distillation for curiosity-driven exploration.
    States with high prediction error = novel = should be explored.
    Phase 39 showed: Random 2.5% -> RND 63.5% at difficulty 6.
    """
    def __init__(self, state_dim, hidden=32):
        self.state_dim = state_dim
        # Fixed random target network (never updated)
        self.target_w1 = np.random.randn(state_dim, hidden).astype(np.float32) * 0.1
        self.target_b1 = np.zeros(hidden, dtype=np.float32)
        self.target_w2 = np.random.randn(hidden, hidden // 2).astype(np.float32) * 0.1
        self.target_b2 = np.zeros(hidden // 2, dtype=np.float32)

        # Learnable predictor (trained to match target -> error drops for seen states)
        scale = np.sqrt(2.0 / state_dim)
        self.pred_w1 = np.random.randn(state_dim, hidden).astype(np.float32) * scale
        self.pred_b1 = np.zeros(hidden, dtype=np.float32)
        self.pred_w2 = np.random.randn(hidden, hidden // 2).astype(np.float32) * scale
        self.pred_b2 = np.zeros(hidden // 2, dtype=np.float32)

    def curiosity_score(self, state):
        """Higher = more novel state. Used to bonus unexplored actions."""
        x = np.array(state, dtype=np.float32).reshape(1, -1)
        if x.shape[1] > self.state_dim:
            x = x[:, :self.state_dim]
        elif x.shape[1] < self.state_dim:
            x = np.pad(x, ((0, 0), (0, self.state_dim - x.shape[1])))
        target = np.maximum(x @ self.target_w1 + self.target_b1, 0) @ self.target_w2 + self.target_b2
        pred = np.maximum(x @ self.pred_w1 + self.pred_b1, 0) @ self.pred_w2 + self.pred_b2
        return float(np.mean((target - pred) ** 2))

    def update(self, state, lr=0.01):
        """Train predictor to reduce error for this state (marks it as 'seen')."""
        x = np.array(state, dtype=np.float32).reshape(1, -1)
        if x.shape[1] > self.state_dim:
            x = x[:, :self.state_dim]
        elif x.shape[1] < self.state_dim:
            x = np.pad(x, ((0, 0), (0, self.state_dim - x.shape[1])))
        target = np.maximum(x @ self.target_w1 + self.target_b1, 0) @ self.target_w2 + self.target_b2
        h1 = x @ self.pred_w1 + self.pred_b1
        a1 = np.maximum(h1, 0)
        pred = a1 @ self.pred_w2 + self.pred_b2
        d_pred = 2.0 * (pred - target) / pred.shape[1]
        dw2 = a1.T @ d_pred; db2 = d_pred.sum(0)
        da1 = (d_pred @ self.pred_w2.T) * (h1 > 0)
        dw1 = x.T @ da1; db1 = da1.sum(0)
        self.pred_w2 -= lr * dw2; self.pred_b2 -= lr * db2
        self.pred_w1 -= lr * dw1; self.pred_b1 -= lr * db1


# ==============================================================
# Grid → Text Conversion
# ==============================================================
def grid_to_text(grids, max_size=24):
    """Convert 2D grid arrays to text representation for LLM.

    Uses symbols: 0='.', 1-9='1'-'9', 10-15='A'-'F'
    Truncates to max_size to keep prompts manageable.
    """
    lines = []
    for i, grid in enumerate(grids[:3]):  # max 3 grids
        arr = np.array(grid, dtype=np.int32)
        h, w = arr.shape

        # Truncate if too large
        show_h = min(h, max_size)
        show_w = min(w, max_size)

        lines.append(f"Grid {i} ({h}x{w}):")

        # Column indices header
        col_indices = ' '.join(f'{c:1d}' if c < 10 else chr(ord('A') + c - 10)
                               for c in range(show_w))
        lines.append(f"  C: {col_indices}")

        for r in range(show_h):
            symbols = []
            for c in range(show_w):
                v = arr[r, c]
                if v == 0:
                    symbols.append('.')
                elif v <= 9:
                    symbols.append(str(v))
                else:
                    symbols.append(chr(ord('A') + v - 10))
            lines.append(f"  {r:2d}: {' '.join(symbols)}")

        if h > max_size or w > max_size:
            lines.append(f"  (truncated from {h}x{w})")

    return '\n'.join(lines)


def grid_diff_text(prev_grids, curr_grids):
    """Describe what changed between two grid states."""
    if not prev_grids or not curr_grids:
        return "No previous state to compare."

    try:
        prev = np.array(prev_grids[0], dtype=np.int32)
        curr = np.array(curr_grids[0], dtype=np.int32)

        if prev.shape != curr.shape:
            return f"Grid shape changed: {prev.shape} -> {curr.shape}"

        diff = (prev != curr)
        n_changed = diff.sum()

        if n_changed == 0:
            return "Grid unchanged."

        # Find changed positions
        changed_pos = np.argwhere(diff)[:5]  # show up to 5 changes
        changes = []
        for r, c in changed_pos:
            changes.append(f"({r},{c}): {prev[r,c]}->{curr[r,c]}")

        return f"{n_changed} cells changed: {', '.join(changes)}"
    except Exception:
        return "Could not compute diff."


# ==============================================================
# LLM Interface
# ==============================================================
class LLMEngine:
    """Wrapper for llama.cpp GGUF model inference."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
        self.model = None
        self.available = False

        try:
            from llama_cpp import Llama
            if os.path.exists(model_path):
                logger.info(f"Loading LLM from {model_path}...")
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,  # -1 = use all GPU layers
                    n_threads=4,
                    verbose=False,
                )
                self.available = True
                logger.info("LLM loaded successfully!")
            else:
                logger.warning(f"Model file not found: {model_path}")
        except ImportError:
            logger.warning("llama_cpp not available, falling back to random")
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}")

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 100) -> str:
        """Generate text from prompt."""
        if not self.available or self.model is None:
            return ""

        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["User:", "\n\n", "---"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return ""


# ==============================================================
# Action Parser
# ==============================================================
def parse_action(response: str):
    """Parse LLM response to extract GameAction and optional coordinates.

    Returns:
        (action_name, x, y) or (action_name, None, None) or None
    """
    if not response:
        return None

    # Try to find ACTION pattern
    # Match patterns like: "ACTION3", "ACTION6 x=10 y=5", "action 3"
    response_upper = response.upper()

    # Pattern 1: ACTIONN with optional coordinates
    match = re.search(r'ACTION\s*(\d+)', response_upper)
    if match:
        action_num = int(match.group(1))
        if 1 <= action_num <= 7:
            action_name = f"ACTION{action_num}"

            # Check for coordinates (for complex actions)
            x_match = re.search(r'X\s*[=:]\s*(\d+)', response_upper)
            y_match = re.search(r'Y\s*[=:]\s*(\d+)', response_upper)

            x = int(x_match.group(1)) if x_match else None
            y = int(y_match.group(1)) if y_match else None

            return (action_name, x, y)

    # Pattern 2: Just a number 1-7
    match = re.search(r'\b([1-7])\b', response)
    if match:
        action_num = int(match.group(1))
        return (f"ACTION{action_num}", None, None)

    return None


def build_prompt(grid_text: str, action_history: list,
                 levels_completed: int, attempt: int,
                 grid_diff: str = "") -> str:
    """Build the LLM prompt for action selection."""
    # Keep action history short
    recent_actions = action_history[-10:] if action_history else []
    action_str = ', '.join(recent_actions) if recent_actions else 'None yet'

    prompt = f"""You are an expert AI agent playing an interactive puzzle game on a grid.
Your goal is to clear all levels by choosing the right actions.

Available actions:
- ACTION1 to ACTION5: Simple actions (each does something different - explore to learn!)
- ACTION6: Complex action that targets a specific cell. Requires x,y coordinates.
- ACTION7: Another simple action.

RULES:
- Study the grid carefully for patterns, objects, and clues
- Try different actions to understand what each one does
- If the grid isn't changing, try a different action
- ACTION6 can target specific colored cells

Current state:
{grid_text}

{f"Changes from last action: {grid_diff}" if grid_diff else ""}
Levels completed: {levels_completed}
Attempt #{attempt}
Recent actions: {action_str}

Choose ONE action. If using ACTION6, include coordinates.
Respond with just the action, e.g.: "ACTION3" or "ACTION6 x=5 y=3"

Action:"""
    return prompt


# ==============================================================
# SNN-Synthesis v12 Agent: LLM + Test-Time ExIt + RND Curiosity
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v12: LLM-ExIt + Test-Time SNN-ExIt + RND Curiosity.

    4-tier action selection:
    1. LLM (Phi-3-mini) for intelligent reasoning (if available)
    2. Test-Time CNN learned from miracles (if trained)
    3. RND Curiosity-guided exploration (Phase 39: +61pp)
    4. Random fallback

    Key innovations:
    - CNN learns *during gameplay* from miracles (no pre-training)
    - RND novelty scoring replaces simple visit-counting
    - sigma-diverse NBS for maximum exploration diversity
    """
    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RNG
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32))

        # Initialize LLM (gracefully handles missing llama_cpp)
        self.llm = LLMEngine(MODEL_PATH)

        # State tracking
        self.sa_visits = {}
        self.action_history = []
        self.current_episode_actions = []
        self.current_episode_states = []
        self.prev_levels = 0
        self.attempt_count = 0
        self.prev_grids = None

        # Miracle memory (for Test-Time ExIt)
        self.miracle_actions = []
        self.miracle_trajectories = []
        self.miracle_prompts = []

        # Test-Time CNN
        self.cnn_model = None
        self.state_dim = None

        # RND Curiosity (Phase 39)
        self.rnd = RNDCuriosity(state_dim=200, hidden=32)

        # NBS temperature
        self.curiosity_bonus = 2.0

        logger.info(
            f"SNN-Synthesis v12 initialized for {self.game_id} "
            f"(LLM available: {self.llm.available})"
        )

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Only stop on WIN."""
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """3-tier action selection: LLM -> CNN -> Curiosity."""

        # --- RESET when needed ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.current_episode_actions = []
            self.current_episode_states = []
            self.prev_grids = None
            self.attempt_count += 1

            # Retrain CNN periodically
            if (len(self.miracle_trajectories) >= MIN_MIRACLES_TO_TRAIN and
                self.attempt_count % 5 == 0):
                self._train_cnn()

            logger.info(
                f"Game {self.game_id}: RESET (attempt #{self.attempt_count}, "
                f"miracles={len(self.miracle_trajectories)}, "
                f"cnn={'yes' if self.cnn_model else 'no'})"
            )
            return GameAction.RESET

        # --- Check for miracle ---
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            self.miracle_actions.extend(self.current_episode_actions)
            self.miracle_trajectories.append({
                'states': list(self.current_episode_states),
                'actions': list(self.current_episode_actions),
            })
            logger.info(
                f"MIRACLE #{len(self.miracle_trajectories)}! Level {current_levels}! "
                f"Trajectory: {len(self.current_episode_actions)} actions."
            )
            self.current_episode_actions = []
            self.current_episode_states = []
            self.prev_levels = current_levels

            # Train CNN on first batch of miracles
            if len(self.miracle_trajectories) == MIN_MIRACLES_TO_TRAIN:
                self._train_cnn()
                logger.info("Test-Time CNN trained on first miracles!")

        # --- Extract state features for CNN ---
        state_features = self._extract_features(latest_frame)
        self.current_episode_states.append(state_features)

        # --- 4-tier action selection ---
        if self.llm.available and latest_frame.frame:
            # Tier 1: LLM reasoning
            action = self._llm_action(latest_frame)
        elif (self.cnn_model is not None and
              len(self.miracle_trajectories) >= CNN_USE_THRESHOLD):
            # Tier 2: Learned CNN policy
            action = self._cnn_action(state_features)
        elif self.miracle_actions and random.random() < 0.3:
            # Exploit miracle memory
            action = self._exploit_miracle()
        else:
            # Tier 3: RND Curiosity-guided exploration
            action = self._curiosity_action(latest_frame, state_features)

        # Update RND predictor (mark this state as 'seen')
        self.rnd.update(state_features, lr=0.005)

        # Record
        self.current_episode_actions.append(action.name)
        self.action_history.append(action.name)

        # Save previous grid for diff
        self.prev_grids = latest_frame.frame if latest_frame.frame else None

        return action

    def _llm_action(self, frame: FrameData) -> GameAction:
        """Use LLM for action selection."""
        try:
            # Build text representation
            grid_text = grid_to_text(frame.frame)

            # Compute grid diff
            diff_text = ""
            if self.prev_grids:
                diff_text = grid_diff_text(self.prev_grids, frame.frame)

            # Build prompt
            prompt = build_prompt(
                grid_text=grid_text,
                action_history=self.current_episode_actions,
                levels_completed=frame.levels_completed or 0,
                attempt=self.attempt_count,
                grid_diff=diff_text,
            )

            # NBS: sigma-diverse per attempt (Phase 37a)
            sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
            sigma = SIGMA_SCHEDULE[sigma_idx]
            temperature = sigma_to_temperature(sigma)

            # Generate
            response = self.llm.generate(prompt, temperature=temperature, max_tokens=50)
            logger.debug(f"LLM response (temp={temperature}): {response}")

            # Parse action
            parsed = parse_action(response)
            if parsed:
                action_name, x, y = parsed
                action = self._name_to_action(action_name)
                if action is not None:
                    if action.is_simple():
                        action.reasoning = (
                            f"LLM-ExIt (t={temperature:.1f}): {response[:80]}"
                        )
                    elif action.is_complex():
                        # Use LLM coordinates or smart fallback
                        if x is None or y is None:
                            x, y = self._smart_coords(frame)
                        x = max(0, min(63, x))
                        y = max(0, min(63, y))
                        action.set_data({"x": x, "y": y})
                        action.reasoning = {
                            "desired_action": f"{action.value}",
                            "my_reason": f"LLM-ExIt: ({x},{y}), {response[:50]}",
                        }
                    return action

            # Parse failed: fall through to curiosity
            logger.debug(f"LLM parse failed: {response}")

        except Exception as e:
            logger.warning(f"LLM action error: {e}")

        # Fallback
        return self._curiosity_action(frame)

    def _curiosity_action(self, frame: FrameData, state_features: list = None) -> GameAction:
        """RND Curiosity-guided action with sigma-diverse noise.
        Uses RND prediction error as novelty bonus (Phase 39: +61pp improvement).
        """
        state_hash = self._grid_hash(frame)
        candidates = [a for a in GameAction if a is not GameAction.RESET]

        # Get current sigma from schedule
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        # RND curiosity score for current state
        rnd_bonus = 0.0
        if state_features:
            rnd_bonus = self.rnd.curiosity_score(state_features)

        scores = []
        for action in candidates:
            sa_key = f"{state_hash}_{action.name}"
            visits = self.sa_visits.get(sa_key, 0)
            # Combine visit-count novelty with RND curiosity
            visit_novelty = 1.0 / (visits + 1)
            # RND bonus decays with visits (novel states get more exploration)
            combined_novelty = visit_novelty * (1.0 + rnd_bonus * 3.0)
            noise = random.gauss(0, max(0.05, sigma))
            scores.append(combined_novelty + noise)

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

        self._configure_action(action, frame)
        return action

    def _exploit_miracle(self) -> GameAction:
        """Replay random action from miracle memory."""
        miracle_name = random.choice(self.miracle_actions)
        for action in GameAction:
            if action.name == miracle_name:
                self._configure_action(action, self.frames[-1] if self.frames else None)
                return action
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action, None)
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

        logits = self.cnn_model.forward(X)
        acc = (logits.argmax(axis=1) == y).mean()
        logger.info(f"CNN retrained: {len(all_states)} samples, acc={acc:.3f}")

    def _cnn_action(self, state_features: list) -> GameAction:
        """Use learned CNN with sigma-diverse noise."""
        candidates = [a for a in GameAction if a is not GameAction.RESET]
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx]

        x = np.array(state_features[:self.state_dim], dtype=np.float32).reshape(1, -1)
        logits = self.cnn_model.forward(x, noise_sigma=sigma)
        logits = logits[0, :len(candidates)]
        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()

        action_idx = np.random.choice(len(candidates), p=probs)
        action = candidates[action_idx]
        self._configure_action(action, self.frames[-1] if self.frames else None)
        return action

    def _configure_action(self, action: GameAction,
                          frame: Optional[FrameData]) -> None:
        """Configure action with reasoning and coordinates."""
        if action.is_simple():
            action.reasoning = f"SNN-v12: attempt={self.attempt_count}, miracles={len(self.miracle_trajectories)}"
        elif action.is_complex():
            x, y = self._smart_coords(frame) if frame else (random.randint(0, 63), random.randint(0, 63))
            action.set_data({"x": x, "y": y})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"SNN-v12: ({x},{y}), cnn={'yes' if self.cnn_model else 'no'}",
            }

    def _smart_coords(self, frame: FrameData):
        """Pick coordinates targeting non-zero grid cells."""
        x, y = random.randint(0, 63), random.randint(0, 63)
        if frame and frame.frame:
            try:
                arr = np.array(frame.frame[0], dtype=np.int32)
                h, w = arr.shape
                if random.random() < 0.5:
                    nz = np.argwhere(arr != 0)
                    if len(nz) > 0:
                        idx = random.randint(0, len(nz) - 1)
                        y, x = int(nz[idx][0]), int(nz[idx][1])
                else:
                    x = random.randint(0, min(w - 1, 63))
                    y = random.randint(0, min(h - 1, 63))
            except Exception:
                pass
        return max(0, min(63, x)), max(0, min(63, y))

    def _name_to_action(self, name: str) -> Optional[GameAction]:
        """Convert action name string to GameAction enum."""
        for action in GameAction:
            if action.name == name:
                return action
        return None

    def _grid_hash(self, frame: FrameData) -> str:
        """Hash grid state for novelty tracking."""
        if not frame.frame:
            return "empty"
        try:
            arr = np.array(frame.frame[0], dtype=np.int32)
            return hashlib.md5(arr.tobytes()).hexdigest()[:12]
        except Exception:
            return "error"


# Alias for Kaggle runner compatibility
MyAgent = StochasticGoose
