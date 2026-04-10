%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v8 Agent for ARC-AGI-3
# Smart Random: Curiosity-Guided Exploration (No CNN overhead)
#
# Paper: https://doi.org/10.5281/zenodo.19481773
# GitHub: https://github.com/hifunsk/snn-synthesis
# Author: Hiroto Funasaki
# ==============================================================
import hashlib
import logging
import random
import time
from typing import Any

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# Sigma-Diverse NBS Schedule (Phase 37a: proven equivalent to best-tuned)
SIGMA_SCHEDULE = [
    0.0, 0.05, 0.15, 0.30, 0.50, 0.01, 0.10, 0.20, 0.75, 1.00,
]


# ==============================================================
# SNN-Synthesis v8: Smart Random Agent
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v8: Curiosity-Guided Smart Random.

    Strategy close to Random baseline but with:
    1. Visit counting to avoid repeating same state-action pairs
    2. Miracle memory to bias toward successful action patterns
    3. Proper RESET handling (only on NOT_PLAYED/GAME_OVER)
    4. NO CNN overhead (pure lightweight logic)
    5. Handle both simple and complex actions correctly
    """
    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RNG with game-specific seed
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

        # State-action visit counts for novelty
        self.sa_visits = {}

        # Miracle memory: action sequences that led to level clears
        self.miracle_actions = []           # flattened list of successful actions
        self.current_episode_actions = []   # current episode's action history

        # Progress tracking
        self.prev_levels = 0
        self.attempt_count = 0

        logger.info(f"SNN-Synthesis v8 initialized for {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Only stop on WIN. Allow retries via RESET after GAME_OVER."""
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose action with curiosity-guided exploration."""

        # --- RESET when game not started or after game over ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.current_episode_actions = []
            self.attempt_count += 1
            return GameAction.RESET

        # --- Check for miracle (level completed) ---
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            # Save successful action sequence
            self.miracle_actions.extend(self.current_episode_actions)
            logger.info(
                f"MIRACLE! Level {current_levels}! "
                f"Saved {len(self.current_episode_actions)} actions. "
                f"Total miracle data: {len(self.miracle_actions)}"
            )
            self.current_episode_actions = []
            self.prev_levels = current_levels

        # --- Compute state hash for novelty tracking ---
        state_hash = self._grid_hash(latest_frame)

        # --- Choose action ---
        if self.miracle_actions and random.random() < 0.3:
            # 30% chance: replay a miracle action (exploitation)
            action = self._exploit_miracle()
        else:
            # 70% chance: curiosity-guided exploration
            action = self._curiosity_action(state_hash)

        # --- Record action ---
        self.current_episode_actions.append(action.name)

        return action

    def _curiosity_action(self, state_hash: str) -> GameAction:
        """Select action favoring less-visited state-action pairs."""
        candidates = [a for a in GameAction if a is not GameAction.RESET]

        # Get current sigma from schedule
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        # Score each action by novelty (less visits = higher score)
        scores = []
        for action in candidates:
            sa_key = f"{state_hash}_{action.name}"
            visits = self.sa_visits.get(sa_key, 0)
            # UCB-style: novelty + sigma-diverse noise
            novelty = 1.0 / (visits + 1)
            noise = random.gauss(0, max(0.05, sigma))
            scores.append(novelty + noise)

        # Softmax-like selection (favor higher scores)
        max_score = max(scores)
        weights = [2.718 ** ((s - max_score) / 0.3) for s in scores]
        total = sum(weights)
        probs = [w / total for w in weights]

        # Weighted random choice
        r = random.random()
        cumsum = 0
        selected_idx = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                selected_idx = i
                break

        action = candidates[selected_idx]

        # Update visit count
        sa_key = f"{state_hash}_{action.name}"
        self.sa_visits[sa_key] = self.sa_visits.get(sa_key, 0) + 1

        # Configure action
        self._configure_action(action)

        return action

    def _exploit_miracle(self) -> GameAction:
        """Replay a random action from miracle memory."""
        miracle_name = random.choice(self.miracle_actions)

        # Find matching GameAction
        for action in GameAction:
            if action.name == miracle_name:
                self._configure_action(action)
                return action

        # Fallback: random action
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action)
        return action

    def _configure_action(self, action: GameAction) -> None:
        """Configure action with proper data (reasoning, coordinates)."""
        if action.is_simple():
            action.reasoning = f"SNN-v8: attempt={self.attempt_count}"
        elif action.is_complex():
            # Random coordinates within reasonable bounds
            x = random.randint(0, 63)
            y = random.randint(0, 63)

            # If grid data available, prefer coordinates near content
            if self.frames and self.frames[-1].frame:
                try:
                    arr = np.array(self.frames[-1].frame[0], dtype=np.int32)
                    h, w = arr.shape
                    # 50% chance: target a non-zero cell
                    if random.random() < 0.5:
                        nonzero = np.argwhere(arr != 0)
                        if len(nonzero) > 0:
                            idx = random.randint(0, len(nonzero) - 1)
                            y, x = int(nonzero[idx][0]), int(nonzero[idx][1])
                    else:
                        # Random within grid bounds
                        x = random.randint(0, min(w - 1, 63))
                        y = random.randint(0, min(h - 1, 63))
                except Exception:
                    pass

            action.set_data({"x": x, "y": y})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"SNN-v8: target ({x},{y})",
            }

    def _grid_hash(self, frame: FrameData) -> str:
        """Hash the grid state for novelty tracking."""
        if not frame.frame:
            return "empty"
        try:
            arr = np.array(frame.frame[0], dtype=np.int32)
            return hashlib.md5(arr.tobytes()).hexdigest()[:12]
        except Exception:
            return "error"


# Alias for Kaggle runner compatibility
MyAgent = StochasticGoose
