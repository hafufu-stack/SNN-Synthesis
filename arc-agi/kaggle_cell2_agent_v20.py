%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v20 Agent for ARC-AGI-3
# "Immortal Goose" - Crash-proof, timeout-proof final agent
#
# Fixes over v19.1:
# 1. GIVE UP after MAX_ATTEMPTS (no more Timeout Starvation)
# 2. Full try-except armor around choose_action
# 3. Safe coordinate setting for ALL complex action paths
# 4. _replay_sequence_d4 properly configures complex actions
#
# Core formula (v5 proven at 0.13):
# - ALL actions (simple + complex)
# - UCB curiosity + miracle memory
# - Sigma-diverse NBS exploration
#
# Paper: https://doi.org/10.5281/zenodo.19343952
# GitHub: https://github.com/hafufu-stack/SNN-Synthesis
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

# Phase 37a: Sigma-Diverse NBS Schedule
SIGMA_SCHEDULE = [
    0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00,
    0.01, 0.03, 0.08, 0.12, 0.25, 0.35, 0.45, 0.60, 0.85, 0.40,
]

# v20: Give up threshold (Phase 42 lesson: "appropriate surrender")
MAX_ATTEMPTS_PER_LEVEL = 5


# ==============================================================
# Grid Intelligence Module
# ==============================================================
class GridIntelligence:
    """Grid analysis with pattern awareness."""

    def __init__(self):
        self.prev_grid = None
        self.action_effects = {}
        self.hot_regions = []
        self.crystallized_count = 0
        self.color_rarity = {}

    def update(self, grid: np.ndarray, last_action: str = None):
        if self.prev_grid is not None and grid.shape == self.prev_grid.shape:
            diff = (grid != self.prev_grid)
            if diff.any():
                changed_ys, changed_xs = np.where(diff)
                if last_action:
                    if last_action not in self.action_effects:
                        self.action_effects[last_action] = []
                    cy = int(np.mean(changed_ys))
                    cx = int(np.mean(changed_xs))
                    self.action_effects[last_action].append(
                        (cy, cx, int(diff.sum())))
                    self.hot_regions.append((cy, cx))
                    if len(self.hot_regions) > 50:
                        self.hot_regions = self.hot_regions[-30:]
                self.crystallized_count = 0
            else:
                self.crystallized_count += 1

        flat = grid.ravel().astype(np.int32)
        counts = np.bincount(flat, minlength=10)[:10]
        total = max(flat.size, 1)
        self.color_rarity = {}
        for c in range(10):
            if counts[c] > 0:
                self.color_rarity[c] = 1.0 - counts[c] / total

        self.prev_grid = grid.copy()

    def suggest_target(self, grid: np.ndarray) -> tuple:
        """Safe (y, x) selection - always returns valid coordinates."""
        try:
            h, w = grid.shape
            if h <= 0 or w <= 0:
                return 0, 0

            strategy = random.random()

            if self.crystallized_count > 5:
                return random.randint(0, h-1), random.randint(0, w-1)

            if strategy < 0.20 and self.hot_regions:
                cy, cx = random.choice(self.hot_regions)
                y = max(0, min(h-1, cy + random.randint(-2, 2)))
                x = max(0, min(w-1, cx + random.randint(-2, 2)))
                return y, x

            if strategy < 0.40 and self.color_rarity:
                flat = grid.ravel().astype(np.int32)
                bg = np.bincount(flat, minlength=10).argmax()
                rare_colors = sorted(
                    [(c, r) for c, r in self.color_rarity.items() if c != bg],
                    key=lambda x: x[1], reverse=True)
                if rare_colors:
                    target_color = rare_colors[0][0]
                    positions = np.argwhere(grid == target_color)
                    if len(positions) > 0:
                        idx = random.randint(0, len(positions) - 1)
                        return int(positions[idx][0]), int(positions[idx][1])

            if strategy < 0.55:
                flat = grid.ravel().astype(np.int32)
                if len(flat) > 0:
                    bg = np.bincount(flat, minlength=10).argmax()
                    nonbg = np.argwhere(grid != bg)
                    if len(nonbg) > 0:
                        idx = random.randint(0, len(nonbg) - 1)
                        return int(nonbg[idx][0]), int(nonbg[idx][1])

            if strategy < 0.75:
                edges_y, edges_x = [], []
                if h > 1:
                    ey, ex = np.where(grid[1:, :] != grid[:-1, :])
                    edges_y.extend(ey.tolist())
                    edges_x.extend(ex.tolist())
                if w > 1:
                    ey, ex = np.where(grid[:, 1:] != grid[:, :-1])
                    edges_y.extend(ey.tolist())
                    edges_x.extend(ex.tolist())
                if edges_y:
                    idx = random.randint(0, len(edges_y) - 1)
                    return edges_y[idx], edges_x[idx]

            return random.randint(0, max(0, h-1)), random.randint(0, max(0, w-1))

        except Exception:
            return 0, 0


# ==============================================================
# Agent Implementation
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v20: Immortal Goose.

    v19.1 + three critical fixes:
    1. Give up after MAX_ATTEMPTS (no Timeout Starvation)
    2. try-except armor (never crash Kaggle engine)
    3. Safe coordinate setting on ALL code paths
    """
    MAX_ACTIONS = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        self.rng = random.Random(seed)

        self.grid_intel = GridIntelligence()
        self.sa_visits = {}
        self.miracle_actions = []
        self.miracle_sequences = []
        self.current_episode_actions = []
        self.prev_levels = 0
        self.attempt_count = 0
        self.steps = 0
        self.frames = []
        self.last_grid_hash = None
        self.stale_counter = 0
        self.explore_temp = 0.5

        logger.info(f"SNN-Synthesis v20 (Immortal) init for {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """v20 FIX #1: Give up after MAX_ATTEMPTS to avoid Timeout Starvation.
        
        v19.1 bug: only returned True on WIN, causing the agent to
        loop forever on unsolvable problems, wasting the entire 9-hour
        Kaggle time budget on a single task.
        """
        if latest_frame.state is GameState.WIN:
            return True
        # Give up after enough attempts to move on to next problem
        if (latest_frame.state is GameState.GAME_OVER
                and self.attempt_count >= MAX_ATTEMPTS_PER_LEVEL):
            logger.info(f"v20: Giving up after {self.attempt_count} attempts")
            return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """v20 FIX #2: Full try-except armor. Never crash."""
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as e:
            logger.error(f"v20: choose_action crashed: {e}")
            # Return RESET as the safest possible fallback
            return GameAction.RESET

    def _choose_action_inner(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self.frames = frames

        # --- RESET when game not started or after game over ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.current_episode_actions = []
            self.attempt_count += 1
            self.steps = 0
            self.stale_counter = 0
            self.last_grid_hash = None
            self.grid_intel.prev_grid = None
            self.grid_intel.crystallized_count = 0
            self.explore_temp = max(0.1, 0.5 * (0.95 ** self.attempt_count))
            return GameAction.RESET

        self.steps += 1

        # --- Force RESET on action limit ---
        if self.steps >= self.MAX_ACTIONS:
            self.current_episode_actions = []
            self.attempt_count += 1
            self.steps = 0
            self.stale_counter = 0
            self.last_grid_hash = None
            self.grid_intel.prev_grid = None
            self.grid_intel.crystallized_count = 0
            self.explore_temp = max(0.1, 0.5 * (0.95 ** self.attempt_count))
            return GameAction.RESET

        # --- Check for miracle (level completed) ---
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            self.miracle_sequences.append(list(self.current_episode_actions))
            self.miracle_actions.extend(self.current_episode_actions)
            logger.info(
                f"MIRACLE! Level {current_levels}! "
                f"Saved {len(self.current_episode_actions)} actions."
            )
            self.current_episode_actions = []
            self.prev_levels = current_levels
            self.steps = 0

        # --- Extract and analyze grid ---
        grid = self._extract_grid(latest_frame)
        last_action = (self.current_episode_actions[-1]
                       if self.current_episode_actions else None)
        self.grid_intel.update(grid, last_action)

        # --- Stale / Crystallization detection ---
        grid_hash = hash(grid.tobytes())
        if grid_hash == self.last_grid_hash:
            self.stale_counter += 1
            if self.stale_counter > 15:
                self.stale_counter = 0
                action = self.rng.choice(
                    [a for a in GameAction if a is not GameAction.RESET])
                self._configure_action(action, grid, "v20:anti_crystal")
                self.current_episode_actions.append(action.name)
                return action
        else:
            self.stale_counter = 0
        self.last_grid_hash = grid_hash

        # --- Compute state hash ---
        state_hash = hashlib.md5(grid.tobytes()).hexdigest()[:12]

        # --- Action selection ---
        exploit_rate = min(0.45, 0.15 + 0.05 * len(self.miracle_sequences))

        if self.miracle_sequences and self.rng.random() < exploit_rate * 0.5:
            action = self._replay_sequence_d4(grid)
        elif self.miracle_actions and self.rng.random() < exploit_rate:
            action = self._exploit_miracle(grid)
        else:
            action = self._curiosity_action(state_hash, grid)

        self.current_episode_actions.append(action.name)
        return action

    def _curiosity_action(self, state_hash: str, grid: np.ndarray) -> GameAction:
        """Continuous-thought action selection with temperature decay."""
        candidates = [a for a in GameAction if a is not GameAction.RESET]

        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        scores = []
        for action in candidates:
            sa_key = f"{state_hash}_{action.name}"
            visits = self.sa_visits.get(sa_key, 0)
            novelty = 1.0 / (visits + 1)
            noise = self.rng.gauss(0, max(0.05, sigma))
            scores.append(novelty + noise)

        temp = max(0.1, self.explore_temp)
        max_score = max(scores)
        weights = [2.718 ** ((s - max_score) / temp) for s in scores]
        total = sum(weights)
        probs = [w / total for w in weights]

        r = self.rng.random()
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

        self._configure_action(action, grid, f"v20:explore_s{sigma:.2f}")
        return action

    def _exploit_miracle(self, grid: np.ndarray) -> GameAction:
        """Replay a random action from miracle memory."""
        miracle_name = self.rng.choice(self.miracle_actions)
        for action in GameAction:
            if action.name == miracle_name:
                self._configure_action(action, grid, "v20:miracle_exploit")
                return action
        action = self.rng.choice(
            [a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action, grid, "v20:miracle_fallback")
        return action

    def _replay_sequence_d4(self, grid: np.ndarray) -> GameAction:
        """v20 FIX #3: D4 sequence replay with SAFE coordinate setting.
        
        v19.1 bug: complex actions returned without set_data(), causing
        Kaggle engine to crash on missing coordinates.
        Now ALL actions go through _configure_action for safety.
        """
        seq = self.rng.choice(self.miracle_sequences)
        step_in_seq = self.steps % max(1, len(seq))

        if step_in_seq < len(seq):
            a_name = seq[step_in_seq]
            for a in GameAction:
                if a.name == a_name:
                    # v20: ALWAYS configure, even for replayed actions
                    self._configure_action(a, grid, "v20:seq_replay")
                    return a

        action = self.rng.choice(
            [a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action, grid, "v20:seq_fallback")
        return action

    def _configure_action(self, action: GameAction, grid: np.ndarray,
                          reason_str: str = "v26") -> None:
        """Configure action with intelligent targeting.
        
        CRITICAL: For complex actions, reasoning MUST be a dict with
        set_data() called. For simple actions, reasoning is a string.
        """
        if action.is_simple():
            action.reasoning = reason_str
        elif action.is_complex():
            y, x = self.grid_intel.suggest_target(grid)
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": reason_str,
            }

    def _extract_grid(self, frame: FrameData) -> np.ndarray:
        """Extract 2D grid from frame data."""
        if not frame.frame:
            return np.zeros((1, 1), dtype=np.float32)
        try:
            grid = np.array(frame.frame[0], dtype=np.float32)
            if grid.ndim == 1:
                side = int(np.sqrt(len(grid)))
                if side * side == len(grid):
                    grid = grid.reshape(side, side)
                else:
                    grid = grid.reshape(1, -1)
            return grid
        except Exception:
            return np.zeros((1, 1), dtype=np.float32)


# Alias for Kaggle runner compatibility
MyAgent = StochasticGoose
