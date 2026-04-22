%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v27 Agent for ARC-AGI-3
# "Metabolic Goose" - Entropy-aware action with hormonal memory
#
# Upgrades over v20 (Immortal Goose):
# 1. METABOLIC SLEEP (P175): skip exploration on crystallized
#    regions, focus actions on uncertain/changing areas
# 2. HORMONAL BROADCAST (P174): global grid summary broadcast
#    to action selection for non-local awareness
# 3. ANNEALED EXPLORATION (P176): temperature decay schedule
#    from hot (diverse) to cold (exploitative) over attempts
# 4. ENTROPY-AWARE TARGETING: target high-entropy grid regions
#    where the most change/uncertainty exists
#
# Core formula (v5 proven at 0.13):
# - ALL actions (simple + complex)
# - UCB curiosity + miracle memory
# - Sigma-diverse NBS exploration
# - NEW: Entropy-guided targeting + metabolic efficiency
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

# v20: Give up threshold
MAX_ATTEMPTS_PER_LEVEL = 5


# ==============================================================
# Grid Intelligence Module (v27: + Entropy + Hormone)
# ==============================================================
class GridIntelligence:
    """Grid analysis with entropy-aware metabolic targeting."""

    def __init__(self):
        self.prev_grid = None
        self.action_effects = {}
        self.hot_regions = []
        self.crystallized_count = 0
        self.color_rarity = {}
        # v27: Metabolic Sleep state
        self.pixel_entropy = None    # per-pixel change entropy
        self.hormone = None          # global grid summary (P174)
        self.change_history = []     # track which regions change most

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

                # v27: Update pixel-level entropy (change frequency map)
                self._update_entropy(diff)
            else:
                self.crystallized_count += 1
        else:
            # First frame - initialize entropy map
            h, w = grid.shape
            self.pixel_entropy = np.zeros((h, w), dtype=np.float32)

        flat = grid.ravel().astype(np.int32)
        counts = np.bincount(flat, minlength=10)[:10]
        total = max(flat.size, 1)
        self.color_rarity = {}
        for c in range(10):
            if counts[c] > 0:
                self.color_rarity[c] = 1.0 - counts[c] / total

        # v27: Hormonal Broadcast - compute global grid summary
        self._update_hormone(grid)

        self.prev_grid = grid.copy()

    def _update_entropy(self, diff_mask: np.ndarray):
        """P175: Track per-pixel change entropy (metabolic activity)."""
        h, w = diff_mask.shape
        if self.pixel_entropy is None or self.pixel_entropy.shape != (h, w):
            self.pixel_entropy = np.zeros((h, w), dtype=np.float32)
        # Exponential moving average of change mask
        self.pixel_entropy = self.pixel_entropy * 0.8 + diff_mask.astype(np.float32) * 0.2

    def _update_hormone(self, grid: np.ndarray):
        """P174: Global Average Pooling as 'hormone' signal."""
        flat = grid.ravel().astype(np.int32)
        counts = np.bincount(flat, minlength=10)[:10]
        total = max(flat.size, 1)
        # Hormone = color distribution (global summary, broadcast to all decisions)
        self.hormone = counts.astype(np.float32) / total

    def suggest_target(self, grid: np.ndarray) -> tuple:
        """v27: Entropy-aware targeting - preferentially target uncertain regions."""
        try:
            h, w = grid.shape
            if h <= 0 or w <= 0:
                return 0, 0

            strategy = random.random()

            # v27: METABOLIC SLEEP - if grid is crystallized, don't waste
            # compute on stable regions. Target high-entropy (active) zones.
            if (strategy < 0.30 and self.pixel_entropy is not None
                    and self.pixel_entropy.shape == (h, w)):
                entropy = self.pixel_entropy
                max_e = entropy.max()
                if max_e > 0.05:
                    # Target region with highest change entropy
                    # (this is where the "uncertain" cells are)
                    hot_mask = entropy > max_e * 0.5
                    hot_ys, hot_xs = np.where(hot_mask)
                    if len(hot_ys) > 0:
                        idx = random.randint(0, len(hot_ys) - 1)
                        return int(hot_ys[idx]), int(hot_xs[idx])

            if self.crystallized_count > 5:
                return random.randint(0, h-1), random.randint(0, w-1)

            if strategy < 0.45 and self.hot_regions:
                cy, cx = random.choice(self.hot_regions)
                y = max(0, min(h-1, cy + random.randint(-2, 2)))
                x = max(0, min(w-1, cx + random.randint(-2, 2)))
                return y, x

            if strategy < 0.60 and self.color_rarity:
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

            if strategy < 0.75:
                flat = grid.ravel().astype(np.int32)
                if len(flat) > 0:
                    bg = np.bincount(flat, minlength=10).argmax()
                    nonbg = np.argwhere(grid != bg)
                    if len(nonbg) > 0:
                        idx = random.randint(0, len(nonbg) - 1)
                        return int(nonbg[idx][0]), int(nonbg[idx][1])

            if strategy < 0.90:
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
    SNN-Synthesis v27: Metabolic Goose.

    v20 (Immortal) + Season 25 upgrades:
    1. Entropy-aware targeting (P175 Metabolic Sleep)
    2. Hormonal grid awareness (P174 non-local info)
    3. Annealed temperature schedule (P176 crystallization)
    4. Metabolic efficiency: skip redundant actions
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
        # v27: Annealed exploration (P176)
        # Start hot (diverse), cool down (exploitative)
        self.explore_temp = 1.0  # higher initial temp than v20

        logger.info(f"SNN-Synthesis v27 (Metabolic) init for {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """v20 logic: Give up after MAX_ATTEMPTS."""
        if latest_frame.state is GameState.WIN:
            return True
        if (latest_frame.state is GameState.GAME_OVER
                and self.attempt_count >= MAX_ATTEMPTS_PER_LEVEL):
            logger.info(f"v27: Giving up after {self.attempt_count} attempts")
            return True
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Full try-except armor. Never crash."""
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as e:
            logger.error(f"v27: choose_action crashed: {e}")
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
            self.grid_intel.pixel_entropy = None
            # v27: Annealed cooling (P176 simulated annealing)
            # Temperature decays exponentially: hot -> cold over attempts
            self.explore_temp = max(0.05, 1.0 * (0.85 ** self.attempt_count))
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
            self.grid_intel.pixel_entropy = None
            self.explore_temp = max(0.05, 1.0 * (0.85 ** self.attempt_count))
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

        # --- v27: Metabolic Skip (P175) ---
        # If grid is totally crystallized (no changes for many steps),
        # do a random drastic action to break symmetry
        if self.grid_intel.crystallized_count > 10:
            action = self.rng.choice(
                [a for a in GameAction if a is not GameAction.RESET])
            self._configure_action(action, grid, "v27:metabolic_break")
            self.current_episode_actions.append(action.name)
            self.grid_intel.crystallized_count = 0
            return action

        # --- Stale detection ---
        grid_hash = hash(grid.tobytes())
        if grid_hash == self.last_grid_hash:
            self.stale_counter += 1
            if self.stale_counter > 12:
                self.stale_counter = 0
                action = self.rng.choice(
                    [a for a in GameAction if a is not GameAction.RESET])
                self._configure_action(action, grid, "v27:anti_stale")
                self.current_episode_actions.append(action.name)
                return action
        else:
            self.stale_counter = 0
        self.last_grid_hash = grid_hash

        # --- Compute state hash ---
        state_hash = hashlib.md5(grid.tobytes()).hexdigest()[:12]

        # --- Action selection with annealed exploitation ---
        exploit_rate = min(0.50, 0.15 + 0.06 * len(self.miracle_sequences))

        if self.miracle_sequences and self.rng.random() < exploit_rate * 0.5:
            action = self._replay_sequence_d4(grid)
        elif self.miracle_actions and self.rng.random() < exploit_rate:
            action = self._exploit_miracle(grid)
        else:
            action = self._curiosity_action(state_hash, grid)

        self.current_episode_actions.append(action.name)
        return action

    def _curiosity_action(self, state_hash: str, grid: np.ndarray) -> GameAction:
        """v27: Curiosity with annealed temperature (P176)."""
        candidates = [a for a in GameAction if a is not GameAction.RESET]

        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        # v27: Hormone-informed action bonus (P174)
        # If hormone detects high color diversity, prefer complex actions
        hormone_bonus = 0.0
        if self.grid_intel.hormone is not None:
            color_diversity = (self.grid_intel.hormone > 0.01).sum()
            if color_diversity >= 4:
                hormone_bonus = 0.15  # more colors = more complex exploration

        scores = []
        for action in candidates:
            sa_key = f"{state_hash}_{action.name}"
            visits = self.sa_visits.get(sa_key, 0)
            novelty = 1.0 / (visits + 1)
            noise = self.rng.gauss(0, max(0.05, sigma))
            # v27: bonus for complex actions when grid is colorful
            bonus = hormone_bonus if action.is_complex() else 0
            scores.append(novelty + noise + bonus)

        # v27: Use annealed temperature (P176)
        temp = max(0.05, self.explore_temp)
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

        self._configure_action(action, grid, f"v27:explore_s{sigma:.2f}_t{temp:.2f}")
        return action

    def _exploit_miracle(self, grid: np.ndarray) -> GameAction:
        """Replay a random action from miracle memory."""
        miracle_name = self.rng.choice(self.miracle_actions)
        for action in GameAction:
            if action.name == miracle_name:
                self._configure_action(action, grid, "v27:miracle_exploit")
                return action
        action = self.rng.choice(
            [a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action, grid, "v27:miracle_fallback")
        return action

    def _replay_sequence_d4(self, grid: np.ndarray) -> GameAction:
        """D4 sequence replay with safe coordinate setting."""
        seq = self.rng.choice(self.miracle_sequences)
        step_in_seq = self.steps % max(1, len(seq))

        if step_in_seq < len(seq):
            a_name = seq[step_in_seq]
            for a in GameAction:
                if a.name == a_name:
                    self._configure_action(a, grid, "v27:seq_replay")
                    return a

        action = self.rng.choice(
            [a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action, grid, "v27:seq_fallback")
        return action

    def _configure_action(self, action: GameAction, grid: np.ndarray,
                          reason_str: str = "v27") -> None:
        """Configure action with entropy-aware targeting (v27)."""
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
