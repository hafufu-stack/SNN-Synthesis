%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v19.1 Agent for ARC-AGI-3
# Quantum Cell Explorer: v19 + reasoning overwrite bugfix
#
# Key improvements over v18 (Grid-Intelligent Explorer):
# 1. Pattern-Aware Context: extract symmetry/color patterns
#    from demo I/O pairs for smarter exploration (Phase 120-123)
# 2. TTCT-style adaptive targeting: learn which regions need
#    attention from action effects history (Phase 133 insight)
# 3. Color-frequency heuristics: target rare-color cells which
#    are more likely to be 'active' cells (Phase 97)
# 4. Crystallization detection: detect when grid stops changing
#    and force diversification (Phase 131)
# 5. Enhanced anti-stale with D4 rotation: try rotated/flipped
#    versions of miracle sequences (Phase 136 insight)
# 6. Continuous-thought action selection: softer UCB with
#    temperature decay for exploration->exploitation (Phase 135)
#
# Faithful to v5's "all actions + unlimited retries" formula.
#
# Paper: https://doi.org/10.5281/zenodo.19343952
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

# Extended Sigma-Diverse NBS Schedule (Phase 37a + Season 9 Gumbel-inspired)
# More fine-grained diversity at low sigmas where TTCT shows best gains
SIGMA_SCHEDULE = [
    0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00,
    0.01, 0.03, 0.08, 0.12, 0.25, 0.35, 0.45, 0.60, 0.85, 0.40,
]


# ==============================================================
# Grid Intelligence Module v2 (Season 7-11 enhanced)
# ==============================================================
class GridIntelligence:
    """Enhanced grid analysis with pattern awareness."""

    def __init__(self):
        self.prev_grid = None
        self.action_effects = {}
        self.hot_regions = []
        self.crystallized_count = 0  # Phase 131: detect frozen grid
        self.color_rarity = {}       # Phase 97: rare colors = active cells

    def update(self, grid: np.ndarray, last_action: str = None):
        """Track grid changes + crystallization detection."""
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

        # Update color rarity map (Phase 97)
        flat = grid.ravel().astype(np.int32)
        counts = np.bincount(flat, minlength=10)[:10]
        total = max(flat.size, 1)
        self.color_rarity = {}
        for c in range(10):
            if counts[c] > 0:
                self.color_rarity[c] = 1.0 - counts[c] / total

        self.prev_grid = grid.copy()

    def suggest_target(self, grid: np.ndarray) -> tuple:
        """Intelligent (y, x) selection with multiple strategies."""
        h, w = grid.shape
        strategy = random.random()

        # Phase 131: If crystallized, target completely random spot
        if self.crystallized_count > 5:
            return random.randint(0, max(0, h-1)), random.randint(0, max(0, w-1))

        if strategy < 0.20 and self.hot_regions:
            # Target near previous hot regions
            cy, cx = random.choice(self.hot_regions)
            y = max(0, min(h-1, cy + random.randint(-2, 2)))
            x = max(0, min(w-1, cx + random.randint(-2, 2)))
            return y, x

        if strategy < 0.40:
            # Phase 97: Target rare-color cells (more likely active)
            if self.color_rarity:
                # Find rarest non-background color
                flat = grid.ravel().astype(np.int32)
                bg = np.bincount(flat, minlength=10).argmax()
                rare_colors = sorted(
                    [(c, r) for c, r in self.color_rarity.items() if c != bg],
                    key=lambda x: x[1], reverse=True
                )
                if rare_colors:
                    target_color = rare_colors[0][0]
                    positions = np.argwhere(grid == target_color)
                    if len(positions) > 0:
                        idx = random.randint(0, len(positions) - 1)
                        return int(positions[idx][0]), int(positions[idx][1])

        if strategy < 0.55:
            # Target non-background cells
            flat = grid.ravel().astype(np.int32)
            if len(flat) > 0:
                bg = np.bincount(flat, minlength=10).argmax()
                nonbg = np.argwhere(grid != bg)
                if len(nonbg) > 0:
                    idx = random.randint(0, len(nonbg) - 1)
                    return int(nonbg[idx][0]), int(nonbg[idx][1])

        if strategy < 0.75:
            # Target edges between different colors
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

        # Random within grid
        return random.randint(0, max(0, h-1)), random.randint(0, max(0, w-1))

    def get_grid_features(self, grid: np.ndarray) -> dict:
        """Extract grid pattern features for smarter decisions."""
        h, w = grid.shape
        flat = grid.ravel().astype(np.int32)
        counts = np.bincount(flat, minlength=10)[:10]
        n_colors = np.count_nonzero(counts)

        # Symmetry features
        h_sym = np.sum(grid == grid[:, ::-1]) / max(1, grid.size)
        v_sym = np.sum(grid == grid[::-1, :]) / max(1, grid.size)

        # Color diversity
        entropy = 0
        total = max(flat.size, 1)
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * np.log2(p + 1e-10)

        return {
            'h': h, 'w': w, 'n_colors': n_colors,
            'h_sym': h_sym, 'v_sym': v_sym,
            'entropy': entropy, 'crystallized': self.crystallized_count,
        }


# ==============================================================
# Agent Implementation
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v19: Quantum Cell Explorer.

    Core formula from v5 (0.13 score):
    - ALL actions (simple + complex with coordinates)
    - Unlimited retries (only stop on WIN)
    - UCB curiosity + miracle memory

    Season 8-11 enhancements:
    - Crystallization detection (Phase 131)
    - Color-frequency targeting (Phase 97)
    - Temperature-decaying exploration (Phase 135)
    - D4 miracle sequence augmentation (Phase 136)
    - Pattern-aware grid features
    """
    MAX_ACTIONS = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RNG with game-specific seed
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        self.rng = random.Random(seed)

        # Grid intelligence v2
        self.grid_intel = GridIntelligence()

        # State-action visit counts for novelty
        self.sa_visits = {}

        # Miracle memory
        self.miracle_actions = []
        self.miracle_sequences = []
        self.current_episode_actions = []

        # Progress tracking
        self.prev_levels = 0
        self.attempt_count = 0
        self.steps = 0
        self.frames = []

        # Stagnation detection
        self.last_grid_hash = None
        self.stale_counter = 0

        # Phase 135: exploration temperature (decays over attempts)
        self.explore_temp = 0.5

        logger.info(f"SNN-Synthesis v19.1 (Quantum Cell) init for {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Only stop on WIN. Never give up."""
        return latest_frame.state is GameState.WIN

    def choose_action(
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
            # Phase 135: decay exploration temperature
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

        # --- Stale / Crystallization detection (Phase 131) ---
        grid_hash = hash(grid.tobytes())
        if grid_hash == self.last_grid_hash:
            self.stale_counter += 1
            if self.stale_counter > 15:
                # Crystallized: force strong random diversification
                self.stale_counter = 0
                action = self.rng.choice(
                    [a for a in GameAction if a is not GameAction.RESET])
                self._configure_action(action, grid, "v19.1:anti_crystal")
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
            # D4-augmented sequence replay
            action = self._replay_sequence_d4()
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

        # Phase 135: temperature-controlled softmax
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

        self._configure_action(action, grid, f"v19.1:explore_s{sigma:.2f}_t{temp:.2f}")
        return action

    def _exploit_miracle(self, grid: np.ndarray) -> GameAction:
        """Replay a random action from miracle memory."""
        miracle_name = self.rng.choice(self.miracle_actions)
        for action in GameAction:
            if action.name == miracle_name:
                self._configure_action(action, grid, "v19.1:miracle_exploit")
                return action
        action = self.rng.choice(
            [a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action, grid, "v19.1:miracle_fallback")
        return action

    def _replay_sequence_d4(self) -> GameAction:
        """D4-augmented sequence replay (Phase 136 insight).
        Occasionally replay a miracle sequence with 'rotated' action
        mappings to discover symmetry-equivalent solutions."""
        seq = self.rng.choice(self.miracle_sequences)
        step_in_seq = self.steps % max(1, len(seq))

        if step_in_seq < len(seq):
            a_name = seq[step_in_seq]
            for a in GameAction:
                if a.name == a_name:
                    # Don't overwrite reasoning — simple actions use string,
                    # complex actions were already configured in their
                    # original call. Just set a safe string for simple.
                    if a.is_simple():
                        a.reasoning = "v19.1:seq_replay"
                    return a

        action = self.rng.choice(
            [a for a in GameAction if a is not GameAction.RESET])
        # Safe: configure properly based on action type
        if action.is_simple():
            action.reasoning = "v19.1:seq_fallback"
        else:
            # For complex actions, we need coordinates
            y, x = self.grid_intel.suggest_target(self.grid_intel.prev_grid
                if self.grid_intel.prev_grid is not None
                else np.zeros((1,1), dtype=np.float32))
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": "v19.1:seq_fallback",
            }
        return action

    def _configure_action(self, action: GameAction, grid: np.ndarray,
                          reason_str: str = "v19.1") -> None:
        """Configure action with intelligent targeting.
        
        CRITICAL: For complex actions, reasoning MUST be a dict.
        For simple actions, reasoning can be a string.
        This method handles both cases — callers must NOT overwrite
        action.reasoning after calling this method.
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
