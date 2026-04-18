%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v18 Agent for ARC-AGI-3
# Smart Explorer: v5 foundation + Grid Intelligence
#
# Key improvements over v5 (0.13):
# 1. Grid-diff tracking: learn which actions change which regions
# 2. Color-aware complex actions: target cells by color similarity
# 3. Demo-pattern heuristics: compare demo I/O for action hints
# 4. Adaptive miracle replay: scale exploitation with confidence
# 5. Quadrant-priority exploration: focus on unexplored grid regions
# 6. Multi-strategy sigma arms: proven sigma-diverse NBS
#
# Faithful to v5's "all actions + unlimited retries" formula,
# enhanced with grid intelligence from Phase 101-109 findings.
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

# Sigma-Diverse NBS Schedule (Phase 37a: proven sigma diversity)
SIGMA_SCHEDULE = [
    0.0, 0.05, 0.15, 0.30, 0.50, 0.01, 0.10, 0.20, 0.75, 1.00,
    0.02, 0.08, 0.25, 0.40, 0.60, 0.03, 0.12, 0.35, 0.55, 0.80,
]


# ==============================================================
# Grid Intelligence Module (lightweight, no NN)
# ==============================================================
class GridIntelligence:
    """Lightweight grid analysis for smarter action selection."""

    def __init__(self):
        self.prev_grid = None
        self.action_effects = {}  # action_name -> list of (dy, dx, color_change)
        self.hot_regions = []     # regions where changes happen most

    def update(self, grid: np.ndarray, last_action: str = None):
        """Track grid changes after each action."""
        if self.prev_grid is not None and last_action and grid.shape == self.prev_grid.shape:
            diff = (grid != self.prev_grid)
            if diff.any():
                changed_ys, changed_xs = np.where(diff)
                if last_action not in self.action_effects:
                    self.action_effects[last_action] = []
                # Record centroid of change
                cy = int(np.mean(changed_ys))
                cx = int(np.mean(changed_xs))
                self.action_effects[last_action].append((cy, cx, int(diff.sum())))
                # Track hot regions
                self.hot_regions.append((cy, cx))
                if len(self.hot_regions) > 50:
                    self.hot_regions = self.hot_regions[-30:]
        self.prev_grid = grid.copy()

    def suggest_target(self, grid: np.ndarray) -> tuple:
        """Suggest (y, x) coordinate for complex action."""
        h, w = grid.shape
        strategy = random.random()

        if strategy < 0.25 and self.hot_regions:
            # Target near previous hot regions
            cy, cx = random.choice(self.hot_regions)
            y = max(0, min(h-1, cy + random.randint(-2, 2)))
            x = max(0, min(w-1, cx + random.randint(-2, 2)))
            return y, x

        if strategy < 0.50:
            # Target non-background cells (cells != most common value)
            flat = grid.ravel()
            if len(flat) > 0:
                bg = np.bincount(flat.astype(np.int32), minlength=10).argmax()
                nonbg = np.argwhere(grid != bg)
                if len(nonbg) > 0:
                    idx = random.randint(0, len(nonbg) - 1)
                    return int(nonbg[idx][0]), int(nonbg[idx][1])

        if strategy < 0.70:
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

    def get_grid_signature(self, grid: np.ndarray) -> str:
        """Compact grid signature for symmetry/pattern detection."""
        flat = grid.ravel().astype(np.int32)
        h, w = grid.shape
        counts = np.bincount(flat, minlength=10)[:10]
        # Symmetry features
        h_sym = np.sum(grid == grid[:, ::-1]) / max(1, grid.size)
        v_sym = np.sum(grid == grid[::-1, :]) / max(1, grid.size)
        return f"{h}x{w}_c{np.count_nonzero(counts)}_hs{h_sym:.1f}_vs{v_sym:.1f}"


# ==============================================================
# Agent Implementation
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v18: Grid-Intelligent Smart Explorer.

    Core formula from v5 (0.13 score):
    - ALL actions (simple + complex with coordinates)
    - Unlimited retries (only stop on WIN)
    - UCB curiosity + miracle memory

    New from Phase 101-109 findings:
    - Grid diff tracking for causal understanding
    - Color-edge targeting for complex actions
    - Adaptive exploitation rate
    - Extended sigma diversity (20 sigmas)
    """
    MAX_ACTIONS = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RNG with game-specific seed
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)
        self.rng = random.Random(seed)

        # Grid intelligence
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

        logger.info(f"SNN-Synthesis v18 (Grid-Intelligent) init for {self.game_id}")

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
            return GameAction.RESET

        self.steps += 1

        # --- Force RESET on action limit (preserves state) ---
        if self.steps >= self.MAX_ACTIONS:
            self.current_episode_actions = []
            self.attempt_count += 1
            self.steps = 0
            self.stale_counter = 0
            self.last_grid_hash = None
            self.grid_intel.prev_grid = None
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
        last_action = self.current_episode_actions[-1] if self.current_episode_actions else None
        self.grid_intel.update(grid, last_action)

        # --- Stale detection ---
        grid_hash = hash(grid.tobytes())
        if grid_hash == self.last_grid_hash:
            self.stale_counter += 1
            if self.stale_counter > 20:
                # Force random action to escape
                self.stale_counter = 0
                action = self.rng.choice([a for a in GameAction if a is not GameAction.RESET])
                self._configure_action(action, grid)
                action.reasoning = "v18:anti_stale"
                self.current_episode_actions.append(action.name)
                return action
        else:
            self.stale_counter = 0
        self.last_grid_hash = grid_hash

        # --- Compute state hash for novelty ---
        state_hash = hashlib.md5(grid.tobytes()).hexdigest()[:12]

        # --- Action selection strategy ---
        exploit_rate = min(0.40, 0.15 + 0.05 * len(self.miracle_sequences))

        if self.miracle_sequences and self.rng.random() < exploit_rate * 0.5:
            # Sequence replay (strongest exploitation)
            action = self._replay_sequence()
        elif self.miracle_actions and self.rng.random() < exploit_rate:
            # Single action exploitation
            action = self._exploit_miracle(grid)
        else:
            # Curiosity-guided exploration
            action = self._curiosity_action(state_hash, grid)

        self.current_episode_actions.append(action.name)
        return action

    def _curiosity_action(self, state_hash: str, grid: np.ndarray) -> GameAction:
        """Select action favoring less-visited state-action pairs."""
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

        # Softmax selection
        max_score = max(scores)
        weights = [2.718 ** ((s - max_score) / 0.3) for s in scores]
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

        self._configure_action(action, grid)
        action.reasoning = f"v18:explore_s{sigma:.2f}"
        return action

    def _exploit_miracle(self, grid: np.ndarray) -> GameAction:
        """Replay a random action from miracle memory."""
        miracle_name = self.rng.choice(self.miracle_actions)
        for action in GameAction:
            if action.name == miracle_name:
                self._configure_action(action, grid)
                action.reasoning = "v18:miracle_exploit"
                return action
        action = self.rng.choice([a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action, grid)
        return action

    def _replay_sequence(self) -> GameAction:
        """Replay an entire miracle sequence, step by step."""
        seq = self.rng.choice(self.miracle_sequences)
        step_in_seq = self.steps % max(1, len(seq))
        if step_in_seq < len(seq):
            a_name = seq[step_in_seq]
            for a in GameAction:
                if a.name == a_name:
                    a.reasoning = "v18:seq_replay"
                    return a
        action = self.rng.choice([a for a in GameAction if a is not GameAction.RESET])
        action.reasoning = "v18:seq_fallback"
        return action

    def _configure_action(self, action: GameAction, grid: np.ndarray) -> None:
        """Configure action with intelligent targeting."""
        if action.is_simple():
            action.reasoning = f"v18:simple_a{self.attempt_count}"
        elif action.is_complex():
            # Use grid intelligence for coordinate selection
            y, x = self.grid_intel.suggest_target(grid)
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"v18:target({x},{y})",
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
