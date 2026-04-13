%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v15 Agent for ARC-AGI-3
# Thermodynamic Explorer: Macro-State + Bounded Retries + O(1) ExIt
#
# Empirical Kaggle score analysis:
#   v5  = 0.13 (UCB, macro-stats hash, GAME_OVER=give up, 4 actions)
#   v14 = 0.10 (CfC + spatial features, 120 steps)
#   v12 = 0.07 (LLM + CNN + RND, 80 steps)
#   v7  = 0.07 (CNN + SFT, 80 steps)
#   v13 = 0.02 (SimHash flatten, 120 steps)
#
# Root cause analysis (why v5 beat all "smarter" agents):
#   1. Timeout Trap: v5 gives up on GAME_OVER -> sees all 100 puzzles.
#      v13/14 retry forever via RESET -> starve on first hard puzzle.
#   2. Noise Trap: v5's macro-stats (mean, std, unique) perfectly ignore
#      pixel-level noise. SimHash/SpatialFeatures are too sensitive.
#   3. Action Space: v5 used only 4 actions, reducing search space.
#
# v15 = v5's survival instincts + O(1) Dictionary ExIt + sigma-diverse NBS
#
# Paper: SNN-Synthesis v7+ (Thermodynamic Explorer)
# GitHub: https://github.com/hafufu-stack/snn-synthesis
# Author: Hiroto Funasaki
# ==============================================================
import hashlib
import logging
import random
import time
import numpy as np
from typing import Any, Optional

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# Sigma-Diverse NBS Schedule (Natural Selection of Noise)
SIGMA_SCHEDULE = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]


class StochasticGoose(Agent):
    """
    SNN-Synthesis v15: Thermodynamic Explorer

    Three survival rules derived from v5 (0.13) vs v13 (0.02) analysis:
    1. GIVE UP after MAX_ATTEMPTS -> don't starve on one hard puzzle
    2. Use MACRO statistical hash -> ignore pixel-level noise
    3. O(1) Dictionary ExIt -> instant learning from miracles

    Core formula: score = UCB_novelty(3.0/sqrt(visits+1)) + sigma_noise
    """
    MAX_ACTIONS_PER_ATTEMPT = 150
    MAX_ATTEMPTS = 5  # The golden survival rule: give up and save time!

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000) + hash(self.game_id) % 1000
        self.rng = random.Random(seed)

        # UCB visit counting (v5's 0.13 champion formula)
        self.sa_visits = {}

        # O(1) ExIt: Dictionary mapping state_hash -> action_name
        # No backprop, no gradient, just instant memorization
        self.miracle_dict = {}

        # Episode tracking
        self.current_ep = []   # list of (hash, action_name) for current episode

        # State tracking
        self.attempt_count = 0
        self.steps = 0
        self.prev_levels = 0
        self.stale_counter = 0
        self.last_hash = None
        self.gave_up = False

        logger.info(f"SNN-Synthesis v15 (Thermodynamic Explorer) initialized for {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """The #1 lesson from v5: GIVE UP on hard puzzles to save time.

        v5 (0.13): returned True on GAME_OVER -> saw all 100 puzzles
        v13 (0.02): returned True only on WIN -> starved on puzzle #1

        v15: give up after MAX_ATTEMPTS retries -> move to next puzzle
        """
        if latest_frame.state == GameState.WIN or self.gave_up:
            return True
        # After MAX_ATTEMPTS retries, accept defeat and move on
        if (latest_frame.state == GameState.GAME_OVER
                and self.attempt_count >= self.MAX_ATTEMPTS - 1):
            self.gave_up = True
            return True
        return False

    def _get_thermodynamic_hash(self, frame: FrameData) -> str:
        """v5's accidental genius: macroscopic stats ignore microscopic noise.

        ARC puzzles have noise characters that move around randomly.
        Pixel-level hashes (SimHash, SpatialFeatures) see them as new states.
        Macro stats (mean, std, nonzero, unique) are invariant to noise motion.

        This is the thermodynamic coarse-graining principle:
        Ignore microstates, track only macrostates.
        """
        features = [float(frame.levels_completed or 0)]
        if frame.frame and len(frame.frame) > 0:
            try:
                arr = np.array(frame.frame[0], dtype=np.float32)
                if arr.size > 0:
                    features.extend([
                        float(arr.shape[0]), float(arr.shape[1]),
                        float(np.mean(arr)), float(np.std(arr)),
                        float(np.count_nonzero(arr)), float(len(np.unique(arr)))
                    ])
                else:
                    features.extend([0.0] * 6)
            except Exception:
                features.extend([0.0] * 6)
        else:
            features.extend([0.0] * 6)

        # Round to 1 decimal: absorbs tiny noise fluctuations (coarse-graining)
        rounded = tuple(round(f, 1) for f in features)
        return hashlib.md5(str(rounded).encode()).hexdigest()[:10]

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:

        # --- Give up if exceeded max attempts ---
        if self.attempt_count >= self.MAX_ATTEMPTS:
            self.gave_up = True
            return GameAction.RESET

        # --- Fast RESET ---
        if (latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]
                or self.steps >= self.MAX_ACTIONS_PER_ATTEMPT):
            self.attempt_count += 1
            self.steps = 0
            self.current_ep = []
            self.stale_counter = 0
            self.last_hash = None
            return GameAction.RESET

        self.steps += 1

        # --- O(1) ExIt: Miracle Detection & Learning ---
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            logger.info(
                f"MIRACLE! Level {current_levels} on attempt {self.attempt_count} "
                f"({len(self.current_ep)} actions)")
            # Instant learning: store entire trajectory in dictionary
            # O(1) per entry, no backprop, no gradient computation
            for h, a_name in self.current_ep:
                self.miracle_dict[h] = a_name
            self.current_ep = []
            self.prev_levels = current_levels
            self.steps = 0
            self.attempt_count = 0  # Reward success with fresh retry budget

        # --- Dynamic Action Masking ---
        if hasattr(latest_frame, 'available_actions') and latest_frame.available_actions:
            valid_actions = latest_frame.available_actions
        else:
            valid_actions = [a for a in GameAction if a is not GameAction.RESET]
        if not valid_actions:
            return GameAction.RESET

        # --- Thermodynamic State Hash ---
        state_hash = self._get_thermodynamic_hash(latest_frame)

        # Stuck detection (on macro hash)
        if state_hash == self.last_hash:
            self.stale_counter += 1
            if self.stale_counter > 15:
                self.stale_counter = 0
                return GameAction.RESET
        else:
            self.stale_counter = 0
        self.last_hash = state_hash

        # --- 1. Exploit: O(1) Dictionary Memory (System 2) ---
        # If we've seen this exact macro-state before and solved it, replay
        if state_hash in self.miracle_dict and self.rng.random() < 0.85:
            learned_name = self.miracle_dict[state_hash]
            action = next(
                (a for a in valid_actions if a.name == learned_name),
                self.rng.choice(valid_actions))
            self._configure_action(action, latest_frame, "exploit")
            self.current_ep.append((state_hash, action.name))
            return action

        # --- 2. Explore: UCB Curiosity + Sigma-Diverse NBS (System 1) ---
        sigma = SIGMA_SCHEDULE[self.attempt_count % len(SIGMA_SCHEDULE)]

        best_score = float('-inf')
        best_action = valid_actions[0]

        for a in valid_actions:
            sa_key = f"{state_hash}_{a.name}"
            visits = self.sa_visits.get(sa_key, 0)

            # UCB novelty (v5's champion formula: 3.0 / sqrt(visits+1))
            ucb = 3.0 / ((visits + 1) ** 0.5)
            # sigma-diverse Gaussian noise (NBS natural selection)
            noise = self.rng.gauss(0, max(0.01, sigma))

            score = ucb + noise

            if score > best_score:
                best_score = score
                best_action = a

        # Update visit count
        sa_key = f"{state_hash}_{best_action.name}"
        self.sa_visits[sa_key] = self.sa_visits.get(sa_key, 0) + 1

        # Bound memory to prevent dict bloat
        if len(self.sa_visits) > 30000:
            self.sa_visits.clear()

        self._configure_action(best_action, latest_frame, f"explore_s{sigma:.2f}")
        self.current_ep.append((state_hash, best_action.name))

        return best_action

    def _configure_action(
        self, action: GameAction, frame: FrameData, mode: str
    ) -> None:
        """Smart coordinate targeting: target the RAREST color.

        Rare colors on the grid are likely goals, items, or interactive objects.
        This is a spatial prior that doesn't require expensive CNN processing.
        """
        if action.is_simple():
            action.reasoning = f"v15:{mode}"
        elif action.is_complex():
            x, y = self.rng.randint(0, 63), self.rng.randint(0, 63)

            # 70% chance: target rarest non-background color
            if frame.frame and self.rng.random() < 0.7:
                try:
                    arr = np.array(frame.frame[0])
                    # Find non-background pixels
                    nz_mask = arr > 0
                    if np.any(nz_mask):
                        colors, counts = np.unique(arr[nz_mask], return_counts=True)
                        # Target the rarest color (most likely a goal/item)
                        rarest = colors[np.argmin(counts)]
                        ys, xs = np.where(arr == rarest)
                        idx = self.rng.randint(0, len(ys) - 1)
                        y, x = int(ys[idx]), int(xs[idx])
                except Exception:
                    pass

            x = max(0, min(63, x))
            y = max(0, min(63, y))
            action.set_data({"x": x, "y": y})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"v15:{mode}",
            }


# Alias for Kaggle runner
MyAgent = StochasticGoose
