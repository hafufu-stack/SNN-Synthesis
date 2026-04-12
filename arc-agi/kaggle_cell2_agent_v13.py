%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v13 Agent for ARC-AGI-3
# The Ultimate Spinal Reflex Explorer (O(1) Compute)
# SimHash Curiosity + sigma-Diverse NBS + Maximize Throughput
#
# Key insight from Phase 44-58 ("The Bitter Lesson"):
#   Any overhead > 0.5ms/action loses to pure Random.
#   SimHash curiosity = 0.005ms/action -> thousands more attempts.
#
# Paper: SNN-Synthesis v7 (Phase 44-58: The Bitter Lesson)
# GitHub: https://github.com/hafufu-stack/snn-synthesis
# Author: Hiroto Funasaki
# ==============================================================
import logging
import random
import time
from typing import Any, Optional

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# Sigma-Diverse NBS Schedule (Phase 37a/40)
# Rotate noise magnitude per attempt to find the optimal Activation Energy
SIGMA_SCHEDULE = [
    0.0,   # attempt 1: greedy (no noise)
    0.01,  # attempt 2: precision exploration
    0.05,  # attempt 3: minimal exploration
    0.10,  # attempt 4: gentle
    0.15,  # attempt 5: moderate (Hanoi-optimal)
    0.20,  # attempt 6: moderate-high
    0.30,  # attempt 7: strong exploration
    0.50,  # attempt 8: wild
    0.75,  # attempt 9: very aggressive
    1.00,  # attempt 10: maximum chaos
]

# ==============================================================
# SimHash Curiosity (Phase 51)
# O(1) Overhead, Noise-Robust Locality-Sensitive Hashing
# ==============================================================
class SimHashCuriosity:
    """Locality-sensitive hash for fast novelty detection.
    
    Phase 51 showed SimHash achieves same solve rate as complex
    curiosity methods while keeping overhead at 0.005ms/action.
    Unlike XOR-Hash, SimHash is robust to small state perturbations.
    """
    def __init__(self, state_dim=1024, hash_bits=32, seed=42):
        self.hash_bits = hash_bits
        self.state_dim = state_dim
        # Fixed random projection matrix (computed once, never changes)
        rng = np.random.RandomState(seed)
        self.projection = rng.randn(state_dim, hash_bits).astype(np.float32)
        self.projection /= (np.linalg.norm(self.projection, axis=0, keepdims=True) + 1e-8)
        self.seen = set()

    def get_hash(self, state_flat: np.ndarray) -> int:
        """O(1) noise-robust hash via random projection."""
        dim = self.state_dim
        if len(state_flat) > dim:
            x = state_flat[:dim]
        elif len(state_flat) < dim:
            x = np.zeros(dim, dtype=np.float32)
            x[:len(state_flat)] = state_flat
        else:
            x = state_flat

        projected = x @ self.projection
        # Convert to bit tuple and hash (Python built-in, extremely fast)
        bits = tuple((projected > 0).astype(np.int8))
        return hash(bits)

    def is_novel(self, state_hash: int, action_idx: int) -> bool:
        """Check if (state, action) pair has been seen before."""
        sa_hash = state_hash ^ (action_idx * 2654435761)
        return sa_hash not in self.seen

    def mark_seen(self, state_hash: int, action_idx: int):
        """Record (state, action) pair as visited."""
        sa_hash = state_hash ^ (action_idx * 2654435761)
        self.seen.add(sa_hash)
        # Keep memory bounded to avoid Python set bloat
        if len(self.seen) > 30000:
            # Keep the most recent half
            seen_list = list(self.seen)
            self.seen = set(seen_list[-15000:])


# ==============================================================
# SNN-Synthesis v13 Agent
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v13: SimHash Curiosity + sigma-Diverse NBS

    Adheres strictly to the "Crossover Law" (Phase 46):
    Overhead per action is kept under 0.005ms to maximize total
    exploration actions within the Kaggle time limit.

    Architecture:
    - NO LLM (saves minutes of load time + seconds per inference)
    - NO CNN/MLP training (saves gradient computation overhead)
    - NO RND weight updates (saves matrix ops per step)
    - YES SimHash curiosity (0.005ms, noise-robust, O(1))
    - YES sigma-diverse noise schedule (automatic sigma tuning)
    - YES miracle memory (exploit successful action sequences)
    """
    MAX_ACTIONS = 120  # more actions per episode than v12's 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Fast RNG (isolated from global state)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        self.rng = random.Random(seed)

        # O(1) Curiosity Module
        self.curiosity = SimHashCuriosity(
            state_dim=1024, hash_bits=32, seed=seed % (2**31)
        )

        # State tracking
        self.attempt_count = 0
        self.steps_in_episode = 0
        self.prev_levels = 0
        self.stale_counter = 0
        self.last_state_hash = None

        # Exploit tracking (miracles)
        self.miracle_actions = []         # flat list of action names from successful levels
        self.current_episode_actions = []  # actions in current episode

        # Cache valid actions (exclude RESET)
        self.valid_actions = [a for a in GameAction if a is not GameAction.RESET]

        logger.info(f"SNN-Synthesis v13 (O(1) Explorer) initialized for {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:

        # --- Fast RESET ---
        if (latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]
                or self.steps_in_episode >= self.MAX_ACTIONS):
            self.attempt_count += 1
            self.steps_in_episode = 0
            self.current_episode_actions = []
            self.stale_counter = 0
            self.last_state_hash = None
            return GameAction.RESET

        self.steps_in_episode += 1

        # --- Track Level Completion (Miracle) ---
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            logger.info(
                f"MIRACLE! Level {current_levels} cleared on attempt "
                f"{self.attempt_count} ({len(self.current_episode_actions)} actions)"
            )
            # Save successful action sequence for exploitation
            self.miracle_actions.extend(self.current_episode_actions)
            self.current_episode_actions = []
            self.prev_levels = current_levels
            self.steps_in_episode = 0

        # --- Extract State (Minimal) ---
        state_flat = np.array([], dtype=np.float32)
        if latest_frame.frame:
            try:
                state_flat = np.array(
                    latest_frame.frame[0], dtype=np.float32
                ).flatten()
            except Exception:
                pass

        state_hash = self.curiosity.get_hash(state_flat)

        # --- Stuck Detection ---
        # If hash hasn't changed for 15 steps, we're stuck -> RESET
        if state_hash == self.last_state_hash:
            self.stale_counter += 1
            if self.stale_counter > 15:
                self.stale_counter = 0
                return GameAction.RESET
        else:
            self.stale_counter = 0
        self.last_state_hash = state_hash

        # --- 1. Exploit (20% chance if we have miracle memory) ---
        if self.miracle_actions and self.rng.random() < 0.2:
            action = self._exploit_miracle()
            self._configure_action(action, state_flat, "exploit")
            self.current_episode_actions.append(action.name)
            return action

        # --- 2. Explore: SimHash Curiosity + sigma-Diverse Noise ---
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        # Score each action by novelty + noise
        n_actions = len(self.valid_actions)
        best_score = float('-inf')
        best_idx = 0

        for a_idx in range(n_actions):
            # Novelty: 1.0 if unseen, 0.0 if seen
            nov = 1.0 if self.curiosity.is_novel(state_hash, a_idx) else 0.0
            # sigma-diverse Gaussian noise
            noise = self.rng.gauss(0, max(0.01, sigma))
            score = nov + noise

            if score > best_score:
                best_score = score
                best_idx = a_idx

        action = self.valid_actions[best_idx]

        # Update curiosity (mark as visited)
        self.curiosity.mark_seen(state_hash, best_idx)

        # Configure and return
        self._configure_action(action, state_flat, f"explore_s{sigma:.2f}")
        self.current_episode_actions.append(action.name)

        return action

    def _exploit_miracle(self) -> GameAction:
        """Replay a random action from miracle memory."""
        miracle_name = self.rng.choice(self.miracle_actions)
        for a in self.valid_actions:
            if a.name == miracle_name:
                return a
        # Fallback: random valid action
        return self.rng.choice(self.valid_actions)

    def _configure_action(
        self, action: GameAction, state_flat: np.ndarray, mode: str
    ) -> None:
        """Fast coordinate targeting for complex actions."""
        if action.is_simple():
            action.reasoning = f"v13:{mode}"
        elif action.is_complex():
            x, y = self.rng.randint(0, 63), self.rng.randint(0, 63)

            # 50% chance to target a non-zero cell (smarter than pure random)
            if len(state_flat) > 0 and self.rng.random() < 0.5:
                try:
                    nz_indices = np.nonzero(state_flat)[0]
                    if len(nz_indices) > 0:
                        chosen_idx = int(self.rng.choice(nz_indices))
                        side_len = int(np.sqrt(len(state_flat)))
                        if side_len > 0:
                            y = chosen_idx // side_len
                            x = chosen_idx % side_len
                except Exception:
                    pass

            x = max(0, min(63, x))
            y = max(0, min(63, y))
            action.set_data({"x": x, "y": y})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"v13:{mode}",
            }


# Alias for Kaggle runner
MyAgent = StochasticGoose
