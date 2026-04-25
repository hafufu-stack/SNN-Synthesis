%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v25 Agent for ARC-AGI-3
# "Relentless Goose" - Maximum Miracle Exploitation
#
# v23 (0.17) analysis: score came entirely from Self-Consistency
# miracle replay. Every % of time NOT replaying a known solution
# is wasted. v25 maximizes replay efficiency.
#
# Changes from v23:
# 1. IMMEDIATE LOCK: Once a level is solved, 100% replay forever
#    (v23 wasted 30-70% of attempts re-exploring solved levels)
# 2. SHORTEST FIRST: Try shortest miracle sequence first
#    (faster replay = more time for unsolved levels)
# 3. LONGER EPISODES: MAX_ACTIONS 120->200 (complex tasks need it)
# 4. DIVERSE TARGETING: Cycle through 5 coordinate strategies
#    per attempt (center/edge/corner/non-bg/random)
# 5. ANTI-REPEAT: Penalize recently-used actions to avoid loops
# 6. FASTER STALE RESET: 15->10 stale steps before reset
#
# Scoring: v5=0.13 > v23=0.17 > v22=0.11 > v14=0.10
#
# Paper: https://doi.org/10.5281/zenodo.19343952
# GitHub: https://github.com/hafufu-stack/SNN-Synthesis
# Author: Hiroto Funasaki
# ==============================================================
import logging
import random
import time
from typing import Any, Optional
from collections import deque

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# Extended sigma schedule (30 values, P211 diversity)
SIGMA_SCHEDULE = [
    0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00,
    0.02, 0.03, 0.08, 0.12, 0.25, 0.35, 0.45, 0.60, 0.85, 0.40,
    0.07, 0.18, 0.55, 0.90, 0.04, 0.22, 0.65, 0.13, 0.38, 0.70,
]


# ==============================================================
# Spatial Feature Extractor (v14 proven)
# ==============================================================
class SpatialFeatures:
    N_COLORS = 10

    def extract(self, grid):
        if grid.ndim != 2 or grid.size == 0:
            return np.zeros(96, dtype=np.float32)
        h, w = grid.shape
        total = float(h * w)
        grid_int = grid.astype(np.int32)
        flat = grid_int.ravel()
        ys_flat = np.repeat(np.arange(h, dtype=np.float32), w)
        xs_flat = np.tile(np.arange(w, dtype=np.float32), h)
        counts = np.bincount(flat, minlength=self.N_COLORS)[:self.N_COLORS]
        hist = counts.astype(np.float32) / total
        centroids = np.zeros(self.N_COLORS * 2, dtype=np.float32)
        bboxes = np.zeros(self.N_COLORS * 2, dtype=np.float32)
        norm_w = max(1.0, float(w - 1))
        norm_h = max(1.0, float(h - 1))
        sum_x = np.bincount(flat, weights=xs_flat, minlength=self.N_COLORS)[:self.N_COLORS]
        sum_y = np.bincount(flat, weights=ys_flat, minlength=self.N_COLORS)[:self.N_COLORS]
        present = counts > 0
        centroids[0::2] = np.where(present, sum_x / np.maximum(counts, 1) / norm_w, 0)
        centroids[1::2] = np.where(present, sum_y / np.maximum(counts, 1) / norm_h, 0)
        for c in np.where((counts > 1))[0]:
            mask = flat == c
            cx = xs_flat[mask]; cy = ys_flat[mask]
            bboxes[c*2] = (cx[-1]-cx[0])/norm_w
            bboxes[c*2+1] = (cy.max()-cy.min())/norm_h
        sym = np.zeros(4, dtype=np.float32)
        sym[0] = np.sum(grid_int == grid_int[:, ::-1]) / total
        sym[1] = np.sum(grid_int == grid_int[::-1, :]) / total
        if h == w: sym[2] = np.sum(grid_int == grid_int.T) / total
        sym[3] = len(np.unique(flat)) / float(self.N_COLORS)
        edges = np.zeros(2, dtype=np.float32)
        if h > 1: edges[0] = np.sum(grid_int[1:, :] != grid_int[:-1, :]) / total
        if w > 1: edges[1] = np.sum(grid_int[:, 1:] != grid_int[:, :-1]) / total
        dims = np.array([h / 30.0, w / 30.0], dtype=np.float32)
        qh, qw = h // 2, w // 2
        quad_hist = np.zeros(40, dtype=np.float32)
        if qh > 0 and qw > 0:
            for i, (qi, qj) in enumerate([(0,0),(0,qw),(qh,0),(qh,qw)]):
                qflat = grid_int[qi:qi+qh, qj:qj+qw].ravel()
                qc = np.bincount(qflat, minlength=self.N_COLORS)[:self.N_COLORS]
                quad_hist[i*10:(i+1)*10] = qc.astype(np.float32) / max(1.0, float(len(qflat)))
        result = np.concatenate([hist, centroids, bboxes, sym, edges, dims, quad_hist])
        fixed = np.zeros(96, dtype=np.float32)
        fixed[:min(len(result), 96)] = result[:96]
        return fixed


# ==============================================================
# CfC Cell (v14 proven)
# ==============================================================
class CfCCell:
    def __init__(self, input_dim, hidden_dim=32, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(input_dim + hidden_dim)
        self.W_in = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.b_in = np.zeros(hidden_dim, dtype=np.float32)
        self.W_hh = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.W_hh *= 0.9 / max(1.0, np.max(np.abs(np.linalg.eigvals(self.W_hh))))
        self.W_tau = rng.randn(input_dim + hidden_dim, hidden_dim).astype(np.float32) * scale
        self.b_tau = np.ones(hidden_dim, dtype=np.float32) * 0.5
        self.W_out = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.h = np.zeros(hidden_dim, dtype=np.float32)
        self.dt = 1.0

    def reset(self):
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)

    def forward(self, x, sigma=0.0):
        assert len(x) == self.input_dim
        xh = np.concatenate([x, self.h])
        tau_logit = xh @ self.W_tau + self.b_tau
        tau = 1.0 / (1.0 + np.exp(-np.clip(tau_logit, -10, 10)))
        tau = np.clip(tau, 0.01, 1.0)
        pre_act = x @ self.W_in + self.h @ self.W_hh + self.b_in
        h_candidate = np.tanh(pre_act)
        alpha = np.clip(self.dt / tau, 0.0, 1.0)
        self.h = (1.0 - alpha) * self.h + alpha * h_candidate
        if sigma > 0:
            noise = np.random.randn(self.hidden_dim).astype(np.float32) * sigma
            self.h = np.clip(self.h + noise, -3.0, 3.0)
        return self.h @ self.W_out


class LiquidBrain:
    def __init__(self, n_actions, seed=42):
        self.spatial = SpatialFeatures()
        self.feature_dim = 96
        self.n_actions = n_actions
        self.cfc = CfCCell(input_dim=self.feature_dim, hidden_dim=32, seed=seed)
        rng = np.random.RandomState(seed + 1)
        self.W_action = rng.randn(32, n_actions).astype(np.float32) * 0.1
        self.seen_hashes = set()

    def reset(self):
        self.cfc.reset()

    def score_actions(self, grid, sigma=0.0):
        features = self.spatial.extract(grid)
        hidden = self.cfc.forward(features, sigma=sigma * 0.5)
        action_scores = hidden @ self.W_action
        feat_hash = hash(tuple(np.round(features * 10).astype(np.int8)))
        is_novel = feat_hash not in self.seen_hashes
        self.seen_hashes.add(feat_hash)
        if len(self.seen_hashes) > 30000:
            self.seen_hashes = set(list(self.seen_hashes)[-15000:])
        if is_novel:
            action_scores += 0.5
        return action_scores, is_novel


# ==============================================================
# SNN-Synthesis v25 Agent: Relentless Goose
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v25: Relentless Goose.

    Core insight: 100% of Kaggle score comes from miracle replay.
    Every second spent NOT replaying a known solution is wasted.

    Strategy:
    - Solved levels: 100% deterministic replay (shortest sequence)
    - Unsolved levels: diverse exploration with anti-repeat
    """
    MAX_ACTIONS = 200  # v25: extended from 120

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed % (2**31))
        self.base_seed = seed % (2**31)

        self.valid_actions = [a for a in GameAction if a is not GameAction.RESET]
        self.brain = LiquidBrain(
            n_actions=len(self.valid_actions),
            seed=self.base_seed
        )

        self.attempt_count = 0
        self.steps_in_episode = 0
        self.prev_levels = 0
        self.stale_counter = 0
        self.last_feat_hash = None

        # v25: Miracle storage (per level, sorted by length)
        self.level_sequences = {}
        self.current_episode_actions = []

        # v25: Replay state
        self.replay_mode = False
        self.replay_seq = None
        self.replay_idx = 0

        # v25: Anti-repeat (recent action penalty)
        self.recent_actions = deque(maxlen=8)

        # v25: Track which targeting strategy to use per attempt
        self.target_strategies = ['edge', 'nonbg', 'center', 'random', 'corner']

        logger.info(f"SNN-Synthesis v25 (Relentless Goose) init for {self.game_id}")

    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame):
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as e:
            logger.error(f"v25: crashed: {e}")
            return GameAction.RESET

    def _choose_action_inner(self, frames, latest_frame):
        # Reset on game boundary
        if (latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]
                or self.steps_in_episode >= self.MAX_ACTIONS):
            self.attempt_count += 1
            self.steps_in_episode = 0
            self.current_episode_actions = []
            self.stale_counter = 0
            self.last_feat_hash = None
            self.replay_mode = False
            self.replay_seq = None
            self.replay_idx = 0
            self.recent_actions.clear()

            # v25: Re-seed brain per attempt (P211 diversity)
            new_seed = (self.base_seed + self.attempt_count * 7919) % (2**31)
            self.brain = LiquidBrain(
                n_actions=len(self.valid_actions),
                seed=new_seed
            )

            # v25: IMMEDIATE LOCK - if current level was already solved,
            # ALWAYS replay (100% probability, shortest sequence)
            current_level = self.prev_levels
            if current_level in self.level_sequences and self.level_sequences[current_level]:
                self._start_replay(current_level)

            return GameAction.RESET

        self.steps_in_episode += 1

        # Track miracles
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            completed_level = self.prev_levels
            logger.info(
                f"MIRACLE! Level {current_levels} on attempt "
                f"{self.attempt_count} ({len(self.current_episode_actions)} actions)")

            seq = list(self.current_episode_actions)
            if completed_level not in self.level_sequences:
                self.level_sequences[completed_level] = []
            self.level_sequences[completed_level].append(seq)
            # v25: Keep only 3 shortest (most efficient)
            self.level_sequences[completed_level].sort(key=len)
            self.level_sequences[completed_level] = self.level_sequences[completed_level][:3]

            self.current_episode_actions = []
            self.prev_levels = current_levels
            self.steps_in_episode = 0
            self.replay_mode = False

        # Extract grid
        grid = np.zeros((1, 1), dtype=np.float32)
        if latest_frame.frame:
            try:
                grid = np.array(latest_frame.frame[0], dtype=np.float32)
                if grid.ndim == 1:
                    side = int(np.sqrt(len(grid)))
                    if side * side == len(grid):
                        grid = grid.reshape(side, side)
                    else:
                        grid = grid.reshape(1, -1)
            except Exception:
                pass

        # v25: Faster stale detection (10 instead of 15)
        feat_hash = hash(grid.tobytes())
        if feat_hash == self.last_feat_hash:
            self.stale_counter += 1
            if self.stale_counter > 10:
                self.stale_counter = 0
                self.replay_mode = False
                return GameAction.RESET
        else:
            self.stale_counter = 0
        self.last_feat_hash = feat_hash

        # v25: Replay mode (deterministic, in-order)
        if self.replay_mode and self.replay_seq:
            action = self._replay_step(grid)
            if action is not None:
                self.current_episode_actions.append(action.name)
                return action
            self.replay_mode = False

        # Explore (for unsolved levels)
        action = self._explore(grid)
        self.current_episode_actions.append(action.name)
        self.recent_actions.append(action.name)
        return action

    def _start_replay(self, level):
        """Start replaying the shortest known sequence for a level."""
        sequences = self.level_sequences.get(level, [])
        if not sequences:
            return
        # v25: Cycle through sequences (shortest first, then alternatives)
        seq_idx = (self.attempt_count - 1) % len(sequences)
        self.replay_seq = list(sequences[seq_idx])
        self.replay_idx = 0
        self.replay_mode = True
        logger.info(f"v25: Replay L{level} seq#{seq_idx} ({len(self.replay_seq)} actions)")

    def _replay_step(self, grid):
        if self.replay_idx >= len(self.replay_seq):
            return None
        a_name = self.replay_seq[self.replay_idx]
        self.replay_idx += 1
        for a in self.valid_actions:
            if a.name == a_name:
                self._configure_action(a, grid, "v25:replay")
                return a
        return None

    def _explore(self, grid):
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        action_scores, _ = self.brain.score_actions(grid, sigma=sigma)
        noise = np.array([self.rng.gauss(0, max(0.01, sigma))
                         for _ in range(len(self.valid_actions))], dtype=np.float32)
        action_scores += noise

        # v25: Anti-repeat penalty
        for i, a in enumerate(self.valid_actions):
            repeat_count = sum(1 for r in self.recent_actions if r == a.name)
            action_scores[i] -= repeat_count * 0.3

        best_idx = int(np.argmax(action_scores))
        action = self.valid_actions[best_idx]
        self._configure_action(action, grid, f"v25:s{sigma:.2f}")
        return action

    def _configure_action(self, action, grid, mode="v25"):
        if action.is_simple():
            action.reasoning = mode
        elif action.is_complex():
            y, x = self._smart_target(grid)
            x = max(0, min(63, x))
            y = max(0, min(63, y))
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": mode,
            }

    def _smart_target(self, grid):
        """v25: Cycle through targeting strategies per attempt."""
        try:
            h, w = grid.shape
            if h <= 1 or w <= 1:
                return random.randint(0, max(0, h-1)), random.randint(0, max(0, w-1))

            strategy_idx = (self.attempt_count - 1) % len(self.target_strategies)
            strategy = self.target_strategies[strategy_idx]
            grid_int = grid.astype(np.int32)

            if strategy == 'edge':
                edge_mask = np.zeros((h, w), dtype=bool)
                if h > 1:
                    edge_mask[:-1, :] |= (grid_int[1:, :] != grid_int[:-1, :])
                    edge_mask[1:, :] |= (grid_int[1:, :] != grid_int[:-1, :])
                if w > 1:
                    edge_mask[:, :-1] |= (grid_int[:, 1:] != grid_int[:, :-1])
                    edge_mask[:, 1:] |= (grid_int[:, 1:] != grid_int[:, :-1])
                ys, xs = np.where(edge_mask)
                if len(ys) > 0:
                    idx = random.randint(0, len(ys) - 1)
                    return int(ys[idx]), int(xs[idx])

            elif strategy == 'nonbg':
                bg = np.bincount(grid_int.ravel(), minlength=10).argmax()
                nonbg = np.argwhere(grid_int != bg)
                if len(nonbg) > 0:
                    idx = random.randint(0, len(nonbg) - 1)
                    return int(nonbg[idx][0]), int(nonbg[idx][1])

            elif strategy == 'center':
                cy, cx = h // 2, w // 2
                dy = random.randint(-min(2, cy), min(2, h-1-cy))
                dx = random.randint(-min(2, cx), min(2, w-1-cx))
                return cy + dy, cx + dx

            elif strategy == 'corner':
                corners = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]
                cy, cx = random.choice(corners)
                dy = random.randint(0, min(2, h-1))
                dx = random.randint(0, min(2, w-1))
                return min(cy + dy, h-1), min(cx + dx, w-1)

            # Fallback: random
            return random.randint(0, h-1), random.randint(0, w-1)
        except Exception:
            return 0, 0


MyAgent = StochasticGoose
