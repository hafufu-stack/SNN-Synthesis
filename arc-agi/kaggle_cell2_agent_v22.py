%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v22 Agent for ARC-AGI-3
# "Crystal Goose" - Best-of-N + Margin-Ranked CfC
#
# Key insight from Season 25-29:
# - v14 (CfC + Spatial, 0.10) is the ONLY agent that scored
# - v15-v21 added complexity but scored 0.00
# - SIMPLICITY wins under Kaggle's time constraints
#
# v22 strategy: Return to v14's proven CfC core, add:
# 1. MARGIN-AWARE targeting (P184): target low-confidence pixels
# 2. MULTI-ATTEMPT diversity (P182): each reset = different seed
# 3. ANTI-STALE (v20): forced reset on stuck loops
# 4. CRYSTAL EXIT: stop early when grid stops changing
#
# Paper: https://doi.org/10.5281/zenodo.19343952
# GitHub: https://github.com/hafufu-stack/SNN-Synthesis
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

# Sigma-Diverse NBS Schedule (Phase 37a, proven effective)
SIGMA_SCHEDULE = [
    0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00,
    0.02, 0.03, 0.08, 0.12, 0.25, 0.35, 0.45, 0.60, 0.85, 0.40,
]


# ==============================================================
# Spatial Feature Extractor (v14 proven)
# ==============================================================
class SpatialFeatures:
    """Extract spatial features from ARC grid WITHOUT flattening."""
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
            cx = xs_flat[mask]
            cy = ys_flat[mask]
            bboxes[c * 2] = (cx[-1] - cx[0]) / norm_w
            bboxes[c * 2 + 1] = (cy.max() - cy.min()) / norm_h

        sym = np.zeros(4, dtype=np.float32)
        sym[0] = np.sum(grid_int == grid_int[:, ::-1]) / total
        sym[1] = np.sum(grid_int == grid_int[::-1, :]) / total
        if h == w:
            sym[2] = np.sum(grid_int == grid_int.T) / total
        sym[3] = len(np.unique(flat)) / float(self.N_COLORS)

        edges = np.zeros(2, dtype=np.float32)
        if h > 1:
            edges[0] = np.sum(grid_int[1:, :] != grid_int[:-1, :]) / total
        if w > 1:
            edges[1] = np.sum(grid_int[:, 1:] != grid_int[:, :-1]) / total

        dims = np.array([h / 30.0, w / 30.0], dtype=np.float32)

        qh, qw = h // 2, w // 2
        quad_hist = np.zeros(40, dtype=np.float32)
        if qh > 0 and qw > 0:
            for i, (qi, qj) in enumerate([(0, 0), (0, qw), (qh, 0), (qh, qw)]):
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
    """Closed-form Continuous-time RNN cell (MIT Liquid Networks)."""
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


# ==============================================================
# Liquid Brain (v14 proven + v22 margin-targeting)
# ==============================================================
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
            seen_list = list(self.seen_hashes)
            self.seen_hashes = set(seen_list[-15000:])
        if is_novel:
            action_scores += 0.5
        return action_scores, is_novel

    def suggest_target(self, grid):
        """v22: Target cells near color boundaries (margin-aware)."""
        try:
            h, w = grid.shape
            if h <= 1 or w <= 1:
                return random.randint(0, max(0, h-1)), random.randint(0, max(0, w-1))

            grid_int = grid.astype(np.int32)
            # Find edge pixels (neighbors differ)
            edge_mask = np.zeros((h, w), dtype=bool)
            if h > 1:
                edge_mask[:-1, :] |= (grid_int[1:, :] != grid_int[:-1, :])
                edge_mask[1:, :] |= (grid_int[1:, :] != grid_int[:-1, :])
            if w > 1:
                edge_mask[:, :-1] |= (grid_int[:, 1:] != grid_int[:, :-1])
                edge_mask[:, 1:] |= (grid_int[:, 1:] != grid_int[:, :-1])

            edge_ys, edge_xs = np.where(edge_mask)
            if len(edge_ys) > 0 and random.random() < 0.7:
                idx = random.randint(0, len(edge_ys) - 1)
                return int(edge_ys[idx]), int(edge_xs[idx])

            # Fallback: non-background
            bg = np.bincount(grid_int.ravel(), minlength=10).argmax()
            nonbg = np.argwhere(grid_int != bg)
            if len(nonbg) > 0:
                idx = random.randint(0, len(nonbg) - 1)
                return int(nonbg[idx][0]), int(nonbg[idx][1])

            return random.randint(0, h-1), random.randint(0, w-1)
        except Exception:
            return 0, 0


# ==============================================================
# SNN-Synthesis v22 Agent: Crystal Goose
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v22: Crystal Goose.

    Back to v14's proven CfC core with surgical improvements:
    - Margin-aware edge targeting (P184)
    - Multi-seed diversity per attempt (P182)
    - Anti-stale reset (v20)
    - Crystal exit (stop when stable)
    """
    MAX_ACTIONS = 120

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed % (2**31))

        self.valid_actions = [a for a in GameAction if a is not GameAction.RESET]
        self.brain = LiquidBrain(
            n_actions=len(self.valid_actions),
            seed=seed % (2**31)
        )

        self.attempt_count = 0
        self.steps_in_episode = 0
        self.prev_levels = 0
        self.stale_counter = 0
        self.last_feat_hash = None
        self.miracle_actions = []
        self.miracle_sequences = []
        self.current_episode_actions = []

        logger.info(f"SNN-Synthesis v22 (Crystal Goose) init for {self.game_id}")

    def is_done(self, frames, latest_frame):
        if latest_frame.state is GameState.WIN:
            return True
        # v22: Give up after 8 attempts (v14 had no limit)
        if (latest_frame.state is GameState.GAME_OVER
                and self.attempt_count >= 8):
            return True
        return False

    def choose_action(self, frames, latest_frame):
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as e:
            logger.error(f"v22: crashed: {e}")
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
            # v22: Re-seed brain per attempt for diversity (P182 island concept)
            new_seed = (hash(self.game_id) + self.attempt_count * 7919) % (2**31)
            self.brain = LiquidBrain(
                n_actions=len(self.valid_actions),
                seed=new_seed
            )
            return GameAction.RESET

        self.steps_in_episode += 1

        # Track miracles
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            logger.info(
                f"MIRACLE! Level {current_levels} on attempt "
                f"{self.attempt_count} ({len(self.current_episode_actions)} actions)")
            self.miracle_sequences.append(list(self.current_episode_actions))
            self.miracle_actions.extend(self.current_episode_actions)
            self.current_episode_actions = []
            self.prev_levels = current_levels
            self.steps_in_episode = 0

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

        # Stale detection
        feat_hash = hash(grid.tobytes())
        if feat_hash == self.last_feat_hash:
            self.stale_counter += 1
            if self.stale_counter > 15:
                self.stale_counter = 0
                return GameAction.RESET
        else:
            self.stale_counter = 0
        self.last_feat_hash = feat_hash

        # Exploit miracle memory
        exploit_rate = min(0.35, 0.15 + 0.05 * len(self.miracle_sequences))
        if self.miracle_sequences and self.rng.random() < exploit_rate * 0.5:
            action = self._replay_sequence(grid)
        elif self.miracle_actions and self.rng.random() < exploit_rate:
            action = self._exploit_miracle(grid)
        else:
            action = self._explore(grid)

        self.current_episode_actions.append(action.name)
        return action

    def _explore(self, grid):
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        action_scores, _ = self.brain.score_actions(grid, sigma=sigma)
        noise = np.array([self.rng.gauss(0, max(0.01, sigma))
                         for _ in range(len(self.valid_actions))], dtype=np.float32)
        action_scores += noise

        best_idx = int(np.argmax(action_scores))
        action = self.valid_actions[best_idx]
        self._configure_action(action, grid, f"v22:s{sigma:.2f}")
        return action

    def _exploit_miracle(self, grid):
        miracle_name = self.rng.choice(self.miracle_actions)
        for a in self.valid_actions:
            if a.name == miracle_name:
                self._configure_action(a, grid, "v22:miracle")
                return a
        action = self.rng.choice(self.valid_actions)
        self._configure_action(action, grid, "v22:miracle_fb")
        return action

    def _replay_sequence(self, grid):
        seq = self.rng.choice(self.miracle_sequences)
        step_in_seq = self.steps_in_episode % max(1, len(seq))
        if step_in_seq < len(seq):
            a_name = seq[step_in_seq]
            for a in self.valid_actions:
                if a.name == a_name:
                    self._configure_action(a, grid, "v22:seq")
                    return a
        action = self.rng.choice(self.valid_actions)
        self._configure_action(action, grid, "v22:seq_fb")
        return action

    def _configure_action(self, action, grid, mode="v22"):
        if action.is_simple():
            action.reasoning = mode
        elif action.is_complex():
            y, x = self.brain.suggest_target(grid)
            x = max(0, min(63, x))
            y = max(0, min(63, y))
            action.set_data({"x": int(x), "y": int(y)})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": mode,
            }


# Alias for Kaggle runner
MyAgent = StochasticGoose
