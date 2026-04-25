%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v26 Agent for ARC-AGI-3
# "The Prophet" - Equation-Driven Multi-Brain Architecture
#
# v25 (Relentless Goose) + Season 43-45 insights:
#
# From P231 (Grand Unified Equation): C matters most (+0.052)
#   -> CfC hidden_dim 32 -> 64 (doubled synapses per cell)
#
# From P211 (Diversity): Multiple seeds increase oracle PA
#   -> 3 independent CfC brains, rotate per step (ensemble)
#
# From P234 (Neuro-Symbolic): identity/flip/swap are top rules
#   -> Symmetry-aware features added to spatial extractor
#
# From P232 (Dynamic Allocation): progressive sigma within episode
#   -> Start conservative (sigma=0), ramp up as episode progresses
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
# Enhanced Spatial Features (v26: +symmetry from P234)
# ==============================================================
class SpatialFeatures:
    N_COLORS = 10

    def extract(self, grid):
        if grid.ndim != 2 or grid.size == 0:
            return np.zeros(112, dtype=np.float32)
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

        # v26: Enhanced symmetry features (P234 insight)
        sym_ext = np.zeros(8, dtype=np.float32)
        # Rotational symmetry (90, 180, 270)
        if h == w:
            r90 = np.rot90(grid_int)
            r180 = np.rot90(grid_int, 2)
            r270 = np.rot90(grid_int, 3)
            sym_ext[0] = np.sum(grid_int == r90) / total
            sym_ext[1] = np.sum(grid_int == r180) / total
            sym_ext[2] = np.sum(grid_int == r270) / total
        # Border uniformity (how uniform are the edges?)
        if h >= 2 and w >= 2:
            top = grid_int[0, :]
            bot = grid_int[-1, :]
            left = grid_int[:, 0]
            right = grid_int[:, -1]
            sym_ext[3] = np.sum(top == bot) / float(w)
            sym_ext[4] = np.sum(left == right) / float(h)
            sym_ext[5] = float(len(np.unique(top))) / self.N_COLORS
        # Color pair dominance (how much of grid is top-2 colors?)
        sorted_counts = np.sort(counts)[::-1]
        sym_ext[6] = (sorted_counts[0] + sorted_counts[1]) / total if len(sorted_counts) > 1 else 1.0
        # Sparsity (fraction of non-zero cells)
        sym_ext[7] = np.sum(grid_int != 0) / total

        # v26: Step/attempt progress features
        progress = np.zeros(8, dtype=np.float32)
        # Will be filled in by the agent before calling extract

        result = np.concatenate([hist, centroids, bboxes, sym, edges, dims,
                                quad_hist, sym_ext, progress])
        fixed = np.zeros(112, dtype=np.float32)
        fixed[:min(len(result), 112)] = result[:112]
        return fixed


# ==============================================================
# CfC Cell (v26: hidden_dim=64, from P231 equation)
# ==============================================================
class CfCCell:
    def __init__(self, input_dim, hidden_dim=64, seed=42):
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
    """v26: Multi-brain ensemble (3 CfC cells with different seeds)."""
    def __init__(self, n_actions, seed=42, n_brains=3):
        self.spatial = SpatialFeatures()
        self.feature_dim = 112  # v26: expanded from 96
        self.n_actions = n_actions
        self.n_brains = n_brains

        # v26: Multiple CfC cells for diversity (P211 insight)
        self.cfcs = [
            CfCCell(input_dim=self.feature_dim, hidden_dim=64, seed=seed + i * 9973)
            for i in range(n_brains)
        ]
        rng = np.random.RandomState(seed + 1)
        # Each brain gets its own action weights
        self.W_actions = [
            rng.randn(64, n_actions).astype(np.float32) * 0.1
            for _ in range(n_brains)
        ]
        self.seen_hashes = set()
        self.step_count = 0

    def reset(self):
        for cfc in self.cfcs:
            cfc.reset()
        self.step_count = 0

    def score_actions(self, grid, sigma=0.0):
        features = self.spatial.extract(grid)
        self.step_count += 1

        # v26: Rotate through brains (each step uses a different brain)
        brain_idx = self.step_count % self.n_brains
        hidden = self.cfcs[brain_idx].forward(features, sigma=sigma * 0.5)
        action_scores = hidden @ self.W_actions[brain_idx]

        # Also feed features to other brains (keep their states warm)
        for i in range(self.n_brains):
            if i != brain_idx:
                self.cfcs[i].forward(features, sigma=0.0)

        feat_hash = hash(tuple(np.round(features * 10).astype(np.int8)))
        is_novel = feat_hash not in self.seen_hashes
        self.seen_hashes.add(feat_hash)
        if len(self.seen_hashes) > 30000:
            self.seen_hashes = set(list(self.seen_hashes)[-15000:])
        if is_novel:
            action_scores += 0.5
        return action_scores, is_novel


# ==============================================================
# SNN-Synthesis v26 Agent: The Prophet
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v26: The Prophet.

    Equation-driven design:
    - 3x CfC brains (C=64 hidden) rotating per step
    - 112-dim features with symmetry detection
    - Progressive sigma within episode
    - All v25 miracle replay mechanics
    """
    MAX_ACTIONS = 200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed % (2**31))
        self.base_seed = seed % (2**31)

        self.valid_actions = [a for a in GameAction if a is not GameAction.RESET]
        self.brain = LiquidBrain(
            n_actions=len(self.valid_actions),
            seed=self.base_seed,
            n_brains=3
        )

        self.attempt_count = 0
        self.steps_in_episode = 0
        self.prev_levels = 0
        self.stale_counter = 0
        self.last_feat_hash = None

        self.level_sequences = {}
        self.current_episode_actions = []

        self.replay_mode = False
        self.replay_seq = None
        self.replay_idx = 0

        self.recent_actions = deque(maxlen=10)  # v26: longer memory (8->10)
        self.target_strategies = ['edge', 'nonbg', 'center', 'random', 'corner',
                                  'symmetry']  # v26: +symmetry strategy

        logger.info(f"SNN-Synthesis v26 (The Prophet) init for {self.game_id}")

    def is_done(self, frames, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame):
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as e:
            logger.error(f"v26: crashed: {e}")
            return GameAction.RESET

    def _choose_action_inner(self, frames, latest_frame):
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

            # v26: Re-seed with more diverse prime multiplication
            new_seed = (self.base_seed + self.attempt_count * 104729) % (2**31)
            self.brain = LiquidBrain(
                n_actions=len(self.valid_actions),
                seed=new_seed,
                n_brains=3
            )

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

        # Stale detection
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

        # Replay mode
        if self.replay_mode and self.replay_seq:
            action = self._replay_step(grid)
            if action is not None:
                self.current_episode_actions.append(action.name)
                return action
            self.replay_mode = False

        # Explore
        action = self._explore(grid)
        self.current_episode_actions.append(action.name)
        self.recent_actions.append(action.name)
        return action

    def _start_replay(self, level):
        sequences = self.level_sequences.get(level, [])
        if not sequences:
            return
        seq_idx = (self.attempt_count - 1) % len(sequences)
        self.replay_seq = list(sequences[seq_idx])
        self.replay_idx = 0
        self.replay_mode = True
        logger.info(f"v26: Replay L{level} seq#{seq_idx} ({len(self.replay_seq)} actions)")

    def _replay_step(self, grid):
        if self.replay_idx >= len(self.replay_seq):
            return None
        a_name = self.replay_seq[self.replay_idx]
        self.replay_idx += 1
        for a in self.valid_actions:
            if a.name == a_name:
                self._configure_action(a, grid, "v26:replay")
                return a
        return None

    def _explore(self, grid):
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        base_sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        # v26: Progressive sigma within episode (P232 insight)
        # Start conservative, ramp up as episode progresses
        progress = min(1.0, self.steps_in_episode / self.MAX_ACTIONS)
        sigma = base_sigma * (0.3 + 0.7 * progress)

        action_scores, _ = self.brain.score_actions(grid, sigma=sigma)
        noise = np.array([self.rng.gauss(0, max(0.01, sigma))
                         for _ in range(len(self.valid_actions))], dtype=np.float32)
        action_scores += noise

        # Anti-repeat penalty (v26: stronger, longer memory)
        for i, a in enumerate(self.valid_actions):
            repeat_count = sum(1 for r in self.recent_actions if r == a.name)
            action_scores[i] -= repeat_count * 0.4  # v26: 0.3->0.4

        best_idx = int(np.argmax(action_scores))
        action = self.valid_actions[best_idx]
        self._configure_action(action, grid, f"v26:s{sigma:.2f}")
        return action

    def _configure_action(self, action, grid, mode="v26"):
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
        """v26: 6 strategies including symmetry-aware targeting."""
        try:
            h, w = grid.shape
            if h <= 1 or w <= 1:
                return random.randint(0, max(0, h-1)), random.randint(0, max(0, w-1))

            strategy_idx = (self.attempt_count + self.steps_in_episode) % len(self.target_strategies)
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

            elif strategy == 'symmetry':
                # v26: Target pixels that break symmetry (P234 insight)
                # Check horizontal symmetry breaks
                flipped = grid_int[:, ::-1]
                diff_mask = grid_int != flipped
                ys, xs = np.where(diff_mask)
                if len(ys) > 0:
                    idx = random.randint(0, len(ys) - 1)
                    return int(ys[idx]), int(xs[idx])
                # Check vertical symmetry breaks
                flipped_v = grid_int[::-1, :]
                diff_mask_v = grid_int != flipped_v
                ys, xs = np.where(diff_mask_v)
                if len(ys) > 0:
                    idx = random.randint(0, len(ys) - 1)
                    return int(ys[idx]), int(xs[idx])

            # Fallback: random
            return random.randint(0, h-1), random.randint(0, w-1)
        except Exception:
            return 0, 0


MyAgent = StochasticGoose
