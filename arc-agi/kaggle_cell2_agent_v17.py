%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v17 Agent for ARC-AGI-3
# Liquid Temporal SR: LNN (CfC) with tau-noise + spatial-noise
# + 4-Action Constraint + Bounded Retries + State Leak Fix
#
# v16 -> v17 upgrades:
# 1. Temporal Stochastic Resonance: inject noise into tau-gates
#    (time-constant perturbation) in addition to spatial noise
# 2. Adaptive sigma schedule based on attempt history
# 3. Thermodynamic hash for smarter stuck detection
# 4. Miracle trajectory replay with action-sequence memory
#
# Combines: v14's LNN brain + v5's action compression +
#           v16's bug fixes + NEW temporal noise injection
# ==============================================================
import logging
import random
import time
import numpy as np
from typing import Any, Optional

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

SIGMA_SCHEDULE = [0.0, 0.05, 0.15, 0.30, 0.50, 0.01, 0.10, 0.20, 0.75, 1.00]
TAU_NOISE_SCHEDULE = [0.0, 0.02, 0.05, 0.10, 0.20, 0.01, 0.03, 0.08, 0.15, 0.30]

# ==============================================================
# Spatial Feature Extractor
# ==============================================================
class SpatialFeatures:
    N_COLORS = 10

    def extract(self, grid: np.ndarray) -> np.ndarray:
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
# CfC Cell with Temporal Noise (Liquid Stochastic Resonance)
# ==============================================================
class CfCCell:
    def __init__(self, input_dim, hidden_dim=32, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(input_dim + hidden_dim)
        
        self.W_in = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.W_hh = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.W_hh *= 0.9 / max(1.0, np.max(np.abs(np.linalg.eigvals(self.W_hh))))
        
        self.W_tau = rng.randn(input_dim + hidden_dim, hidden_dim).astype(np.float32) * scale
        self.b_tau = np.ones(hidden_dim, dtype=np.float32) * 0.5
        
        self.W_out = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        self.h = np.zeros(hidden_dim, dtype=np.float32)
        self.dt = 1.0
        
    def reset(self):
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def forward(self, x: np.ndarray, sigma: float = 0.0, tau_noise: float = 0.0) -> np.ndarray:
        """Forward with BOTH spatial noise AND temporal noise on tau-gates."""
        xh = np.concatenate([x, self.h])
        tau_logit = xh @ self.W_tau + self.b_tau
        
        # TEMPORAL STOCHASTIC RESONANCE: perturb the time-constant gate
        if tau_noise > 0:
            tau_logit += np.random.randn(self.hidden_dim).astype(np.float32) * tau_noise
        
        tau = 1.0 / (1.0 + np.exp(-np.clip(tau_logit, -10, 10)))
        tau = np.clip(tau, 0.01, 1.0)
        
        pre_act = x @ self.W_in + self.h @ self.W_hh
        h_candidate = np.tanh(pre_act)
        
        alpha = np.clip(self.dt / tau, 0.0, 1.0)
        self.h = (1.0 - alpha) * self.h + alpha * h_candidate
        
        # SPATIAL noise (standard SNN noise injection)
        if sigma > 0:
            self.h += np.random.randn(self.hidden_dim).astype(np.float32) * sigma
            self.h = np.clip(self.h, -3.0, 3.0)
            
        return self.h @ self.W_out


class LiquidBrain:
    def __init__(self, n_actions, seed=42):
        self.spatial = SpatialFeatures()
        self.cfc = CfCCell(input_dim=96, hidden_dim=32, seed=seed)
        rng = np.random.RandomState(seed + 1)
        self.W_action = rng.randn(32, n_actions).astype(np.float32) * 0.1
        self.seen_hashes = set()
        
    def reset(self):
        self.cfc.reset()
        self.seen_hashes.clear()
        
    def score_actions(self, grid: np.ndarray, sigma: float = 0.0,
                     tau_noise: float = 0.0) -> np.ndarray:
        features = self.spatial.extract(grid)
        hidden = self.cfc.forward(features, sigma=sigma * 0.5, tau_noise=tau_noise)
        action_scores = hidden @ self.W_action
        
        feat_hash = hash(tuple(np.round(features, 2)))
        is_novel = feat_hash not in self.seen_hashes
        self.seen_hashes.add(feat_hash)
        
        if len(self.seen_hashes) > 20000:
            seen_list = list(self.seen_hashes)
            self.seen_hashes = set(seen_list[-10000:])
        
        if is_novel:
            action_scores += 0.5
        
        return action_scores

# ==============================================================
# Agent Implementation
# ==============================================================
class StochasticGoose(Agent):
    MAX_ACTIONS = 150
    MAX_ATTEMPTS = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000) + hash(self.game_id) % 1000
        self.rng = random.Random(seed)
        
        # 4-ACTION COMPRESSION (v5's winning formula)
        self.valid_actions = [
            GameAction.ACTION1, GameAction.ACTION2, 
            GameAction.ACTION3, GameAction.ACTION4
        ]
        
        self.brain = LiquidBrain(len(self.valid_actions), seed=seed)
        self.miracle_actions = []
        self.miracle_sequences = []   # Store successful action SEQUENCES
        self._init_episode_state()

        logger.info(f"v17 (Liquid Temporal SR) init: {len(self.valid_actions)} actions")

    def _init_episode_state(self):
        """Reset per-puzzle state. Fixes v15 State Leak bug."""
        self.attempt_count = 0
        self.steps = 0
        self.prev_levels = 0
        self.stale_counter = 0
        self.last_grid_hash = None
        self.gave_up = False
        self.current_episode_actions = []
        self.brain.reset()

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state == GameState.WIN or self.gave_up:
            return True
        if latest_frame.state == GameState.GAME_OVER and self.attempt_count >= self.MAX_ATTEMPTS - 1:
            self.gave_up = True
            return True
        return False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # STATE LEAK PREVENTION
        if len(frames) <= 1 and latest_frame.state == GameState.NOT_PLAYED:
            self._init_episode_state()
            
        if self.attempt_count >= self.MAX_ATTEMPTS:
            self.gave_up = True
            return GameAction.RESET

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER] or self.steps >= self.MAX_ACTIONS:
            self.attempt_count += 1
            self.steps = 0
            self.stale_counter = 0
            self.last_grid_hash = None
            self.current_episode_actions = []
            self.brain.reset()
            return GameAction.RESET

        self.steps += 1

        # Miracle detection (level up)
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            self.miracle_sequences.append(list(self.current_episode_actions))
            self.miracle_actions.extend(self.current_episode_actions)
            self.current_episode_actions = []
            self.prev_levels = current_levels
            self.steps = 0
            self.attempt_count = 0

        # Extract grid
        grid = np.zeros((1, 1), dtype=np.float32)
        if latest_frame.frame:
            try:
                grid = np.array(latest_frame.frame[0], dtype=np.float32)
                if grid.ndim == 1:
                    side = int(np.sqrt(len(grid)))
                    grid = grid.reshape(side, side) if side*side == len(grid) else grid.reshape(1, -1)
            except Exception:
                pass

        # EXACT HASH stuck detection (fixes v15 suicide bug)
        grid_hash = hash(grid.tobytes())
        if grid_hash == self.last_grid_hash:
            self.stale_counter += 1
            if self.stale_counter > 15:
                self.stale_counter = 0
                return GameAction.RESET
        else:
            self.stale_counter = 0
        self.last_grid_hash = grid_hash
        
        # MIRACLE SEQUENCE REPLAY (stronger than single-action replay)
        if self.miracle_sequences and self.rng.random() < 0.15:
            seq = self.rng.choice(self.miracle_sequences)
            step_in_seq = self.steps % max(1, len(seq))
            if step_in_seq < len(seq):
                a_name = seq[step_in_seq]
                action = next((a for a in self.valid_actions if a.name == a_name),
                             self.rng.choice(self.valid_actions))
                action.reasoning = "v17:replay"
                self.current_episode_actions.append(action.name)
                return action

        # Miracle single-action exploit
        if self.miracle_actions and self.rng.random() < 0.15:
            a_name = self.rng.choice(self.miracle_actions)
            action = next((a for a in self.valid_actions if a.name == a_name),
                         self.rng.choice(self.valid_actions))
            action.reasoning = "v17:exploit"
            self.current_episode_actions.append(action.name)
            return action

        # DUAL NOISE: Spatial sigma + Temporal tau-noise
        attempt_idx = self.attempt_count % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[attempt_idx]
        tau_noise = TAU_NOISE_SCHEDULE[attempt_idx]
        
        action_scores = self.brain.score_actions(grid, sigma=sigma, tau_noise=tau_noise)

        # Add exploration noise
        noise = np.array([self.rng.gauss(0, max(0.01, sigma)) 
                         for _ in range(len(self.valid_actions))], dtype=np.float32)
        best_idx = int(np.argmax(action_scores + noise))
        action = self.valid_actions[best_idx]

        action.reasoning = f"v17:LNN_s{sigma:.2f}_t{tau_noise:.2f}"
        self.current_episode_actions.append(action.name)

        return action

MyAgent = StochasticGoose
