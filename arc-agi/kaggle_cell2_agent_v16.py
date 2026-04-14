%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v16 Agent for ARC-AGI-3
# Liquid Minimalist: LNN (CfC) + 4-Action Constraint + Bounded Retries
#
# Root Cause of v15's 0.00:
# 1. The 15-Step Suicide: Macro-stats don't change when the agent moves.
#    v15 thought it was stuck and reset itself every 15 steps (suicide).
# 2. State Leak: `self.gave_up` was never reset on new puzzles. v15 gave up
#    on puzzle 1 and slept through puzzles 2-100.
#
# Why v14 (0.10) is the true champion:
# LNN (CfC) achieved 0.10 *even while struggling with the huge action space*.
# Its liquid time-constants natively handle temporal dynamics perfectly.
#
# v16 = v14's LiquidBrain + v5's Action Compression + Fixed Bounded Retries
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

# ==============================================================
# Spatial Feature Extractor (From v14)
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
# CfC Cell (Liquid Neural Network)
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
    
    def forward(self, x: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        xh = np.concatenate([x, self.h])
        tau_logit = xh @ self.W_tau + self.b_tau
        tau = 1.0 / (1.0 + np.exp(-np.clip(tau_logit, -10, 10)))
        tau = np.clip(tau, 0.01, 1.0)
        
        pre_act = x @ self.W_in + self.h @ self.W_hh
        h_candidate = np.tanh(pre_act)
        
        alpha = np.clip(self.dt / tau, 0.0, 1.0)
        self.h = (1.0 - alpha) * self.h + alpha * h_candidate
        
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
        
    def score_actions(self, grid: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        features = self.spatial.extract(grid)
        hidden = self.cfc.forward(features, sigma=sigma * 0.5)
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
    MAX_ATTEMPTS = 5  # Give up and move on after 5 attempts!

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000) + hash(self.game_id) % 1000
        self.rng = random.Random(seed)
        
        # --- ACTION SPACE COMPRESSION ---
        # v5 accidentally restricted itself to 4 actions.
        # This makes navigation puzzles trivially easy to explore!
        self.valid_actions = [
            GameAction.ACTION1, GameAction.ACTION2, 
            GameAction.ACTION3, GameAction.ACTION4
        ]
        
        self.brain = LiquidBrain(len(self.valid_actions), seed=seed)
        self.miracle_actions = []
        self._init_episode_state()

        logger.info(f"v16 (Liquid Explorer) init: {len(self.valid_actions)} actions")

    def _init_episode_state(self):
        """Fixes State Leak bug from v15. Called on every NEW puzzle."""
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
        # --- Prevent State Leak Across Puzzles ---
        # len(frames) <= 1 means it's a completely new puzzle from the Kaggle runner
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

        # Check for miracle (level up)
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
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

        # Stuck detection USING EXACT HASH (Fixes 15-step suicide bug!)
        grid_hash = hash(grid.tobytes())
        if grid_hash == self.last_grid_hash:
            self.stale_counter += 1
            if self.stale_counter > 15:
                self.stale_counter = 0
                return GameAction.RESET
        else:
            self.stale_counter = 0
        self.last_grid_hash = grid_hash
        
        # --- Exploit Miracle Memory ---
        if self.miracle_actions and self.rng.random() < 0.2:
            a_name = self.rng.choice(self.miracle_actions)
            action = next((a for a in self.valid_actions if a.name == a_name), self.rng.choice(self.valid_actions))
            action.reasoning = "v16:exploit"
            self.current_episode_actions.append(action.name)
            return action

        # Liquid Brain Scoring + Sigma Diverse Noise
        sigma = SIGMA_SCHEDULE[self.attempt_count % len(SIGMA_SCHEDULE)]
        action_scores = self.brain.score_actions(grid, sigma=sigma)

        # Add Noise & Select
        noise = np.array([self.rng.gauss(0, max(0.01, sigma)) for _ in range(len(self.valid_actions))], dtype=np.float32)
        best_idx = int(np.argmax(action_scores + noise))
        action = self.valid_actions[best_idx]

        action.reasoning = f"v16:LNN_s{sigma:.2f}"
        self.current_episode_actions.append(action.name)

        return action

MyAgent = StochasticGoose
