%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v14 Agent for ARC-AGI-3
# Liquid Stochastic Resonance (LSR)
# CfC (Closed-form Continuous-time) + Spatial Features + sigma-diverse NBS
#
# v13 lesson: SimHash flattens 2D grid -> destroys spatial topology
# v14 fix: Lightweight spatial features + CfC temporal memory
#
# CfC = MIT's Closed-form Continuous-time neural network
#   - Worm-brain inspired (C. elegans, 302 neurons)
#   - Dynamic time constants adapt during inference (no backprop)
#   - O(1) overhead per step (~0.05ms for 32 neurons)
#
# Paper: SNN-Synthesis v7+ (Liquid Stochastic Resonance)
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
SIGMA_SCHEDULE = [
    0.0,   # attempt 1: greedy
    0.01,  # attempt 2: precision
    0.05,  # attempt 3: minimal
    0.10,  # attempt 4: gentle
    0.15,  # attempt 5: moderate
    0.20,  # attempt 6: moderate-high
    0.30,  # attempt 7: strong
    0.50,  # attempt 8: wild
    0.75,  # attempt 9: aggressive
    1.00,  # attempt 10: chaos
]


# ==============================================================
# Spatial Feature Extractor (replaces SimHash's blind flatten)
# Preserves 2D topology via lightweight statistics
# ==============================================================
class SpatialFeatures:
    """Extract spatial features from ARC grid WITHOUT flattening.
    
    Instead of destroying 2D structure, we extract:
    - Color histogram (what colors exist)
    - Per-color centroid (where each color is centered)
    - Symmetry scores (horizontal/vertical)
    - Edge density (complexity measure)
    - Bounding box per color
    
    Total feature dimension: ~80-100 (tiny, but spatially meaningful)
    """
    N_COLORS = 10  # ARC uses colors 0-9

    def extract(self, grid: np.ndarray) -> np.ndarray:
        """Extract spatial features from 2D grid. Optimized ~0.10ms."""
        if grid.ndim != 2 or grid.size == 0:
            return np.zeros(96, dtype=np.float32)

        h, w = grid.shape
        total = float(h * w)
        grid_int = grid.astype(np.int32)
        flat = grid_int.ravel()

        # Pre-compute coordinate arrays (once)
        ys_flat = np.repeat(np.arange(h, dtype=np.float32), w)
        xs_flat = np.tile(np.arange(w, dtype=np.float32), h)

        # 1. Color histogram via bincount (10 values)
        counts = np.bincount(flat, minlength=self.N_COLORS)[:self.N_COLORS]
        hist = counts.astype(np.float32) / total

        # 2+3. Per-color centroid + bbox via bincount with weights (40 values)
        centroids = np.zeros(self.N_COLORS * 2, dtype=np.float32)
        bboxes = np.zeros(self.N_COLORS * 2, dtype=np.float32)
        norm_w = max(1.0, float(w - 1))
        norm_h = max(1.0, float(h - 1))
        # Vectorized centroid via weighted bincount
        sum_x = np.bincount(flat, weights=xs_flat, minlength=self.N_COLORS)[:self.N_COLORS]
        sum_y = np.bincount(flat, weights=ys_flat, minlength=self.N_COLORS)[:self.N_COLORS]
        present = counts > 0
        centroids[0::2] = np.where(present, sum_x / np.maximum(counts, 1) / norm_w, 0)
        centroids[1::2] = np.where(present, sum_y / np.maximum(counts, 1) / norm_h, 0)
        # Bbox requires min/max per color - keep loop but skip empty colors
        for c in np.where((counts > 1))[0]:
            mask = flat == c
            cx = xs_flat[mask]
            cy = ys_flat[mask]
            bboxes[c * 2] = (cx[-1] - cx[0]) / norm_w  # sorted by row
            bboxes[c * 2 + 1] = (cy.max() - cy.min()) / norm_h

        # 4. Symmetry scores (4 values)
        sym = np.zeros(4, dtype=np.float32)
        sym[0] = np.sum(grid_int == grid_int[:, ::-1]) / total
        sym[1] = np.sum(grid_int == grid_int[::-1, :]) / total
        if h == w:
            sym[2] = np.sum(grid_int == grid_int.T) / total
        sym[3] = len(np.unique(flat)) / float(self.N_COLORS)

        # 5. Edge density (2 values)
        edges = np.zeros(2, dtype=np.float32)
        if h > 1:
            edges[0] = np.sum(grid_int[1:, :] != grid_int[:-1, :]) / total
        if w > 1:
            edges[1] = np.sum(grid_int[:, 1:] != grid_int[:, :-1]) / total

        # 6. Grid dimensions (2 values)
        dims = np.array([h / 30.0, w / 30.0], dtype=np.float32)

        # 7. Quadrant histograms (40 values)
        qh, qw = h // 2, w // 2
        quad_hist = np.zeros(40, dtype=np.float32)
        if qh > 0 and qw > 0:
            for i, (qi, qj) in enumerate([(0, 0), (0, qw), (qh, 0), (qh, qw)]):
                qflat = grid_int[qi:qi+qh, qj:qj+qw].ravel()
                qc = np.bincount(qflat, minlength=self.N_COLORS)[:self.N_COLORS]
                quad_hist[i*10:(i+1)*10] = qc.astype(np.float32) / max(1.0, float(len(qflat)))

        # Concatenate (10+20+20+4+2+2+40 = 98 -> 96)
        result = np.concatenate([hist, centroids, bboxes, sym, edges, dims, quad_hist])
        fixed = np.zeros(96, dtype=np.float32)
        fixed[:min(len(result), 96)] = result[:96]
        return fixed


# ==============================================================
# CfC Cell (Closed-form Continuous-time)
# MIT Liquid Neural Networks - implemented from scratch
# No external dependencies (ncps, etc.)
# ==============================================================
class CfCCell:
    """Closed-form Continuous-time (CfC) RNN cell.
    
    Based on: Hasani et al., "Liquid Time-constant Networks" (AAAI 2021)
    and "Closed-form Continuous-time Neural Networks" (Nature ML 2022)
    
    Key property: Time constants tau adapt based on input,
    giving the network "liquid" adaptive dynamics.
    
    Forward pass only (no backprop needed for adaptation).
    The liquid dynamics provide implicit in-context learning.
    """
    def __init__(self, input_dim, hidden_dim=32, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(input_dim + hidden_dim)
        
        # Input-to-hidden weights
        self.W_in = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.b_in = np.zeros(hidden_dim, dtype=np.float32)
        
        # Hidden-to-hidden (recurrent) weights
        self.W_hh = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        # Make slightly contractive for stability
        self.W_hh *= 0.9 / max(1.0, np.max(np.abs(np.linalg.eigvals(self.W_hh))))
        
        # Time-constant network (makes tau input-dependent -> "liquid")
        self.W_tau = rng.randn(input_dim + hidden_dim, hidden_dim).astype(np.float32) * scale
        self.b_tau = np.ones(hidden_dim, dtype=np.float32) * 0.5  # default tau ~ 0.6
        
        # Output projection
        self.W_out = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
        
        # Hidden state
        self.h = np.zeros(hidden_dim, dtype=np.float32)
        
        # Time step (fixed for discrete setting)
        self.dt = 1.0
        
    def reset(self):
        """Reset hidden state for new episode."""
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
    
    def forward(self, x: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        """Forward pass with optional SNN noise injection.
        
        CfC update rule:
            tau = sigmoid(W_tau @ [x; h] + b_tau)   # adaptive time constant
            h_new = sigmoid(W_in @ x + W_hh @ h + b_in)  # candidate
            h = (1 - dt/tau) * h + (dt/tau) * h_new      # liquid interpolation
            
        With SNN noise: h += N(0, sigma) before output
        
        ~0.03ms for hidden_dim=32 (well under 0.5ms threshold)
        """
        assert len(x) == self.input_dim, f"Expected {self.input_dim}, got {len(x)}"
        
        # 1. Compute input-dependent time constant (the "liquid" part)
        xh = np.concatenate([x, self.h])
        tau_logit = xh @ self.W_tau + self.b_tau
        tau = 1.0 / (1.0 + np.exp(-np.clip(tau_logit, -10, 10)))  # sigmoid
        tau = np.clip(tau, 0.01, 1.0)  # bounded time constants
        
        # 2. Compute candidate hidden state
        pre_act = x @ self.W_in + self.h @ self.W_hh + self.b_in
        h_candidate = np.tanh(pre_act)
        
        # 3. CfC liquid interpolation (smooth blending based on tau)
        alpha = self.dt / tau  # how much to move toward candidate
        alpha = np.clip(alpha, 0.0, 1.0)
        self.h = (1.0 - alpha) * self.h + alpha * h_candidate
        
        # 4. Inject SNN stochastic resonance noise
        if sigma > 0:
            noise = np.random.randn(self.hidden_dim).astype(np.float32) * sigma
            self.h = np.clip(self.h + noise, -3.0, 3.0)
        
        # 5. Output via projection
        out = self.h @ self.W_out
        return out


# ==============================================================
# Liquid Brain: CfC + Spatial Features + Action Scoring
# ==============================================================
class LiquidBrain:
    """Combines spatial feature extraction with CfC temporal processing.
    
    Architecture:
        Grid -> SpatialFeatures(96d) -> CfC(32 neurons) -> action_scores
                                                          -> curiosity_signal
    
    Total overhead: ~0.05ms per step (spatial features + CfC forward)
    well within the 0.5ms Crossover Law threshold.
    """
    def __init__(self, n_actions, seed=42):
        self.spatial = SpatialFeatures()
        self.feature_dim = 96
        self.n_actions = n_actions
        
        # CfC brain (32 liquid neurons)
        self.cfc = CfCCell(input_dim=self.feature_dim, hidden_dim=32, seed=seed)
        
        # Action scoring: hidden -> action preferences
        rng = np.random.RandomState(seed + 1)
        self.W_action = rng.randn(32, n_actions).astype(np.float32) * 0.1
        
        # Novelty memory (lightweight hash set, same as v13 but on spatial features)
        self.seen_hashes = set()
        self.step_count = 0
        
    def reset(self):
        """Reset for new episode (keep weights, reset state)."""
        self.cfc.reset()
        self.step_count = 0
        
    def score_actions(self, grid: np.ndarray, sigma: float = 0.0) -> np.ndarray:
        """Score all actions given current grid state.
        
        Returns: action_scores (n_actions,)
        """
        # 1. Extract spatial features (preserves 2D topology)
        features = self.spatial.extract(grid)
        
        # 2. Feed through CfC (temporal context + liquid adaptation)
        hidden = self.cfc.forward(features, sigma=sigma * 0.5)
        
        # 3. Compute action preferences
        action_scores = hidden @ self.W_action
        
        # 4. Add novelty bonus
        feat_hash = hash(tuple(np.round(features * 10).astype(np.int8)))
        is_novel = feat_hash not in self.seen_hashes
        self.seen_hashes.add(feat_hash)
        
        # Bound memory
        if len(self.seen_hashes) > 30000:
            seen_list = list(self.seen_hashes)
            self.seen_hashes = set(seen_list[-15000:])
        
        # Novelty bonus to all actions if state is new
        if is_novel:
            action_scores += 0.5
        
        self.step_count += 1
        return action_scores, is_novel


# ==============================================================
# SNN-Synthesis v14 Agent
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v14: Liquid Stochastic Resonance (LSR)
    
    v13 flaw: SimHash flattens 2D grid -> destroys spatial topology.
    v14 fix: Lightweight spatial features + CfC temporal memory.
    
    Architecture:
    - SpatialFeatures: 96D vector preserving 2D topology (~0.02ms)
    - CfC (32 neurons): Liquid time-constant memory (~0.03ms)
    - sigma-diverse NBS: Automatic noise schedule
    - Miracle replay: Exploit successful action sequences
    
    Total overhead: ~0.05ms/action (within 0.5ms Crossover Law)
    """
    MAX_ACTIONS = 120

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed % (2**31))

        # Liquid Brain (CfC + Spatial Features)
        self.valid_actions = [a for a in GameAction if a is not GameAction.RESET]
        self.brain = LiquidBrain(
            n_actions=len(self.valid_actions),
            seed=seed % (2**31)
        )

        # State tracking
        self.attempt_count = 0
        self.steps_in_episode = 0
        self.prev_levels = 0
        self.stale_counter = 0
        self.last_feat_hash = None

        # Exploit tracking (miracles)
        self.miracle_actions = []
        self.current_episode_actions = []

        logger.info(f"SNN-Synthesis v14 (Liquid SR) initialized for {self.game_id}")

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
            self.last_feat_hash = None
            self.brain.reset()  # Reset CfC hidden state
            return GameAction.RESET

        self.steps_in_episode += 1

        # --- Track Level Completion (Miracle) ---
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            logger.info(
                f"MIRACLE! Level {current_levels} cleared on attempt "
                f"{self.attempt_count} ({len(self.current_episode_actions)} actions)"
            )
            self.miracle_actions.extend(self.current_episode_actions)
            self.current_episode_actions = []
            self.prev_levels = current_levels
            self.steps_in_episode = 0

        # --- Extract 2D Grid (PRESERVE spatial structure) ---
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

        # --- Stuck Detection (on spatial features) ---
        feat_hash = hash(grid.tobytes())
        if feat_hash == self.last_feat_hash:
            self.stale_counter += 1
            if self.stale_counter > 15:
                self.stale_counter = 0
                return GameAction.RESET
        else:
            self.stale_counter = 0
        self.last_feat_hash = feat_hash

        # --- 1. Exploit (20% chance if we have miracle memory) ---
        if self.miracle_actions and self.rng.random() < 0.2:
            action = self._exploit_miracle()
            self._configure_action(action, grid, "exploit")
            self.current_episode_actions.append(action.name)
            return action

        # --- 2. Explore: CfC + sigma-Diverse Noise ---
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        # Score actions via Liquid Brain (CfC + spatial features)
        action_scores, is_novel = self.brain.score_actions(grid, sigma=sigma)

        # Add sigma-diverse Gaussian noise to scores
        noise = np.array([self.rng.gauss(0, max(0.01, sigma))
                         for _ in range(len(self.valid_actions))], dtype=np.float32)
        action_scores += noise

        # Select best scoring action
        best_idx = int(np.argmax(action_scores))
        action = self.valid_actions[best_idx]

        # Configure and return
        self._configure_action(action, grid, f"lsr_s{sigma:.2f}")
        self.current_episode_actions.append(action.name)
        return action

    def _exploit_miracle(self) -> GameAction:
        """Replay a random action from miracle memory."""
        miracle_name = self.rng.choice(self.miracle_actions)
        for a in self.valid_actions:
            if a.name == miracle_name:
                return a
        return self.rng.choice(self.valid_actions)

    def _configure_action(
        self, action: GameAction, grid: np.ndarray, mode: str
    ) -> None:
        """Smart coordinate targeting using spatial knowledge."""
        if action.is_simple():
            action.reasoning = f"v14:{mode}"
        elif action.is_complex():
            x, y = self.rng.randint(0, 63), self.rng.randint(0, 63)

            # Use spatial awareness: target non-background cells
            if grid.size > 1 and self.rng.random() < 0.6:
                try:
                    # Find non-zero (non-background) cells
                    nz = np.argwhere(grid > 0)
                    if len(nz) > 0:
                        # Pick a random non-zero cell
                        chosen = nz[self.rng.randint(0, len(nz) - 1)]
                        y, x = int(chosen[0]), int(chosen[1])
                    else:
                        # All zeros: pick random position within grid
                        h, w = grid.shape
                        y = self.rng.randint(0, max(0, h - 1))
                        x = self.rng.randint(0, max(0, w - 1))
                except Exception:
                    pass

            x = max(0, min(63, x))
            y = max(0, min(63, y))
            action.set_data({"x": x, "y": y})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"v14:{mode}",
            }


# Alias for Kaggle runner
MyAgent = StochasticGoose
