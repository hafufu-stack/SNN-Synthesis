%%writefile /kaggle/working/my_agent.py
# ==============================================================
# SNN-Synthesis v9 Agent for ARC-AGI-3
# LLM-ExIt: Phi-3-mini + Noisy Beam Search + Curiosity Fallback
#
# Paper: https://doi.org/10.5281/zenodo.19481773
# GitHub: https://github.com/hifunsk/snn-synthesis
# Author: Hiroto Funasaki
# ==============================================================
import hashlib
import logging
import os
import random
import re
import time
from typing import Any, Optional

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# Path to GGUF model (uploaded as Kaggle Dataset)
MODEL_PATH = "/kaggle/input/snn-synthesis-models/Phi-3-mini-4k-instruct-Q4_K_M.gguf"

# Sigma-Diverse NBS Schedule (Phase 37a: proven equivalent to best-tuned)
# Each attempt uses a different noise level -> natural selection finds optimal
SIGMA_SCHEDULE = [
    0.0,   # attempt 1: greedy (no noise)
    0.05,  # attempt 2: minimal exploration
    0.15,  # attempt 3: moderate (Hanoi-optimal)
    0.30,  # attempt 4: strong exploration
    0.50,  # attempt 5: wild
    0.01,  # attempt 6: precision (GSM8K-optimal)
    0.10,  # attempt 7: gentle
    0.20,  # attempt 8: moderate-high
    0.75,  # attempt 9: very aggressive
    1.00,  # attempt 10: maximum chaos
]

# Map sigma to LLM temperature (higher sigma = higher temperature)
def sigma_to_temperature(sigma):
    """Convert NBS sigma to LLM temperature for inference."""
    return max(0.1, min(2.0, 0.3 + sigma * 2.0))


# ==============================================================
# Grid → Text Conversion
# ==============================================================
def grid_to_text(grids, max_size=24):
    """Convert 2D grid arrays to text representation for LLM.

    Uses symbols: 0='.', 1-9='1'-'9', 10-15='A'-'F'
    Truncates to max_size to keep prompts manageable.
    """
    lines = []
    for i, grid in enumerate(grids[:3]):  # max 3 grids
        arr = np.array(grid, dtype=np.int32)
        h, w = arr.shape

        # Truncate if too large
        show_h = min(h, max_size)
        show_w = min(w, max_size)

        lines.append(f"Grid {i} ({h}x{w}):")

        # Column indices header
        col_indices = ' '.join(f'{c:1d}' if c < 10 else chr(ord('A') + c - 10)
                               for c in range(show_w))
        lines.append(f"  C: {col_indices}")

        for r in range(show_h):
            symbols = []
            for c in range(show_w):
                v = arr[r, c]
                if v == 0:
                    symbols.append('.')
                elif v <= 9:
                    symbols.append(str(v))
                else:
                    symbols.append(chr(ord('A') + v - 10))
            lines.append(f"  {r:2d}: {' '.join(symbols)}")

        if h > max_size or w > max_size:
            lines.append(f"  (truncated from {h}x{w})")

    return '\n'.join(lines)


def grid_diff_text(prev_grids, curr_grids):
    """Describe what changed between two grid states."""
    if not prev_grids or not curr_grids:
        return "No previous state to compare."

    try:
        prev = np.array(prev_grids[0], dtype=np.int32)
        curr = np.array(curr_grids[0], dtype=np.int32)

        if prev.shape != curr.shape:
            return f"Grid shape changed: {prev.shape} -> {curr.shape}"

        diff = (prev != curr)
        n_changed = diff.sum()

        if n_changed == 0:
            return "Grid unchanged."

        # Find changed positions
        changed_pos = np.argwhere(diff)[:5]  # show up to 5 changes
        changes = []
        for r, c in changed_pos:
            changes.append(f"({r},{c}): {prev[r,c]}->{curr[r,c]}")

        return f"{n_changed} cells changed: {', '.join(changes)}"
    except Exception:
        return "Could not compute diff."


# ==============================================================
# LLM Interface
# ==============================================================
class LLMEngine:
    """Wrapper for llama.cpp GGUF model inference."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1):
        self.model = None
        self.available = False

        try:
            from llama_cpp import Llama
            if os.path.exists(model_path):
                logger.info(f"Loading LLM from {model_path}...")
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,  # -1 = use all GPU layers
                    n_threads=4,
                    verbose=False,
                )
                self.available = True
                logger.info("LLM loaded successfully!")
            else:
                logger.warning(f"Model file not found: {model_path}")
        except ImportError:
            logger.warning("llama_cpp not available, falling back to random")
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}")

    def generate(self, prompt: str, temperature: float = 0.7,
                 max_tokens: int = 100) -> str:
        """Generate text from prompt."""
        if not self.available or self.model is None:
            return ""

        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["User:", "\n\n", "---"],
                echo=False,
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return ""


# ==============================================================
# Action Parser
# ==============================================================
def parse_action(response: str):
    """Parse LLM response to extract GameAction and optional coordinates.

    Returns:
        (action_name, x, y) or (action_name, None, None) or None
    """
    if not response:
        return None

    # Try to find ACTION pattern
    # Match patterns like: "ACTION3", "ACTION6 x=10 y=5", "action 3"
    response_upper = response.upper()

    # Pattern 1: ACTIONN with optional coordinates
    match = re.search(r'ACTION\s*(\d+)', response_upper)
    if match:
        action_num = int(match.group(1))
        if 1 <= action_num <= 7:
            action_name = f"ACTION{action_num}"

            # Check for coordinates (for complex actions)
            x_match = re.search(r'X\s*[=:]\s*(\d+)', response_upper)
            y_match = re.search(r'Y\s*[=:]\s*(\d+)', response_upper)

            x = int(x_match.group(1)) if x_match else None
            y = int(y_match.group(1)) if y_match else None

            return (action_name, x, y)

    # Pattern 2: Just a number 1-7
    match = re.search(r'\b([1-7])\b', response)
    if match:
        action_num = int(match.group(1))
        return (f"ACTION{action_num}", None, None)

    return None


def build_prompt(grid_text: str, action_history: list,
                 levels_completed: int, attempt: int,
                 grid_diff: str = "") -> str:
    """Build the LLM prompt for action selection."""
    # Keep action history short
    recent_actions = action_history[-10:] if action_history else []
    action_str = ', '.join(recent_actions) if recent_actions else 'None yet'

    prompt = f"""You are an expert AI agent playing an interactive puzzle game on a grid.
Your goal is to clear all levels by choosing the right actions.

Available actions:
- ACTION1 to ACTION5: Simple actions (each does something different - explore to learn!)
- ACTION6: Complex action that targets a specific cell. Requires x,y coordinates.
- ACTION7: Another simple action.

RULES:
- Study the grid carefully for patterns, objects, and clues
- Try different actions to understand what each one does
- If the grid isn't changing, try a different action
- ACTION6 can target specific colored cells

Current state:
{grid_text}

{f"Changes from last action: {grid_diff}" if grid_diff else ""}
Levels completed: {levels_completed}
Attempt #{attempt}
Recent actions: {action_str}

Choose ONE action. If using ACTION6, include coordinates.
Respond with just the action, e.g.: "ACTION3" or "ACTION6 x=5 y=3"

Action:"""
    return prompt


# ==============================================================
# SNN-Synthesis v9 Agent: LLM-ExIt
# ==============================================================
class StochasticGoose(Agent):
    """
    SNN-Synthesis v9: LLM-ExIt Agent.

    Uses Phi-3-mini (GGUF) for intelligent action selection,
    with curiosity-based fallback when LLM is unavailable.
    Implements sequential Noisy Beam Search via temperature scheduling.
    """
    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RNG
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

        # Initialize LLM
        self.llm = LLMEngine(MODEL_PATH)

        # State tracking
        self.sa_visits = {}
        self.action_history = []
        self.current_episode_actions = []
        self.prev_levels = 0
        self.attempt_count = 0
        self.prev_grids = None

        # Miracle memory
        self.miracle_actions = []
        self.miracle_prompts = []  # for future QLoRA

        # NBS temperature
        self.curiosity_bonus = 2.0

        logger.info(
            f"SNN-Synthesis v9 initialized for {self.game_id} "
            f"(LLM available: {self.llm.available})"
        )

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Only stop on WIN."""
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """LLM-ExIt action selection with curiosity fallback."""

        # --- RESET when needed ---
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.current_episode_actions = []
            self.prev_grids = None
            self.attempt_count += 1
            logger.info(
                f"Game {self.game_id}: RESET (attempt #{self.attempt_count}, "
                f"miracles={len(self.miracle_actions)})"
            )
            return GameAction.RESET

        # --- Check for miracle ---
        current_levels = latest_frame.levels_completed or 0
        if current_levels > self.prev_levels:
            self.miracle_actions.extend(self.current_episode_actions)
            logger.info(
                f"MIRACLE! Level {current_levels}! "
                f"Saved {len(self.current_episode_actions)} actions."
            )
            self.current_episode_actions = []
            self.prev_levels = current_levels

        # --- Choose action ---
        if self.llm.available and latest_frame.frame:
            action = self._llm_action(latest_frame)
        elif self.miracle_actions and random.random() < 0.3:
            action = self._exploit_miracle()
        else:
            action = self._curiosity_action(latest_frame)

        # Record
        self.current_episode_actions.append(action.name)
        self.action_history.append(action.name)

        # Save previous grid for diff
        self.prev_grids = latest_frame.frame if latest_frame.frame else None

        return action

    def _llm_action(self, frame: FrameData) -> GameAction:
        """Use LLM for action selection."""
        try:
            # Build text representation
            grid_text = grid_to_text(frame.frame)

            # Compute grid diff
            diff_text = ""
            if self.prev_grids:
                diff_text = grid_diff_text(self.prev_grids, frame.frame)

            # Build prompt
            prompt = build_prompt(
                grid_text=grid_text,
                action_history=self.current_episode_actions,
                levels_completed=frame.levels_completed or 0,
                attempt=self.attempt_count,
                grid_diff=diff_text,
            )

            # NBS: sigma-diverse per attempt (Phase 37a)
            sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
            sigma = SIGMA_SCHEDULE[sigma_idx]
            temperature = sigma_to_temperature(sigma)

            # Generate
            response = self.llm.generate(prompt, temperature=temperature, max_tokens=50)
            logger.debug(f"LLM response (temp={temperature}): {response}")

            # Parse action
            parsed = parse_action(response)
            if parsed:
                action_name, x, y = parsed
                action = self._name_to_action(action_name)
                if action is not None:
                    if action.is_simple():
                        action.reasoning = (
                            f"LLM-ExIt (t={temperature:.1f}): {response[:80]}"
                        )
                    elif action.is_complex():
                        # Use LLM coordinates or smart fallback
                        if x is None or y is None:
                            x, y = self._smart_coords(frame)
                        x = max(0, min(63, x))
                        y = max(0, min(63, y))
                        action.set_data({"x": x, "y": y})
                        action.reasoning = {
                            "desired_action": f"{action.value}",
                            "my_reason": f"LLM-ExIt: ({x},{y}), {response[:50]}",
                        }
                    return action

            # Parse failed: fall through to curiosity
            logger.debug(f"LLM parse failed: {response}")

        except Exception as e:
            logger.warning(f"LLM action error: {e}")

        # Fallback
        return self._curiosity_action(frame)

    def _curiosity_action(self, frame: FrameData) -> GameAction:
        """Curiosity-guided random action with sigma-diverse noise."""
        state_hash = self._grid_hash(frame)
        candidates = [a for a in GameAction if a is not GameAction.RESET]

        # Get current sigma from schedule
        sigma_idx = (self.attempt_count - 1) % len(SIGMA_SCHEDULE)
        sigma = SIGMA_SCHEDULE[sigma_idx] if self.attempt_count > 0 else 0.15

        scores = []
        for action in candidates:
            sa_key = f"{state_hash}_{action.name}"
            visits = self.sa_visits.get(sa_key, 0)
            novelty = 1.0 / (visits + 1)
            noise = random.gauss(0, max(0.05, sigma))  # sigma-diverse noise
            scores.append(novelty + noise)

        max_score = max(scores)
        weights = [2.718 ** ((s - max_score) / 0.3) for s in scores]
        total = sum(weights)
        probs = [w / total for w in weights]

        r = random.random()
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

        self._configure_action(action, frame)
        return action

    def _exploit_miracle(self) -> GameAction:
        """Replay random action from miracle memory."""
        miracle_name = random.choice(self.miracle_actions)
        for action in GameAction:
            if action.name == miracle_name:
                self._configure_action(action, self.frames[-1] if self.frames else None)
                return action
        action = random.choice([a for a in GameAction if a is not GameAction.RESET])
        self._configure_action(action, None)
        return action

    def _configure_action(self, action: GameAction,
                          frame: Optional[FrameData]) -> None:
        """Configure action with reasoning and coordinates."""
        if action.is_simple():
            action.reasoning = f"SNN-v9: curiosity, attempt={self.attempt_count}"
        elif action.is_complex():
            x, y = self._smart_coords(frame) if frame else (random.randint(0, 63), random.randint(0, 63))
            action.set_data({"x": x, "y": y})
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": f"SNN-v9: target ({x},{y})",
            }

    def _smart_coords(self, frame: FrameData):
        """Pick coordinates targeting non-zero grid cells."""
        x, y = random.randint(0, 63), random.randint(0, 63)
        if frame and frame.frame:
            try:
                arr = np.array(frame.frame[0], dtype=np.int32)
                h, w = arr.shape
                if random.random() < 0.5:
                    nz = np.argwhere(arr != 0)
                    if len(nz) > 0:
                        idx = random.randint(0, len(nz) - 1)
                        y, x = int(nz[idx][0]), int(nz[idx][1])
                else:
                    x = random.randint(0, min(w - 1, 63))
                    y = random.randint(0, min(h - 1, 63))
            except Exception:
                pass
        return max(0, min(63, x)), max(0, min(63, y))

    def _name_to_action(self, name: str) -> Optional[GameAction]:
        """Convert action name string to GameAction enum."""
        for action in GameAction:
            if action.name == name:
                return action
        return None

    def _grid_hash(self, frame: FrameData) -> str:
        """Hash grid state for novelty tracking."""
        if not frame.frame:
            return "empty"
        try:
            arr = np.array(frame.frame[0], dtype=np.int32)
            return hashlib.md5(arr.tobytes()).hexdigest()[:12]
        except Exception:
            return "error"


# Alias for Kaggle runner compatibility
MyAgent = StochasticGoose
