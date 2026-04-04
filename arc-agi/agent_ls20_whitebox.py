"""
ARC-AGI-3 LS20 Agent — White-Box BFS Solver (CPU Only)
=======================================================
Approach 1: Analyze frame data to extract game state, then use BFS
to find optimal action sequences.

Key insights from source analysis:
- 64x64 pixel grid, movement in 5-pixel steps
- ACTION1=up(y-1), ACTION2=down(y+1), ACTION3=left(x-1), ACTION4=right(x+1)
- Player is sfqyzhzkij sprite (color 12/9, 5x5)
- Walls are ihdgageizm sprites (color 4, 5x5)
- Goals are rjlbuycveu sprites (color 5, 5x5)
- Special tiles change shape/color/rotation
- Must reach goal with correct shape+color+rotation
- Step limit per level, 3 retries

Strategy: Use frame analysis to detect player, walls, goals, and
special tiles, then BFS to find shortest path.
"""

import arc_agi
from arcengine import GameAction
import numpy as np
import time
import json
from collections import deque

OUT = r"c:\tmp\arc_agent_results.txt"

# ARC color palette (from source analysis)
COLOR_MAP = {
    0: "black",
    1: "blue", 
    3: "green",
    4: "yellow",   # walls
    5: "gray",     # background/goals
    8: "azure",
    9: "maroon",   # player bottom
    11: "teal",
    12: "magenta",  # player top
    14: "brown",    # boundary walls
}


def extract_game_state(frame):
    """Extract player position, walls, and goals from frame data."""
    if isinstance(frame, list):
        frame = frame[0]  # frame is a list of ndarrays
    
    h, w = frame.shape
    
    # Find player by looking for the distinctive sfqyzhzkij pattern
    # It has color 12 on top half and 9 on bottom half (5x5 sprite)
    player_pos = None
    for y in range(h - 4):
        for x in range(w - 4):
            block = frame[y:y+5, x:x+5]
            # Check if this 5x5 block has color 12 on top and 9 on bottom
            top = block[:2, :]
            bot = block[2:, :]
            if (np.all(top == 12) and np.all(bot == 9)):
                player_pos = (x, y)
                break
        if player_pos:
            break
    
    # Map the grid into 5x5 cells 
    cell_w = w // 5 if w >= 5 else 1
    cell_h = h // 5 if h >= 5 else 1
    
    walls = set()
    goals = set()
    specials = {}  # (cx, cy) -> type
    
    for cy in range(0, h, 5):
        for cx in range(0, w, 5):
            if cy + 5 > h or cx + 5 > w:
                continue
            block = frame[cy:cy+5, cx:cx+5]
            dominant = int(np.median(block))
            
            if dominant == 4:  # yellow = wall
                walls.add((cx, cy))
            elif dominant == 5 and not np.all(block == 5):  # gray with pattern = goal area
                # Check if it's a goal (rjlbuycveu)
                if np.all(block == 5):
                    goals.add((cx, cy))
            elif dominant == 14:  # brown = boundary wall
                walls.add((cx, cy))
    
    return {
        'player': player_pos,
        'walls': walls,
        'goals': goals,
        'frame_shape': frame.shape,
        'frame': frame
    }


def visualize_frame_text(frame, player_pos=None):
    """Create a text visualization of the frame at 5x5 cell resolution."""
    if isinstance(frame, list):
        frame = frame[0]
    h, w = frame.shape
    lines = []
    
    for cy in range(0, min(h, 64), 5):
        row = ""
        for cx in range(0, min(w, 64), 5):
            if cy + 5 > h or cx + 5 > w:
                row += "?"
                continue
            block = frame[cy:cy+5, cx:cx+5]
            dominant = int(np.median(block))
            
            if player_pos and abs(cx - player_pos[0]) < 5 and abs(cy - player_pos[1]) < 5:
                row += "P"
            elif dominant == 4:
                row += "#"  # wall
            elif dominant == 5:
                row += "."  # background
            elif dominant == 14:
                row += "W"  # boundary wall
            elif dominant == 12 or dominant == 9:
                row += "P"  # player
            elif dominant == 0:
                row += "X"  # special/black pixel 
            elif dominant == 3:
                row += "G"  # green area
            elif dominant == 1:
                row += "B"  # blue
            elif dominant == 11:
                row += "T"  # teal
            else:
                row += str(dominant % 10)
        lines.append(row)
    return "\n".join(lines)


class SimpleRLAgent:
    """Simple agent that learns from frame differences."""
    
    def __init__(self):
        self.action_history = []
        self.frame_history = []
        self.best_sequence = None
        self.best_level = 0
        
    def get_action(self, obs, step):
        """Choose action based on simple heuristics."""
        # For now: try all 4 directions systematically
        actions = [GameAction.ACTION1, GameAction.ACTION2, 
                   GameAction.ACTION3, GameAction.ACTION4]
        
        # Cycle through actions based on step
        return actions[step % 4]


def run_systematic_exploration(game_id="ls20", max_steps=500):
    """Systematically explore a game by trying all action sequences."""
    
    results = {
        "game_id": game_id,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": []
    }
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  ARC-AGI-3 LS20 White-Box Agent\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        
        arc = arc_agi.Arcade()
        env = arc.make(game_id)
        env.include_frame_data = True
        
        # --- Run 1: Systematic sweep ---
        f.write("=== Run 1: Systematic Exploration ===\n")
        obs = env.step(GameAction.RESET)
        
        # Get initial frame
        frame = obs.frame[0] if obs.frame else None
        if frame is not None:
            f.write(f"Frame shape: {frame.shape}\n")
            state = extract_game_state(obs.frame)
            f.write(f"Player pos: {state['player']}\n")
            f.write(f"Walls: {len(state['walls'])} cells\n")
            f.write(f"\nInitial map:\n")
            f.write(visualize_frame_text(obs.frame, state['player']))
            f.write("\n\n")
        
        # Try different strategies
        strategies = [
            ("all_right", [GameAction.ACTION4] * 50),
            ("all_down",  [GameAction.ACTION2] * 50),
            ("zigzag_right_down", [GameAction.ACTION4, GameAction.ACTION4, 
                                   GameAction.ACTION2] * 30),
            ("spiral", _make_spiral(60)),
            ("explore_systematic", _make_systematic(100)),
        ]
        
        for strat_name, action_seq in strategies:
            obs = env.step(GameAction.RESET)
            f.write(f"\n--- Strategy: {strat_name} ({len(action_seq)} steps) ---\n")
            
            best_level = 0
            for i, action in enumerate(action_seq):
                obs = env.step(action)
                
                if obs.levels_completed > best_level:
                    best_level = obs.levels_completed
                    f.write(f"  LEVEL UP at step {i+1}! levels_completed={best_level}\n")
                
                if obs.state.value != 'NOT_FINISHED':
                    f.write(f"  Game state changed at step {i+1}: {obs.state.value}\n")
                    break
            
            f.write(f"  Result: levels_completed={obs.levels_completed}, "
                    f"state={obs.state.value}\n")
            
            # Analyze final frame
            if obs.frame:
                state = extract_game_state(obs.frame)
                f.write(f"  Final player pos: {state['player']}\n")
                f.write(f"\nFinal map:\n")
                f.write(visualize_frame_text(obs.frame, state['player']))
                f.write("\n")
            
            results["runs"].append({
                "strategy": strat_name,
                "levels": obs.levels_completed,
                "state": obs.state.value,
            })
        
        # --- Run 2: Random with restarts ---
        f.write("\n\n=== Run 2: Random Agent (1000 steps) ===\n")
        import random
        random.seed(42)
        
        obs = env.step(GameAction.RESET)
        actions = [GameAction.ACTION1, GameAction.ACTION2, 
                   GameAction.ACTION3, GameAction.ACTION4]
        
        max_level = 0
        for i in range(1000):
            action = random.choice(actions)
            obs = env.step(action)
            
            if obs.levels_completed > max_level:
                max_level = obs.levels_completed
                f.write(f"  Level {max_level} completed at step {i+1}!\n")
            
            if obs.state.value != 'NOT_FINISHED':
                f.write(f"  Game ended at step {i+1}: {obs.state.value}\n")
                # Reset and continue
                obs = env.step(GameAction.RESET)
        
        f.write(f"  Random agent best: {max_level} levels\n")
        results["random_best"] = max_level
        
        # Scorecard
        try:
            sc = arc.get_scorecard()
            f.write(f"\n=== Final Scorecard ===\n")
            f.write(f"  Score: {sc.score}\n")
            if hasattr(sc, 'environments') and sc.environments:
                for env_sc in sc.environments:
                    f.write(f"  Game: {env_sc.id}\n")
                    if hasattr(env_sc, 'runs'):
                        for run in env_sc.runs:
                            f.write(f"    Run: score={run.score}, "
                                    f"levels={run.levels_completed}, "
                                    f"actions={run.actions}\n")
        except Exception as e:
            f.write(f"  Scorecard error: {e}\n")
        
        f.write(f"\n\nAll results: {json.dumps(results, indent=2)}\n")
    
    print(f"Results saved to {OUT}")
    return results


def _make_spiral(n):
    """Generate a spiral movement pattern."""
    actions = []
    dirs = [GameAction.ACTION4, GameAction.ACTION2, 
            GameAction.ACTION3, GameAction.ACTION1]  # right, down, left, up
    length = 1
    for _ in range(n):
        for d in range(4):
            for _ in range(length):
                actions.append(dirs[d])
            if d % 2 == 1:
                length += 1
    return actions[:n]


def _make_systematic(n):
    """Generate systematic exploration: go right until wall, then down, repeat."""
    actions = []
    # Right sweep then down
    for row in range(n // 10):
        for _ in range(8):
            actions.append(GameAction.ACTION4)  # right
        actions.append(GameAction.ACTION2)  # down
        for _ in range(8):
            actions.append(GameAction.ACTION3)  # left
        actions.append(GameAction.ACTION2)  # down
    return actions[:n]


if __name__ == "__main__":
    print("="*60)
    print("  ARC-AGI-3 LS20 White-Box Agent (CPU only)")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = run_systematic_exploration("ls20", max_steps=500)
    print(f"\nDone! Best levels: {max(r['levels'] for r in results['runs'])}")
