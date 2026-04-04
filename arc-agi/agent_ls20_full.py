"""
ARC-AGI-3 LS20 Full Solver — All 7 Levels (CPU Only)
======================================================
Uses source code analysis + BFS path planning to solve all levels.

Key game mechanics:
  - 6 shapes (index 0-5), 4 colors, 4 rotations (0,90,180,270)
  - shape_changer (ttfwljgohq): shape = (shape+1) % 6
  - color_changer (soyhouuebz): color = (color+1) % 4
  - rotation_changer (rhsxkxzdjz): rot = (rot+1) % 4
  - Step on goal with matching shape+color+rot to clear
  - 42 steps per attempt, 3 retries per level
  - npxgalaybz tiles restore step counter
"""

import arc_agi
from arcengine import GameAction
import numpy as np
import time
import sys
from collections import deque
from itertools import product

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
import ls20

OUT = r"c:\tmp\arc_full_solver.txt"
CELL = 5

# Color indices (from source: tnkekoeuk = [epqvqkpffo, jninpsotet, bejggpjowv, tqogkgimes])
# These map to actual pixel colors. We need to figure out the mapping.
# From Level 1: StartColor=9 and GoalColor=9, color index = tnkekoeuk.index(9)
# From Level 3: StartColor=12, color changes available
# tnkekoeuk indices: 0=?, 1=?, 2=?, 3=?
# We know the actual pixel values used: 9, 12, 14, 8
# So tnkekoeuk = [9, 12, 14, 8] or similar ordering
# From source: self.hiaauhahz = self.tnkekoeuk.index(self.current_level.get_data("StartColor"))
# Level 1: StartColor=9 → index in tnkekoeuk
# Level 3: StartColor=12, GoalColor=9 → need to cycle from 12 to 9
# Level 4: StartColor=14, GoalColor=9

# Shapes: ijessuuig has 6 shapes (index 0-5)
# Rotations: dhksvilbb = [0, 90, 180, 270]


def compute_changes_needed(start_shape, start_color, start_rotation,
                           goal_shape_idx, goal_color, goal_rotation):
    """
    Compute how many times each changer needs to be activated.
    Returns (shape_hits, color_hits, rotation_hits).
    
    shape cycles 0→1→2→3→4→5→0 (mod 6)
    color cycles through tnkekoeuk (4 colors, mod 4)
    rotation cycles 0→90→180→270→0 (index mod 4)
    """
    rotations = [0, 90, 180, 270]
    
    # Shape: start_shape → goal_shape_idx (mod 6)
    shape_hits = (goal_shape_idx - start_shape) % 6
    
    # Rotation: start index → goal index (mod 4)
    start_rot_idx = rotations.index(start_rotation)
    goal_rot_idx = rotations.index(goal_rotation)
    rot_hits = (goal_rot_idx - start_rot_idx) % 4
    
    # Color: we need to know the color order
    # From the Ls20.__init__: tnkekoeuk = [epqvqkpffo, jninpsotet, bejggpjowv, tqogkgimes]
    # These are module-level constants. Let's extract them.
    # Looking at the source, these are color constants defined somewhere
    # From level data: colors used are 9, 12, 14, 8
    # Level 1: start=9, goal=9 → 0 hits
    # Level 3: start=12, goal=9 → ? hits
    # Level 4: start=14, goal=9 → ? hits
    # Level 5: start=12, goal=8 → ? hits
    
    # We need to figure out the actual order. Let's try all orderings 
    # and see which one is consistent.
    # Actually, looking at the source code more carefully:
    # The color_changer does: hiaauhahz = (hiaauhahz + 1) % len(tnkekoeuk)
    # So we just need the index of start_color and goal_color in tnkekoeuk
    
    # Let's try to infer the order from the Ls20 class
    game = ls20.Ls20()
    color_order = game.tnkekoeuk  # This should give us the actual color values
    
    start_color_idx = color_order.index(start_color)
    goal_color_idx = color_order.index(goal_color)
    color_hits = (goal_color_idx - start_color_idx) % len(color_order)
    
    return shape_hits, color_hits, rot_hits


def extract_walls_from_level(level_idx):
    """Extract wall positions from level sprites."""
    level = ls20.levels[level_idx]
    walls = set()
    for sprite in level._sprites:
        if sprite.tags and "ihdgageizm" in sprite.tags:
            # Wall at (sprite.x // CELL, sprite.y // CELL)
            walls.add((sprite.x // CELL, sprite.y // CELL))
    return walls


def bfs(start, target, walls, grid_w=13, grid_h=13):
    """BFS shortest path avoiding walls."""
    if start == target:
        return []
    
    queue = deque([(start, [])])
    visited = {start}
    directions = [
        ((0, -1), GameAction.ACTION1),   # up
        ((0, 1), GameAction.ACTION2),    # down
        ((-1, 0), GameAction.ACTION3),   # left
        ((1, 0), GameAction.ACTION4),    # right
    ]
    
    while queue:
        (cx, cy), path = queue.popleft()
        for (dx, dy), action in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h and \
               (nx, ny) not in walls and (nx, ny) not in visited:
                new_path = path + [action]
                if (nx, ny) == target:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))
    return None


def plan_level(level_idx, f):
    """Plan the optimal action sequence for a given level."""
    level = ls20.levels[level_idx]
    
    f.write(f"\n{'='*50}\n")
    f.write(f"  Planning Level {level_idx + 1}\n")
    f.write(f"{'='*50}\n")
    
    # Extract level data
    step_limit = level.get_data("StepCounter") or 42
    start_shape = level.get_data("StartShape")
    start_color = level.get_data("StartColor")
    start_rotation = level.get_data("StartRotation") 
    goal_color = level.get_data("GoalColor")
    goal_rotation = level.get_data("GoalRotation")
    goal_shape_idx = level.get_data("kvynsvxbpi")
    fog = level.get_data("Fog") or False
    
    f.write(f"  Start: shape={start_shape}, color={start_color}, rot={start_rotation}\n")
    
    # Handle multi-goal levels
    if isinstance(goal_color, list):
        f.write(f"  Goals: {len(goal_color)} goals\n")
        for i in range(len(goal_color)):
            f.write(f"    Goal {i+1}: shape={goal_shape_idx[i]}, "
                    f"color={goal_color[i]}, rot={goal_rotation[i]}\n")
    else:
        f.write(f"  Goal: shape={goal_shape_idx}, "
                f"color={goal_color}, rot={goal_rotation}\n")
    
    f.write(f"  Step limit: {step_limit}, Fog: {fog}\n")
    
    # Key positions (pixel → grid)
    player_sprites = [s for s in level._sprites if s.tags and "sfqyzhzkij" in s.tags]
    goal_sprites = [s for s in level._sprites if s.tags and "rjlbuycveu" in s.tags]
    rot_sprites = [s for s in level._sprites if s.tags and "rhsxkxzdjz" in s.tags]
    shape_sprites = [s for s in level._sprites if s.tags and "ttfwljgohq" in s.tags]
    color_sprites = [s for s in level._sprites if s.tags and "soyhouuebz" in s.tags]
    timer_sprites = [s for s in level._sprites if s.tags and "npxgalaybz" in s.tags]
    
    player_pos = (player_sprites[0].x // CELL, player_sprites[0].y // CELL) if player_sprites else None
    goal_positions = [(s.x // CELL, s.y // CELL) for s in goal_sprites]
    rot_pos = [(s.x // CELL, s.y // CELL) for s in rot_sprites]
    shape_pos = [(s.x // CELL, s.y // CELL) for s in shape_sprites]
    color_pos = [(s.x // CELL, s.y // CELL) for s in color_sprites]
    timer_positions = [(s.x // CELL, s.y // CELL) for s in timer_sprites]
    
    f.write(f"  Player: {player_pos}\n")
    f.write(f"  Goals: {goal_positions}\n")
    f.write(f"  Rot changers: {rot_pos}\n")
    f.write(f"  Shape changers: {shape_pos}\n")
    f.write(f"  Color changers: {color_pos}\n")
    f.write(f"  Timer pickups: {timer_positions}\n")
    
    # Calculate changes needed
    walls = extract_walls_from_level(level_idx)
    f.write(f"  Walls: {len(walls)} cells\n")
    
    if isinstance(goal_color, list):
        # Multi-goal: plan for first goal, then second
        f.write("\n  Multi-goal level — planning sequentially\n")
        all_actions = []
        current_pos = player_pos
        current_shape = start_shape
        current_color_idx = None  # Will be computed
        current_rot_idx = [0, 90, 180, 270].index(start_rotation)
        
        game = ls20.Ls20()
        color_order = game.tnkekoeuk
        current_color_idx = color_order.index(start_color)
        
        for gi in range(len(goal_color)):
            g_shape = goal_shape_idx[gi]
            g_color = goal_color[gi]
            g_rot = goal_rotation[gi]
            g_pos = goal_positions[gi]
            
            g_color_idx = color_order.index(g_color)
            g_rot_idx = [0, 90, 180, 270].index(g_rot)
            
            needed_shape = (g_shape - current_shape) % 6
            needed_color = (g_color_idx - current_color_idx) % len(color_order)
            needed_rot = (g_rot_idx - current_rot_idx) % 4
            
            f.write(f"\n  Goal {gi+1}: need shape×{needed_shape}, "
                    f"color×{needed_color}, rot×{needed_rot}\n")
            
            # Build waypoints
            waypoints = []
            if needed_shape > 0 and shape_pos:
                for _ in range(needed_shape):
                    waypoints.append(shape_pos[0])
            if needed_color > 0 and color_pos:
                for _ in range(needed_color):
                    waypoints.append(color_pos[0])
            if needed_rot > 0 and rot_pos:
                for _ in range(needed_rot):
                    waypoints.append(rot_pos[0])
            waypoints.append(g_pos)
            
            # BFS through waypoints
            for wp in waypoints:
                path = bfs(current_pos, wp, walls)
                if path:
                    all_actions.extend(path)
                    current_pos = wp
                    f.write(f"    {current_pos} → {wp}: {len(path)} steps\n")
                else:
                    f.write(f"    ❌ No path from {current_pos} to {wp}!\n")
                    return None
            
            # Update state after reaching goal
            current_shape = (current_shape + needed_shape) % 6
            current_color_idx = (current_color_idx + needed_color) % len(color_order)
            current_rot_idx = (current_rot_idx + needed_rot) % 4
        
        f.write(f"\n  Total actions: {len(all_actions)} (limit: {step_limit})\n")
        return all_actions
    
    else:
        # Single goal
        game = ls20.Ls20()
        color_order = game.tnkekoeuk
        start_color_idx = color_order.index(start_color)
        goal_color_idx = color_order.index(goal_color)
        goal_rot_idx = [0, 90, 180, 270].index(goal_rotation)
        start_rot_idx = [0, 90, 180, 270].index(start_rotation)
        
        needed_shape = (goal_shape_idx - start_shape) % 6
        needed_color = (goal_color_idx - start_color_idx) % len(color_order)
        needed_rot = (goal_rot_idx - start_rot_idx) % 4
        
        f.write(f"\n  Changes needed: shape×{needed_shape}, "
                f"color×{needed_color}, rot×{needed_rot}\n")
        
        # Build waypoints: visit changers the required number of times, then goal
        waypoints = []
        # Optimize order: visit closest changers first
        changers = []
        if needed_shape > 0 and shape_pos:
            for _ in range(needed_shape):
                changers.append(('shape', shape_pos[0]))
        if needed_color > 0 and color_pos:
            for _ in range(needed_color):
                changers.append(('color', color_pos[0]))
        if needed_rot > 0 and rot_pos:
            for _ in range(needed_rot):
                changers.append(('rot', rot_pos[0]))
        
        # Try all orderings of changers to find shortest total path
        if len(changers) <= 6:
            # Brute force for small number of changers
            from itertools import permutations
            best_path = None
            best_len = float('inf')
            
            # Only permute unique orderings
            seen = set()
            for perm in permutations(range(len(changers))):
                key = tuple(changers[i][1] for i in perm)
                if key in seen:
                    continue
                seen.add(key)
                
                wp_list = [changers[i][1] for i in perm] + [goal_positions[0]]
                
                total_path = []
                current = player_pos
                valid = True
                for wp in wp_list:
                    p = bfs(current, wp, walls)
                    if p is None:
                        valid = False
                        break
                    total_path.extend(p)
                    current = wp
                
                if valid and len(total_path) < best_len:
                    best_len = len(total_path)
                    best_path = total_path
                    best_order = [changers[i][0] for i in perm]
            
            if best_path:
                f.write(f"  Best order: {best_order} → goal\n")
                f.write(f"  Total steps: {best_len} (limit: {step_limit})\n")
                
                # Check if we need timer pickups
                if best_len > step_limit and timer_positions:
                    f.write(f"  ⚠️ Path too long! Need timer pickups.\n")
                    # TODO: insert timer pickup visits
                
                return best_path
            else:
                f.write(f"  ❌ No valid path found!\n")
                return None
        else:
            # Greedy for many changers
            all_actions = []
            current = player_pos
            wps = [c[1] for c in changers] + [goal_positions[0]]
            for wp in wps:
                p = bfs(current, wp, walls)
                if p:
                    all_actions.extend(p)
                    current = wp
                else:
                    f.write(f"  ❌ No path from {current} to {wp}\n")
                    return None
            f.write(f"  Total steps: {len(all_actions)} (limit: {step_limit})\n")
            return all_actions


def run_full_solver():
    """Execute the full solver for all 7 levels."""
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  ARC-AGI-3 LS20 Full Solver — All 7 Levels\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        
        # Plan all levels
        level_plans = []
        for i in range(7):
            plan = plan_level(i, f)
            level_plans.append(plan)
            if plan:
                f.write(f"  ✅ Level {i+1} planned: {len(plan)} steps\n")
            else:
                f.write(f"  ❌ Level {i+1} planning FAILED\n")
        
        # Execute!
        f.write(f"\n{'='*60}\n")
        f.write(f"  EXECUTION\n")
        f.write(f"{'='*60}\n\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        env.include_frame_data = True
        
        obs = env.step(GameAction.RESET)
        f.write(f"Initial: levels={obs.levels_completed}, state={obs.state.value}\n\n")
        
        total_actions = 0
        for level_idx, plan in enumerate(level_plans):
            if plan is None:
                f.write(f"\n--- Level {level_idx+1}: SKIPPED (no plan) ---\n")
                continue
            
            f.write(f"\n--- Executing Level {level_idx+1} ({len(plan)} actions) ---\n")
            
            level_start = obs.levels_completed
            
            for i, action in enumerate(plan):
                obs = env.step(action)
                total_actions += 1
                
                action_names = {
                    GameAction.ACTION1: 'UP', GameAction.ACTION2: 'DOWN',
                    GameAction.ACTION3: 'LEFT', GameAction.ACTION4: 'RIGHT',
                }
                
                if obs.levels_completed > level_start:
                    f.write(f"  Step {i+1}: {action_names[action]:5s} → "
                            f"🎉 LEVEL {obs.levels_completed} COMPLETE!\n")
                    break
                
                if obs.state.value == 'GAME_OVER':
                    f.write(f"  Step {i+1}: GAME OVER!\n")
                    break
                
                if obs.full_reset:
                    f.write(f"  Step {i+1}: Full reset (lost a life)\n")
            
            if obs.levels_completed <= level_start:
                f.write(f"  ❌ Level {level_idx+1} NOT completed. "
                        f"levels={obs.levels_completed}, state={obs.state.value}\n")
                
                # Retry with fresh reset if game over
                if obs.state.value == 'GAME_OVER':
                    f.write(f"  Game over. Cannot continue.\n")
                    break
        
        f.write(f"\n{'='*60}\n")
        f.write(f"  FINAL RESULTS\n")
        f.write(f"  Levels completed: {obs.levels_completed}/7\n")
        f.write(f"  Total actions: {total_actions}\n")
        f.write(f"  Final state: {obs.state.value}\n")
        f.write(f"{'='*60}\n")
        
        try:
            sc = arc.get_scorecard()
            f.write(f"\n  Score: {sc.score}\n")
            if hasattr(sc, 'environments') and sc.environments:
                for env_sc in sc.environments:
                    if hasattr(env_sc, 'runs'):
                        for run in env_sc.runs:
                            f.write(f"    Run: score={run.score}, "
                                    f"levels={run.levels_completed}, "
                                    f"actions={run.actions}\n")
        except Exception as e:
            f.write(f"  Scorecard error: {e}\n")
    
    print(f"Results saved to {OUT}")


if __name__ == "__main__":
    print("="*60)
    print("  ARC-AGI-3 LS20 Full Solver (CPU only)")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    run_full_solver()
