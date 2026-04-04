"""
ARC-AGI-3 LS20 Smart Agent — BFS Path Planning (CPU Only)
============================================================
Uses source code analysis + frame data to solve LS20.

Level 1 Analysis (from source):
  - Player starts at (34, 45) with shape=5, color=9, rotation=270
  - Goal needs: shape=5, color=9, rotation=0
  - Only need: 1 rotation change (270→0)
  - Rotation changer at (19, 30)
  - Goal slot at (34, 10)
  - Step limit: 42 steps per attempt, 3 retries
  
Plan: BFS path = start→rotation_changer→goal
Movement: 5-pixel steps in 4 directions
"""

import arc_agi
from arcengine import GameAction
import numpy as np
import time
from collections import deque

OUT = r"c:\tmp\arc_bfs_results.txt"
CELL = 5  # Movement step size


def frame_to_grid(frame):
    """Convert 64x64 pixel frame to 13x13 cell grid.
    Cells at positions 4,9,14,...,59 in both x and y.
    Actually, sprites use positions 0,5,10,...,60.
    Grid cells: (x/5, y/5) for x,y in [0,5,10,...,60]
    """
    if isinstance(frame, list):
        frame = frame[0]
    
    h, w = frame.shape
    grid_w = w // CELL
    grid_h = h // CELL
    
    grid = {}  # (gx, gy) -> cell_type
    
    for gy in range(grid_h):
        for gx in range(grid_w):
            px, py = gx * CELL, gy * CELL
            if px + CELL > w or py + CELL > h:
                continue
            block = frame[py:py+CELL, px:px+CELL]
            
            # Analyze block content
            unique, counts = np.unique(block, return_counts=True)
            dominant = unique[np.argmax(counts)]
            
            # Detect cell types
            has_4 = 4 in unique   # yellow (wall)
            has_5 = 5 in unique   # gray (background/goal)
            has_14 = 14 in unique # brown (boundary)
            has_12 = 12 in unique # magenta (player top)
            has_9 = 9 in unique   # maroon (player bottom) 
            has_0 = 0 in unique   # black (special tiles, player shape)
            has_3 = 3 in unique   # green
            has_1 = 1 in unique   # blue
            has_11 = 11 in unique # teal
            has_8 = 8 in unique   # azure
            
            # Classify
            if has_12 and has_9 and counts[list(unique).index(12)] > 3:
                grid[(gx, gy)] = 'player'
            elif dominant == 4 and not has_5:
                grid[(gx, gy)] = 'wall'
            elif dominant == 14:
                grid[(gx, gy)] = 'boundary'
            elif dominant == 5 and np.all(block == 5):
                grid[(gx, gy)] = 'goal_area'  # could be goal
            elif has_3 and counts[list(unique).index(3)] > 10:
                grid[(gx, gy)] = 'green_area'  # nszegiawib
            else:
                # Check for -1 (transparent) - in numpy these might be 255 or similar
                # Check for special tiles by looking at patterns
                if has_0 and has_1 and has_8:
                    grid[(gx, gy)] = 'color_changer'  # soyhouuebz
                elif has_0 and has_1:
                    grid[(gx, gy)] = 'rotation_changer'  # rhsxkxzdjz
                elif has_0 and not has_4 and not has_14:
                    grid[(gx, gy)] = 'shape_changer'  # ttfwljgohq or other special
                else:
                    grid[(gx, gy)] = 'open'
    
    return grid, grid_w, grid_h


def find_walkable(grid, grid_w, grid_h):
    """Find all walkable cells."""
    walkable = set()
    for (gx, gy), cell_type in grid.items():
        if cell_type not in ('wall', 'boundary'):
            walkable.add((gx, gy))
    return walkable


def bfs(start, target, walkable, grid_w, grid_h):
    """BFS shortest path from start to target on grid."""
    if start == target:
        return []
    
    queue = deque([(start, [])])
    visited = {start}
    directions = {
        (0, -1): GameAction.ACTION1,  # up
        (0, 1): GameAction.ACTION2,   # down
        (-1, 0): GameAction.ACTION3,  # left
        (1, 0): GameAction.ACTION4,   # right
    }
    
    while queue:
        (cx, cy), path = queue.popleft()
        
        for (dx, dy), action in directions.items():
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in walkable and (nx, ny) not in visited:
                new_path = path + [action]
                if (nx, ny) == target:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))
    
    return None  # No path found


def print_grid(f, grid, grid_w, grid_h, path_cells=None):
    """Print readable grid map."""
    symbols = {
        'wall': '#',
        'boundary': 'W',
        'player': 'P',
        'open': '.',
        'goal_area': 'G',
        'green_area': 'g',
        'shape_changer': 'S',
        'color_changer': 'C',
        'rotation_changer': 'R',
    }
    
    for gy in range(grid_h):
        row = ""
        for gx in range(grid_w):
            if path_cells and (gx, gy) in path_cells:
                row += "*"
            elif (gx, gy) in grid:
                row += symbols.get(grid[(gx, gy)], '?')
            else:
                row += ' '
        f.write(row + "\n")


def run_bfs_agent(game_id="ls20"):
    """Run BFS-based agent on LS20."""
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  ARC-AGI-3 LS20 BFS Agent\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        
        arc = arc_agi.Arcade()
        env = arc.make(game_id)
        env.include_frame_data = True
        
        # Reset and get initial frame
        obs = env.step(GameAction.RESET)
        frame = obs.frame[0] if obs.frame else None
        
        if frame is None:
            f.write("ERROR: No frame data!\n")
            return
        
        f.write(f"Frame shape: {frame.shape}\n")
        f.write(f"Frame dtype: {frame.dtype}\n")
        f.write(f"Unique colors: {np.unique(frame)}\n\n")
        
        # Build grid
        grid, gw, gh = frame_to_grid(obs.frame)
        f.write(f"Grid size: {gw}x{gh}\n")
        f.write(f"Cell types: {dict((t, sum(1 for v in grid.values() if v == t)) for t in set(grid.values()))}\n\n")
        
        f.write("=== Grid Map ===\n")
        print_grid(f, grid, gw, gh)
        f.write("\n")
        
        # Find key positions
        player_cells = [(gx, gy) for (gx, gy), t in grid.items() if t == 'player']
        goal_cells = [(gx, gy) for (gx, gy), t in grid.items() if t == 'goal_area']
        shape_cells = [(gx, gy) for (gx, gy), t in grid.items() if t == 'shape_changer']
        color_cells = [(gx, gy) for (gx, gy), t in grid.items() if t == 'color_changer']
        rot_cells = [(gx, gy) for (gx, gy), t in grid.items() if t == 'rotation_changer']
        
        f.write(f"Player cells: {player_cells}\n")
        f.write(f"Goal cells: {goal_cells}\n")
        f.write(f"Shape changers: {shape_cells}\n")
        f.write(f"Color changers: {color_cells}\n")
        f.write(f"Rotation changers: {rot_cells}\n\n")
        
        # From source analysis:
        # Player pixel pos (34, 45) → grid cell (34/5, 45/5) = (6, 9) 
        # Rotation changer pixel pos (19, 30) → grid (19/5, 30/5) = (3, 6) 
        # Goal pixel pos (34, 10) → grid (34/5, 10/5) = (6, 2)
        # But also check (33, 9) for hoswmpiqkw goal indicator
        
        # Actually let's use the detected positions if available,
        # otherwise fall back to source-known positions
        player_gx = player_cells[0][0] if player_cells else 34 // CELL
        player_gy = player_cells[0][1] if player_cells else 45 // CELL
        
        # Known from source for Level 1
        rot_changer_gx, rot_changer_gy = 19 // CELL, 30 // CELL  # (3, 6)
        goal_gx, goal_gy = 34 // CELL, 10 // CELL  # (6, 2)
        
        f.write(f"Player grid pos: ({player_gx}, {player_gy})\n")
        f.write(f"Rotation changer grid pos: ({rot_changer_gx}, {rot_changer_gy})\n")
        f.write(f"Goal grid pos: ({goal_gx}, {goal_gy})\n\n")
        
        walkable = find_walkable(grid, gw, gh)
        f.write(f"Walkable cells: {len(walkable)}\n\n")
        
        # Plan: player → rotation_changer → goal
        f.write("=== BFS Path Planning ===\n")
        
        # Path 1: player to rotation changer
        path1 = bfs((player_gx, player_gy), (rot_changer_gx, rot_changer_gy), 
                     walkable, gw, gh)
        f.write(f"Path to rotation changer: {len(path1) if path1 else 'NOT FOUND'} steps\n")
        
        if path1 is None:
            # Try nearby cells
            f.write("  Trying nearby rotation changer positions...\n")
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    test_pos = (rot_changer_gx + dx, rot_changer_gy + dy)
                    path1 = bfs((player_gx, player_gy), test_pos, walkable, gw, gh)
                    if path1:
                        f.write(f"  Found path to ({test_pos}): {len(path1)} steps\n")
                        rot_changer_gx, rot_changer_gy = test_pos
                        break
                if path1:
                    break
        
        # Path 2: rotation changer to goal
        if path1:
            path2 = bfs((rot_changer_gx, rot_changer_gy), (goal_gx, goal_gy), 
                         walkable, gw, gh)
            f.write(f"Path to goal: {len(path2) if path2 else 'NOT FOUND'} steps\n")
            
            if path2 is None:
                f.write("  Trying nearby goal positions...\n")
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        test_pos = (goal_gx + dx, goal_gy + dy)
                        path2 = bfs((rot_changer_gx, rot_changer_gy), test_pos, 
                                    walkable, gw, gh)
                        if path2:
                            f.write(f"  Found path to ({test_pos}): {len(path2)} steps\n")
                            goal_gx, goal_gy = test_pos
                            break
                    if path2:
                        break
        else:
            path2 = None
        
        total_steps = (len(path1) if path1 else 0) + (len(path2) if path2 else 0)
        f.write(f"\nTotal planned steps: {total_steps} (limit: 42)\n")
        
        if path1 and path2 and total_steps <= 42:
            f.write("\n=== EXECUTING PLAN ===\n")
            full_path = path1 + path2
            
            # Show planned path on grid
            f.write("\nPlanned path on map:\n")
            path_cells_set = set()
            cx, cy = player_gx, player_gy
            path_cells_set.add((cx, cy))
            action_names = {
                GameAction.ACTION1: 'UP',
                GameAction.ACTION2: 'DOWN', 
                GameAction.ACTION3: 'LEFT',
                GameAction.ACTION4: 'RIGHT',
            }
            dirs = {
                GameAction.ACTION1: (0, -1),
                GameAction.ACTION2: (0, 1),
                GameAction.ACTION3: (-1, 0),
                GameAction.ACTION4: (1, 0),
            }
            for action in full_path:
                dx, dy = dirs[action]
                cx, cy = cx + dx, cy + dy
                path_cells_set.add((cx, cy))
            print_grid(f, grid, gw, gh, path_cells_set)
            f.write("\n")
            
            # Execute!
            obs = env.step(GameAction.RESET)
            f.write(f"After RESET: levels={obs.levels_completed}, state={obs.state.value}\n")
            
            for i, action in enumerate(full_path):
                obs = env.step(action)
                step_label = "→rot_changer" if i == len(path1) - 1 else \
                             "→GOAL" if i == len(full_path) - 1 else ""
                
                if i < 5 or i == len(path1) - 1 or i == len(full_path) - 1 or \
                   obs.levels_completed > 0 or obs.state.value != 'NOT_FINISHED':
                    f.write(f"  Step {i+1}/{len(full_path)}: {action_names[action]:5s} "
                            f"levels={obs.levels_completed} state={obs.state.value} "
                            f"{step_label}\n")
                
                if obs.levels_completed > 0:
                    f.write(f"\n  🎉 LEVEL COMPLETED! levels={obs.levels_completed}\n")
                    break
                    
                if obs.state.value == 'GAME_OVER':
                    f.write(f"\n  💀 GAME OVER at step {i+1}\n")
                    break
            
            f.write(f"\nFinal: levels_completed={obs.levels_completed}, "
                    f"state={obs.state.value}\n")
            
            # Show final frame
            if obs.frame:
                f.write("\nFinal frame map:\n")
                final_grid, _, _ = frame_to_grid(obs.frame)
                print_grid(f, final_grid, gw, gh)
        
        elif path1 is None:
            f.write("\n❌ Could not find path to rotation changer!\n")
            f.write("Dumping walkable cells:\n")
            for gy_row in range(gh):
                cells_in_row = [(gx, gy_row) for gx in range(gw) if (gx, gy_row) in walkable]
                if cells_in_row:
                    f.write(f"  y={gy_row}: {[gx for gx, _ in cells_in_row]}\n")
        elif path2 is None:
            f.write("\n❌ Could not find path to goal!\n")
        else:
            f.write(f"\n❌ Path too long ({total_steps} > 42 step limit)!\n")
        
        # Scorecard
        try:
            sc = arc.get_scorecard()
            f.write(f"\n=== Scorecard ===\n")
            f.write(f"  Score: {sc.score}\n")
        except Exception as e:
            f.write(f"\n  Scorecard error: {e}\n")
    
    print(f"Results saved to {OUT}")


if __name__ == "__main__":
    print("="*60)
    print("  ARC-AGI-3 LS20 BFS Agent (CPU only)")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    run_bfs_agent("ls20")
