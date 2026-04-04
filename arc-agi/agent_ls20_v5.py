"""
ARC-AGI-3 LS20 Solver v5 — Pixel-accurate BFS
================================================
Root cause: BFS grid used x//5 for wall positions but sprites have 
x%5=4 offset. This means wall at pixel (4,0) maps to grid 0, 
but wall at pixel (4,4) also maps to grid 0 but blocks differently.

Fix: Use PIXEL coordinates directly for BFS, with 5-pixel step sizes.
Player moves from current (x,y) to (x+dx*5, y+dy*5).
Collision check: player sprite at new position overlaps with wall sprites.
"""

import arc_agi
from arcengine import GameAction
import numpy as np
import time
import sys
from collections import deque
from itertools import permutations

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
import ls20

OUT = r"c:\tmp\arc_solver_v5.txt"


def get_wall_pixels(level_idx):
    """Get all wall sprite bounding boxes as a set of occupied pixel positions."""
    walls = set()
    for s in ls20.levels[level_idx]._sprites:
        if s.tags and "ihdgageizm" in s.tags:
            # Each wall sprite is a 5x5 block
            # The player position (x,y) is blocked if moving there would overlap
            # Since both player and wall are 5x5, player at (px,py) collides with
            # wall at (wx,wy) if |px-wx| < 5 && |py-wy| < 5
            walls.add((s.x, s.y))
    return walls


def can_move_to(px, py, walls, pw=5, ph=5):
    """Check if player at (px,py) would collide with any wall."""
    for wx, wy in walls:
        if abs(px - wx) < pw and abs(py - wy) < ph:
            return False
    return True


def bfs_pixel(start_x, start_y, target_x, target_y, walls, step=5):
    """BFS in pixel space with 5-pixel steps."""
    if start_x == target_x and start_y == target_y:
        return []
    
    queue = deque([((start_x, start_y), [])])
    visited = {(start_x, start_y)}
    dirs = [
        (0, -step, GameAction.ACTION1),   # up
        (0, step, GameAction.ACTION2),    # down
        (-step, 0, GameAction.ACTION3),   # left
        (step, 0, GameAction.ACTION4),    # right
    ]
    
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy, action in dirs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < 60 and 0 <= ny < 60 and \
               (nx, ny) not in visited and can_move_to(nx, ny, walls):
                new_path = path + [action]
                if nx == target_x and ny == target_y:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))
    return None


def revisit_pixel(px, py, walls, step=5):
    """Step away and back."""
    dirs = [(0,-step, GameAction.ACTION1, GameAction.ACTION2),
            (0,step, GameAction.ACTION2, GameAction.ACTION1),
            (-step,0, GameAction.ACTION3, GameAction.ACTION4),
            (step,0, GameAction.ACTION4, GameAction.ACTION3)]
    for dx, dy, go, back in dirs:
        nx, ny = px+dx, py+dy
        if 0 <= nx < 60 and 0 <= ny < 60 and can_move_to(nx, ny, walls):
            return [go, back]
    return None


def find_tag(level_idx, tag):
    """Find sprite pixel positions by tag."""
    return [(s.x, s.y) for s in ls20.levels[level_idx]._sprites
            if s.tags and tag in s.tags]


def run_v5():
    game = ls20.Ls20()
    color_order = game.tnkekoeuk
    rotations = [0, 90, 180, 270]
    
    action_names = {GameAction.ACTION1: 'UP', GameAction.ACTION2: 'DOWN',
                    GameAction.ACTION3: 'LEFT', GameAction.ACTION4: 'RIGHT'}
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n  LS20 Solver v5 (Pixel-Accurate BFS)\n")
        f.write(f"  Color order: {color_order}\n{'='*60}\n")
        
        all_plans = []
        
        for li in range(7):
            level = ls20.levels[li]
            step_limit = level.get_data("StepCounter") or 42
            steps_dec = level.get_data("StepsDecrement")
            if steps_dec is None:
                steps_dec = 2
            budget = step_limit // steps_dec
            
            player_pos = find_tag(li, "sfqyzhzkij")[0]
            goal_positions = find_tag(li, "rjlbuycveu")
            rot_ch = find_tag(li, "rhsxkxzdjz")
            shape_ch = find_tag(li, "ttfwljgohq")
            color_ch = find_tag(li, "soyhouuebz")
            timers = find_tag(li, "npxgalaybz")
            walls = get_wall_pixels(li)
            
            gc = level.get_data("GoalColor")
            gr = level.get_data("GoalRotation")
            gs = level.get_data("kvynsvxbpi")
            start_shape = level.get_data("StartShape")
            start_color = level.get_data("StartColor")
            start_rot = level.get_data("StartRotation")
            
            f.write(f"\n{'='*50}\n  Level {li+1}  (budget={budget})\n{'='*50}\n")
            f.write(f"  Player: {player_pos} (PIXEL)\n")
            f.write(f"  Start: shape={start_shape}, color={start_color}, rot={start_rot}\n")
            f.write(f"  Walls: {len(walls)}\n")
            
            # Handle single vs multi goal
            goal_list = []
            if isinstance(gc, list):
                for i in range(len(gc)):
                    goal_list.append({'shape': gs[i], 'color': gc[i], 
                                      'rot': gr[i], 'pos': goal_positions[i]})
            else:
                goal_list.append({'shape': gs, 'color': gc, 
                                  'rot': gr, 'pos': goal_positions[0]})
            
            full_plan = []
            cur_pos = player_pos
            cur_shape = start_shape
            cur_ci = color_order.index(start_color)
            cur_ri = rotations.index(start_rot)
            avail_timers = list(timers)
            
            all_ok = True
            for gi, goal in enumerate(goal_list):
                g_si = goal['shape']
                g_ci = color_order.index(goal['color'])
                g_ri = rotations.index(goal['rot'])
                
                sh = (g_si - cur_shape) % 6
                ch = (g_ci - cur_ci) % len(color_order)
                rh = (g_ri - cur_ri) % 4
                
                f.write(f"  Goal {gi+1}: shape*{sh}, color*{ch}, rot*{rh} -> {goal['pos']}\n")
                
                # Build changer visits
                changers = []
                if sh > 0 and shape_ch:
                    changers.append(('shape', shape_ch[0], sh))
                if ch > 0 and color_ch:
                    changers.append(('color', color_ch[0], ch))
                if rh > 0 and rot_ch:
                    changers.append(('rot', rot_ch[0], rh))
                
                # Try all orderings
                best_plan = None
                best_len = float('inf')
                best_desc = None
                
                perms = list(permutations(range(len(changers)))) if changers else [()]
                
                for perm in perms:
                    ordered = [changers[i] for i in perm] if perm else []
                    
                    # Try without timer
                    plan, desc = _build_plan(cur_pos, ordered, goal['pos'], walls)
                    if plan and len(plan) < best_len:
                        if len(plan) <= budget:
                            best_plan = plan
                            best_len = len(plan)
                            best_desc = desc
                    
                    # Try with each timer
                    for ti, tp in enumerate(avail_timers):
                        for insert_at in range(len(ordered) + 1):
                            mod = list(ordered)
                            mod.insert(insert_at, ('TIMER', tp, 1))
                            plan2, desc2 = _build_plan(cur_pos, mod, goal['pos'], walls)
                            if plan2 and len(plan2) < best_len:
                                # Check segments
                                if _check_segments(cur_pos, mod, goal['pos'], 
                                                    walls, budget):
                                    best_plan = plan2
                                    best_len = len(plan2)
                                    best_desc = desc2
                
                if best_plan is None:
                    f.write(f"  X Goal {gi+1} FAILED\n")
                    all_ok = False
                    break
                
                f.write(f"  Route ({len(best_plan)} moves): "
                        f"{' -> '.join(best_desc)}\n")
                full_plan.extend(best_plan)
                cur_pos = goal['pos']
                cur_shape = (cur_shape + sh) % 6
                cur_ci = (cur_ci + ch) % len(color_order)
                cur_ri = (cur_ri + rh) % 4
            
            if all_ok:
                f.write(f"  Total: {len(full_plan)} moves\n")
                all_plans.append(full_plan)
            else:
                all_plans.append(None)
        
        # Summary & Execute
        f.write(f"\n{'='*60}\n  SUMMARY & EXECUTION\n{'='*60}\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        obs = env.step(GameAction.RESET)
        
        levels_cleared = 0
        total = 0
        
        for li, plan in enumerate(all_plans):
            if plan is None:
                f.write(f"\n  Level {li+1}: SKIPPED\n")
                break
            
            f.write(f"\n  Level {li+1}: {len(plan)} actions\n")
            start_lvl = obs.levels_completed
            
            for attempt in range(3):
                if attempt > 0:
                    f.write(f"  Retry #{attempt}\n")
                
                done = False
                for i, action in enumerate(plan):
                    obs = env.step(action)
                    total += 1
                    
                    if obs.levels_completed > start_lvl:
                        levels_cleared = obs.levels_completed
                        f.write(f"  Step {i+1}: LEVEL {levels_cleared} COMPLETE!\n")
                        done = True
                        break
                    if obs.state.value == 'GAME_OVER':
                        f.write(f"  Step {i+1}: GAME OVER\n")
                        done = True
                        break
                    if obs.full_reset:
                        f.write(f"  Step {i+1}: Life lost, retrying...\n")
                        break
                
                if done:
                    break
            
            if obs.state.value == 'GAME_OVER':
                break
        
        f.write(f"\n{'='*60}\n  FINAL: {levels_cleared}/7 levels, "
                f"{total} actions, {obs.state.value}\n{'='*60}\n")
        
        try:
            sc = arc.get_scorecard()
            f.write(f"  Score: {sc.score}\n")
        except:
            pass
    
    print(f"Results saved to {OUT}")


def _build_plan(start, visits, goal, walls):
    """Build path through visits to goal in pixel coords."""
    path = []
    desc = []
    cur = start
    
    for name, pos, count in visits:
        p = bfs_pixel(cur[0], cur[1], pos[0], pos[1], walls)
        if p is None:
            return None, None
        path.extend(p)
        cur = pos
        desc.append(f"{name}@{pos}")
        
        for _ in range(count - 1):
            rv = revisit_pixel(cur[0], cur[1], walls)
            if rv is None:
                return None, None
            path.extend(rv)
    
    p = bfs_pixel(cur[0], cur[1], goal[0], goal[1], walls)
    if p is None:
        return None, None
    path.extend(p)
    desc.append(f"goal@{goal}")
    
    return path, desc


def _check_segments(start, visits, goal, walls, budget):
    """Check if path fits within budget per segment (timer resets)."""
    cur = start
    segment = 0
    
    for name, pos, count in visits:
        dist = bfs_pixel(cur[0], cur[1], pos[0], pos[1], walls)
        if dist is None:
            return False
        segment += len(dist) + (count - 1) * 2
        cur = pos
        
        if name == 'TIMER':
            if segment > budget:
                return False
            segment = 0  # Reset
    
    # Final segment to goal
    dist = bfs_pixel(cur[0], cur[1], goal[0], goal[1], walls)
    if dist is None:
        return False
    segment += len(dist)
    
    return segment <= budget


if __name__ == "__main__":
    print("ARC-AGI-3 LS20 Solver v5 (Pixel-Accurate)")
    run_v5()
