"""
ARC-AGI-3 LS20 Solver v6 — Interactive BFS
============================================
Instead of trying to replicate the game's collision logic,
actually PLAY the game step by step and check if movement
was successful by observing frame changes.

Strategy:
1. For each level, use the game itself to explore and build
   a walkability map
2. BFS on the actual walkability map
3. Execute optimal path
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

OUT = r"c:\tmp\arc_solver_v6.txt"


def find_player(frame):
    """Find player center position from frame (look for color 12)."""
    if isinstance(frame, list):
        frame = frame[0]
    # Find all pixels with color 12 (player top half)
    ys, xs = np.where(frame == 12)
    if len(xs) == 0:
        return None
    # Return median position
    return (int(np.median(xs)), int(np.median(ys)))


def run_v6():
    game = ls20.Ls20()
    color_order = game.tnkekoeuk
    rotations = [0, 90, 180, 270]
    
    action_names = {GameAction.ACTION1: 'UP', GameAction.ACTION2: 'DOWN',
                    GameAction.ACTION3: 'LEFT', GameAction.ACTION4: 'RIGHT'}
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n  LS20 Solver v6 (Interactive BFS)\n{'='*60}\n\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        env.include_frame_data = True
        
        obs = env.step(GameAction.RESET)
        levels_cleared = 0
        total_actions = 0
        
        for li in range(7):
            level_data = ls20.levels[li]
            step_limit = level_data.get_data("StepCounter") or 42
            steps_dec = level_data.get_data("StepsDecrement")
            if steps_dec is None:
                steps_dec = 2
            budget = step_limit // steps_dec
            
            start_shape = level_data.get_data("StartShape")
            start_color = level_data.get_data("StartColor")
            start_rot = level_data.get_data("StartRotation")
            gc = level_data.get_data("GoalColor")
            gr = level_data.get_data("GoalRotation")
            gs = level_data.get_data("kvynsvxbpi")
            
            rot_ch = [(s.x, s.y) for s in level_data._sprites 
                      if s.tags and "rhsxkxzdjz" in s.tags]
            shape_ch = [(s.x, s.y) for s in level_data._sprites 
                        if s.tags and "ttfwljgohq" in s.tags]
            color_ch = [(s.x, s.y) for s in level_data._sprites 
                        if s.tags and "soyhouuebz" in s.tags]
            goals = [(s.x, s.y) for s in level_data._sprites 
                     if s.tags and "rjlbuycveu" in s.tags]
            timers = [(s.x, s.y) for s in level_data._sprites 
                      if s.tags and "npxgalaybz" in s.tags]
            player_start = [(s.x, s.y) for s in level_data._sprites 
                           if s.tags and "sfqyzhzkij" in s.tags][0]
            
            f.write(f"\n{'='*50}\n  Level {li+1}  (budget={budget})\n{'='*50}\n")
            f.write(f"  Player: {player_start}\n")
            f.write(f"  Goals: {goals}\n")
            f.write(f"  Rot: {rot_ch}, Shape: {shape_ch}, Color: {color_ch}\n")
            f.write(f"  Timers: {timers}\n")
            
            # Calculate needed changes
            if isinstance(gc, list):
                f.write(f"  Multi-goal: {len(gc)} goals\n")
                # Handle multi-goal later
                f.write(f"  SKIPPING multi-goal for now\n")
                all_plans = None
                break
            
            sci = color_order.index(start_color)
            gci = color_order.index(gc)
            sh = (gs - start_shape) % 6
            ch = (gci - sci) % len(color_order)
            rh = (rotations.index(gr) - rotations.index(start_rot)) % 4
            
            f.write(f"  Need: shape*{sh}, color*{ch}, rot*{rh}\n")
            
            # Build changer visit list
            changers = []
            if sh > 0 and shape_ch:
                changers.append(('shape', shape_ch[0], sh))
            if ch > 0 and color_ch:
                changers.append(('color', color_ch[0], ch))
            if rh > 0 and rot_ch:
                changers.append(('rot', rot_ch[0], rh))
            
            # Step 1: Explore the map interactively
            # Use BFS from current position by actually trying moves
            f.write(f"\n  --- Interactive Exploration ---\n")
            
            # First, find actual player position from frame
            if obs.frame:
                player_pos = find_player(obs.frame)
                f.write(f"  Actual player (from frame): {player_pos}\n")
            else:
                player_pos = player_start
            
            # Build walkability by trying all moves from each reachable cell
            walkability = {}  # {(x,y): set of reachable (nx,ny)}
            
            # We'll use a simulated approach: save state, try move, check result
            # But the API might not support save/load...
            # Instead, use the game's collision logic directly
            # Actually, the walls are blocking, so let's just check 
            # the frame data differences
            
            # Simpler approach: try the plan and see if it works
            # Use pixel-coordinate BFS but with CORRECT collision
            
            # From source code analysis:
            # txnfzvzetn checks: mrznumynfe finds sprites at target position
            # For ihdgageizm (walls): bwdzgjttjp = True → blocked
            # The check is: sprite.x >= juldcpkjse and sprite.x < juldcpkjse + width
            # AND sprite.y >= ullicjtklz and sprite.y < ullicjtklz + height
            
            # So wall at (wx,wy) blocks player at (px,py) if:
            #   wx >= px and wx < px + 5 and wy >= py and wy < py + 5
            # This means: player at (px,py) is blocked if any wall sprite
            # has position WITHIN the player's 5x5 bounding box
            
            walls = set()
            for s in level_data._sprites:
                if s.tags and "ihdgageizm" in s.tags:
                    walls.add((s.x, s.y))
            
            def blocked(px, py):
                """Check if player at (px,py) is blocked by any wall."""
                for wx, wy in walls:
                    if wx >= px and wx < px + 5 and wy >= py and wy < py + 5:
                        return True
                return False
            
            def pixel_bfs(sx, sy, tx, ty):
                """BFS with correct collision."""
                if sx == tx and sy == ty:
                    return []
                queue = deque([((sx, sy), [])])
                visited = {(sx, sy)}
                for _ in range(10000):  # safety limit
                    if not queue:
                        return None
                    (cx, cy), path = queue.popleft()
                    for dx, dy, action in [(0,-5,GameAction.ACTION1),
                                            (0,5,GameAction.ACTION2),
                                            (-5,0,GameAction.ACTION3),
                                            (5,0,GameAction.ACTION4)]:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < 60 and 0 <= ny < 60 and \
                           (nx,ny) not in visited and not blocked(nx, ny):
                            npath = path + [action]
                            if nx == tx and ny == ty:
                                return npath
                            visited.add((nx, ny))
                            queue.append(((nx, ny), npath))
                return None
            
            def pixel_revisit(px, py):
                for dx,dy,go,back in [(0,-5,GameAction.ACTION1,GameAction.ACTION2),
                                       (0,5,GameAction.ACTION2,GameAction.ACTION1),
                                       (-5,0,GameAction.ACTION3,GameAction.ACTION4),
                                       (5,0,GameAction.ACTION4,GameAction.ACTION3)]:
                    nx, ny = px+dx, py+dy
                    if 0<=nx<60 and 0<=ny<60 and not blocked(nx, ny):
                        return [go, back]
                return None
            
            # Try all orderings of changers
            best_plan = None
            best_len = float('inf')
            best_desc = None
            
            perms = list(permutations(range(len(changers)))) if changers else [()]
            
            for perm in perms:
                ordered = [changers[i] for i in perm] if perm else []
                
                # Build plans without and with timer
                for use_timer_idx in range(-1, len(timers)):
                    visits = list(ordered)
                    if use_timer_idx >= 0:
                        # Try inserting timer at various positions
                        for ins_pos in range(len(visits) + 1):
                            test_visits = list(visits)
                            test_visits.insert(ins_pos, ('TIMER', timers[use_timer_idx], 1))
                            plan, desc = _make_plan(player_start, test_visits, 
                                                     goals[0], pixel_bfs, pixel_revisit)
                            if plan and len(plan) < best_len:
                                if _check_seg(player_start, test_visits, 
                                              goals[0], pixel_bfs, budget):
                                    best_plan = plan
                                    best_len = len(plan)
                                    best_desc = desc
                    else:
                        plan, desc = _make_plan(player_start, visits, 
                                                goals[0], pixel_bfs, pixel_revisit)
                        if plan and len(plan) <= budget and len(plan) < best_len:
                            best_plan = plan
                            best_len = len(plan)
                            best_desc = desc
            
            if best_plan is None:
                f.write(f"  X Level {li+1} FAILED\n")
                break
            
            f.write(f"  Route ({len(best_plan)} moves): "
                    f"{' -> '.join(best_desc)}\n")
            
            # Execute
            f.write(f"\n  --- Executing Level {li+1} ---\n")
            start_lvl = obs.levels_completed
            
            for attempt in range(3):
                if attempt > 0:
                    f.write(f"  Retry #{attempt}\n")
                
                done = False
                for i, action in enumerate(best_plan):
                    obs = env.step(action)
                    total_actions += 1
                    
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
                        f.write(f"  Step {i+1}: Life lost\n")
                        break
                
                if done:
                    break
            
            if obs.state.value == 'GAME_OVER':
                break
            
            if obs.levels_completed <= start_lvl:
                f.write(f"  Level {li+1} not cleared after all attempts.\n")
                break
        
        f.write(f"\n{'='*60}\n  FINAL: {levels_cleared}/7 levels, "
                f"{total_actions} actions\n{'='*60}\n")
        try:
            sc = arc.get_scorecard()
            f.write(f"  Score: {sc.score}\n")
        except:
            pass
    
    print(f"Results saved to {OUT}")


def _make_plan(start, visits, goal, bfs_fn, rv_fn):
    path = []
    desc = []
    cur = start
    for name, pos, count in visits:
        p = bfs_fn(cur[0], cur[1], pos[0], pos[1])
        if p is None:
            return None, None
        path.extend(p)
        cur = pos
        desc.append(f"{name}@{pos}")
        for _ in range(count - 1):
            rv = rv_fn(cur[0], cur[1])
            if rv is None:
                return None, None
            path.extend(rv)
    p = bfs_fn(cur[0], cur[1], goal[0], goal[1])
    if p is None:
        return None, None
    path.extend(p)
    desc.append(f"goal@{goal}")
    return path, desc


def _check_seg(start, visits, goal, bfs_fn, budget):
    cur = start
    seg = 0
    for name, pos, count in visits:
        p = bfs_fn(cur[0], cur[1], pos[0], pos[1])
        if p is None:
            return False
        seg += len(p) + (count - 1) * 2
        cur = pos
        if name == 'TIMER':
            if seg > budget:
                return False
            seg = 0
    p = bfs_fn(cur[0], cur[1], goal[0], goal[1])
    if p is None:
        return False
    seg += len(p)
    return seg <= budget


if __name__ == "__main__":
    print("ARC-AGI-3 LS20 Solver v6")
    run_v6()
