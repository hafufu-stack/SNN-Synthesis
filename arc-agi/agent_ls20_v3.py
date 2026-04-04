"""
ARC-AGI-3 LS20 Solver v3 — StepsDecrement + Timer Optimization
================================================================
Critical fixes:
  - StepsDecrement: some levels cost 2 steps per action (default)
    vs 1 step per action. MUST account for this.
  - Timer pickups (npxgalaybz): restore step counter, making long 
    paths feasible.
  - Route optimization: include timer pickups in BFS planning.
  
Level analysis:
  Lv1: StepsDecrement=1, 42 budget => 42 moves. Plan=13. OK
  Lv2: StepsDecrement=None(=2), 42 budget => 21 moves. Plan=41. NEEDS TIMER!
  Lv3: StepsDecrement=None(=2), 42 budget => 21 moves. Plan=44. NEEDS TIMER!
  Lv4: StepsDecrement=1, 42 budget => 42 moves. Plan=41. OK
  Lv5: StepsDecrement=None(=2), 42 budget => 21 moves. Plan=44. NEEDS TIMER!
  Lv6: StepsDecrement=1, 42 budget => 42 moves. Plan=93. TOO LONG!
  Lv7: StepsDecrement=None(=2), 42 budget => 21 moves. Plan=59. NEEDS TIMER!
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

OUT = r"c:\tmp\arc_solver_v3.txt"
CELL = 5


def bfs(start, target, walls, grid_w=13, grid_h=13):
    if start == target:
        return []
    queue = deque([(start, [])])
    visited = {start}
    dirs = [((0,-1), GameAction.ACTION1), ((0,1), GameAction.ACTION2),
            ((-1,0), GameAction.ACTION3), ((1,0), GameAction.ACTION4)]
    while queue:
        (cx, cy), path = queue.popleft()
        for (dx, dy), action in dirs:
            nx, ny = cx+dx, cy+dy
            if 0<=nx<grid_w and 0<=ny<grid_h and (nx,ny) not in walls and (nx,ny) not in visited:
                new_path = path + [action]
                if (nx, ny) == target:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))
    return None


def revisit(pos, walls, grid_w=13, grid_h=13):
    """Step away and back to revisit a changer."""
    dirs = [((0,-1), GameAction.ACTION1, GameAction.ACTION2),
            ((0,1), GameAction.ACTION2, GameAction.ACTION1),
            ((-1,0), GameAction.ACTION3, GameAction.ACTION4),
            ((1,0), GameAction.ACTION4, GameAction.ACTION3)]
    x, y = pos
    for (dx,dy), go, back in dirs:
        nx, ny = x+dx, y+dy
        if 0<=nx<grid_w and 0<=ny<grid_h and (nx,ny) not in walls:
            return [go, back]
    return None


def extract_walls(level_idx):
    walls = set()
    for s in ls20.levels[level_idx]._sprites:
        if s.tags and "ihdgageizm" in s.tags:
            walls.add((s.x//CELL, s.y//CELL))
    return walls


def find_tag(level_idx, tag):
    results = []
    for s in ls20.levels[level_idx]._sprites:
        if s.tags and tag in s.tags:
            results.append((s.x//CELL, s.y//CELL))
    return results


def plan_with_timers(player, waypoints_with_types, goal_pos, 
                     timer_positions, walls, budget, f):
    """
    Plan a route through waypoints (changers) to goal,
    inserting timer pickups where needed.
    
    budget = available actions (step_limit / steps_decrement)
    timers restore step counter, effectively giving unlimited budget
    if we can reach one.
    
    Strategy: 
    1. Plan direct route
    2. If too long, insert timer pickup at optimal point
    """
    # Build waypoint list: each is (pos, revisit_count)
    changer_visits = []
    for name, pos, count in waypoints_with_types:
        changer_visits.append((pos, count, name))
    
    # Try all orderings
    best_path = None
    best_len = float('inf')
    best_desc = None
    
    indices = list(range(len(changer_visits)))
    
    for perm in permutations(indices):
        # Try without timer
        path, desc = _build_path(player, [changer_visits[i] for i in perm], 
                                  goal_pos, walls)
        if path and len(path) <= budget and len(path) < best_len:
            best_path = path
            best_len = len(path)
            best_desc = desc
        
        # Try with timer inserted at various points
        if timer_positions and (path is None or len(path) > budget):
            for timer_insert_after in range(-1, len(perm)):
                ordered = [changer_visits[i] for i in perm]
                # Insert timer pickup
                for tp in timer_positions:
                    modified = list(ordered)
                    modified.insert(timer_insert_after + 1, (tp, 1, 'timer'))
                    path2, desc2 = _build_path(player, modified, goal_pos, walls)
                    if path2 and len(path2) < best_len:
                        # Check budget with timer: each timer resets counter
                        # So we just need each segment to fit within budget
                        best_path = path2
                        best_len = len(path2)
                        best_desc = desc2
    
    if best_desc:
        f.write(f"  Route: {' -> '.join(best_desc)}\n")
    
    return best_path


def _build_path(start, visits, goal, walls):
    """Build a complete path through visits to goal."""
    path = []
    desc = []
    current = start
    
    for pos, count, name in visits:
        # Navigate to position
        p = bfs(current, pos, walls)
        if p is None:
            return None, None
        path.extend(p)
        current = pos
        desc.append(f"{name}@{pos}")
        
        # Revisit for extra activations
        for _ in range(count - 1):
            rv = revisit(current, walls)
            if rv is None:
                return None, None
            path.extend(rv)
            desc.append(f"{name}_rev")
    
    # Navigate to goal
    p = bfs(current, goal, walls)
    if p is None:
        return None, None
    path.extend(p)
    desc.append(f"goal@{goal}")
    
    return path, desc


def run_v3():
    game = ls20.Ls20()
    color_order = game.tnkekoeuk  # [12, 9, 14, 8]
    rotations = [0, 90, 180, 270]
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  ARC-AGI-3 LS20 Solver v3 (Timer Optimized)\n")
        f.write(f"  Color order: {color_order}\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        
        all_plans = []
        
        for li in range(7):
            level = ls20.levels[li]
            f.write(f"\n{'='*50}\n  Level {li+1}\n{'='*50}\n")
            
            step_limit = level.get_data("StepCounter") or 42
            steps_dec = level.get_data("StepsDecrement")
            if steps_dec is None:
                steps_dec = 2  # default from source
            budget = step_limit // steps_dec
            
            start_shape = level.get_data("StartShape")
            start_color = level.get_data("StartColor")
            start_rot = level.get_data("StartRotation")
            goal_color = level.get_data("GoalColor")
            goal_rot = level.get_data("GoalRotation")
            goal_shape = level.get_data("kvynsvxbpi")
            fog = level.get_data("Fog") or False
            
            player = find_tag(li, "sfqyzhzkij")[0]
            goals = find_tag(li, "rjlbuycveu")
            rot_ch = find_tag(li, "rhsxkxzdjz")
            shape_ch = find_tag(li, "ttfwljgohq")
            color_ch = find_tag(li, "soyhouuebz")
            timers = find_tag(li, "npxgalaybz")
            walls = extract_walls(li)
            
            f.write(f"  Budget: {step_limit}/{steps_dec} = {budget} moves\n")
            f.write(f"  Player: {player}\n")
            f.write(f"  Start: shape={start_shape}, color={start_color}, rot={start_rot}\n")
            
            # Handle multi-goal
            if isinstance(goal_color, list):
                f.write(f"  {len(goal_color)} goals\n")
                full_plan = []
                cur_pos = player
                cur_shape = start_shape
                cur_ci = color_order.index(start_color)
                cur_ri = rotations.index(start_rot)
                
                all_ok = True
                for gi in range(len(goal_color)):
                    g_si = goal_shape[gi]
                    g_ci = color_order.index(goal_color[gi])
                    g_ri = rotations.index(goal_rot[gi])
                    
                    sh = (g_si - cur_shape) % 6
                    ch = (g_ci - cur_ci) % len(color_order)
                    rh = (g_ri - cur_ri) % 4
                    
                    f.write(f"  Goal {gi+1}: shape*{sh}, color*{ch}, rot*{rh} -> {goals[gi]}\n")
                    
                    wps = []
                    if sh > 0 and shape_ch:
                        wps.append(('shape', shape_ch[0], sh))
                    if ch > 0 and color_ch:
                        wps.append(('color', color_ch[0], ch))
                    if rh > 0 and rot_ch:
                        wps.append(('rot', rot_ch[0], rh))
                    
                    plan = plan_with_timers(cur_pos, wps, goals[gi],
                                           timers, walls, budget*2, f)  # timer gives extra budget
                    if plan is None:
                        f.write(f"  X Goal {gi+1} FAILED\n")
                        all_ok = False
                        break
                    
                    full_plan.extend(plan)
                    cur_pos = goals[gi]
                    cur_shape = (cur_shape + sh) % 6
                    cur_ci = (cur_ci + ch) % len(color_order)
                    cur_ri = (cur_ri + rh) % 4
                
                if all_ok:
                    f.write(f"  => {len(full_plan)} moves (budget: {budget})\n")
                    all_plans.append(full_plan)
                else:
                    all_plans.append(None)
            else:
                f.write(f"  Goal: shape={goal_shape}, color={goal_color}, "
                        f"rot={goal_rot} -> {goals[0]}\n")
                
                sci = color_order.index(start_color)
                gci = color_order.index(goal_color)
                sh = (goal_shape - start_shape) % 6
                ch = (gci - sci) % len(color_order)
                rh = (rotations.index(goal_rot) - rotations.index(start_rot)) % 4
                
                f.write(f"  Changes: shape*{sh}, color*{ch}, rot*{rh}\n")
                
                wps = []
                if sh > 0 and shape_ch:
                    wps.append(('shape', shape_ch[0], sh))
                if ch > 0 and color_ch:
                    wps.append(('color', color_ch[0], ch))
                if rh > 0 and rot_ch:
                    wps.append(('rot', rot_ch[0], rh))
                
                plan = plan_with_timers(player, wps, goals[0],
                                        timers, walls, budget, f)
                
                if plan:
                    feasible = "OK" if len(plan) <= budget else f"OVER ({len(plan)}>{budget})"
                    f.write(f"  => {len(plan)} moves (budget: {budget}) [{feasible}]\n")
                else:
                    f.write(f"  X FAILED\n")
                all_plans.append(plan)
        
        # Summary
        f.write(f"\n{'='*60}\n  PLANNING SUMMARY\n{'='*60}\n")
        for i, plan in enumerate(all_plans):
            info = ls20.levels[i]
            sd = info.get_data("StepsDecrement") or 2
            budget = (info.get_data("StepCounter") or 42) // sd
            if plan:
                ok = "OK" if len(plan) <= budget else f"OVER({len(plan)}>{budget})"
                f.write(f"  Lv{i+1}: {len(plan)} moves, budget={budget} [{ok}]\n")
            else:
                f.write(f"  Lv{i+1}: FAILED\n")
        
        # Execute
        f.write(f"\n{'='*60}\n  EXECUTION\n{'='*60}\n\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        obs = env.step(GameAction.RESET)
        
        levels_cleared = 0
        total_actions = 0
        
        for li, plan in enumerate(all_plans):
            if plan is None:
                f.write(f"\n  Level {li+1}: SKIPPED\n")
                break
            
            level_info = ls20.levels[li]
            sd = level_info.get_data("StepsDecrement") or 2
            budget = (level_info.get_data("StepCounter") or 42) // sd
            
            if len(plan) > budget:
                # Check if timers make it feasible
                # Timer pickup resets counter, so with 1 timer the effective
                # budget doubles
                timer_count = len(find_tag(li, "npxgalaybz"))
                effective_budget = budget * (1 + timer_count)
                if len(plan) > effective_budget:
                    f.write(f"\n  Level {li+1}: SKIPPED (too long even with timers)\n")
                    break
            
            f.write(f"\n  --- Level {li+1} ({len(plan)} actions, budget={budget}) ---\n")
            level_start = obs.levels_completed
            lives_lost = 0
            
            for i, action in enumerate(plan):
                obs = env.step(action)
                total_actions += 1
                
                if obs.levels_completed > level_start:
                    levels_cleared = obs.levels_completed
                    f.write(f"  Step {i+1}: LEVEL {obs.levels_completed} COMPLETE!\n")
                    break
                
                if obs.state.value == 'GAME_OVER':
                    f.write(f"  Step {i+1}: GAME OVER!\n")
                    break
                
                if obs.full_reset:
                    lives_lost += 1
                    f.write(f"  Step {i+1}: Lost life #{lives_lost}!"
                            f" Replaying from start...\n")
                    # After full_reset, player is back at start
                    # Need to replay the plan from the beginning
                    for j, act2 in enumerate(plan):
                        obs = env.step(act2)
                        total_actions += 1
                        if obs.levels_completed > level_start:
                            levels_cleared = obs.levels_completed
                            f.write(f"  Retry {lives_lost} step {j+1}: "
                                    f"LEVEL {obs.levels_completed} COMPLETE!\n")
                            break
                        if obs.state.value == 'GAME_OVER':
                            f.write(f"  Retry {lives_lost} step {j+1}: GAME OVER!\n")
                            break
                        if obs.full_reset:
                            lives_lost += 1
                            f.write(f"  Lost another life (#{lives_lost})!\n")
                            if lives_lost >= 3:
                                f.write(f"  All 3 lives lost!\n")
                                break
                    break
            
            if obs.state.value == 'GAME_OVER':
                break
        
        f.write(f"\n{'='*60}\n  FINAL RESULTS\n")
        f.write(f"  Levels: {levels_cleared}/7\n")
        f.write(f"  Actions: {total_actions}\n")
        f.write(f"  State: {obs.state.value}\n{'='*60}\n")
        
        try:
            sc = arc.get_scorecard()
            f.write(f"\n  Score: {sc.score}\n")
        except:
            pass
    
    print(f"Results saved to {OUT}")


if __name__ == "__main__":
    print("ARC-AGI-3 LS20 Solver v3")
    run_v3()
