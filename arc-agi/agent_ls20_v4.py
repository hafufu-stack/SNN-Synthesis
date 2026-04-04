"""
ARC-AGI-3 LS20 Solver v4 — Segmented Budget BFS
=================================================
Key insight: Timer pickups RESET the step counter.
So the path must be planned in segments:
  Segment 1: start → [changers] → timer → (budget resets)
  Segment 2: timer → [changers] → timer/goal → (budget resets)  
  ...
Each segment must fit within the budget (step_limit / steps_decrement).

This is a graph search problem where the state includes:
  (position, shape, color_idx, rot_idx, goals_completed, timers_collected)
"""

import arc_agi
from arcengine import GameAction
import time
import sys
from collections import deque
from itertools import permutations

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
import ls20

OUT = r"c:\tmp\arc_solver_v4.txt"
CELL = 5


def bfs_dist(start, target, walls, gw=13, gh=13):
    """Return shortest path length, or -1 if unreachable."""
    if start == target:
        return 0
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (cx, cy), d = queue.popleft()
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if 0<=nx<gw and 0<=ny<gh and (nx,ny) not in walls and (nx,ny) not in visited:
                if (nx, ny) == target:
                    return d + 1
                visited.add((nx, ny))
                queue.append(((nx, ny), d+1))
    return -1


def bfs_path(start, target, walls, gw=13, gh=13):
    """Return shortest path as list of actions."""
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
            if 0<=nx<gw and 0<=ny<gh and (nx,ny) not in walls and (nx,ny) not in visited:
                new_path = path + [action]
                if (nx, ny) == target:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))
    return None


def revisit_cost(pos, walls, gw=13, gh=13):
    """Cost of stepping away and back (always 2 if possible)."""
    x, y = pos
    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
        nx, ny = x+dx, y+dy
        if 0<=nx<gw and 0<=ny<gh and (nx,ny) not in walls:
            return 2
    return -1  # Impossible


def revisit_actions_at(pos, walls, gw=13, gh=13):
    """Return 2-action sequence to step away and back."""
    x, y = pos
    pairs = [((0,-1), GameAction.ACTION1, GameAction.ACTION2),
             ((0,1), GameAction.ACTION2, GameAction.ACTION1),
             ((-1,0), GameAction.ACTION3, GameAction.ACTION4),
             ((1,0), GameAction.ACTION4, GameAction.ACTION3)]
    for (dx,dy), go, back in pairs:
        nx, ny = x+dx, y+dy
        if 0<=nx<gw and 0<=ny<gh and (nx,ny) not in walls:
            return [go, back]
    return None


def plan_level_v4(level_idx, color_order, f):
    """
    Plan level using segmented budget approach.
    
    State: (pos, shape, color_idx, rot_idx, goals_done_mask)
    For each state, track remaining timer pickups available.
    
    For simplicity, use a greedy approach:
    1. Determine required changer visits
    2. Try all orderings of changers
    3. Insert timer pickups greedily when budget would be exceeded
    4. Check if entire path is feasible segment by segment
    """
    level = ls20.levels[level_idx]
    step_limit = level.get_data("StepCounter") or 42
    steps_dec = level.get_data("StepsDecrement")
    if steps_dec is None:
        steps_dec = 2
    budget = step_limit // steps_dec  # moves per segment
    
    rotations = [0, 90, 180, 270]
    
    start_shape = level.get_data("StartShape")
    start_color = level.get_data("StartColor")
    start_rot = level.get_data("StartRotation")
    goal_color = level.get_data("GoalColor")
    goal_rot = level.get_data("GoalRotation")
    goal_shape = level.get_data("kvynsvxbpi")
    
    def find_tag(tag):
        r = []
        for s in level._sprites:
            if s.tags and tag in s.tags:
                r.append((s.x//CELL, s.y//CELL))
        return r
    
    player = find_tag("sfqyzhzkij")[0]
    goals = find_tag("rjlbuycveu")
    rot_ch = find_tag("rhsxkxzdjz")
    shape_ch = find_tag("ttfwljgohq")
    color_ch = find_tag("soyhouuebz")
    timers = find_tag("npxgalaybz")
    walls = set()
    for s in level._sprites:
        if s.tags and "ihdgageizm" in s.tags:
            walls.add((s.x//CELL, s.y//CELL))
    
    f.write(f"\n{'='*50}\n  Level {level_idx+1}\n{'='*50}\n")
    f.write(f"  Budget: {step_limit}/{steps_dec} = {budget} moves/segment\n")
    f.write(f"  Timers: {timers}\n")
    f.write(f"  Player: {player}\n")
    
    # Handle multi-goal
    goal_list = []
    if isinstance(goal_color, list):
        for i in range(len(goal_color)):
            goal_list.append({
                'shape': goal_shape[i],
                'color': goal_color[i],
                'rot': goal_rot[i],
                'pos': goals[i]
            })
    else:
        goal_list.append({
            'shape': goal_shape,
            'color': goal_color,
            'rot': goal_rot,
            'pos': goals[0]
        })
    
    # Plan for all goals sequentially
    full_actions = []
    cur_pos = player
    cur_shape = start_shape
    cur_ci = color_order.index(start_color)
    cur_ri = rotations.index(start_rot)
    available_timers = list(timers)
    
    for gi, goal in enumerate(goal_list):
        g_si = goal['shape']
        g_ci = color_order.index(goal['color'])
        g_ri = rotations.index(goal['rot'])
        
        sh = (g_si - cur_shape) % 6
        ch = (g_ci - cur_ci) % len(color_order)
        rh = (g_ri - cur_ri) % 4
        
        f.write(f"\n  Goal {gi+1}: shape*{sh}, color*{ch}, rot*{rh} -> {goal['pos']}\n")
        
        # Build changer visit list
        changer_visits = []
        if sh > 0 and shape_ch:
            changer_visits.append(('shape', shape_ch[0], sh))
        if ch > 0 and color_ch:
            changer_visits.append(('color', color_ch[0], ch))
        if rh > 0 and rot_ch:
            changer_visits.append(('rot', rot_ch[0], rh))
        
        # Try all orderings of changer types
        best_actions = None
        best_total = float('inf')
        best_desc = None
        
        indices = list(range(len(changer_visits)))
        perms = list(permutations(indices)) if indices else [()]
        
        for perm in perms:
            ordered = [changer_visits[i] for i in perm] if perm else []
            
            # Build waypoint list: [(pos, revisit_count)]
            waypoints = []
            for name, pos, count in ordered:
                waypoints.append((name, pos, count))
            waypoints.append(('goal', goal['pos'], 1))
            
            # Try to build segmented path
            result = _segmented_path(cur_pos, waypoints, available_timers, 
                                      budget, walls)
            if result and len(result[0]) < best_total:
                best_actions = result[0]
                best_total = len(result[0])
                best_desc = result[1]
                best_timers_used = result[2]
        
        if best_actions is None:
            f.write(f"  X Goal {gi+1} FAILED!\n")
            return None
        
        f.write(f"  Route ({len(best_actions)} moves): {' -> '.join(best_desc)}\n")
        
        # Remove used timers
        if best_timers_used:
            for t in best_timers_used:
                if t in available_timers:
                    available_timers.remove(t)
        
        full_actions.extend(best_actions)
        cur_pos = goal['pos']
        cur_shape = (cur_shape + sh) % 6
        cur_ci = (cur_ci + ch) % len(color_order)
        cur_ri = (cur_ri + rh) % 4
    
    f.write(f"\n  Total: {len(full_actions)} moves\n")
    return full_actions


def _segmented_path(start, waypoints, available_timers, budget, walls):
    """
    Build a path through waypoints, inserting timer pickups to stay within budget.
    
    Returns (actions, descriptions, timers_used) or None.
    """
    # First try without timers
    actions, desc = _build_direct_path(start, waypoints, walls)
    if actions is not None and len(actions) <= budget:
        return (actions, desc, [])
    
    # Need timers. Try inserting timer before each waypoint.
    if not available_timers:
        # No timers available — try anyway (might work with the direct path)
        if actions is not None:
            return (actions, desc, [])
        return None
    
    # Strategy: greedily insert timer when current segment would overflow
    best = None
    
    # Try each timer as the refuel point
    for timer_pos in available_timers:
        # Find best insertion point
        for insert_idx in range(len(waypoints)):
            modified_wps = list(waypoints)
            modified_wps.insert(insert_idx, ('timer', timer_pos, 1))
            
            # Check segmented feasibility
            result = _check_segmented(start, modified_wps, budget, walls)
            if result:
                acts, descs = result
                if best is None or len(acts) < len(best[0]):
                    best = (acts, descs, [timer_pos])
    
    # Try 2 timers
    if best is None and len(available_timers) >= 2:
        for t1_idx, t1 in enumerate(available_timers):
            for t2_idx, t2 in enumerate(available_timers):
                if t1_idx == t2_idx:
                    continue
                for i1 in range(len(waypoints)):
                    for i2 in range(i1, len(waypoints) + 1):
                        modified = list(waypoints)
                        modified.insert(i2, ('timer', t2, 1))
                        modified.insert(i1, ('timer', t1, 1))
                        result = _check_segmented(start, modified, budget, walls)
                        if result:
                            acts, descs = result
                            if best is None or len(acts) < len(best[0]):
                                best = (acts, descs, [t1, t2])
    
    return best


def _check_segmented(start, waypoints, budget, walls):
    """Check if waypoints can be traversed within budget per segment.
    Timer waypoints reset the segment counter.
    Returns (actions, descriptions) or None.
    """
    all_actions = []
    all_desc = []
    cur_pos = start
    segment_cost = 0
    
    for name, pos, count in waypoints:
        # Cost to reach this waypoint
        dist = bfs_dist(cur_pos, pos, walls)
        if dist < 0:
            return None
        
        # Cost of revisits (count-1 extra visits)
        revisit_c = 0
        if count > 1:
            rc = revisit_cost(pos, walls)
            if rc < 0:
                return None
            revisit_c = (count - 1) * rc
        
        total_cost = dist + revisit_c
        
        if name == 'timer':
            # Timer resets budget. Check if we can reach it within current segment
            if segment_cost + dist > budget:
                return None  # Can't reach timer
            
            # Get actual path
            p = bfs_path(cur_pos, pos, walls)
            if p is None:
                return None
            all_actions.extend(p)
            all_desc.append(f"TIMER@{pos}")
            cur_pos = pos
            segment_cost = 0  # Reset!
        else:
            # Check if this waypoint fits in current segment
            if segment_cost + total_cost > budget:
                return None
            
            # Get actual path
            p = bfs_path(cur_pos, pos, walls)
            if p is None:
                return None
            all_actions.extend(p)
            cur_pos = pos
            segment_cost += dist
            
            # Add revisits
            for _ in range(count - 1):
                rv = revisit_actions_at(cur_pos, walls)
                if rv is None:
                    return None
                all_actions.extend(rv)
                segment_cost += 2
            
            all_desc.append(f"{name}@{pos}" + (f"x{count}" if count > 1 else ""))
    
    return (all_actions, all_desc)


def _build_direct_path(start, waypoints, walls):
    """Build direct path without timer insertions."""
    actions = []
    desc = []
    cur = start
    for name, pos, count in waypoints:
        p = bfs_path(cur, pos, walls)
        if p is None:
            return None, None
        actions.extend(p)
        cur = pos
        desc.append(f"{name}@{pos}")
        for _ in range(count - 1):
            rv = revisit_actions_at(cur, walls)
            if rv is None:
                return None, None
            actions.extend(rv)
            desc.append(f"{name}_rev")
    return actions, desc


def run_v4():
    game = ls20.Ls20()
    color_order = game.tnkekoeuk
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  ARC-AGI-3 LS20 Solver v4 (Segmented Budget)\n")
        f.write(f"  Color order: {color_order}\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        
        all_plans = []
        for li in range(7):
            plan = plan_level_v4(li, color_order, f)
            all_plans.append(plan)
        
        # Summary
        f.write(f"\n{'='*60}\n  SUMMARY\n{'='*60}\n")
        for i, plan in enumerate(all_plans):
            level = ls20.levels[i]
            sd = level.get_data("StepsDecrement") or 2
            budget = (level.get_data("StepCounter") or 42) // sd
            if plan:
                f.write(f"  Lv{i+1}: {len(plan)} moves, budget={budget}/seg\n")
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
            
            f.write(f"\n  --- Level {li+1} ({len(plan)} actions) ---\n")
            level_start = obs.levels_completed
            
            attempt = 0
            while attempt < 3:
                if attempt > 0:
                    f.write(f"  Retry #{attempt}\n")
                
                success = False
                for i, action in enumerate(plan):
                    obs = env.step(action)
                    total_actions += 1
                    
                    if obs.levels_completed > level_start:
                        levels_cleared = obs.levels_completed
                        f.write(f"  Step {i+1}: LEVEL {obs.levels_completed} COMPLETE!\n")
                        success = True
                        break
                    
                    if obs.state.value == 'GAME_OVER':
                        f.write(f"  Step {i+1}: GAME OVER!\n")
                        break
                    
                    if obs.full_reset:
                        f.write(f"  Step {i+1}: Lost a life, replaying...\n")
                        attempt += 1
                        break
                
                if success or obs.state.value == 'GAME_OVER':
                    break
                if not obs.full_reset:
                    break
            
            if obs.state.value == 'GAME_OVER':
                break
        
        f.write(f"\n{'='*60}\n  FINAL\n")
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
    print("ARC-AGI-3 LS20 Solver v4 (Segmented Budget)")
    run_v4()
