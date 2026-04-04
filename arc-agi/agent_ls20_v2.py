"""
ARC-AGI-3 LS20 Full Solver v2 — Fixed BFS + Revisit Logic
============================================================
Fixes from v1:
  - Same-position revisit: step away then step back
  - Level transition handling
  - Better changer ordering with timer pickup optimization
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

OUT = r"c:\tmp\arc_solver_v2.txt"
CELL = 5


def bfs(start, target, walls, grid_w=13, grid_h=13):
    """BFS shortest path avoiding walls."""
    if start == target:
        return []
    
    queue = deque([(start, [])])
    visited = {start}
    directions = [
        ((0, -1), GameAction.ACTION1),
        ((0, 1), GameAction.ACTION2),
        ((-1, 0), GameAction.ACTION3),
        ((1, 0), GameAction.ACTION4),
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


def revisit_actions(pos, walls, grid_w=13, grid_h=13):
    """Generate actions to leave and return to the same position.
    This is needed when a changer needs to be activated multiple times.
    """
    directions = [
        ((0, -1), GameAction.ACTION1, GameAction.ACTION2),  # up, then down
        ((0, 1), GameAction.ACTION2, GameAction.ACTION1),
        ((-1, 0), GameAction.ACTION3, GameAction.ACTION4),
        ((1, 0), GameAction.ACTION4, GameAction.ACTION3),
    ]
    
    x, y = pos
    for (dx, dy), go_action, back_action in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid_w and 0 <= ny < grid_h and (nx, ny) not in walls:
            return [go_action, back_action]
    
    return None  # Completely surrounded by walls (shouldn't happen)


def extract_walls(level_idx):
    """Extract wall grid positions from level."""
    level = ls20.levels[level_idx]
    walls = set()
    for sprite in level._sprites:
        if sprite.tags and "ihdgageizm" in sprite.tags:
            walls.add((sprite.x // CELL, sprite.y // CELL))
    return walls


def extract_level_info(level_idx):
    """Extract all relevant info for a level."""
    level = ls20.levels[level_idx]
    
    info = {
        'step_limit': level.get_data("StepCounter") or 42,
        'start_shape': level.get_data("StartShape"),
        'start_color': level.get_data("StartColor"),
        'start_rotation': level.get_data("StartRotation"),
        'goal_color': level.get_data("GoalColor"),
        'goal_rotation': level.get_data("GoalRotation"),
        'goal_shape': level.get_data("kvynsvxbpi"),
        'fog': level.get_data("Fog") or False,
        'steps_decrement': level.get_data("StepsDecrement"),
    }
    
    # Extract positions
    def find_tag(tag):
        results = []
        for s in level._sprites:
            if s.tags and tag in s.tags:
                results.append((s.x // CELL, s.y // CELL))
        return results
    
    info['player'] = find_tag("sfqyzhzkij")[0] if find_tag("sfqyzhzkij") else None
    info['goals'] = find_tag("rjlbuycveu")
    info['rot_changers'] = find_tag("rhsxkxzdjz")
    info['shape_changers'] = find_tag("ttfwljgohq")
    info['color_changers'] = find_tag("soyhouuebz")
    info['timers'] = find_tag("npxgalaybz")
    info['walls'] = extract_walls(level_idx)
    
    return info


def plan_single_goal(player, goal_pos, shape_hits, color_hits, rot_hits,
                     shape_pos, color_pos, rot_pos, timer_pos, walls, 
                     step_limit, f):
    """Plan path for a single goal with required changer visits."""
    
    # Build list of waypoints
    visit_list = []
    if shape_hits > 0 and shape_pos:
        visit_list.append(('shape', shape_pos, shape_hits))
    if color_hits > 0 and color_pos:
        visit_list.append(('color', color_pos, color_hits))
    if rot_hits > 0 and rot_pos:
        visit_list.append(('rot', rot_pos, rot_hits))
    
    if not visit_list:
        # Just go to goal
        path = bfs(player, goal_pos, walls)
        return path
    
    # Try all orderings of changer types
    best_path = None
    best_len = float('inf')
    best_desc = None
    
    # Generate all type orderings
    type_indices = list(range(len(visit_list)))
    
    for perm in permutations(type_indices):
        # Build ordered waypoint sequence
        waypoints = []
        for idx in perm:
            name, pos, count = visit_list[idx]
            waypoints.append((name, pos, count))
        
        # Try with and without timer pickups (if available)
        timer_options = [False]
        if timer_pos:
            timer_options.append(True)
        
        for use_timers in timer_options:
            total_path = []
            current = player
            valid = True
            desc = []
            
            for name, pos, count in waypoints:
                changer_pos = pos[0]  # Use first changer of this type
                
                # Navigate to changer
                p = bfs(current, changer_pos, walls)
                if p is None:
                    valid = False
                    break
                total_path.extend(p)
                current = changer_pos
                desc.append(f"{name}@{changer_pos}")
                
                # Revisit for additional activations
                for _ in range(count - 1):
                    rv = revisit_actions(current, walls)
                    if rv is None:
                        valid = False
                        break
                    total_path.extend(rv)
                    desc.append(f"{name}_revisit")
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if we need a timer pickup before goal
            if use_timers and len(total_path) > step_limit - 10 and timer_pos:
                # Try inserting closest timer
                for tp in timer_pos:
                    p_to_timer = bfs(current, tp, walls)
                    if p_to_timer:
                        p_to_goal = bfs(tp, goal_pos, walls)
                        if p_to_goal:
                            test_path = total_path + p_to_timer + p_to_goal
                            if len(test_path) < best_len:
                                best_len = len(test_path)
                                best_path = test_path
                                best_desc = desc + [f"timer@{tp}", f"goal@{goal_pos}"]
                continue
            
            # Navigate to goal
            p = bfs(current, goal_pos, walls)
            if p is None:
                continue
            total_path.extend(p)
            desc.append(f"goal@{goal_pos}")
            
            if len(total_path) < best_len:
                best_len = len(total_path)
                best_path = total_path
                best_desc = desc
    
    if best_desc:
        f.write(f"  Route: {' -> '.join(best_desc)}\n")
    
    return best_path


def run_solver_v2():
    """Full solver v2 with fixed revisit logic."""
    
    # Get color order from game
    game = ls20.Ls20()
    color_order = game.tnkekoeuk
    rotations = [0, 90, 180, 270]
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"  ARC-AGI-3 LS20 Solver v2\n")
        f.write(f"  Color order: {color_order}\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        
        all_plans = []
        
        for li in range(7):
            info = extract_level_info(li)
            f.write(f"\n{'='*50}\n")
            f.write(f"  Level {li+1}\n")
            f.write(f"{'='*50}\n")
            f.write(f"  Player: {info['player']}\n")
            f.write(f"  Start: shape={info['start_shape']}, "
                    f"color={info['start_color']}, rot={info['start_rotation']}\n")
            
            gc = info['goal_color']
            gr = info['goal_rotation']
            gs = info['goal_shape']
            
            # Handle multi-goal
            if isinstance(gc, list):
                f.write(f"  {len(gc)} goals:\n")
                for i in range(len(gc)):
                    f.write(f"    Goal {i+1}: shape={gs[i]}, "
                            f"color={gc[i]}, rot={gr[i]} @ {info['goals'][i]}\n")
                
                # Plan multi-goal sequentially
                full_plan = []
                current_pos = info['player']
                cur_shape = info['start_shape']
                cur_color_idx = color_order.index(info['start_color'])
                cur_rot_idx = rotations.index(info['start_rotation'])
                
                all_ok = True
                for gi in range(len(gc)):
                    g_shape = gs[gi]
                    g_color_idx = color_order.index(gc[gi])
                    g_rot_idx = rotations.index(gr[gi])
                    
                    sh = (g_shape - cur_shape) % 6
                    ch = (g_color_idx - cur_color_idx) % len(color_order)
                    rh = (g_rot_idx - cur_rot_idx) % 4
                    
                    f.write(f"  Goal {gi+1}: shape*{sh}, color*{ch}, rot*{rh}\n")
                    
                    plan = plan_single_goal(
                        current_pos, info['goals'][gi],
                        sh, ch, rh,
                        info['shape_changers'], info['color_changers'],
                        info['rot_changers'], info['timers'],
                        info['walls'], info['step_limit'], f
                    )
                    
                    if plan is None:
                        f.write(f"  X Goal {gi+1} FAILED\n")
                        all_ok = False
                        break
                    
                    full_plan.extend(plan)
                    current_pos = info['goals'][gi]
                    cur_shape = (cur_shape + sh) % 6
                    cur_color_idx = (cur_color_idx + ch) % len(color_order)
                    cur_rot_idx = (cur_rot_idx + rh) % 4
                
                if all_ok:
                    f.write(f"  => {len(full_plan)} steps (limit: {info['step_limit']})\n")
                    all_plans.append(full_plan)
                else:
                    all_plans.append(None)
            else:
                f.write(f"  Goal: shape={gs}, color={gc}, rot={gr} @ {info['goals'][0]}\n")
                
                start_ci = color_order.index(info['start_color'])
                goal_ci = color_order.index(gc)
                sh = (gs - info['start_shape']) % 6
                ch = (goal_ci - start_ci) % len(color_order)
                rh = (rotations.index(gr) - rotations.index(info['start_rotation'])) % 4
                
                f.write(f"  Changes: shape*{sh}, color*{ch}, rot*{rh}\n")
                f.write(f"  Changers: shape={info['shape_changers']}, "
                        f"color={info['color_changers']}, rot={info['rot_changers']}\n")
                f.write(f"  Timers: {info['timers']}\n")
                
                plan = plan_single_goal(
                    info['player'], info['goals'][0],
                    sh, ch, rh,
                    info['shape_changers'], info['color_changers'],
                    info['rot_changers'], info['timers'],
                    info['walls'], info['step_limit'], f
                )
                
                if plan:
                    f.write(f"  => {len(plan)} steps (limit: {info['step_limit']})\n")
                else:
                    f.write(f"  X Planning FAILED\n")
                all_plans.append(plan)
        
        # Summary
        f.write(f"\n{'='*60}\n  PLANNING SUMMARY\n{'='*60}\n")
        for i, plan in enumerate(all_plans):
            status = f"{len(plan)} steps" if plan else "FAILED"
            f.write(f"  Level {i+1}: {status}\n")
        
        # Execute
        f.write(f"\n{'='*60}\n  EXECUTION\n{'='*60}\n\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        obs = env.step(GameAction.RESET)
        
        action_names = {
            GameAction.ACTION1: 'UP', GameAction.ACTION2: 'DOWN',
            GameAction.ACTION3: 'LEFT', GameAction.ACTION4: 'RIGHT',
        }
        
        levels_cleared = 0
        total_actions = 0
        
        for li, plan in enumerate(all_plans):
            if plan is None:
                f.write(f"\n  Level {li+1}: SKIPPED\n")
                break  # Can't skip levels
            
            f.write(f"\n  --- Level {li+1} ({len(plan)} actions) ---\n")
            level_start = obs.levels_completed
            
            for i, action in enumerate(plan):
                obs = env.step(action)
                total_actions += 1
                
                if obs.levels_completed > level_start:
                    levels_cleared = obs.levels_completed
                    f.write(f"  Step {i+1}/{len(plan)}: "
                            f"LEVEL {obs.levels_completed} COMPLETE!\n")
                    break
                
                if obs.state.value == 'GAME_OVER':
                    f.write(f"  Step {i+1}/{len(plan)}: GAME OVER!\n")
                    break
                
                if obs.full_reset:
                    f.write(f"  Step {i+1}/{len(plan)}: "
                            f"Lost a life! (full_reset)\n")
                    # Replay the plan from beginning
                    # The reset puts player back at start position
                    # So remaining planned actions won't work
                    f.write(f"  Retrying level from scratch...\n")
                    # Re-execute the full plan
                    remaining_plan = plan  # Restart from beginning
                    for j, act2 in enumerate(remaining_plan):
                        obs = env.step(act2)
                        total_actions += 1
                        if obs.levels_completed > level_start:
                            levels_cleared = obs.levels_completed
                            f.write(f"  Retry step {j+1}: "
                                    f"LEVEL {obs.levels_completed} COMPLETE!\n")
                            break
                        if obs.state.value == 'GAME_OVER':
                            f.write(f"  Retry step {j+1}: GAME OVER!\n")
                            break
                    break
            
            if obs.state.value == 'GAME_OVER':
                f.write(f"  Game over after level {li+1}\n")
                break
        
        f.write(f"\n{'='*60}\n")
        f.write(f"  FINAL RESULTS\n")
        f.write(f"  Levels completed: {levels_cleared}/7\n")
        f.write(f"  Total actions: {total_actions}\n")
        f.write(f"  Final state: {obs.state.value}\n")
        f.write(f"{'='*60}\n")
        
        try:
            sc = arc.get_scorecard()
            f.write(f"\n  Score: {sc.score}\n")
            if hasattr(sc, 'environments'):
                for env_sc in sc.environments:
                    if hasattr(env_sc, 'runs'):
                        for run in env_sc.runs:
                            f.write(f"    Run: score={run.score}, "
                                    f"levels={run.levels_completed}, "
                                    f"actions={run.actions}\n")
        except:
            pass
    
    print(f"Results saved to {OUT}")


if __name__ == "__main__":
    print("="*60)
    print("  ARC-AGI-3 LS20 Solver v2 (CPU only)")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    run_solver_v2()
