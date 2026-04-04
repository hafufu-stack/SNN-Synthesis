"""
ARC-AGI-3 LS20 Solver v13 — Dijkstra + Danger Heatmap
=======================================================
KEY UPGRADE: Replace BFS with cost-weighted Dijkstra.
When pushed by an enemy, increase the cost of cells on that
enemy's patrol line. Solver will then autonomously find 
alternative safer routes.

Also logs trajectory data for future LLM training (Mission 2 prep).
"""

import arc_agi
from arcengine import GameAction
import time
import sys
import heapq
from collections import deque, defaultdict
from itertools import permutations
import json

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
import ls20

OUT = r"c:\tmp\arc_solver_v13.txt"
TRAJECTORY_OUT = r"c:\tmp\arc_trajectories.jsonl"


class DijkstraSolver:
    def __init__(self, f, traj_f=None):
        self.f = f
        self.traj_f = traj_f  # trajectory log file
        self.color_order = None
        self.total_actions = 0
        self.danger_map = defaultdict(float)  # (x,y) -> danger score
        self.replan_log = []  # Mission 3: Aha! moments
    
    def get_level_info(self, level_idx):
        level = ls20.levels[level_idx]
        info = {
            'walls': set(), 'start': None, 'goals': [], 'rot_ch': [],
            'shape_ch': [], 'color_ch': [], 'timers': [], 'interactive': set(),
            'timer_pickups': {}, 'budget': 21, 'dec': 2, 'level': level,
            'enemies': [], 'enemy_cells': set(), 'push_map': {}
        }
        
        # Raw wall+goal positions (for push distance calculation)
        fjzuynaokm = set()
        
        for s in level._sprites:
            if not s.tags: continue
            if "ihdgageizm" in s.tags:
                info['walls'].add((s.x, s.y))
                fjzuynaokm.add((s.x, s.y))
            if "sfqyzhzkij" in s.tags: info['start'] = (s.x, s.y)
            if "rjlbuycveu" in s.tags:
                info['goals'].append((s.x, s.y))
                fjzuynaokm.add((s.x, s.y))
            if "rhsxkxzdjz" in s.tags: info['rot_ch'].append((s.x, s.y))
            if "ttfwljgohq" in s.tags: info['shape_ch'].append((s.x, s.y))
            if "soyhouuebz" in s.tags: info['color_ch'].append((s.x, s.y))
            if "npxgalaybz" in s.tags: info['timers'].append((s.x, s.y))
        
        # Enemy detection — enemies are STATIC traps at fixed positions
        px_off = info['start'][0] % 5 if info['start'] else 4
        py_off = info['start'][1] % 5 if info['start'] else 0
        
        for s in level._sprites:
            name = s.name if hasattr(s, 'name') else ""
            if s.tags and "gbvqrjtaqo" in s.tags:
                dx, dy = 0, 0
                if name.endswith("_r"): dx = 1
                elif name.endswith("_l"): dx = -1
                elif name.endswith("_t"): dy = -1
                elif name.endswith("_b"): dy = 1
                info['enemies'].append({'name': name, 'x': s.x, 'y': s.y, 'dx': dx, 'dy': dy})
                
                ex, ey = s.x, s.y
                
                # Compute push distance (simulate ullzqnksoj)
                wall_cx = ex + dx
                wall_cy = ey + dy
                push_dist = 0
                for d in range(1, 12):
                    nx = wall_cx + dx * 5 * d
                    ny = wall_cy + dy * 5 * d
                    if (nx, ny) in fjzuynaokm:
                        push_dist = max(0, d - 1)
                        break
                
                push_delta = (dx * 5 * push_dist, dy * 5 * push_dist)
                
                # Map enemy to player grid cells and compute push destinations
                for px in range(px_off, 60, 5):
                    for py in range(py_off, 60, 5):
                        if (px < ex + 5 and px + 5 > ex and py < ey + 5 and py + 5 > ey):
                            info['enemy_cells'].add((px, py))
                            # Push teleports player from (px,py) to destination
                            dest = (px + push_delta[0], py + push_delta[1])
                            # Only valid if destination is in bounds
                            if 0 <= dest[0] < 60 and 0 <= dest[1] < 60:
                                info['push_map'][(px, py)] = dest
        
        for pos in info['rot_ch'] + info['shape_ch'] + info['color_ch'] + info['goals']:
            info['interactive'].add(pos)
        
        sc = level.get_data("StepCounter") or 42
        sd = level.get_data("StepsDecrement")
        info['dec'] = sd if sd is not None else 2
        info['budget'] = sc // info['dec']
        
        # Timer pickups
        for tx, ty in info['timers']:
            pps = set()
            for px in range(px_off, 60, 5):
                for py in range(py_off, 60, 5):
                    if tx >= px and tx < px + 5 and ty >= py and ty < py + 5:
                        if not self._blocked(info['walls'], px, py):
                            pps.add((px, py))
            info['timer_pickups'][(tx, ty)] = pps
        
        return info
    
    def _blocked(self, walls, px, py):
        for wx, wy in walls:
            if wx >= px and wx < px + 5 and wy >= py and wy < py + 5: return True
        return False
    
    def _dijkstra(self, walls, sx, sy, tx, ty, avoid=None):
        """Dijkstra with danger-weighted costs."""
        if sx == tx and sy == ty: return []
        avoid_set = avoid or set()
        pm = self.push_map if hasattr(self, 'push_map') else {}
        
        # Priority queue: (cost, counter, x, y, path)
        counter = 0
        heap = [(0, counter, sx, sy, [])]
        visited = {}  # (x,y) -> best cost
        dirs = [(0,-5,GameAction.ACTION1),(0,5,GameAction.ACTION2),
                (-5,0,GameAction.ACTION3),(5,0,GameAction.ACTION4)]
        
        while heap:
            cost, _, cx, cy, path = heapq.heappop(heap)
            
            if (cx, cy) in visited and visited[(cx, cy)] <= cost:
                continue
            visited[(cx, cy)] = cost
            
            if cx == tx and cy == ty:
                return path
            
            for dx, dy, act in dirs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < 60 and 0 <= ny < 60): continue
                if self._blocked(walls, nx, ny): continue
                if (nx, ny) in avoid_set and not (nx == tx and ny == ty):
                    continue
                
                # Check push chain: if entering enemy cell, simulate full chain
                final_x, final_y = nx, ny
                push_chain_len = 0
                while (final_x, final_y) in pm and (final_x, final_y) != (tx, ty):
                    dest = pm[(final_x, final_y)]
                    final_x, final_y = dest
                    push_chain_len += 1
                    if push_chain_len > 10: break  # safety
                
                if push_chain_len > 0:
                    # Single push = normal cost (push is free teleport!)
                    # Chain push (2+ links) = heavily penalized
                    chain_penalty = max(0, (push_chain_len - 1)) * 100
                    step_cost = 1 + chain_penalty + self.danger_map.get((final_x, final_y), 0) * 20
                    new_cost = cost + step_cost
                    if (final_x, final_y) not in visited or visited[(final_x, final_y)] > new_cost:
                        counter += 1
                        heapq.heappush(heap, (new_cost, counter, final_x, final_y, path + [act]))
                else:
                    # Normal step
                    step_cost = 1 + self.danger_map[(nx, ny)] * 20
                    new_cost = cost + step_cost
                    if (nx, ny) not in visited or visited[(nx, ny)] > new_cost:
                        counter += 1
                        heapq.heappush(heap, (new_cost, counter, nx, ny, path + [act]))
        
        return None
    
    def _bfs_fallback(self, walls, sx, sy, tx, ty, avoid=None, push_map=None):
        """BFS with optional push teleport support."""
        if sx == tx and sy == ty: return []
        avoid_set = avoid or set()
        pm = push_map or {}
        queue = deque([((sx, sy), [])]); vis = {(sx, sy)}
        dirs = [(0,-5,GameAction.ACTION1),(0,5,GameAction.ACTION2),
                (-5,0,GameAction.ACTION3),(5,0,GameAction.ACTION4)]
        while queue:
            (cx, cy), path = queue.popleft()
            for dx, dy, act in dirs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < 60 and 0 <= ny < 60): continue
                if self._blocked(walls, nx, ny): continue
                
                # Check if stepping into enemy cell with push
                if (nx, ny) in pm and (nx, ny) != (tx, ty):
                    # Push teleports us to destination
                    dest = pm[(nx, ny)]
                    if dest not in vis and 0 <= dest[0] < 60 and 0 <= dest[1] < 60:
                        if dest == (tx, ty): return path + [act]
                        if dest not in avoid_set:
                            vis.add(dest)
                            queue.append((dest, path + [act]))
                    continue
                
                if (nx, ny) in vis: continue
                if nx == tx and ny == ty: return path + [act]
                if (nx, ny) in avoid_set: continue
                vis.add((nx, ny)); queue.append(((nx, ny), path + [act]))
        return None
    
    def _revisit(self, walls, px, py, avoid=None):
        avoid_set = avoid or set()
        dirs = [(0,-5,GameAction.ACTION1,GameAction.ACTION2),(0,5,GameAction.ACTION2,GameAction.ACTION1),
                (-5,0,GameAction.ACTION3,GameAction.ACTION4),(5,0,GameAction.ACTION4,GameAction.ACTION3)]
        # Prefer low-danger directions
        scored = []
        for dx,dy,go,back in dirs:
            nx,ny = px+dx,py+dy
            if 0<=nx<60 and 0<=ny<60 and not self._blocked(walls,nx,ny) and (nx,ny) not in avoid_set:
                scored.append((self.danger_map[(nx,ny)], [go, back]))
        scored.sort(key=lambda x: x[0])
        return scored[0][1] if scored else None
    
    def _mark_danger(self, info, pushed_from, pushed_to, expected_target=None):
        """Add danger to the pushed-from cell and nearby cells."""
        # Heavily penalize the exact cell where push happened
        self.danger_map[pushed_from] += 10.0
        
        # Also penalize adjacent cells in the push direction
        pfx, pfy = pushed_from
        ptx, pty = pushed_to
        push_dx = 1 if ptx > pfx else (-1 if ptx < pfx else 0)
        push_dy = 1 if pty > pfy else (-1 if pty < pfy else 0)
        # Penalize cells near push-from in the approach direction
        for offset in range(-2, 3):
            if push_dx != 0:  # horizontal push
                cx = pfx + offset * 5
                if 0 <= cx < 60:
                    self.danger_map[(cx, pfy)] += 3.0
            if push_dy != 0:  # vertical push
                cy = pfy + offset * 5
                if 0 <= cy < 60:
                    self.danger_map[(pfx, cy)] += 3.0
        
        # If stuck (pushed to same position), block the expected target cell
        if pushed_from == pushed_to and expected_target:
            self.danger_map[expected_target] += 100.0  # effectively impassable
    
    def _execute_to(self, env, game, info, sx, sy, tx, ty, start_lvl, max_retries=30):
        """Navigate using Dijkstra, adapting to pushes."""
        action_map = {GameAction.ACTION1:(0,-5), GameAction.ACTION2:(0,5),
                      GameAction.ACTION3:(-5,0), GameAction.ACTION4:(5,0)}
        walls = info['walls']
        interactive = info['interactive']
        cur_x, cur_y = sx, sy
        consecutive_stuck = 0
        last_stuck_pos = None
        
        for retry in range(max_retries):
            # Dijkstra handles enemy cells via push_map (teleport), not avoidance
            avoid = interactive - {(tx, ty)}
            path = self._dijkstra(walls, cur_x, cur_y, tx, ty, avoid)
            if path is None:
                path = self._bfs_fallback(walls, cur_x, cur_y, tx, ty, avoid)
                if path is None:
                    path = self._bfs_fallback(walls, cur_x, cur_y, tx, ty)
                    if path is None:
                        return None, cur_x, cur_y
            
            pushed = False
            for act_i, act in enumerate(path):
                dx, dy = action_map[act]
                expected_x, expected_y = cur_x + dx, cur_y + dy
                
                obs = env.step(act)
                self.total_actions += 1
                
                actual_x = game.gudziatsk.x
                actual_y = game.gudziatsk.y
                
                # Log trajectory
                if self.traj_f:
                    self._log_step(game, info, cur_x, cur_y, act, actual_x, actual_y,
                                   expected_x, expected_y, start_lvl)
                
                if obs.levels_completed > start_lvl:
                    return 'COMPLETE', actual_x, actual_y
                if obs.state.value == 'GAME_OVER':
                    return 'GAME_OVER', actual_x, actual_y
                if obs.full_reset:
                    return 'LIFE_LOST', actual_x, actual_y
                
                if actual_x != expected_x or actual_y != expected_y:
                    # PUSHED! Wait for animation to complete
                    for anim in range(20):
                        if not game.euemavvxz: break
                        obs = env.step(act)
                        self.total_actions += 1
                        if obs.levels_completed > start_lvl:
                            return 'COMPLETE', game.gudziatsk.x, game.gudziatsk.y
                        if obs.state.value == 'GAME_OVER':
                            return 'GAME_OVER', game.gudziatsk.x, game.gudziatsk.y
                        if obs.full_reset:
                            return 'LIFE_LOST', game.gudziatsk.x, game.gudziatsk.y
                    
                    pushed_from = (cur_x, cur_y)
                    cur_x, cur_y = game.gudziatsk.x, game.gudziatsk.y
                    is_stuck = (pushed_from == (cur_x, cur_y))
                    
                    # Track consecutive stucks at same position
                    if is_stuck:
                        if last_stuck_pos == pushed_from:
                            consecutive_stuck += 1
                        else:
                            consecutive_stuck = 1
                            last_stuck_pos = pushed_from
                        # After 3 consecutive stucks, bail out immediately
                        if consecutive_stuck >= 3:
                            self.f.write(f"      STUCK x{consecutive_stuck} at {pushed_from}, bailing\n")
                            return 'TIMEOUT', cur_x, cur_y
                    else:
                        consecutive_stuck = 0
                        last_stuck_pos = None
                    
                    # Mark danger
                    self._mark_danger(info, pushed_from, (cur_x, cur_y),
                                      expected_target=(expected_x, expected_y))
                    self.replan_log.append({
                        'step': self.total_actions,
                        'level': start_lvl + 1,
                        'pushed_from': pushed_from,
                        'pushed_to': (cur_x, cur_y),
                        'target': (tx, ty),
                        'type': 'PUSH_STUCK' if is_stuck else 'PUSH_REPLAN'
                    })
                    stuck_str = " STUCK!" if is_stuck else ""
                    self.f.write(f"      Push ({pushed_from})->({cur_x},{cur_y}){stuck_str}, "
                                f"replan #{retry+1}\n")
                    pushed = True
                    break
                
                cur_x, cur_y = actual_x, actual_y
            
            if not pushed:
                if cur_x == tx and cur_y == ty:
                    return 'OK', cur_x, cur_y
                continue
        
        return 'TIMEOUT', cur_x, cur_y
    
    def _execute_revisit(self, env, game, info, px, py, start_lvl):
        walls = info['walls']
        interactive = info['interactive']
        avoid = interactive - {(px, py)}
        rv = self._revisit(walls, px, py, avoid)
        if rv is None: return None, px, py
        
        cur_x, cur_y = px, py
        for act in rv:
            obs = env.step(act)
            self.total_actions += 1
            cur_x, cur_y = game.gudziatsk.x, game.gudziatsk.y
            if obs.levels_completed > start_lvl: return 'COMPLETE', cur_x, cur_y
            if obs.state.value == 'GAME_OVER': return 'GAME_OVER', cur_x, cur_y
            if obs.full_reset: return 'LIFE_LOST', cur_x, cur_y
        return 'OK', cur_x, cur_y
    
    def _log_step(self, game, info, from_x, from_y, action, to_x, to_y, exp_x, exp_y, lvl):
        act_names = {GameAction.ACTION1:'UP', GameAction.ACTION2:'DOWN',
                     GameAction.ACTION3:'LEFT', GameAction.ACTION4:'RIGHT'}
        pushed = (to_x != exp_x or to_y != exp_y)
        entry = {
            'step': self.total_actions,
            'level': lvl + 1,
            'from': [from_x, from_y],
            'to': [to_x, to_y],
            'action': act_names.get(action, '?'),
            'pushed': pushed,
            'steps_remaining': game._step_counter_ui.current_steps,
            'shape': game.fwckfzsyc,
            'color': game.hiaauhahz,
            'rot': game.cklxociuu,
        }
        if pushed:
            entry['expected'] = [exp_x, exp_y]
            entry['event'] = f"PUSHED from ({from_x},{from_y}) to ({to_x},{to_y})"
        self.traj_f.write(json.dumps(entry) + '\n')
    
    def solve_level(self, env, game, level_idx, start_lvl):
        info = self.get_level_info(level_idx)
        level = info['level']
        self.push_map = info.get('push_map', {})  # For Dijkstra push chain awareness
        
        gc = level.get_data("GoalColor")
        gr = level.get_data("GoalRotation")
        gs = level.get_data("kvynsvxbpi")
        rotations = [0, 90, 180, 270]
        
        if isinstance(gc, list):
            goal_list = [{'shape': gs[j], 'color': gc[j], 'rot': gr[j], 'pos': info['goals'][j]} for j in range(len(gc))]
        else:
            goal_list = [{'shape': gs, 'color': gc, 'rot': gr, 'pos': info['goals'][0]}]
        
        self.f.write(f"\n  Level {level_idx+1}: budget={info['budget']}, dec={info['dec']}, "
                     f"enemies={len(info['enemies'])}\n")
        
        # Reset danger map per level (enemies change)
        self.danger_map = defaultdict(float)
        used_timers = set()
        
        for gi, goal in enumerate(goal_list):
            # Re-read actual game state (critical after pushes that trigger changers)
            for attempt in range(3):
                cur_shape = game.fwckfzsyc
                cur_ci = game.hiaauhahz
                cur_ri = game.cklxociuu
                
                g_ci = self.color_order.index(goal['color'])
                g_ri = rotations.index(goal['rot'])
                sh = (goal['shape'] - cur_shape) % 6
                ch = (g_ci - cur_ci) % len(self.color_order)
                rh = (g_ri - cur_ri) % 4
                
                self.f.write(f"    Goal {gi+1}: need shape*{sh} color*{ch} rot*{rh}"
                             f" (cur: s={cur_shape} c={cur_ci} r={cur_ri})\n")
                
                changers = []
                if sh > 0 and info['shape_ch']: changers.append(('shape', info['shape_ch'][0], sh))
                if ch > 0 and info['color_ch']: changers.append(('color', info['color_ch'][0], ch))
                if rh > 0 and info['rot_ch']: changers.append(('rot', info['rot_ch'][0], rh))
                
                best = self._plan_goal(info, game.gudziatsk.x, game.gudziatsk.y,
                                        changers, goal['pos'], used_timers)
                if best is None:
                    self.f.write(f"      NO ROUTE found\n")
                    return False, None
                
                waypoints, desc, t_used = best
                self.f.write(f"      Route: {' -> '.join(desc)}\n")
                
                cur_x, cur_y = game.gudziatsk.x, game.gudziatsk.y
                state_changed = False
                
                for wp_idx, (wp_name, wp_pos, wp_count) in enumerate(waypoints):
                    # Snapshot state before this segment  
                    pre_shape = game.fwckfzsyc
                    pre_ci = game.hiaauhahz
                    pre_ri = game.cklxociuu
                    
                    result, cur_x, cur_y = self._execute_to(env, game, info,
                                                              cur_x, cur_y, wp_pos[0], wp_pos[1],
                                                              start_lvl)
                    if result == 'COMPLETE': return True, None
                    if result in ('GAME_OVER', 'LIFE_LOST'):
                        self.f.write(f"      {result} at ({cur_x},{cur_y})\n")
                        return False, None
                    
                    if result not in ('OK',):
                        # Check if state changed during failed navigation
                        if (game.fwckfzsyc != pre_shape or game.hiaauhahz != pre_ci 
                            or game.cklxociuu != pre_ri):
                            self.f.write(f"      State changed during {wp_name} nav! Re-planning...\n")
                            state_changed = True
                            break
                        self.f.write(f"      {wp_name}@{wp_pos}: {result}\n")
                        return False, None
                    
                    # Check if push altered state unexpectedly
                    post_shape = game.fwckfzsyc
                    post_ci = game.hiaauhahz
                    post_ri = game.cklxociuu
                    
                    if wp_name == 'TIMER':
                        # Timer shouldn't change state, but pushes during nav might have
                        if post_shape != pre_shape or post_ci != pre_ci or post_ri != pre_ri:
                            self.f.write(f"      State changed by push during T nav! Re-planning...\n")
                            state_changed = True
                            break
                    else:
                        # Revisit for changer
                        for rv_i in range(wp_count - 1):
                            result, cur_x, cur_y = self._execute_revisit(env, game, info,
                                                                           cur_x, cur_y, start_lvl)
                            if result == 'COMPLETE': return True, None
                            if result in ('GAME_OVER', 'LIFE_LOST'): return False, None
                
                if state_changed:
                    continue  # Retry with re-read state
                
                # All waypoints done, go to goal
                result, cur_x, cur_y = self._execute_to(env, game, info,
                                                          cur_x, cur_y, goal['pos'][0], goal['pos'][1],
                                                          start_lvl)
                if result == 'COMPLETE': return True, None
                if result in ('GAME_OVER', 'LIFE_LOST'):
                    self.f.write(f"      {result} going to goal\n")
                    return False, None
                if result not in ('OK',):
                    # Goal unreachable - maybe state is wrong, re-plan
                    if attempt < 2:
                        self.f.write(f"      Goal unreachable ({result}), re-reading state...\n")
                        continue
                    return False, None
                
                used_timers.update(t_used)
                break  # Goal reached successfully
        
        # Check completion
        obs_check = env.step(GameAction.ACTION1)
        self.total_actions += 1
        if obs_check.levels_completed > start_lvl:
            return True, obs_check
        return False, obs_check
    
    def _plan_goal(self, info, sx, sy, changers, goal, used_timers):
        """Find best route using BFS for planning (Dijkstra used only during execution)."""
        best = None; best_len = float('inf')
        walls = info['walls']; interactive = info['interactive']
        enemy_cells = info.get('enemy_cells', set())
        push_map = info.get('push_map', {})
        avail_timers = [t for t in info['timers'] if t not in used_timers]
        tp_pp = {tp: info['timer_pickups'].get(tp, set()) for tp in avail_timers}
        
        perms = list(permutations(range(len(changers)))) if changers else [()]
        
        for perm in perms:
            ordered = [changers[i] for i in perm] if perm else []
            
            r = self._calc_route(walls, interactive, enemy_cells, push_map, sx, sy, ordered, goal, info['budget'])
            if r and r[0] < best_len:
                best_len = r[0]; best = (ordered, r[1], set())
            
            for tp in avail_timers:
                for pp in tp_pp.get(tp, set()):
                    for ins in range(len(ordered) + 1):
                        mod = list(ordered); mod.insert(ins, ('TIMER', pp, 1))
                        r = self._calc_route(walls, interactive, enemy_cells, push_map, sx, sy, mod, goal, info['budget'])
                        if r and r[0] < best_len:
                            best_len = r[0]; best = (mod, r[1], {tp})
            
            if len(avail_timers) >= 2:
                for i, tp1 in enumerate(avail_timers):
                    for j, tp2 in enumerate(avail_timers):
                        if i == j: continue
                        for pp1 in tp_pp.get(tp1, set()):
                            for pp2 in tp_pp.get(tp2, set()):
                                for ins1 in range(len(ordered) + 1):
                                    for ins2 in range(ins1, len(ordered) + 2):
                                        mod = list(ordered)
                                        mod.insert(ins2, ('TIMER', pp2, 1))
                                        mod.insert(ins1, ('TIMER', pp1, 1))
                                        r = self._calc_route(walls, interactive, enemy_cells, push_map, sx, sy, mod, goal, info['budget'])
                                        if r and r[0] < best_len:
                                            best_len = r[0]; best = (mod, r[1], {tp1, tp2})
        # Fallback: if no route found with enemy avoidance, try without
        if best is None:
            self.f.write(f"      No safe route, trying with enemy cells allowed...\n")
            for perm in perms:
                ordered = [changers[i] for i in perm] if perm else []
                
                r = self._calc_route(walls, interactive, set(), push_map, sx, sy, ordered, goal, info['budget'])
                if r and r[0] < best_len:
                    best_len = r[0]; best = (ordered, r[1], set())
                
                for tp in avail_timers:
                    for pp in tp_pp.get(tp, set()):
                        for ins in range(len(ordered) + 1):
                            mod = list(ordered); mod.insert(ins, ('TIMER', pp, 1))
                            r = self._calc_route(walls, interactive, set(), push_map, sx, sy, mod, goal, info['budget'])
                            if r and r[0] < best_len:
                                best_len = r[0]; best = (mod, r[1], {tp})
                
                if len(avail_timers) >= 2:
                    for i, tp1 in enumerate(avail_timers):
                        for j, tp2 in enumerate(avail_timers):
                            if i == j: continue
                            for pp1 in tp_pp.get(tp1, set()):
                                for pp2 in tp_pp.get(tp2, set()):
                                    for ins1 in range(len(ordered) + 1):
                                        for ins2 in range(ins1, len(ordered) + 2):
                                            mod = list(ordered)
                                            mod.insert(ins2, ('TIMER', pp2, 1))
                                            mod.insert(ins1, ('TIMER', pp1, 1))
                                            r = self._calc_route(walls, interactive, set(), push_map, sx, sy, mod, goal, info['budget'])
                                            if r and r[0] < best_len:
                                                best_len = r[0]; best = (mod, r[1], {tp1, tp2})
        
        return best
    
    def _calc_route(self, walls, interactive, enemy_cells, push_map, sx, sy, visits, goal, budget):
        total = 0; cur_x, cur_y = sx, sy; seg = 0; desc = []
        for name, pos, count in visits:
            avoid = (interactive | enemy_cells) - {pos}
            p = self._bfs_fallback(walls, cur_x, cur_y, pos[0], pos[1], avoid)
            if p is None: return None
            d = len(p)
            if name == 'TIMER':
                desc.append(f"T@{pos}"); seg += max(0, d - 1)
                if seg > budget: return None
                seg = 0
            else:
                desc.append(f"{name}@{pos}"); seg += d + (count - 1) * 2
            total += d + (0 if name == 'TIMER' else (count - 1) * 2)
            cur_x, cur_y = pos
        avoid = (interactive | enemy_cells) - {goal}
        p = self._bfs_fallback(walls, cur_x, cur_y, goal[0], goal[1], avoid)
        if p is None: return None
        seg += len(p); total += len(p); desc.append(f"goal@{goal}")
        if seg > budget: return None
        return total, desc


def run():
    with open(OUT, "w", encoding="utf-8") as f, \
         open(TRAJECTORY_OUT, "w", encoding="utf-8") as traj_f:
        f.write(f"{'='*60}\n  LS20 Solver v13 (Dijkstra + Danger Heatmap)\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        obs = env.step(GameAction.RESET)
        game = env._game
        
        solver = DijkstraSolver(f, traj_f)
        solver.color_order = game.tnkekoeuk
        
        for li in range(7):
            start_lvl = obs.levels_completed
            success, _ = solver.solve_level(env, game, li, start_lvl)
            
            obs_check = env.step(GameAction.ACTION1)
            solver.total_actions += 1
            completed = obs_check.levels_completed
            
            f.write(f"  -> Level {li+1}: {'CLEARED' if completed > start_lvl else 'FAILED'} "
                    f"(completed={completed})\n")
            
            if obs_check.state.value == 'GAME_OVER':
                f.write(f"  GAME OVER\n"); break
            if completed <= start_lvl:
                f.write(f"  STOPPED (could not clear)\n"); break
            obs = obs_check
        
        # Summary
        final = obs_check if 'obs_check' in dir() else obs
        f.write(f"\n{'='*60}\n  FINAL: {final.levels_completed}/7, "
                f"{solver.total_actions} actions, {final.state.value}\n{'='*60}\n")
        try:
            sc = arc.get_scorecard(); f.write(f"  Score: {sc.score}\n")
        except: pass
        
        # Aha! moments summary
        if solver.replan_log:
            f.write(f"\n  Aha! Moments (Re-plan events): {len(solver.replan_log)}\n")
            for ev in solver.replan_log:
                f.write(f"    Step {ev['step']}: Lv{ev['level']} "
                        f"{ev['pushed_from']}->{ev['pushed_to']} "
                        f"(target={ev['target']})\n")
    
    print(f"Results: {OUT}")
    print(f"Trajectories: {TRAJECTORY_OUT}")

if __name__ == "__main__":
    print("ARC-AGI-3 LS20 Solver v13 (Dijkstra + Danger Heatmap)")
    run()
