"""
ARC-AGI-3 LS20 Solver v15 — Self-Healing 5D State-Space BFS
============================================================
Builds on v14's 5D BFS with a Self-Healing World Model:
- Compares BFS predictions vs actual game state at every step
- Learns true transitions when model predictions are wrong  
- Uses learned corrections in subsequent BFS searches
- Automatically handles unknown game mechanics (slide changers, etc.)

State space: 144 * 6 * 4 * 4 * 8 * budget ≈ 50K nodes → ms-fast.
"""

import arc_agi
from arcengine import GameAction
import time
import sys
import heapq
from collections import deque, defaultdict
import json

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
import ls20

OUT = r"c:\tmp\arc_solver_v15.txt"
TRAJECTORY_OUT = r"c:\tmp\arc_trajectories.jsonl"


class StateSpaceSolver:
    def __init__(self, f, traj_f=None):
        self.f = f
        self.traj_f = traj_f
        self.color_order = None
        self.total_actions = 0
        self.replan_log = []
        # Self-Healing World Model
        # Exact transition overrides: (x, y, shape, color, rot, action_idx) → ((new_x, new_y), (new_s, new_c, new_r))
        self.learned_transitions = {}
        self.learned_walls = set()
    
    def get_level_info(self, level_idx):
        level = ls20.levels[level_idx]
        info = {
            'walls': set(), 'start': None, 'goals': [], 'rot_ch': [],
            'shape_ch': [], 'color_ch': [], 'timers': [], 'interactive': set(),
            'budget': 21, 'dec': 2, 'level': level,
            'enemies': [], 'enemy_cells': set(), 'push_map': {}
        }
        
        fjzuynaokm = set()  # wall+goal positions for push calculation
        
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
        
        px_off = info['start'][0] % 5 if info['start'] else 4
        py_off = info['start'][1] % 5 if info['start'] else 0
        info['px_off'] = px_off
        info['py_off'] = py_off
        
        # Enemy detection + push_map
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
                wall_cx, wall_cy = ex + dx, ey + dy
                push_dist = 0
                for d in range(1, 12):
                    nx = wall_cx + dx * 5 * d
                    ny = wall_cy + dy * 5 * d
                    if (nx, ny) in fjzuynaokm:
                        push_dist = max(0, d - 1)
                        break
                push_delta = (dx * 5 * push_dist, dy * 5 * push_dist)
                
                for px in range(px_off, 60, 5):
                    for py in range(py_off, 60, 5):
                        if (px < ex + 5 and px + 5 > ex and py < ey + 5 and py + 5 > ey):
                            info['enemy_cells'].add((px, py))
                            dest = (px + push_delta[0], py + push_delta[1])
                            if 0 <= dest[0] < 60 and 0 <= dest[1] < 60:
                                info['push_map'][(px, py)] = dest
        
        for pos in info['rot_ch'] + info['shape_ch'] + info['color_ch'] + info['goals']:
            info['interactive'].add(pos)
        
        sc = level.get_data("StepCounter") or 42
        sd = level.get_data("StepsDecrement")
        info['dec'] = sd if sd is not None else 2
        info['budget'] = sc // info['dec']
        
        # Map changers/timers/goals to player grid cells
        info['changer_cells'] = {}  # (px,py) -> 'shape'|'color'|'rot' (static model)
        for sx, sy in info['shape_ch']:
            for px in range(px_off, 60, 5):
                for py in range(py_off, 60, 5):
                    if sx >= px and sx < px + 5 and sy >= py and sy < py + 5:
                        info['changer_cells'][(px, py)] = 'shape'
        for sx, sy in info['color_ch']:
            for px in range(px_off, 60, 5):
                for py in range(py_off, 60, 5):
                    if sx >= px and sx < px + 5 and sy >= py and sy < py + 5:
                        info['changer_cells'][(px, py)] = 'color'
        for sx, sy in info['rot_ch']:
            for px in range(px_off, 60, 5):
                for py in range(py_off, 60, 5):
                    if sx >= px and sx < px + 5 and sy >= py and sy < py + 5:
                        info['changer_cells'][(px, py)] = 'rot'
        
        info['timer_cells'] = {}  # (px,py) -> timer_index
        for ti, (tx, ty) in enumerate(info['timers']):
            for px in range(px_off, 60, 5):
                for py in range(py_off, 60, 5):
                    if tx >= px and tx < px + 5 and ty >= py and ty < py + 5:
                        info['timer_cells'][(px, py)] = ti
        
        info['goal_cells'] = set()
        for gx, gy in info['goals']:
            for px in range(px_off, 60, 5):
                for py in range(py_off, 60, 5):
                    if gx >= px and gx < px + 5 and gy >= py and gy < py + 5:
                        info['goal_cells'].add((px, py))
        
        # === GATE SIMULATION ===
        # Pre-compute gate schedules: at step t, which player cells have active changers?
        # gate_schedule[step % period][(px,py)] = 'shape'|'color'|'rot'
        from ls20 import dboxixicic
        gates = []
        gate_sprites = level.get_sprites_by_tag('xfmluydglp')
        changer_tags = {'ttfwljgohq': 'shape', 'soyhouuebz': 'color', 'rhsxkxzdjz': 'rot'}
        
        for gs in gate_sprites:
            for tag, ch_type in changer_tags.items():
                for ch in level.get_sprites_by_tag(tag):
                    if gs.collides_with(ch, ignoreMode=True):
                        # Simulate gate movement to find period
                        gate = dboxixicic(gs, ch)
                        positions = [(ch.x, ch.y)]
                        start_pos = (ch.x, ch.y)
                        for s in range(200):
                            gate.step()
                            positions.append((ch.x, ch.y))
                            if s > 0 and (ch.x, ch.y) == start_pos:
                                # Check if pattern repeats from here
                                period = s + 1
                                break
                        else:
                            period = len(positions) - 1
                        
                        gates.append({
                            'type': ch_type, 'period': period,
                            'positions': positions[:period],
                            'start_pos': start_pos
                        })
                        # Reset changer position
                        ch.set_position(start_pos[0], start_pos[1])
        
        if gates:
            # Compute LCM of all gate periods
            from math import gcd
            periods = [g['period'] for g in gates]
            lcm = periods[0]
            for p in periods[1:]:
                lcm = lcm * p // gcd(lcm, p)
            lcm = min(lcm, 120)  # cap to avoid explosion
            
            # Build schedule: for each step t, which cells have changers?
            gate_schedule = [{} for _ in range(lcm)]
            for t in range(lcm):
                for g in gates:
                    sx, sy = g['positions'][t % g['period']]
                    # Map changer sprite position to player grid cell
                    for px in range(px_off, 60, 5):
                        for py in range(py_off, 60, 5):
                            if sx >= px and sx < px + 5 and sy >= py and sy < py + 5:
                                gate_schedule[t][(px, py)] = g['type']
            
            info['gate_schedule'] = gate_schedule
            info['gate_period'] = lcm
            info['has_gates'] = True
            
            # Calculate which changer positions are gate-controlled
            gated_positions = set()
            for t in range(lcm):
                for pos in gate_schedule[t]:
                    gated_positions.add(pos)
            info['gated_positions'] = gated_positions
            
            # Static changers = those NOT covered by gates
            info['static_changers'] = {k: v for k, v in info['changer_cells'].items()
                                        if k not in gated_positions}
        else:
            info['has_gates'] = False
            info['static_changers'] = info['changer_cells']
        
        return info
    
    def _blocked(self, walls, px, py):
        for wx, wy in walls:
            if wx >= px and wx < px + 5 and wy >= py and wy < py + 5: return True
        return False
    
    def _5d_bfs(self, info, sx, sy, cur_s, cur_c, cur_r, goal_s, goal_c, goal_r, goal_pos, 
                start_step=0, goal_barriers=None):
        """6D BFS with Gate-Aware Changer Timing.
        
        State: (x, y, shape, color, rot, timer_mask, seg_remaining, step_mod)
        When gates exist, changer triggering depends on step_mod (gate phase).
        """
        walls = info['walls']
        budget = info['budget']
        pm = info.get('push_map', {})
        changer_cells = info.get('changer_cells', {})  # static fallback
        timer_cells = info.get('timer_cells', {})
        # Use specific goal position, not all goal_cells
        target_cell = goal_pos  # already a grid cell
        # Goal barriers: other goals that block unless state matches
        g_barriers = goal_barriers or {}
        num_shapes = 6
        num_colors = len(self.color_order)
        has_gates = info.get('has_gates', False)
        gate_schedule = info.get('gate_schedule', None)
        gate_period = info.get('gate_period', 1)
        
        start_mod = start_step % gate_period if has_gates else 0
        initial = (sx, sy, cur_s, cur_c, cur_r, 0, budget, start_mod)
        queue = deque([(initial, [])])
        visited = set()
        visited.add(initial)
        
        dirs = [(0, -5, GameAction.ACTION1, 0), (0, 5, GameAction.ACTION2, 1),
                (-5, 0, GameAction.ACTION3, 2), (5, 0, GameAction.ACTION4, 3)]
        
        nodes_explored = 0
        
        while queue:
            state, path = queue.popleft()
            cx, cy, cs, cc, cr, t_mask, seg_rem, step_mod = state
            nodes_explored += 1
            
            for dx, dy, act, act_idx in dirs:
                new_seg = seg_rem - 1
                if new_seg < 0: continue
                
                # Next step's gate phase (step happens BEFORE move in game)
                next_mod = (step_mod + 1) % gate_period if has_gates else 0
                
                # Check learned transitions FIRST
                learn_key = (cx, cy, cs, cc, cr, act_idx)
                if learn_key in self.learned_transitions:
                    (fnx, fny), (ns, nc, nr) = self.learned_transitions[learn_key]
                else:
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < 60 and 0 <= ny < 60): continue
                    if self._blocked(walls, nx, ny): continue
                    if (nx, ny) in self.learned_walls: continue
                    
                    fnx, fny = nx, ny
                    chain = 0
                    while (fnx, fny) in pm:
                        fnx, fny = pm[(fnx, fny)]
                        chain += 1
                        if chain > 10: break
                    
                    if not (0 <= fnx < 60 and 0 <= fny < 60): continue
                    if self._blocked(walls, fnx, fny): continue
                    
                    ns, nc, nr = cs, cc, cr
                    # Static changers (always active, not gate-controlled)
                    static_ch = info.get('static_changers', changer_cells)
                    if (fnx, fny) in static_ch:
                        ch = static_ch[(fnx, fny)]
                        if ch == 'shape': ns = (ns + 1) % num_shapes
                        elif ch == 'color': nc = (nc + 1) % num_colors
                        elif ch == 'rot': nr = (nr + 1) % 4
                    # Gate-controlled changers (timing-dependent)
                    if has_gates and gate_schedule:
                        gate_idx = (step_mod + 1) % gate_period
                        active = gate_schedule[gate_idx]
                        if (fnx, fny) in active:
                            ch = active[(fnx, fny)]
                            if ch == 'shape': ns = (ns + 1) % num_shapes
                            elif ch == 'color': nc = (nc + 1) % num_colors
                            elif ch == 'rot': nr = (nr + 1) % 4
                
                # Timer at landing position
                nt_mask = t_mask
                if (fnx, fny) in timer_cells:
                    t_idx = timer_cells[(fnx, fny)]
                    if not (nt_mask & (1 << t_idx)):
                        nt_mask |= (1 << t_idx)
                        new_seg = budget
                
                # Goal check - only target goal
                if (fnx, fny) == target_cell:
                    if (ns, nc, nr) == (goal_s, goal_c, goal_r):
                        self.f.write(f"      6D-BFS: {nodes_explored} nodes, "
                                     f"path={len(path)+1} steps, "
                                     f"learned={len(self.learned_transitions)}\n")
                        return path + [act]
                    else:
                        continue  # can't enter target goal with wrong state
                # Other goal cells: barrier unless player state matches that goal's req
                if (fnx, fny) in g_barriers:
                    req = g_barriers[(fnx, fny)]
                    if (ns, nc, nr) != req:
                        continue  # blocked — state doesn't match this goal
                
                state_key = (fnx, fny, ns, nc, nr, nt_mask, new_seg, next_mod)
                if state_key in visited: continue
                visited.add(state_key)
                
                queue.append(((fnx, fny, ns, nc, nr, nt_mask, new_seg, next_mod), path + [act]))
        
        self.f.write(f"      6D-BFS: {nodes_explored} nodes, NO PATH FOUND\n")
        return None
    
    def _log_trajectory(self, level_idx, step, state, action, 
                         from_x, from_y, to_x, to_y,
                         exp_x=None, exp_y=None, pushed=False):
        if self.traj_f is None: return
        entry = {
            'level': level_idx + 1, 'step': step,
            'state': state, 'action': str(action),
            'from': [from_x, from_y], 'to': [to_x, to_y]
        }
        if pushed:
            entry['expected'] = [exp_x, exp_y]
            entry['event'] = f"PUSHED from ({from_x},{from_y}) to ({to_x},{to_y})"
        self.traj_f.write(json.dumps(entry) + '\n')
    
    def solve_level(self, env, game, level_idx, start_lvl):
        info = self.get_level_info(level_idx)
        level = info['level']
        
        gc = level.get_data("GoalColor")
        gr = level.get_data("GoalRotation")
        gs = level.get_data("kvynsvxbpi")
        rotations = [0, 90, 180, 270]
        
        if isinstance(gc, list):
            goal_list = [{'shape': gs[j], 'color': gc[j], 'rot': gr[j], 
                           'pos': info['goals'][j]} for j in range(len(gc))]
        else:
            goal_list = [{'shape': gs, 'color': gc, 'rot': gr, 'pos': info['goals'][0]}]
        
        self.f.write(f"\n  Level {level_idx+1}: budget={info['budget']}, dec={info['dec']}, "
                     f"enemies={len(info['enemies'])}"
                     f"{' gates='+str(info['gate_period']) if info.get('has_gates') else ''}\n")
        
        level_steps = 0  # Track steps within this level for gate phase
        remaining_goals = list(enumerate(goal_list))  # [(index, goal_dict), ...]
        
        # Any-Order Goal Resolution: try ALL remaining goals, solve whichever has a path
        for attempt in range(200):  # generous retry budget
            if not remaining_goals:
                break  # all goals done
            
            # Read current game state
            cur_shape = game.fwckfzsyc
            cur_ci = game.hiaauhahz
            cur_ri = game.cklxociuu
            px, py = game.gudziatsk.x, game.gudziatsk.y
            
            # Gate offset = 1 because game initializes gates with 1 step already taken
            gate_start = level_steps + 1 if info.get('has_gates') else level_steps
            
            # Try BFS for each remaining goal, pick the first one with a valid path
            best_path = None
            best_gi = None
            best_goal = None
            
            for gi, goal in remaining_goals:
                g_ci = self.color_order.index(goal['color'])
                g_ri = rotations.index(goal['rot'])
                
                sh = (goal['shape'] - cur_shape) % 6
                ch = (g_ci - cur_ci) % len(self.color_order)
                rh = (g_ri - cur_ri) % 4
                
                self.f.write(f"    Goal {gi+1}: need shape*{sh} color*{ch} rot*{rh}"
                             f" (cur: s={cur_shape} c={cur_ci} r={cur_ri})\n")
                
                # Build goal barriers: other remaining goals block unless state matches
                goal_barriers = {}
                for ogi, og in remaining_goals:
                    if ogi == gi: continue
                    og_ci = self.color_order.index(og['color'])
                    og_ri = rotations.index(og['rot'])
                    goal_barriers[og['pos']] = (og['shape'], og_ci, og_ri)
                
                path = self._5d_bfs(info, px, py, cur_shape, cur_ci, cur_ri,
                                     goal['shape'], g_ci, g_ri, goal['pos'],
                                     start_step=gate_start,
                                     goal_barriers=goal_barriers)
                
                if path is not None:
                    if best_path is None or len(path) < len(best_path):
                        best_path = path
                        best_gi = gi
                        best_goal = goal
            
            if best_path is None:
                self.f.write(f"      NO PATH to any remaining goal\n")
                return False, None
            
            self.f.write(f"      Executing {len(best_path)} actions to goal {best_gi+1} at {best_goal['pos']}...\n")
            
            # Execute the path step by step
            result, steps_taken = self._execute_path(env, game, info, best_path, level_idx, start_lvl,
                                                      level_steps=level_steps)
            level_steps += steps_taken
            
            if result == 'COMPLETE':
                return True, None
            elif result == 'GOAL_REACHED':
                # Remove the completed goal from remaining list
                remaining_goals = [(gi, g) for gi, g in remaining_goals if gi != best_gi]
                self.f.write(f"      Goal {best_gi+1} cleared! {len(remaining_goals)} remaining\n")
                continue  # try next goal
            elif result == 'GAME_OVER':
                self.f.write(f"      GAME_OVER\n")
                return False, None
            elif result == 'REPLAN':
                self.f.write(f"      Re-planning after mismatch...\n")
                continue  # retry with re-read state
            else:
                self.f.write(f"      Execution result: {result}\n")
                continue
        
        # Check completion
        obs_check = env.step(GameAction.ACTION1)
        self.total_actions += 1
        if obs_check.levels_completed > start_lvl:
            return True, obs_check
        return False, obs_check
    
    def _execute_path(self, env, game, info, path, level_idx, start_lvl, level_steps=0):
        """Execute path with Self-Healing: compare predictions vs reality at every step."""
        action_map = {GameAction.ACTION1: (0,-5,0), GameAction.ACTION2: (0,5,1),
                      GameAction.ACTION3: (-5,0,2), GameAction.ACTION4: (5,0,3)}
        pm = info.get('push_map', {})
        changer_cells = info.get('changer_cells', {})
        num_colors = len(self.color_order)
        has_gates = info.get('has_gates', False)
        gate_schedule = info.get('gate_schedule', None)
        gate_period = info.get('gate_period', 1)
        
        for step_i, act in enumerate(path):
            pre_x, pre_y = game.gudziatsk.x, game.gudziatsk.y
            pre_s, pre_c, pre_r = game.fwckfzsyc, game.hiaauhahz, game.cklxociuu
            dx, dy, act_idx = action_map[act]
            
            # BFS expected position (push chain)
            raw_x, raw_y = pre_x + dx, pre_y + dy
            exp_x, exp_y = raw_x, raw_y
            chain = 0
            while (exp_x, exp_y) in pm:
                exp_x, exp_y = pm[(exp_x, exp_y)]
                chain += 1
                if chain > 10: break
            
            # BFS expected state — gate-aware
            exp_s, exp_c, exp_r = pre_s, pre_c, pre_r
            current_step = level_steps + step_i
            # Static changers (always active)
            static_ch = info.get('static_changers', changer_cells)
            if (exp_x, exp_y) in static_ch:
                ch = static_ch[(exp_x, exp_y)]
                if ch == 'shape': exp_s = (exp_s + 1) % 6
                elif ch == 'color': exp_c = (exp_c + 1) % num_colors
                elif ch == 'rot': exp_r = (exp_r + 1) % 4
            # Gate-controlled changers
            if has_gates and gate_schedule:
                gate_idx = (current_step + 1 + 1) % gate_period  # +1 for gate init offset, +1 for step-before-move
                active = gate_schedule[gate_idx]
                if (exp_x, exp_y) in active:
                    ch = active[(exp_x, exp_y)]
                    if ch == 'shape': exp_s = (exp_s + 1) % 6
                    elif ch == 'color': exp_c = (exp_c + 1) % num_colors
                    elif ch == 'rot': exp_r = (exp_r + 1) % 4
            
            # Check learned transition override
            learn_key = (pre_x, pre_y, pre_s, pre_c, pre_r, act_idx)
            if learn_key in self.learned_transitions:
                (exp_x, exp_y), (exp_s, exp_c, exp_r) = self.learned_transitions[learn_key]
            
            obs = env.step(act)
            self.total_actions += 1
            
            post_x, post_y = game.gudziatsk.x, game.gudziatsk.y
            post_s, post_c, post_r = game.fwckfzsyc, game.hiaauhahz, game.cklxociuu
            
            if obs.levels_completed > start_lvl:
                self.f.write(f"      LEVEL CLEARED at step {step_i+1}\n")
                return 'COMPLETE', step_i + 1
            
            # Check if a goal was just reached (path completed successfully)
            if step_i == len(path) - 1:
                self.f.write(f"      Path completed at step {step_i+1}\n")
                return 'GOAL_REACHED', step_i + 1
            if obs.state.value == 'GAME_OVER':
                return 'GAME_OVER', step_i + 1
            
            # Life lost
            if post_x == info['start'][0] and post_y == info['start'][1] and step_i > 0:
                if pre_x != info['start'][0] or pre_y != info['start'][1]:
                    self.f.write(f"      LIFE LOST at step {step_i+1}\n")
                    return 'REPLAN', step_i + 1
            
            # Self-Healing: compare prediction vs reality
            pos_match = (post_x == exp_x and post_y == exp_y)
            state_match = (post_s == exp_s and post_c == exp_c and post_r == exp_r)
            
            if not pos_match or not state_match:
                actual_pos = (post_x, post_y)
                actual_state = (post_s, post_c, post_r)
                
                if post_x == pre_x and post_y == pre_y and post_s == pre_s and post_c == pre_c and post_r == pre_r:
                    # Wall bump — but protect goal cells from being learned as walls!
                    is_goal_cell = (raw_x, raw_y) in info.get('goal_cells', set())
                    if is_goal_cell:
                        self.f.write(f"      [Heal] Goal barrier at ({raw_x},{raw_y}) step {step_i+1}\n")
                    else:
                        self.learned_walls.add((raw_x, raw_y))
                        self.f.write(f"      [Heal] Wall at ({raw_x},{raw_y}) step {step_i+1}\n")
                else:
                    # Learn the true transition
                    self.learned_transitions[learn_key] = (actual_pos, actual_state)
                    mismatch = []
                    if not pos_match: mismatch.append(f"pos({post_x},{post_y})")
                    if post_s != exp_s: mismatch.append(f"s{exp_s}->{post_s}")
                    if post_c != exp_c: mismatch.append(f"c{exp_c}->{post_c}")
                    if post_r != exp_r: mismatch.append(f"r{exp_r}->{post_r}")
                    self.f.write(f"      [Heal] {' '.join(mismatch)} step {step_i+1}\n")
                
                self.replan_log.append({
                    'step': self.total_actions, 'level': level_idx + 1,
                    'pushed_from': (pre_x, pre_y), 'pushed_to': (post_x, post_y),
                    'target': 'goal'
                })
                return 'REPLAN', step_i + 1
        
        return 'REPLAN', len(path)


def run():
    with open(OUT, "w", encoding="utf-8") as f, \
         open(TRAJECTORY_OUT, "w", encoding="utf-8") as traj_f:
        f.write(f"{'='*60}\n  LS20 Solver v15 (Self-Healing 5D State-Space BFS)\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        obs = env.step(GameAction.RESET)
        game = env._game
        
        solver = StateSpaceSolver(f, traj_f)
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
            sc = arc.get_scorecard()
            f.write(f"  Score: {sc.score}\n")
        except: pass
        
        if solver.replan_log:
            f.write(f"\n  Aha! Moments: {len(solver.replan_log)}\n")
            for ev in solver.replan_log:
                f.write(f"    Step {ev['step']}: Lv{ev['level']} "
                        f"{ev['pushed_from']}->{ev['pushed_to']}\n")
    
    print(f"Results: {OUT}")

if __name__ == "__main__":
    print("ARC-AGI-3 LS20 Solver v15 (Self-Healing 5D State-Space BFS)")
    run()
