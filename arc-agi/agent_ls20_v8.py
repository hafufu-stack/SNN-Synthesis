"""
ARC-AGI-3 LS20 Solver v8 — Timer-Free Move Correction
=======================================================
CRITICAL FIX: Timer pickup moves do NOT decrement the step counter!
Python's `and` short-circuits, so when timer is picked up,
mfyzdfvxsm() is never called.

This means: in a segment, you can make (budget) non-timer moves
+ 1 timer-pickup move (free). The effective budget per segment is
(step_limit / steps_dec) for non-timer moves only.
"""

import arc_agi
from arcengine import GameAction
import time
import sys
from collections import deque
from itertools import permutations

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
import ls20

OUT = r"c:\tmp\arc_solver_v8.txt"


class LevelSolver:
    def __init__(self, level_idx):
        self.level = ls20.levels[level_idx]
        self.idx = level_idx
        
        self.step_limit = self.level.get_data("StepCounter") or 42
        sd = self.level.get_data("StepsDecrement")
        self.steps_dec = sd if sd is not None else 2
        self.budget = self.step_limit // self.steps_dec
        
        self.walls = set()
        self.player = None
        self.goals = []
        self.rot_ch = []
        self.shape_ch = []
        self.color_ch = []
        self.timers = []
        
        for s in self.level._sprites:
            if not s.tags:
                continue
            if "ihdgageizm" in s.tags:
                self.walls.add((s.x, s.y))
            if "sfqyzhzkij" in s.tags:
                self.player = (s.x, s.y)
            if "rjlbuycveu" in s.tags:
                self.goals.append((s.x, s.y))
            if "rhsxkxzdjz" in s.tags:
                self.rot_ch.append((s.x, s.y))
            if "ttfwljgohq" in s.tags:
                self.shape_ch.append((s.x, s.y))
            if "soyhouuebz" in s.tags:
                self.color_ch.append((s.x, s.y))
            if "npxgalaybz" in s.tags:
                self.timers.append((s.x, s.y))
        
        # Timer pickup positions
        # Player grid: x%5=4 (4,9,14,...), y%5=0 (0,5,10,...)
        px_offset = self.player[0] % 5  # should be 4
        py_offset = self.player[1] % 5  # should be 0
        self.timer_pickups = {}
        for tx, ty in self.timers:
            pickups = set()
            for px in range(px_offset, 60, 5):
                for py in range(py_offset, 60, 5):
                    if tx >= px and tx < px + 5 and ty >= py and ty < py + 5:
                        if not self.blocked(px, py):
                            pickups.add((px, py))
            self.timer_pickups[(tx, ty)] = pickups
        
        self._path_cache = {}
    
    def blocked(self, px, py):
        for wx, wy in self.walls:
            if wx >= px and wx < px + 5 and wy >= py and wy < py + 5:
                return True
        return False
    
    def bfs(self, sx, sy, tx, ty):
        key = (sx, sy, tx, ty)
        if key in self._path_cache:
            return self._path_cache[key]
        
        if sx == tx and sy == ty:
            self._path_cache[key] = []
            return []
        
        queue = deque([((sx, sy), [])])
        visited = {(sx, sy)}
        dirs = [(0, -5, GameAction.ACTION1), (0, 5, GameAction.ACTION2),
                (-5, 0, GameAction.ACTION3), (5, 0, GameAction.ACTION4)]
        
        while queue:
            (cx, cy), path = queue.popleft()
            for dx, dy, action in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 60 and 0 <= ny < 60 and \
                   (nx, ny) not in visited and not self.blocked(nx, ny):
                    npath = path + [action]
                    if nx == tx and ny == ty:
                        self._path_cache[key] = npath
                        return npath
                    visited.add((nx, ny))
                    queue.append(((nx, ny), npath))
        
        self._path_cache[key] = None
        return None
    
    def bfs_dist(self, sx, sy, tx, ty):
        p = self.bfs(sx, sy, tx, ty)
        return len(p) if p is not None else -1
    
    def revisit(self, px, py, prefer_dir=None):
        dirs = [(0, -5, GameAction.ACTION1, GameAction.ACTION2),
                (0, 5, GameAction.ACTION2, GameAction.ACTION1),
                (-5, 0, GameAction.ACTION3, GameAction.ACTION4),
                (5, 0, GameAction.ACTION4, GameAction.ACTION3)]
        
        if prefer_dir:
            dx, dy = prefer_dir[0] - px, prefer_dir[1] - py
            if abs(dx) >= abs(dy):
                order = [(5 if dx > 0 else -5, 0), (0, 5 if dy > 0 else -5),
                         (0, -(5 if dy > 0 else -5)), (-(5 if dx > 0 else -5), 0)]
            else:
                order = [(0, 5 if dy > 0 else -5), (5 if dx > 0 else -5, 0),
                         (-(5 if dx > 0 else -5), 0), (0, -(5 if dy > 0 else -5))]
            for odx, ody in order:
                nx, ny = px + odx, py + ody
                if 0 <= nx < 60 and 0 <= ny < 60 and not self.blocked(nx, ny):
                    for ddx, ddy, go, back in dirs:
                        if ddx == odx and ddy == ody:
                            return [go, back]
        
        for dx, dy, go, back in dirs:
            nx, ny = px + dx, py + dy
            if 0 <= nx < 60 and 0 <= ny < 60 and not self.blocked(nx, ny):
                return [go, back]
        return None
    
    def _check_segments(self, start, visits, goal):
        """
        Check segmented budget feasibility.
        KEY: Timer pickup moves are FREE (don't decrement step counter).
        So each segment can have up to self.budget NON-TIMER moves.
        The timer-pickup move itself doesn't count.
        """
        cur = start
        seg_non_timer = 0
        
        for name, pos, count in visits:
            d = self.bfs_dist(cur[0], cur[1], pos[0], pos[1])
            if d < 0:
                return False
            
            if name == 'TIMER':
                # All moves to timer except the last are non-timer
                # The last move picks up the timer (free)
                seg_non_timer += (d - 1)  # d-1 non-timer + 1 free
                if seg_non_timer > self.budget:
                    return False
                seg_non_timer = 0  # Reset after timer
            else:
                seg_non_timer += d + (count - 1) * 2  # distance + revisits
            
            cur = pos
        
        # Final segment to goal
        d = self.bfs_dist(cur[0], cur[1], goal[0], goal[1])
        if d < 0:
            return False
        seg_non_timer += d
        return seg_non_timer <= self.budget
    
    def solve(self, color_order, f):
        rotations = [0, 90, 180, 270]
        
        start_shape = self.level.get_data("StartShape")
        start_color = self.level.get_data("StartColor")
        start_rot = self.level.get_data("StartRotation")
        gc = self.level.get_data("GoalColor")
        gr = self.level.get_data("GoalRotation")
        gs = self.level.get_data("kvynsvxbpi")
        
        f.write(f"\n{'='*50}\n  Level {self.idx+1}  (budget={self.budget}, "
                f"dec={self.steps_dec})\n{'='*50}\n")
        f.write(f"  Player: {self.player}\n")
        f.write(f"  Goals: {self.goals}\n")
        f.write(f"  Timers: {self.timers}\n")
        f.write(f"  Timer pickups: {dict((k, list(v)) for k, v in self.timer_pickups.items())}\n")
        
        goal_list = []
        if isinstance(gc, list):
            for i in range(len(gc)):
                goal_list.append({'shape': gs[i], 'color': gc[i],
                                  'rot': gr[i], 'pos': self.goals[i]})
        else:
            goal_list.append({'shape': gs, 'color': gc,
                              'rot': gr, 'pos': self.goals[0]})
        
        full_plan = []
        cur_pos = self.player
        cur_shape = start_shape
        cur_ci = color_order.index(start_color)
        cur_ri = rotations.index(start_rot)
        used_timers = set()
        
        for gi, goal in enumerate(goal_list):
            g_ci = color_order.index(goal['color'])
            g_ri = rotations.index(goal['rot'])
            
            sh = (goal['shape'] - cur_shape) % 6
            ch = (g_ci - cur_ci) % len(color_order)
            rh = (g_ri - cur_ri) % 4
            
            f.write(f"\n  Goal {gi+1}: shape*{sh}, color*{ch}, rot*{rh} -> {goal['pos']}\n")
            
            changers = []
            if sh > 0 and self.shape_ch:
                changers.append(('shape', self.shape_ch[0], sh))
            if ch > 0 and self.color_ch:
                changers.append(('color', self.color_ch[0], ch))
            if rh > 0 and self.rot_ch:
                changers.append(('rot', self.rot_ch[0], rh))
            
            best = self._find_best_route(cur_pos, changers, goal['pos'],
                                          used_timers, f)
            
            if best is None:
                f.write(f"  X FAILED\n")
                return None
            
            plan, desc, timers_used = best
            f.write(f"  Route ({len(plan)} moves): {' -> '.join(desc)}\n")
            
            full_plan.extend(plan)
            used_timers.update(timers_used)
            cur_pos = goal['pos']
            cur_shape = (cur_shape + sh) % 6
            cur_ci = (cur_ci + ch) % len(color_order)
            cur_ri = (cur_ri + rh) % 4
        
        f.write(f"\n  Total: {len(full_plan)} moves\n")
        return full_plan
    
    def _find_best_route(self, start, changers, goal, used_timers, f):
        best_plan = None
        best_len = float('inf')
        best_desc = None
        best_timers = set()
        
        perms = list(permutations(range(len(changers)))) if changers else [()]
        avail_timers = [t for t in self.timers if t not in used_timers]
        
        timer_pp = {}
        for tp in avail_timers:
            if tp in self.timer_pickups:
                timer_pp[tp] = self.timer_pickups[tp]
        
        for perm in perms:
            ordered = [changers[i] for i in perm] if perm else []
            
            # Without timers
            plan, desc = self._build_route(start, ordered, goal)
            if plan and len(plan) <= self.budget and len(plan) < best_len:
                best_plan = plan
                best_len = len(plan)
                best_desc = desc
                best_timers = set()
            
            # With 1 timer
            for tp in avail_timers:
                for pp in timer_pp.get(tp, set()):
                    for ins in range(len(ordered) + 1):
                        mod = list(ordered)
                        mod.insert(ins, ('TIMER', pp, 1))
                        plan2, desc2 = self._build_route(start, mod, goal)
                        if plan2 and len(plan2) < best_len:
                            if self._check_segments(start, mod, goal):
                                best_plan = plan2
                                best_len = len(plan2)
                                best_desc = desc2
                                best_timers = {tp}
            
            # With 2 timers
            if len(avail_timers) >= 2:
                for i, tp1 in enumerate(avail_timers):
                    for j, tp2 in enumerate(avail_timers):
                        if i == j:
                            continue
                        for pp1 in timer_pp.get(tp1, set()):
                            for pp2 in timer_pp.get(tp2, set()):
                                for ins1 in range(len(ordered) + 1):
                                    for ins2 in range(ins1, len(ordered) + 2):
                                        mod = list(ordered)
                                        mod.insert(ins2, ('TIMER', pp2, 1))
                                        mod.insert(ins1, ('TIMER', pp1, 1))
                                        plan3, desc3 = self._build_route(start, mod, goal)
                                        if plan3 and len(plan3) < best_len:
                                            if self._check_segments(start, mod, goal):
                                                best_plan = plan3
                                                best_len = len(plan3)
                                                best_desc = desc3
                                                best_timers = {tp1, tp2}
        
        if best_plan:
            return best_plan, best_desc, best_timers
        return None
    
    def _build_route(self, start, visits, goal):
        actions = []
        desc = []
        cur = start
        
        for name, pos, count in visits:
            p = self.bfs(cur[0], cur[1], pos[0], pos[1])
            if p is None:
                return None, None
            actions.extend(p)
            cur = pos
            
            if name == 'TIMER':
                desc.append(f"T@{pos}")
            else:
                desc.append(f"{name}@{pos}")
                for _ in range(count - 1):
                    rv = self.revisit(cur[0], cur[1])
                    if rv is None:
                        return None, None
                    actions.extend(rv)
        
        p = self.bfs(cur[0], cur[1], goal[0], goal[1])
        if p is None:
            return None, None
        actions.extend(p)
        desc.append(f"goal@{goal}")
        
        return actions, desc


def run_v8():
    game = ls20.Ls20()
    color_order = game.tnkekoeuk
    
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n  LS20 Solver v8 (Timer-Free Move)\n")
        f.write(f"  Color order: {color_order}\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        
        all_plans = []
        for li in range(7):
            solver = LevelSolver(li)
            plan = solver.solve(color_order, f)
            all_plans.append(plan)
        
        # Summary
        f.write(f"\n{'='*60}\n  PLANNING SUMMARY\n{'='*60}\n")
        for i, plan in enumerate(all_plans):
            if plan:
                f.write(f"  Lv{i+1}: {len(plan)} moves\n")
            else:
                f.write(f"  Lv{i+1}: FAILED\n")
        
        # Execute
        f.write(f"\n{'='*60}\n  EXECUTION\n{'='*60}\n\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        obs = env.step(GameAction.RESET)
        
        levels_cleared = 0
        total = 0
        
        for li, plan in enumerate(all_plans):
            if plan is None:
                f.write(f"\n  Level {li+1}: SKIPPED\n")
                break
            
            f.write(f"  Level {li+1}: {len(plan)} actions\n")
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
                        f.write(f"    Step {i+1}: LEVEL {levels_cleared} COMPLETE!\n")
                        done = True
                        break
                    if obs.state.value == 'GAME_OVER':
                        f.write(f"    Step {i+1}: GAME OVER\n")
                        done = True
                        break
                    if obs.full_reset:
                        f.write(f"    Step {i+1}: Life lost\n")
                        break
                
                if done:
                    break
            
            if obs.state.value == 'GAME_OVER':
                break
            if obs.levels_completed <= start_lvl:
                f.write(f"  Level {li+1}: NOT CLEARED\n")
                break
        
        f.write(f"\n{'='*60}\n  FINAL: {levels_cleared}/7, {total} actions, "
                f"{obs.state.value}\n{'='*60}\n")
        try:
            sc = arc.get_scorecard()
            f.write(f"  Score: {sc.score}\n")
        except:
            pass
    
    print(f"Results saved to {OUT}")


if __name__ == "__main__":
    print("ARC-AGI-3 LS20 Solver v8 (Timer-Free Move)")
    run_v8()
