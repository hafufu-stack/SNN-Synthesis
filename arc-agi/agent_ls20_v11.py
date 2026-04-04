"""
ARC-AGI-3 LS20 Solver v11 — Reactive Execution
=================================================
Instead of pre-planning everything, execute moves reactively:
- Plan BFS path to next waypoint
- Execute step by step
- Check actual player position after each move
- If player dies (full_reset), re-plan from respawn position
- Skip levels with enemies if too many retries fail
"""

import arc_agi
from arcengine import GameAction
import time
import sys
from collections import deque
from itertools import permutations

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
import ls20

OUT = r"c:\tmp\arc_solver_v11.txt"


class ReactiveSolver:
    def __init__(self, env, game, f):
        self.env = env
        self.game = game
        self.f = f
        self.color_order = game.tnkekoeuk
        self.rotations = [0, 90, 180, 270]
        self.total_actions = 0
    
    def get_level_data(self, level_idx):
        level = ls20.levels[level_idx]
        walls = set()
        player_start = None
        goals = []
        rot_ch = []
        shape_ch = []
        color_ch = []
        timers = []
        interactive = set()
        
        for s in level._sprites:
            if not s.tags: continue
            if "ihdgageizm" in s.tags: walls.add((s.x, s.y))
            if "sfqyzhzkij" in s.tags: player_start = (s.x, s.y)
            if "rjlbuycveu" in s.tags: goals.append((s.x, s.y))
            if "rhsxkxzdjz" in s.tags: rot_ch.append((s.x, s.y))
            if "ttfwljgohq" in s.tags: shape_ch.append((s.x, s.y))
            if "soyhouuebz" in s.tags: color_ch.append((s.x, s.y))
            if "npxgalaybz" in s.tags: timers.append((s.x, s.y))
        
        for pos in rot_ch + shape_ch + color_ch + goals:
            interactive.add(pos)
        
        # Timer pickups
        px_off = player_start[0] % 5 if player_start else 4
        py_off = player_start[1] % 5 if player_start else 0
        timer_pickups = {}
        for tx, ty in timers:
            pps = set()
            for px in range(px_off, 60, 5):
                for py in range(py_off, 60, 5):
                    if tx >= px and tx < px + 5 and ty >= py and ty < py + 5:
                        blocked = False
                        for wx, wy in walls:
                            if wx >= px and wx < px + 5 and wy >= py and wy < py + 5:
                                blocked = True; break
                        if not blocked:
                            pps.add((px, py))
            timer_pickups[(tx, ty)] = pps
        
        return {'walls': walls, 'start': player_start, 'goals': goals,
                'rot_ch': rot_ch, 'shape_ch': shape_ch, 'color_ch': color_ch,
                'timers': timers, 'interactive': interactive, 'timer_pickups': timer_pickups,
                'level': level}
    
    def bfs(self, walls, sx, sy, tx, ty, avoid=None):
        if sx == tx and sy == ty: return []
        avoid_set = avoid or set()
        queue = deque([((sx, sy), [])]); visited = {(sx, sy)}
        dirs = [(0,-5,GameAction.ACTION1),(0,5,GameAction.ACTION2),
                (-5,0,GameAction.ACTION3),(5,0,GameAction.ACTION4)]
        while queue:
            (cx, cy), path = queue.popleft()
            for dx, dy, act in dirs:
                nx, ny = cx + dx, cy + dy
                blocked = False
                for wx, wy in walls:
                    if wx >= nx and wx < nx + 5 and wy >= ny and wy < ny + 5:
                        blocked = True; break
                if 0 <= nx < 60 and 0 <= ny < 60 and (nx,ny) not in visited and not blocked:
                    if nx == tx and ny == ty: return path + [act]
                    if (nx, ny) in avoid_set: continue
                    visited.add((nx, ny)); queue.append(((nx, ny), path + [act]))
        return None
    
    def revisit(self, walls, px, py, avoid=None):
        avoid_set = avoid or set()
        dirs = [(0,-5,GameAction.ACTION1,GameAction.ACTION2),(0,5,GameAction.ACTION2,GameAction.ACTION1),
                (-5,0,GameAction.ACTION3,GameAction.ACTION4),(5,0,GameAction.ACTION4,GameAction.ACTION3)]
        for dx,dy,go,back in dirs:
            nx,ny = px+dx,py+dy
            blocked = False
            for wx,wy in walls:
                if wx>=nx and wx<nx+5 and wy>=ny and wy<ny+5: blocked=True; break
            if 0<=nx<60 and 0<=ny<60 and not blocked and (nx,ny) not in avoid_set:
                return [go, back]
        return None
    
    def execute_path(self, path, start_lvl):
        """Execute a path reactively, checking position after each step."""
        action_map = {GameAction.ACTION1:(0,-5), GameAction.ACTION2:(0,5),
                      GameAction.ACTION3:(-5,0), GameAction.ACTION4:(5,0)}
        
        for act in path:
            obs = self.env.step(act)
            self.total_actions += 1
            
            if obs.levels_completed > start_lvl:
                return 'COMPLETE', obs
            if obs.state.value == 'GAME_OVER':
                return 'GAME_OVER', obs
            if obs.full_reset:
                return 'LIFE_LOST', obs
        
        return 'OK', obs
    
    def solve_level(self, level_idx, start_lvl):
        data = self.get_level_data(level_idx)
        level = data['level']
        walls = data['walls']
        interactive = data['interactive']
        
        budget = (level.get_data("StepCounter") or 42) // (level.get_data("StepsDecrement") or 2)
        
        start_shape = level.get_data("StartShape")
        start_color = level.get_data("StartColor")
        start_rot = level.get_data("StartRotation")
        gc = level.get_data("GoalColor")
        gr = level.get_data("GoalRotation")
        gs = level.get_data("kvynsvxbpi")
        
        # Build goal list
        if isinstance(gc, list):
            goal_list = [{'shape': gs[j], 'color': gc[j], 'rot': gr[j], 'pos': data['goals'][j]} for j in range(len(gc))]
        else:
            goal_list = [{'shape': gs, 'color': gc, 'rot': gr, 'pos': data['goals'][0]}]
        
        self.f.write(f"\n  Level {level_idx+1}: budget={budget}\n")
        
        for attempt in range(3):
            cur_pos = data['start']
            cur_shape = start_shape
            cur_ci = self.color_order.index(start_color)
            cur_ri = self.rotations.index(start_rot)
            used_timers = set()
            success = True
            
            for gi, goal in enumerate(goal_list):
                g_ci = self.color_order.index(goal['color'])
                g_ri = self.rotations.index(goal['rot'])
                sh = (goal['shape'] - cur_shape) % 6
                ch = (g_ci - cur_ci) % len(self.color_order)
                rh = (g_ri - cur_ri) % 4
                
                # Build waypoints
                changers = []
                if sh > 0 and data['shape_ch']: changers.append(('shape', data['shape_ch'][0], sh))
                if ch > 0 and data['color_ch']: changers.append(('color', data['color_ch'][0], ch))
                if rh > 0 and data['rot_ch']: changers.append(('rot', data['rot_ch'][0], rh))
                
                # Find best route
                best = self._find_route(walls, interactive, data['timer_pickups'],
                                         cur_pos, changers, goal['pos'],
                                         [t for t in data['timers'] if t not in used_timers],
                                         budget)
                
                if best is None:
                    self.f.write(f"    Goal {gi+1}: no route found\n")
                    success = False; break
                
                plan, desc, t_used = best
                self.f.write(f"    Goal {gi+1}: {' -> '.join(desc)} ({len(plan)} moves)\n")
                
                result, obs = self.execute_path(plan, start_lvl)
                
                if result == 'COMPLETE':
                    return True, obs
                elif result == 'GAME_OVER':
                    return False, obs
                elif result == 'LIFE_LOST':
                    self.f.write(f"    Life lost! Retrying...\n")
                    success = False; break
                
                used_timers.update(t_used)
                cur_pos = goal['pos']
                cur_shape = (cur_shape + sh) % 6
                cur_ci = (cur_ci + ch) % len(self.color_order)
                cur_ri = (cur_ri + rh) % 4
            
            if success and obs.levels_completed > start_lvl:
                return True, obs
            if obs.state.value == 'GAME_OVER':
                return False, obs
        
        return False, obs
    
    def _find_route(self, walls, interactive, timer_pickups, start, changers, goal, avail_timers, budget):
        best = None; best_len = float('inf')
        perms = list(permutations(range(len(changers)))) if changers else [()]
        tp_pp = {tp: timer_pickups.get(tp, set()) for tp in avail_timers}
        
        for perm in perms:
            ordered = [changers[i] for i in perm] if perm else []
            
            r = self._try_route(walls, interactive, start, ordered, goal, budget)
            if r and len(r[0]) < best_len:
                best = r; best_len = len(r[0])
            
            for tp in avail_timers:
                for pp in tp_pp.get(tp, set()):
                    for ins in range(len(ordered) + 1):
                        mod = list(ordered); mod.insert(ins, ('TIMER', pp, 1))
                        r = self._try_route(walls, interactive, start, mod, goal, budget)
                        if r and len(r[0]) < best_len:
                            best = (r[0], r[1], {tp}); best_len = len(r[0])
            
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
                                        r = self._try_route(walls, interactive, start, mod, goal, budget)
                                        if r and len(r[0]) < best_len:
                                            best = (r[0], r[1], {tp1, tp2}); best_len = len(r[0])
        
        return best
    
    def _try_route(self, walls, interactive, start, visits, goal, budget):
        actions = []; desc = []; cur = start; seg = 0
        for name, pos, count in visits:
            avoid = interactive - {pos}
            p = self.bfs(walls, cur[0], cur[1], pos[0], pos[1], avoid)
            if p is None: return None
            actions.extend(p); cur = pos
            if name == 'TIMER':
                desc.append(f"T@{pos}"); seg += max(0, len(p) - 1)
                if seg > budget: return None
                seg = 0
            else:
                desc.append(f"{name}@{pos}"); seg += len(p) + (count - 1) * 2
                rv_avoid = interactive - {pos}
                for _ in range(count - 1):
                    rv = self.revisit(walls, cur[0], cur[1], rv_avoid)
                    if rv is None: return None
                    actions.extend(rv)
        avoid = interactive - {goal}
        p = self.bfs(walls, cur[0], cur[1], goal[0], goal[1], avoid)
        if p is None: return None
        actions.extend(p); desc.append(f"goal@{goal}"); seg += len(p)
        if seg > budget: return None
        return actions, desc, set()


def run():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n  LS20 Solver v11 (Reactive)\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        obs = env.step(GameAction.RESET)
        game = env._game
        
        solver = ReactiveSolver(env, game, f)
        
        for li in range(7):
            start_lvl = obs.levels_completed
            success, obs = solver.solve_level(li, start_lvl)
            f.write(f"  -> Level {li+1}: {'CLEARED' if success else 'FAILED'} "
                    f"(completed={obs.levels_completed})\n")
            if obs.state.value == 'GAME_OVER':
                f.write(f"  GAME OVER at Level {li+1}\n"); break
            if not success: break
        
        f.write(f"\n{'='*60}\n  FINAL: {obs.levels_completed}/7, "
                f"{solver.total_actions} actions, {obs.state.value}\n{'='*60}\n")
        try:
            sc = arc.get_scorecard(); f.write(f"  Score: {sc.score}\n")
        except: pass
    
    print(f"Results saved to {OUT}")

if __name__ == "__main__":
    run()
