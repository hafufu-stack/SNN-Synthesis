"""
ARC-AGI-3 LS20 Solver v12 — Adaptive Re-planning
===================================================
Enemies PUSH players, they don't kill. After being pushed, the player
ends up at a different position. We handle this by:
1. Execute each move and check actual position
2. If position differs from expected (pushed by enemy), re-plan from actual position
3. Track actual state changes from changers/timers visited
"""

import arc_agi
from arcengine import GameAction
import time
import sys
from collections import deque
from itertools import permutations

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
import ls20

OUT = r"c:\tmp\arc_solver_v12.txt"


class AdaptiveSolver:
    def __init__(self, f):
        self.f = f
        self.color_order = None
        self.total_actions = 0
    
    def get_level_info(self, level_idx):
        level = ls20.levels[level_idx]
        info = {
            'walls': set(), 'start': None, 'goals': [], 'rot_ch': [],
            'shape_ch': [], 'color_ch': [], 'timers': [], 'interactive': set(),
            'timer_pickups': {}, 'budget': 21, 'dec': 2, 'level': level
        }
        
        for s in level._sprites:
            if not s.tags: continue
            if "ihdgageizm" in s.tags: info['walls'].add((s.x, s.y))
            if "sfqyzhzkij" in s.tags: info['start'] = (s.x, s.y)
            if "rjlbuycveu" in s.tags: info['goals'].append((s.x, s.y))
            if "rhsxkxzdjz" in s.tags: info['rot_ch'].append((s.x, s.y))
            if "ttfwljgohq" in s.tags: info['shape_ch'].append((s.x, s.y))
            if "soyhouuebz" in s.tags: info['color_ch'].append((s.x, s.y))
            if "npxgalaybz" in s.tags: info['timers'].append((s.x, s.y))
        
        for pos in info['rot_ch'] + info['shape_ch'] + info['color_ch'] + info['goals']:
            info['interactive'].add(pos)
        
        sc = level.get_data("StepCounter") or 42
        sd = level.get_data("StepsDecrement")
        info['dec'] = sd if sd is not None else 2
        info['budget'] = sc // info['dec']
        
        # Timer pickups on player grid
        px_off = info['start'][0] % 5 if info['start'] else 4
        py_off = info['start'][1] % 5 if info['start'] else 0
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
    
    def _bfs(self, walls, sx, sy, tx, ty, avoid=None):
        if sx == tx and sy == ty: return []
        avoid_set = avoid or set()
        queue = deque([((sx, sy), [])]); vis = {(sx, sy)}
        dirs = [(0,-5,GameAction.ACTION1),(0,5,GameAction.ACTION2),
                (-5,0,GameAction.ACTION3),(5,0,GameAction.ACTION4)]
        while queue:
            (cx, cy), path = queue.popleft()
            for dx, dy, act in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 60 and 0 <= ny < 60 and (nx,ny) not in vis and not self._blocked(walls, nx, ny):
                    if nx == tx and ny == ty: return path + [act]
                    if (nx, ny) in avoid_set: continue
                    vis.add((nx, ny)); queue.append(((nx, ny), path + [act]))
        return None
    
    def _revisit(self, walls, px, py, avoid=None):
        avoid_set = avoid or set()
        dirs = [(0,-5,GameAction.ACTION1,GameAction.ACTION2),(0,5,GameAction.ACTION2,GameAction.ACTION1),
                (-5,0,GameAction.ACTION3,GameAction.ACTION4),(5,0,GameAction.ACTION4,GameAction.ACTION3)]
        for dx,dy,go,back in dirs:
            nx,ny = px+dx,py+dy
            if 0<=nx<60 and 0<=ny<60 and not self._blocked(walls,nx,ny) and (nx,ny) not in avoid_set:
                return [go, back]
        return None
    
    def _execute_to(self, env, game, walls, interactive, sx, sy, tx, ty, start_lvl, max_retries=8):
        """Navigate from (sx,sy) to (tx,ty), adapting to pushes."""
        action_map = {GameAction.ACTION1:(0,-5), GameAction.ACTION2:(0,5),
                      GameAction.ACTION3:(-5,0), GameAction.ACTION4:(5,0)}
        cur_x, cur_y = sx, sy
        
        for retry in range(max_retries):
            avoid = interactive - {(tx, ty)}
            path = self._bfs(walls, cur_x, cur_y, tx, ty, avoid)
            if path is None:
                path = self._bfs(walls, cur_x, cur_y, tx, ty)
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
                
                if obs.levels_completed > start_lvl:
                    return 'COMPLETE', actual_x, actual_y
                if obs.state.value == 'GAME_OVER':
                    return 'GAME_OVER', actual_x, actual_y
                if obs.full_reset:
                    return 'LIFE_LOST', actual_x, actual_y
                
                # Check for push (position mismatch)
                if actual_x != expected_x or actual_y != expected_y:
                    # Wait for push animation to complete
                    # During animation, euemavvxz is non-empty and step() just advances animation
                    for anim_frame in range(20):
                        if not game.euemavvxz:
                            break
                        # Send the SAME action that was blocked - it will be consumed by animation
                        obs = env.step(act)
                        self.total_actions += 1
                        if obs.levels_completed > start_lvl:
                            return 'COMPLETE', game.gudziatsk.x, game.gudziatsk.y
                        if obs.state.value == 'GAME_OVER':
                            return 'GAME_OVER', game.gudziatsk.x, game.gudziatsk.y
                        if obs.full_reset:
                            return 'LIFE_LOST', game.gudziatsk.x, game.gudziatsk.y
                    
                    cur_x, cur_y = game.gudziatsk.x, game.gudziatsk.y
                    self.f.write(f"      Pushed to ({cur_x},{cur_y}), re-planning...\n")
                    pushed = True
                    break
                
                cur_x, cur_y = actual_x, actual_y
            
            if not pushed:
                if cur_x == tx and cur_y == ty:
                    return 'OK', cur_x, cur_y
                continue
        
        return 'TIMEOUT', cur_x, cur_y
    
    def _execute_revisit(self, env, game, walls, interactive, px, py, start_lvl):
        """Step away and back, handling pushes."""
        avoid = interactive - {(px, py)}
        rv = self._revisit(walls, px, py, avoid)
        if rv is None: return None, px, py
        
        action_map = {GameAction.ACTION1:(0,-5), GameAction.ACTION2:(0,5),
                      GameAction.ACTION3:(-5,0), GameAction.ACTION4:(5,0)}
        
        cur_x, cur_y = px, py
        for act in rv:
            obs = env.step(act)
            self.total_actions += 1
            cur_x, cur_y = game.gudziatsk.x, game.gudziatsk.y
            if obs.levels_completed > start_lvl: return 'COMPLETE', cur_x, cur_y
            if obs.state.value == 'GAME_OVER': return 'GAME_OVER', cur_x, cur_y
            if obs.full_reset: return 'LIFE_LOST', cur_x, cur_y
        return 'OK', cur_x, cur_y
    
    def solve_level(self, env, game, level_idx, start_lvl):
        info = self.get_level_info(level_idx)
        level = info['level']
        walls = info['walls']
        interactive = info['interactive']
        
        gc = level.get_data("GoalColor")
        gr = level.get_data("GoalRotation")
        gs = level.get_data("kvynsvxbpi")
        start_shape = level.get_data("StartShape")
        start_color = level.get_data("StartColor")
        start_rot = level.get_data("StartRotation")
        rotations = [0, 90, 180, 270]
        
        if isinstance(gc, list):
            goal_list = [{'shape': gs[j], 'color': gc[j], 'rot': gr[j], 'pos': info['goals'][j]} for j in range(len(gc))]
        else:
            goal_list = [{'shape': gs, 'color': gc, 'rot': gr, 'pos': info['goals'][0]}]
        
        self.f.write(f"\n  Level {level_idx+1}: budget={info['budget']}, dec={info['dec']}\n")
        
        cur_shape = start_shape
        cur_ci = self.color_order.index(start_color)
        cur_ri = rotations.index(start_rot)
        used_timers = set()
        
        for gi, goal in enumerate(goal_list):
            g_ci = self.color_order.index(goal['color'])
            g_ri = rotations.index(goal['rot'])
            sh = (goal['shape'] - cur_shape) % 6
            ch = (g_ci - cur_ci) % len(self.color_order)
            rh = (g_ri - cur_ri) % 4
            
            changers = []
            if sh > 0 and info['shape_ch']: changers.append(('shape', info['shape_ch'][0], sh))
            if ch > 0 and info['color_ch']: changers.append(('color', info['color_ch'][0], ch))
            if rh > 0 and info['rot_ch']: changers.append(('rot', info['rot_ch'][0], rh))
            
            # Find best waypoint order + timer usage
            best = self._plan_goal(info, game.gudziatsk.x, game.gudziatsk.y,
                                    changers, goal['pos'], used_timers)
            if best is None:
                self.f.write(f"    Goal {gi+1}: NO ROUTE\n")
                return False, None
            
            waypoints, desc, t_used = best
            self.f.write(f"    Goal {gi+1}: {' -> '.join(desc)}\n")
            
            # Execute waypoints adaptively
            cur_x, cur_y = game.gudziatsk.x, game.gudziatsk.y
            
            for wp_name, wp_pos, wp_count in waypoints:
                avoid = interactive - {wp_pos}
                result, cur_x, cur_y = self._execute_to(env, game, walls, interactive,
                                                          cur_x, cur_y, wp_pos[0], wp_pos[1],
                                                          start_lvl)
                if result == 'COMPLETE': return True, env.step(GameAction.ACTION1)  # dummy
                if result in ('GAME_OVER', 'LIFE_LOST'): return False, None
                if result not in ('OK',):
                    self.f.write(f"      {wp_name}@{wp_pos}: {result}\n")
                    return False, None
                
                if wp_name != 'TIMER':
                    for _ in range(wp_count - 1):
                        result, cur_x, cur_y = self._execute_revisit(env, game, walls, interactive,
                                                                       cur_x, cur_y, start_lvl)
                        if result == 'COMPLETE': return True, None
                        if result in ('GAME_OVER', 'LIFE_LOST'): return False, None
            
            # Go to goal
            result, cur_x, cur_y = self._execute_to(env, game, walls, interactive,
                                                      cur_x, cur_y, goal['pos'][0], goal['pos'][1],
                                                      start_lvl)
            if result == 'COMPLETE': return True, None
            if result in ('GAME_OVER', 'LIFE_LOST'): return False, None
            
            used_timers.update(t_used)
            cur_shape = (cur_shape + sh) % 6
            cur_ci = (cur_ci + ch) % len(self.color_order)
            cur_ri = (cur_ri + rh) % 4
        
        # Check if level completed
        obs_check = env.step(GameAction.ACTION1)
        self.total_actions += 1
        if obs_check.levels_completed > start_lvl:
            return True, obs_check
        return False, obs_check
    
    def _plan_goal(self, info, sx, sy, changers, goal, used_timers):
        best = None; best_len = float('inf')
        walls = info['walls']; interactive = info['interactive']
        avail_timers = [t for t in info['timers'] if t not in used_timers]
        tp_pp = {tp: info['timer_pickups'].get(tp, set()) for tp in avail_timers}
        
        perms = list(permutations(range(len(changers)))) if changers else [()]
        
        for perm in perms:
            ordered = [changers[i] for i in perm] if perm else []
            
            # No timers
            r = self._calc_route(walls, interactive, sx, sy, ordered, goal, info['budget'])
            if r and r[0] < best_len:
                best_len = r[0]; best = (ordered, r[1], set())
            
            # 1 timer
            for tp in avail_timers:
                for pp in tp_pp.get(tp, set()):
                    for ins in range(len(ordered) + 1):
                        mod = list(ordered); mod.insert(ins, ('TIMER', pp, 1))
                        r = self._calc_route(walls, interactive, sx, sy, mod, goal, info['budget'])
                        if r and r[0] < best_len:
                            best_len = r[0]; best = (mod, r[1], {tp})
            
            # 2 timers
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
                                        r = self._calc_route(walls, interactive, sx, sy, mod, goal, info['budget'])
                                        if r and r[0] < best_len:
                                            best_len = r[0]; best = (mod, r[1], {tp1, tp2})
        
        return best
    
    def _calc_route(self, walls, interactive, sx, sy, visits, goal, budget):
        total = 0; cur_x, cur_y = sx, sy; seg = 0; desc = []
        for name, pos, count in visits:
            avoid = interactive - {pos}
            p = self._bfs(walls, cur_x, cur_y, pos[0], pos[1], avoid)
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
        avoid = interactive - {goal}
        p = self._bfs(walls, cur_x, cur_y, goal[0], goal[1], avoid)
        if p is None: return None
        seg += len(p); total += len(p); desc.append(f"goal@{goal}")
        if seg > budget: return None
        return total, desc


def run():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n  LS20 Solver v12 (Adaptive)\n")
        f.write(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        
        arc = arc_agi.Arcade()
        env = arc.make("ls20")
        obs = env.step(GameAction.RESET)
        game = env._game
        
        solver = AdaptiveSolver(f)
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
                break
            obs = obs_check
        
        final_obs = obs_check if 'obs_check' in dir() else obs
        f.write(f"\n{'='*60}\n  FINAL: {final_obs.levels_completed}/7, "
                f"{solver.total_actions} actions, {final_obs.state.value}\n{'='*60}\n")
        try:
            sc = arc.get_scorecard(); f.write(f"  Score: {sc.score}\n")
        except: pass
    
    print(f"Results saved to {OUT}")

if __name__ == "__main__":
    run()
