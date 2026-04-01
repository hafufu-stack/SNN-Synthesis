"""
ARC-AGI LS20 Oracle Dataset Generator v5
=========================================
Hooks env.step() to record game states. Defers static grid computation
until AFTER solver's get_level_info (avoids sprite state corruption).
"""
import torch, numpy as np, sys, io, time

sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi\environment_files\ls20\9607627b")
sys.path.insert(0, r"c:\Users\kyjan\研究\snn-synthesis\arc-agi")

import arc_agi
from arcengine import GameAction
import ls20
from agent_ls20_v14 import StateSpaceSolver

OUT = r"c:\tmp\arc_oracle_data.pt"
ACT_IDX = {GameAction.ACTION1: 0, GameAction.ACTION2: 1,
            GameAction.ACTION3: 2, GameAction.ACTION4: 3}

def p2g(px, py, pxo, pyo):
    return (px - pxo) // 5, (py - pyo) // 5

def build_static_grid(info, pxo, pyo):
    """Build static grid (7ch, 12x12) from solver's level info."""
    grid = np.zeros((7, 12, 12), dtype=np.float32)
    ch_map = {'shape': 2, 'color': 3, 'rot': 4}
    for gx in range(12):
        for gy in range(12):
            cx, cy = pxo + gx*5, pyo + gy*5
            for wx, wy in info['walls']:
                if cx >= wx and cx < wx+5 and cy >= wy and cy < wy+5:
                    grid[0, gy, gx] = 1.0; break
    for g in info.get('goal_cells', set()):
        x, y = p2g(g[0], g[1], pxo, pyo)
        if 0 <= x < 12 and 0 <= y < 12: grid[1, y, x] = 1.0
    for pos, ct in info.get('changer_cells', {}).items():
        x, y = p2g(pos[0], pos[1], pxo, pyo)
        if 0 <= x < 12 and 0 <= y < 12 and ct in ch_map:
            grid[ch_map[ct], y, x] = 1.0
    for ec in info.get('enemy_cells', set()):
        x, y = p2g(ec[0], ec[1], pxo, pyo)
        if 0 <= x < 12 and 0 <= y < 12: grid[5, y, x] = 1.0
    for tc in info.get('timer_cells', {}).keys():
        x, y = p2g(tc[0], tc[1], pxo, pyo)
        if 0 <= x < 12 and 0 <= y < 12: grid[6, y, x] = 1.0
    return grid


class RecordingSolver(StateSpaceSolver):
    """Extends solver to hook get_level_info and build static grid lazily."""
    
    def __init__(self, f, traj_f=None):
        super().__init__(f, traj_f)
        self.static_grids = {}  # li -> (grid, pxo, pyo)
        self.level_goals = {}   # li -> (gs, gci, gri)
    
    def get_level_info(self, level_idx):
        """Override to capture info AFTER parent computes it."""
        info = super().get_level_info(level_idx)
        
        # Build static grid from this level's info
        pxo = info['start'][0] % 5
        pyo = info['start'][1] % 5
        sg = build_static_grid(info, pxo, pyo)
        self.static_grids[level_idx] = (sg, pxo, pyo)
        
        # Get goal info
        rots = [0, 90, 180, 270]
        lev = info['level']
        gc_d = lev.get_data("GoalColor")
        gr_d = lev.get_data("GoalRotation")
        gs_d = lev.get_data("kvynsvxbpi")
        if isinstance(gc_d, list):
            self.level_goals[level_idx] = (
                gs_d[0], self.color_order.index(gc_d[0]), rots.index(gr_d[0]))
        else:
            self.level_goals[level_idx] = (
                gs_d, self.color_order.index(gc_d), rots.index(gr_d))
        
        return info


def main():
    t0 = time.time()
    print("=== Oracle Dataset Generator v5 ===")
    
    arc = arc_agi.Arcade()
    env = arc.make("ls20")
    obs = env.step(GameAction.RESET)
    game = env._game
    
    records = []
    
    # Create recording solver (no pre-computation!)
    f = open(r"c:\tmp\arc_oracle_solver_log.txt", "w")
    solver = RecordingSolver(f)
    solver.color_order = game.tnkekoeuk
    ncol = len(solver.color_order)
    
    # Hook env.step
    original_step = env.step
    current_level = [0]
    step_count = [0]
    
    def hooked_step(action):
        if action != GameAction.RESET:
            li = current_level[0]
            if li in solver.static_grids and li in solver.level_goals:
                sg, pxo, pyo = solver.static_grids[li]
                gs0, gci0, gri0 = solver.level_goals[li]
                
                player_ch = np.zeros((1, 12, 12), dtype=np.float32)
                px, py = game.gudziatsk.x, game.gudziatsk.y
                x, y = p2g(px, py, pxo, pyo)
                if 0 <= x < 12 and 0 <= y < 12:
                    player_ch[0, y, x] = 1.0
                full = np.concatenate([sg, player_ch], axis=0)
                
                state = np.array([
                    game.fwckfzsyc / 5.0, game.hiaauhahz / max(1, ncol-1),
                    game.cklxociuu / 3.0, gs0 / 5.0,
                    gci0 / max(1, ncol-1), gri0 / 3.0,
                    max(0, 1.0 - step_count[0] / 200.0),
                    step_count[0] / 200.0
                ], dtype=np.float32)
                
                ai = ACT_IDX.get(action, -1)
                if ai >= 0:
                    records.append((full.copy(), state.copy(), ai, li))
                step_count[0] += 1
        
        return original_step(action)
    
    env.step = hooked_step
    
    # Run solver
    for li in range(7):
        current_level[0] = li
        step_count[0] = 0
        start_lvl = li
        
        cleared, obs_check = solver.solve_level(env, game, li, start_lvl)
        if obs_check:
            obs = obs_check
        
        n_samples = sum(1 for r in records if r[3] == li)
        f.write(f"  -> Level {li+1}: {'CLEARED' if cleared else 'FAILED'}\n")
        print(f"  Level {li+1}: {'CLEARED' if cleared else 'FAILED'} ({n_samples} samples)")
        
        if not cleared or (obs_check and obs_check.state.value == 'GAME_OVER'):
            break
    
    f.close()
    
    # Build dataset
    if not records:
        print("No data!"); return
    
    grids_t = [torch.from_numpy(r[0]) for r in records]
    states_t = [torch.from_numpy(r[1]) for r in records]
    actions_t = [r[2] for r in records]
    levels_t = [r[3] for r in records]
    
    ds = {
        'grid': torch.stack(grids_t),
        'state': torch.stack(states_t),
        'action': torch.tensor(actions_t, dtype=torch.long),
        'level': torch.tensor(levels_t, dtype=torch.long),
    }
    torch.save(ds, OUT)
    
    print(f"\nSaved {len(records)} samples to {OUT}")
    for i in range(7):
        c = sum(1 for r in records if r[3] == i)
        print(f"  L{i+1}: {c}")
    print(f"Time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
