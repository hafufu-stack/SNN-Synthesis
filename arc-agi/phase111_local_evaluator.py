"""
Phase 111: Local ARC Evaluator (v2)

Test agents locally using installed arc_agi/arcengine packages.
Uses obs (FrameDataRaw) directly as frame input.

Usage:
    python phase111_local_evaluator.py [agent_file] [n_episodes]

Author: Hiroto Funasaki
"""
import os, sys, json, time, random
import numpy as np
import types

# Setup environment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARC_DIR = os.path.join(SCRIPT_DIR, 'environment_files')
os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = ARC_DIR

# Mock agents.agent module (Kaggle-only)
class MockAgent:
    def __init__(self, *args, **kwargs):
        self.game_id = kwargs.get('game_id', 'unknown')
        self.frames = []

agents_mod = types.ModuleType('agents')
agent_submod = types.ModuleType('agents.agent')
agent_submod.Agent = MockAgent
agents_mod.agent = agent_submod
sys.modules['agents'] = agents_mod
sys.modules['agents.agent'] = agent_submod

import arc_agi
from arcengine import GameAction, GameState


def load_agent_class(agent_file):
    """Load MyAgent class from agent file."""
    with open(agent_file, 'r', encoding='utf-8') as f:
        code = f.read()
    lines = code.split('\n')
    if lines[0].startswith('%%writefile'):
        code = '\n'.join(lines[1:])
    ns = {'__name__': '__evaluator__'}
    exec(code, ns)
    return ns.get('MyAgent')


def play_game(agent_class, game_id, max_steps=300, max_time=60):
    """Play one game episode. Returns levels cleared."""
    try:
        arcade = arc_agi.Arcade()
        env = arcade.make(game_id)
    except Exception as e:
        return 0

    try:
        agent = agent_class(game_id=game_id)
    except Exception as e:
        return 0

    t0 = time.time()
    max_lc = 0
    frames = []

    try:
        obs = env.step(GameAction.RESET)
    except:
        return 0

    for step in range(max_steps):
        if time.time() - t0 > max_time:
            break

        frames.append(obs)
        agent.frames = frames  # Mock Kaggle Agent base class

        # Check if done
        try:
            if agent.is_done(frames, obs):
                break
        except:
            pass

        # Choose action
        try:
            action = agent.choose_action(frames, obs)
        except:
            action = GameAction.RESET

        # Step
        try:
            obs = env.step(action)
            if obs.levels_completed > max_lc:
                max_lc = obs.levels_completed
            if obs.state == GameState.WIN:
                max_lc = max(max_lc, obs.levels_completed or 0)
                break
            if obs.state == GameState.GAME_OVER:
                # Let agent handle GAME_OVER via choose_action (RESET)
                pass
        except:
            try:
                obs = env.step(GameAction.RESET)
            except:
                break

    return max_lc


def evaluate_agent(agent_file, n_episodes=10, max_steps=300, max_time_per_game=30):
    """Full evaluation across all games."""
    print(f"{'='*60}")
    print(f"Phase 111: Local ARC Evaluator")
    print(f"  Agent: {os.path.basename(agent_file)}")
    print(f"  Episodes per game: {n_episodes}")
    print(f"  Max steps: {max_steps}, Time/game: {max_time_per_game}s")
    print(f"{'='*60}")

    agent_class = load_agent_class(agent_file)
    if agent_class is None:
        print("ERROR: Could not load MyAgent class!")
        return

    games = sorted([d for d in os.listdir(ARC_DIR)
                   if os.path.isdir(os.path.join(ARC_DIR, d))])
    print(f"  Games: {len(games)} ({', '.join(games)})\n")

    results = {}
    total_start = time.time()

    for game_id in games:
        print(f"  [{game_id.upper():6s}] ", end="", flush=True)
        game_results = []

        for ep in range(n_episodes):
            lc = play_game(agent_class, game_id, max_steps, max_time_per_game)
            game_results.append(lc)
            if lc > 0:
                print(f"L{lc}", end=" ", flush=True)
            else:
                print(".", end="", flush=True)

        best = max(game_results)
        avg = sum(game_results) / len(game_results)
        rate = sum(1 for x in game_results if x > 0) / len(game_results)
        results[game_id] = {'best': best, 'avg': avg, 'miracle_rate': rate,
                           'per_episode': game_results}
        print(f"  best={best} avg={avg:.2f} rate={rate*100:.0f}%")

    total_time = time.time() - total_start
    n_solved = sum(1 for r in results.values() if r['best'] > 0)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {n_solved}/{len(games)} = {n_solved/max(len(games),1)*100:.1f}%")
    print(f"  Time: {total_time:.0f}s")
    for gid in sorted(results, key=lambda g: results[g]['best'], reverse=True):
        r = results[gid]
        s = "SOLVED" if r['best'] > 0 else "FAILED"
        print(f"    {gid:6s}: [{s:6s}] best={r['best']} rate={r['miracle_rate']*100:.0f}%")
    print(f"{'='*60}")

    out_path = os.path.join(SCRIPT_DIR, 'results', 'local_eval_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'agent': os.path.basename(agent_file),
            'n_episodes': n_episodes, 'n_solved': n_solved,
            'n_games': len(games), 'results': results,
            'total_time_s': total_time
        }, f, indent=2)
    return results


if __name__ == '__main__':
    agent_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        SCRIPT_DIR, 'kaggle_cell2_agent.py')
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    evaluate_agent(agent_file, n_episodes=n_ep, max_steps=300, max_time_per_game=30)
