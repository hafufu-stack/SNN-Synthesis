"""
ARC-AGI-3 Explorer: First contact with the environment
=======================================================
Goal: Understand the game structure, action space, observation format.
This runs on CPU only - no GPU needed.
"""
import arc_agi
from arcengine import GameAction
import json

def explore_game(game_id="ls20", n_steps=20):
    """Explore a single ARC-AGI-3 game to understand its structure."""
    print(f"\n{'='*60}")
    print(f"  Exploring game: {game_id}")
    print(f"{'='*60}")
    
    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode="terminal")
    
    # Check available actions
    all_actions = list(GameAction)
    print(f"\n  Available actions ({len(all_actions)}):")
    for a in all_actions:
        print(f"    - {a.name} = {a.value}")
    
    # Take random actions and observe
    print(f"\n  Taking {n_steps} random actions...")
    import random
    for step in range(n_steps):
        action = random.choice(all_actions)
        try:
            obs = env.step(action)
            if step < 5 or step == n_steps - 1:  # Print first 5 and last
                print(f"\n  Step {step+1}: action={action.name}")
                print(f"    Observation type: {type(obs).__name__}")
                if hasattr(obs, '__dict__'):
                    for k, v in obs.__dict__.items():
                        if isinstance(v, (int, float, str, bool)):
                            print(f"    {k}: {v}")
                        elif isinstance(v, (list, tuple)) and len(v) < 20:
                            print(f"    {k}: {v}")
                        else:
                            print(f"    {k}: {type(v).__name__} (len={len(v) if hasattr(v, '__len__') else '?'})")
        except Exception as e:
            print(f"  Step {step+1}: action={action.name} -> ERROR: {e}")
    
    # Get scorecard
    scorecard = arc.get_scorecard()
    print(f"\n  Scorecard: {scorecard}")
    
    return scorecard

def list_available_games():
    """List games we can try."""
    # Known game IDs from docs
    game_ids = ["ls20", "ft09"]
    print("\n  Known game IDs from docs: ls20, ft09")
    print("  Full list at: https://arcprize.org/tasks")
    return game_ids

if __name__ == "__main__":
    print("="*60)
    print("  ARC-AGI-3 Environment Explorer")
    print("  CPU-only, no GPU needed")
    print("="*60)
    
    # List available games
    games = list_available_games()
    
    # Explore the first game
    for gid in games:
        try:
            explore_game(gid, n_steps=10)
        except Exception as e:
            print(f"\n  ERROR exploring {gid}: {e}")
            import traceback
            traceback.print_exc()
