"""Probe ARC-AGI-3 observation object to find visual/frame data."""
import os, sys
os.environ['OPERATION_MODE'] = 'OFFLINE'
os.environ['ENVIRONMENTS_DIR'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'environment_files')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arc_agi
from arcengine import GameAction
import numpy as np

for game_id in ['tr87', 'ls20', 'm0r0']:
    print(f"\n{'='*60}")
    print(f"  Game: {game_id}")
    print(f"{'='*60}")
    
    arcade = arc_agi.Arcade()
    env = arcade.make(game_id)
    obs = env.step(GameAction.RESET)
    game = env._game
    
    # Probe observation object
    print(f"\n  obs type: {type(obs)}")
    print(f"  obs attrs: {[a for a in dir(obs) if not a.startswith('_')]}")
    
    for attr in sorted(dir(obs)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(obs, attr)
            if callable(val):
                continue
            if hasattr(val, 'shape'):
                print(f"  obs.{attr}: shape={val.shape}, dtype={val.dtype}, range=[{val.min():.2f}, {val.max():.2f}]")
            elif hasattr(val, '__len__') and not isinstance(val, str):
                print(f"  obs.{attr}: len={len(val)}, type={type(val).__name__}")
                if len(val) > 0:
                    first = val[0] if hasattr(val, '__getitem__') else list(val)[0]
                    if hasattr(first, 'shape'):
                        print(f"    first element: shape={first.shape}")
                    elif hasattr(first, '__len__'):
                        print(f"    first element: len={len(first)}")
                    else:
                        print(f"    first element: {first}")
            else:
                print(f"  obs.{attr}: {val}")
        except Exception as e:
            print(f"  obs.{attr}: ERROR {e}")
    
    # Probe game object
    print(f"\n  game type: {type(game)}")
    numeric_attrs = []
    grid_attrs = []
    for attr in sorted(dir(game)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(game, attr)
            if callable(val):
                continue
            if hasattr(val, 'shape'):
                grid_attrs.append((attr, val.shape, val.dtype))
            elif isinstance(val, (list, tuple)) and len(val) > 0:
                if isinstance(val[0], (list, tuple)):
                    grid_attrs.append((attr, f"list[{len(val)}][{len(val[0])}]", type(val[0][0]).__name__))
                else:
                    numeric_attrs.append((attr, val[:5] if len(val) <= 5 else f"len={len(val)}"))
            elif isinstance(val, (int, float, bool)):
                numeric_attrs.append((attr, val))
        except:
            pass
    
    print(f"  Numeric attrs ({len(numeric_attrs)}):")
    for name, val in numeric_attrs:
        print(f"    {name}: {val}")
    print(f"  Grid/array attrs ({len(grid_attrs)}):")
    for name, shape, dt in grid_attrs:
        print(f"    {name}: {shape}, {dt}")
    
    # Take a step and check frame
    obs2 = env.step(GameAction.ACTION1)
    print(f"\n  After ACTION1:")
    for attr in sorted(dir(obs2)):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(obs2, attr)
            if callable(val):
                continue
            if hasattr(val, 'shape'):
                print(f"  obs.{attr}: shape={val.shape}, dtype={val.dtype}")
        except:
            pass
    
    del env, arcade
    print()
