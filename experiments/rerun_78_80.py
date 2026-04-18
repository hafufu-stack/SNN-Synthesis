"""Quick re-run for Phase 78 (bugfix) and Phase 80 (bugfix)."""
import sys, os, gc, time, traceback, importlib, winsound

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EXPERIMENT_DIR, "experiments"))

def run_phase(module_name, label):
    print(f"\n{'#'*70}")
    print(f"# {label}")
    print(f"# Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")
    try:
        if module_name in sys.modules:
            mod = importlib.reload(sys.modules[module_name])
        else:
            mod = importlib.import_module(module_name)
        mod.main()
        print(f"\n  >>> {label} COMPLETED SUCCESSFULLY <<<")
    except Exception as e:
        print(f"\n  >>> {label} FAILED: {e} <<<")
        traceback.print_exc()
    gc.collect()

run_phase("phase78_conv_liquid_lif", "Phase 78: Conv-Liquid-LIF (bugfix)")
run_phase("phase80_tau_diverse_nbs", "Phase 80: tau-Diverse NBS (bugfix)")

print("\n\nDone! All re-runs complete.")
try:
    for _ in range(5):
        winsound.Beep(1200, 400)
        time.sleep(0.2)
except:
    pass
