"""
Statistical analysis for Phase 8: ARC-AGI Stochastic Resonance
Fisher exact test + binomial 95% CI for all conditions
"""
import json, os
import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SCRIPT_DIR, "results", "arc_noise_final.json")
OUTPUT = os.path.join(SCRIPT_DIR, "results", "statistical_analysis.txt")
OUTPUT_JSON = os.path.join(SCRIPT_DIR, "results", "statistical_analysis.json")

with open(RESULTS) as f:
    data = json.load(f)

def binomial_ci(k, n, alpha=0.05):
    """Wilson score interval for binomial proportion"""
    if n == 0:
        return 0, 0
    p_hat = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)

lines = []
stats_json = {}

for model_name, model_data in data.items():
    lines.append("=" * 70)
    lines.append(f"  {model_name}")
    lines.append("=" * 70)
    stats_json[model_name] = {}
    
    # baseline = sigma=0.00
    baseline = model_data["0.00"]
    
    for level in ["L1", "L2", "L3", "L4"]:
        lines.append(f"\n  --- {level} ---")
        lines.append(f"  {'σ':>6s}  {'Rate':>7s}  {'c/n':>6s}  {'95% CI':>16s}  {'Fisher p':>12s}  {'vs baseline':>12s}")
        lines.append(f"  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*16}  {'-'*12}  {'-'*12}")
        
        bl_c = baseline[level]["c"]
        bl_n = baseline[level]["n"]
        stats_json[model_name][level] = {}
        
        for sigma in sorted(model_data.keys(), key=float):
            entry = model_data[sigma][level]
            c, n = entry["c"], entry["n"]
            rate = c / n * 100
            ci_lo, ci_hi = binomial_ci(c, n)
            
            # Fisher exact test: 2x2 table
            # [[c_sigma, n_sigma - c_sigma], [c_baseline, n_baseline - c_baseline]]
            table = [[c, n - c], [bl_c, bl_n - bl_c]]
            _, p_val = stats.fisher_exact(table)
            
            sig = ""
            if p_val < 0.001: sig = "***"
            elif p_val < 0.01: sig = "**"
            elif p_val < 0.05: sig = "*"
            
            ci_str = f"[{ci_lo*100:5.1f}%, {ci_hi*100:5.1f}%]"
            
            if sigma == "0.00":
                vs_bl = "(baseline)"
            else:
                vs_bl = f"p={p_val:.4f} {sig}"
            
            lines.append(f"  σ={float(sigma):4.2f}  {rate:6.1f}%  {c:2d}/{n:2d}  {ci_str}  {vs_bl:>24s}")
            
            stats_json[model_name][level][sigma] = {
                "c": c, "n": n, "rate": round(rate, 1),
                "ci_lower": round(ci_lo * 100, 1),
                "ci_upper": round(ci_hi * 100, 1),
                "fisher_p": round(p_val, 6) if sigma != "0.00" else None,
                "significant": sig if sig else None
            }

# Summary
lines.append("\n" + "=" * 70)
lines.append("  SUMMARY: Key Statistical Tests")
lines.append("=" * 70)

for model_name in data:
    bl_c = data[model_name]["0.00"]["L2"]["c"]
    bl_n = data[model_name]["0.00"]["L2"]["n"]
    pk_c = data[model_name]["0.20"]["L2"]["c"]
    pk_n = data[model_name]["0.20"]["L2"]["n"]
    
    table = [[pk_c, pk_n - pk_c], [bl_c, bl_n - bl_c]]
    _, p = stats.fisher_exact(table)
    ci_lo, ci_hi = binomial_ci(pk_c, pk_n)
    
    lines.append(f"\n  {model_name}:")
    lines.append(f"    L2 baseline (σ=0):  {bl_c}/{bl_n} = {bl_c/bl_n*100:.1f}%")
    lines.append(f"    L2 peak (σ=0.2):    {pk_c}/{pk_n} = {pk_c/pk_n*100:.1f}%")
    lines.append(f"    Fisher exact p:     {p:.6f}")
    lines.append(f"    95% CI at peak:     [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
    lines.append(f"    Significant:        {'YES (p < 0.05)' if p < 0.05 else 'NO'}")

output_text = "\n".join(lines)
print(output_text)

with open(OUTPUT, "w") as f:
    f.write(output_text)

with open(OUTPUT_JSON, "w") as f:
    json.dump(stats_json, f, indent=2)

print(f"\nSaved to: {OUTPUT}")
print(f"Saved to: {OUTPUT_JSON}")
