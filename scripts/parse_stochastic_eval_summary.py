import json, os, glob, csv
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
base = PROJECT_ROOT / "results" / "01_initial_main"

methods = [
    ("Operation-Only Strict", "eval_operation_only_strict"),
    ("Operation+Station Strict", "eval_operation_station_strict"),
    ("Homogeneous GraphSAGE Strict", "eval_homogeneous_graphsage_strict"),
    ("FULL-X Initial (Main Method)", "eval_full_x_initial"),
]

scales = ["283", "680", "2338", "3182"]

print("=" * 96)
print("APAL INITIAL SCHEDULING: FULL BENCHMARK (4 SCALES x 5 SEEDS STOCHASTIC + DETERMINISTIC)")
print("=" * 96)

table_rows = []

for name, d in methods:
    method_dir = base / d
    if not method_dir.exists():
        continue
    print(f"\n>>> Method: {name}")
    row = {"Method": name}
    for scale in scales:
        det_path = method_dir / f"real_{scale}" / "temp0_seed42" / "summary.json"
        det_mk = "N/A"
        if det_path.exists():
            try:
                with open(det_path, encoding="utf-8") as f:
                    det_mk = f"{float(json.load(f).get('makespan', 0)):.2f}"
            except Exception:
                det_mk = "Err"
        
        stoch_mks = []
        for seed in range(42, 47):
            p = method_dir / f"real_{scale}" / f"temp001_seed{seed}" / "summary.json"
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        mk = json.load(f).get("makespan")
                        if mk is not None:
                            stoch_mks.append(float(mk))
                except Exception:
                    pass
        if stoch_mks:
            stoch_str = f"{np.mean(stoch_mks):.2f} ± {np.std(stoch_mks):.2f} (min={min(stoch_mks):.2f}, max={max(stoch_mks):.2f})"
        else:
            stoch_str = "Evaluating..."
        row[f"{scale}_det"] = det_mk
        row[f"{scale}_stoch"] = stoch_str
        print(f"  Instance {scale:4s} | Det (temp=0): {det_mk:8s} | Stoch 5-seeds (temp=0.01): {stoch_str}")
    table_rows.append(row)

print("\n" + "=" * 96)

out_csv = base / "full_comparison_table.csv"
if table_rows:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Method"]
        for s in scales:
            fieldnames.extend([f"{s}_det", f"{s}_stoch"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Saved full comparison CSV to: {out_csv}")
