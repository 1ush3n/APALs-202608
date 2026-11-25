import json
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
base = PROJECT_ROOT / "results" / "01_initial_main"

methods = [
    ("Operation-Only Strict", "eval_operation_only_strict"),
    ("Operation+Station Strict", "eval_operation_station_strict"),
    ("Homogeneous GraphSAGE Strict", "eval_homogeneous_graphsage_strict"),
    ("FULL-X Initial (Reference)", "eval_full_x_initial"),
]

scales = ["283", "680", "2338", "3182"]

print("=" * 88)
print("APAL INITIAL STRICT ABLATION & MAIN RESULTS (4 STANDARD BENCHMARKS)")
print("=" * 88)

records = []
for name, d in methods:
    method_dir = base / d
    row = {"Method": name}
    for scale in scales:
        summary_path = method_dir / scale / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, encoding="utf-8") as f:
                    data = json.load(f)
                mk = data.get("makespan")
                row[scale] = f"{float(mk):.2f}" if mk is not None else "N/A"
            except Exception:
                row[scale] = "Error"
        else:
            row[scale] = "N/A"
    records.append(row)

df = pd.DataFrame(records)
print(df.to_string(index=False))
print("=" * 88)

csv_out = base / "initial_ablation_summary.csv"
df.to_csv(csv_out, index=False, encoding="utf-8")
print(f"Summary table saved to: {csv_out}")
