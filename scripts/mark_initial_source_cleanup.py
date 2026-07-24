"""记录初始调度下载副本已在哈希核验后清理。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/01_initial_main_stochastic_20260723"

def main() -> int:
    count = 0
    for path in ARCHIVE.rglob("integrity_check_stochastic.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_cleanup"] = {
            "performed": True,
            "method": "source and archive hashes matched before deleting root-level downloaded duplicate",
            "source_is_expected_to_exist": False,
        }
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        count += 1
    print({"updated_integrity_files": count})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
