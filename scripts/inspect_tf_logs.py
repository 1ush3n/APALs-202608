from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_tf_logs(log_dir: Path | str, last_n: int = 5, filter_group: str | None = None):
    log_dir = Path(log_dir)
    event_files = sorted(list(log_dir.glob("**/events.out.tfevents.*")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not event_files:
        print(f"未找到任何 tfevents 文件于: {log_dir}")
        return

    latest_file = event_files[0]
    print("=" * 80)
    print(f"📊 [TensorBoard Inspector] 最新日志: {latest_file.name}")
    print(f"   文件大小: {latest_file.stat().st_size / 1024:.2f} KB | 路径: {latest_file.parent}")
    print("=" * 80)

    ea = EventAccumulator(str(latest_file))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("当前尚未写入任何标量 (Scalar) 指标。")
        return

    groups = {}
    for tag in sorted(tags):
        prefix = tag.split("/")[0] if "/" in tag else "General"
        if filter_group and filter_group.lower() not in prefix.lower():
            continue
        groups.setdefault(prefix, []).append(tag)

    print(f"匹配标量字段数: {sum(len(v) for v in groups.values())} | 分组: {list(groups.keys())}\n")

    for group_name, group_tags in groups.items():
        print(f"--- 🔹 [{group_name}] ---")
        for tag in group_tags:
            events = ea.Scalars(tag)
            if not events:
                continue
            recent = events[-last_n:] if len(events) >= last_n else events
            steps_vals = [(e.step, e.value) for e in recent]
            latest_val = steps_vals[-1][1]
            if len(steps_vals) > 1:
                trend = f" (最新 step {steps_vals[-1][0]}: {latest_val:.4f} | 初值 step {steps_vals[0][0]}: {steps_vals[0][1]:.4f})"
                sample_str = " -> ".join([f"{v:.4f}" for _, v in steps_vals[-min(5, len(steps_vals)):]])
                print(f"  • {tag:42s}: {latest_val:12.4e} | 最近走势: [{sample_str}]{trend}")
            else:
                print(f"  • {tag:42s}: {latest_val:12.4e} (step {steps_vals[0][0]})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TensorBoard logs")
    parser.add_argument("log_dir", type=str, help="Path to tensorboard log directory")
    parser.add_argument("--last-n", type=int, default=5, help="Number of recent values to show")
    parser.add_argument("--group", type=str, default=None, help="Filter by group prefix")
    args = parser.parse_args()
    parse_tf_logs(args.log_dir, args.last_n, args.group)
