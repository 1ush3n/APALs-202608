"""
批量运行所有测试脚本并记录输出。

用法:
    python tests/run_all_tests.py              # 运行全部测试，输出到 tests/outputs/
    python tests/run_all_tests.py --quick      # 只运行快速测试（跳过 full_pipeline 等慢测试）
    python tests/run_all_tests.py --stream     # 实时打印输出（同时保存到文件）
"""
import subprocess
import sys
import os
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "tests", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_DIR, f"test_output_{timestamp}.txt")

quick_ignore = ""
if "--quick" in sys.argv:
    quick_ignore = " --ignore=tests/test_full_pipeline.py --ignore=tests/test_domain_randomization.py"
    print("[Quick Mode] Skipping slow tests (full_pipeline, domain_randomization)")

stream_mode = "--stream" in sys.argv

print(f"Running pytest on tests/ ...")
print(f"Output will be saved to: {output_file}")

if stream_mode:
    with open(output_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pytest", "tests/", "-v", "-s"] + quick_ignore.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=ROOT_DIR,
        )
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, b""):
            decoded = line.decode("utf-8", errors="replace")
            f.write(decoded)
            f.flush()
            sys.stdout.write(decoded)
            sys.stdout.flush()
        proc.communicate()
else:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"] + quick_ignore.split(),
        capture_output=True,
        cwd=ROOT_DIR,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.stdout.decode("utf-8", errors="replace"))
        if result.stderr:
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr.decode("utf-8", errors="replace"))
    print(result.stdout.decode("utf-8", errors="replace"))

print(f"\nOutput saved to: {output_file}")
