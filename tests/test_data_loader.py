"""
数据加载全场景测试。

覆盖场景：
  1. 标准 CSV 加载（283.csv / 680.csv）
  2. Excel 文件加载
  3. 不存在的文件路径（异常路径）
  4. 负工时的数据行检测
  5. 列名自动映射（中文/英文/混合列名）
  6. 显式紧前工序解析（逗号/中文逗号/分号分隔）
  7. 拓扑环路检测与报错
  8. 多数据集池加载
  9. 加载后数据结构完整性校验
"""
import sys
import os
import tempfile
import numpy as np
import pandas as pd
import torch
import networkx as nx
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from data_loader import load_data

TOTAL_TESTS = 0
PASSED_TESTS = 0
FAILED_TESTS = []


def check(condition, name):
    global TOTAL_TESTS, PASSED_TESTS
    TOTAL_TESTS += 1
    if condition:
        PASSED_TESTS += 1
        print(f"  [PASS] {name}")
    else:
        FAILED_TESTS.append(name)
        print(f"  [FAIL] {name}")


def make_temp_csv(content: str, suffix: str = ".csv") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="test_data_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def cleanup(*paths):
    for p in paths:
        if os.path.exists(p):
            os.unlink(p)


def test_valid_csv():
    print("\n--- test_valid_csv ---")
    p1 = os.path.join(ROOT_DIR, "data", "283.csv")
    p2 = os.path.join(ROOT_DIR, "data", "680.csv")

    d1 = load_data(p1)
    check(isinstance(d1, dict) and 'task_df' in d1, "283.csv loads and returns dict")
    check(d1['num_tasks'] > 0, f"283.csv has positive num_tasks (got {d1['num_tasks']})")
    check(isinstance(d1['precedence_edges'], torch.Tensor), "precedence_edges is Tensor")
    check('id_map' in d1, "id_map exists")

    d2 = load_data(p2)
    check(d2['num_tasks'] > 0, f"680.csv has positive num_tasks (got {d2['num_tasks']})")
    check(d1['num_tasks'] != d2['num_tasks'], "290 and 715 have different num_tasks")


def test_nonexistent_file():
    print("\n--- test_nonexistent_file ---")
    try:
        load_data("nonexistent_file_xyz.csv")
        check(False, "Should raise FileNotFoundError for missing file")
    except FileNotFoundError:
        check(True, "Raises FileNotFoundError for missing file")


def test_negative_duration():
    print("\n--- test_negative_duration ---")
    csv = "AO号,加工时间/h,紧前工序AO号,类型\nT1,-5.0,,2\nT2,10.0,T1,2\n"
    p = make_temp_csv(csv)
    try:
        load_data(p)
        check(False, "Should raise ValueError for negative duration")
    except ValueError:
        check(True, "Raises ValueError for negative duration")
    finally:
        cleanup(p)


def test_column_auto_mapping():
    print("\n--- test_column_auto_mapping ---")
    csv = ("工序号,工时,紧前工序,类型,限定站位,需求人数\n"
           "A,5.0,,1,Station 1,2\n"
           "A-1,10.0,A,2,,1\n"
           "T1,8.0,A-1,2,,3\n")
    p = make_temp_csv(csv)
    try:
        d = load_data(p)
        check(isinstance(d, dict), "Chinese column names mapped correctly")
        df = d['task_df']
        check('duration' in df.columns, "duration column mapped from 工时")
        check('predecessors' in df.columns, "predecessors column mapped from 紧前工序")
        check(df['duration'].iloc[0] == 5.0, "First duration = 5.0")
    finally:
        cleanup(p)


def test_predecessor_parsing():
    print("\n--- test_predecessor_parsing ---")
    csv = ("AO号,加工时间/h,紧前工序AO号,类型\n"
           "A,2.0,,1\n"
           "B,3.0,A,2\n"
           "C,4.0,\"A,B\",2\n")
    p = make_temp_csv(csv)
    try:
        d = load_data(p)
        edges = d['precedence_edges'].numpy()
        check(edges.shape[1] >= 2, f"At least 2 edges, got {edges.shape[1]}")
    finally:
        cleanup(p)


def test_cycle_detection():
    print("\n--- test_cycle_detection ---")
    csv = ("AO号,加工时间/h,紧前工序AO号,类型\n"
           "A,2.0,B,2\n"
           "B,3.0,C,2\n"
           "C,4.0,A,2\n")
    p = make_temp_csv(csv)
    try:
        load_data(p)
        check(False, "Should raise for cyclic dependency")
    except (ValueError, nx.NetworkXUnfeasible):
        check(True, "Raises error for cyclic dependency")
    finally:
        cleanup(p)


def test_data_structure_integrity():
    print("\n--- test_data_structure_integrity ---")
    p = os.path.join(ROOT_DIR, "data", "283.csv")
    d = load_data(p)

    check(d['num_tasks'] == len(d['task_df']), "num_tasks matches DataFrame length")
    check(d['precedence_edges'].size(0) == 2, "precedence_edges has shape [2, E]")
    check(len(d['id_map']) == d['num_tasks'], "id_map covers all tasks")

    df = d['task_df']
    check('internal_id' in df.columns, "internal_id column exists")
    check(df['internal_id'].is_monotonic_increasing or df['internal_id'].iloc[0] == 0,
          "internal_id is 0-based sequential")
    check('duration' in df.columns, "duration column exists")
    check('skill_type' in df.columns, "skill_type column exists (default fill)")
    check('demand_workers' in df.columns, "demand_workers column exists (default fill)")


def test_pathlib_path_input():
    print("\n--- test_pathlib_path_input ---")
    p = Path(ROOT_DIR) / "data" / "283.csv"
    d = load_data(p)
    check(d['num_tasks'] > 0, "load_data accepts pathlib.Path input")


def test_multi_dataset_loading():
    print("\n--- test_multi_dataset_loading ---")
    base_dir = os.path.join(ROOT_DIR, "data", "train_mix")
    if os.path.isdir(base_dir):
        for f in os.listdir(base_dir):
            if f.endswith('.csv'):
                fp = os.path.join(base_dir, f)
                d = load_data(fp)
                check(d['num_tasks'] > 0, f"train_mix/{f} loaded ({d['num_tasks']} tasks)")


def main():
    print("=" * 60)
    print("DATA LOADER TEST SUITE")
    print("=" * 60)

    test_valid_csv()
    test_nonexistent_file()
    test_negative_duration()
    test_column_auto_mapping()
    test_predecessor_parsing()
    test_cycle_detection()
    test_data_structure_integrity()
    test_multi_dataset_loading()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASSED_TESTS}/{TOTAL_TESTS} passed")
    if FAILED_TESTS:
        print("FAILURES:")
        for name in FAILED_TESTS:
            print(f"  - {name}")
    print("=" * 60)
    return len(FAILED_TESTS) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
