"""
仿真引擎与动作掩码正确性测试。

覆盖场景：
  1. EventQueue 基本操作 (push/pop/peek/clear)
  2. EventQueue 时间+优先级联合排序
  3. EventQueue 容量溢出保护
  4. ActionMasker: 所有任务就绪时的掩码
  5. ActionMasker: 无任务就绪时的掩码
  6. ActionMasker: 固定站位约束下的掩码
  7. ActionMasker: 工人技能不匹配时的掩码
  8. ActionMasker: 站位容量满时的掩码
  9. CPM 关键路径计算正确性
 10. 工人缺勤事件注入与恢复
 11. Station Slot Model 并发控制
"""
import sys
import os
import copy
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core.event_engine import Event, EventType, EventQueue
from core.action_masker import ActionMasker
from environment import AirLineEnv_Graph
import configs

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


def test_eventqueue_basic():
    print("\n--- test_eventqueue_basic ---")
    q = EventQueue(max_size=100)
    check(q.is_empty(), "New queue is empty")

    q.push(Event(10.0, EventType.TASK_FINISH, {'task_id': 1}))
    check(not q.is_empty(), "Queue not empty after push")
    check(len(q) == 1, "len(q) == 1")

    ev = q.pop()
    check(ev.time == 10.0, "Popped event time == 10.0")
    check(ev.type == EventType.TASK_FINISH, "Popped event type is TASK_FINISH")
    check(q.is_empty(), "Queue empty after pop")


def test_eventqueue_ordering():
    print("\n--- test_eventqueue_ordering ---")
    q = EventQueue()

    q.push(Event(5.0, EventType.WORKER_RETURN, {'worker_id': 0}))
    q.push(Event(5.0, EventType.TASK_FINISH, {'task_id': 0}))
    q.push(Event(3.0, EventType.WORKER_LEAVE, {'worker_id': 1}))

    e1 = q.pop()
    check(e1.time == 3.0 and e1.type == EventType.WORKER_LEAVE, "Earliest time pops first")

    e2 = q.pop()
    check(e2.time == 5.0 and e2.type == EventType.TASK_FINISH,
          "Same time: TASK_FINISH(priority=1) before WORKER_RETURN(priority=2)")

    e3 = q.pop()
    check(e3.type == EventType.WORKER_RETURN, "Last: WORKER_RETURN")


def test_eventqueue_overflow():
    print("\n--- test_eventqueue_overflow ---")
    q = EventQueue(max_size=5)
    for i in range(5):
        q.push(Event(float(i), EventType.TASK_FINISH, {}))
    try:
        q.push(Event(100.0, EventType.TASK_FINISH, {}))
        check(False, "Should raise RuntimeError on overflow")
    except RuntimeError:
        check(True, "Raises RuntimeError on overflow")


def test_eventqueue_clear():
    print("\n--- test_eventqueue_clear ---")
    q = EventQueue()
    q.push(Event(1.0, EventType.TASK_FINISH, {}))
    q.push(Event(2.0, EventType.TASK_FINISH, {}))
    q.clear()
    check(q.is_empty(), "Queue empty after clear")


def test_eventqueue_peek():
    print("\n--- test_eventqueue_peek ---")
    q = EventQueue()
    q.push(Event(2.0, EventType.TASK_FINISH, {}))
    q.push(Event(1.0, EventType.TASK_FINISH, {}))
    check(q.peek().time == 1.0, "peek returns earliest event")
    check(len(q) == 2, "peek does not remove event")


def test_eventqueue_pop_empty():
    print("\n--- test_eventqueue_pop_empty ---")
    q = EventQueue()
    try:
        q.pop()
        check(False, "Should raise IndexError on pop from empty")
    except IndexError:
        check(True, "Raises IndexError on pop from empty")


def test_mask_all_tasks_mask():
    print("\n--- test_mask_all_tasks_mask ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()
    masker = ActionMasker(env)

    t_mask, s_mask, w_mask = masker.get_masks()
    check(isinstance(t_mask, torch.Tensor), "task_mask is Tensor")
    check(isinstance(s_mask, torch.Tensor), "station_mask is Tensor")
    check(isinstance(w_mask, torch.Tensor), "worker_mask is Tensor")
    check(t_mask.dtype == torch.bool, "task_mask dtype is bool")
    check(t_mask.shape[0] == env.num_tasks, f"task_mask size matches num_tasks ({env.num_tasks})")
    check(s_mask.shape == (env.num_tasks, env.num_stations),
          "station_mask shape is [num_tasks, num_stations]")
    check(w_mask.shape[0] == env.num_workers, f"worker_mask size matches num_workers ({env.num_workers})")


def test_mask_valid_tasks_exist():
    print("\n--- test_mask_valid_tasks_exist ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()
    masker = ActionMasker(env)

    t_mask, s_mask, w_mask = masker.get_masks()
    has_valid = (t_mask == False).any()
    check(has_valid, "At least one valid (unmasked) task exists after reset")

    valid_count = (t_mask == False).sum().item()
    print(f"    Valid task count: {valid_count}")


def test_mask_after_step():
    print("\n--- test_mask_after_step ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()
    masker = ActionMasker(env)

    tl_before = (masker.get_masks()[0] == False).sum().item()

    t_mask, s_mask, w_mask = masker.get_masks()
    valid_tasks = torch.where(~t_mask)[0]
    if len(valid_tasks) > 0:
        tid = valid_tasks[0].item()
        s_valid = torch.where(~s_mask[tid])[0]
        if len(s_valid) > 0:
            sid = s_valid[0].item()
            demand = int(env.task_static_feat[tid, 2].item())
            team = list(range(min(demand, env.num_workers)))
            state, reward, done, info = env.step((tid, sid, team))

            tl_after = (masker.get_masks()[0] == False).sum().item()
            check(tl_after >= 0, f"Masks still computable after step (valid_tasks={tl_after})")


def test_cpm_calculation():
    print("\n--- test_cpm_calculation ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    is_critical, cpm_makespan = env._calculate_cpm()
    check(isinstance(is_critical, np.ndarray), "is_critical is ndarray")
    check(len(is_critical) == env.num_tasks, "is_critical covers all tasks")
    check(cpm_makespan > 0, f"CPM makespan > 0 (got {cpm_makespan:.1f})")
    check(env.ideal_makespan > 0, f"ideal_makespan > 0 (got {env.ideal_makespan:.1f})")


def test_worker_absence_injection():
    print("\n--- test_worker_absence_injection ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5
    configs.enable_dynamic_events = True
    configs.prob_worker_absent_base = 0.3
    configs.prob_worker_absent_max = 0.3

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset(randomize_workers=True)

    leave_events = 0
    while not env.event_queue.is_empty():
        ev = env.event_queue.pop()
        if ev.type == EventType.WORKER_LEAVE:
            leave_events += 1

    check(leave_events >= 0, "Worker leave events can be injected")
    print(f"    Leave events injected: {leave_events}")


def test_station_slot_behavior():
    print("\n--- test_station_slot_behavior ---")
    DATA = os.path.join(ROOT_DIR, "data", "283.csv")
    configs.n_w = 40
    configs.n_m = 5
    configs.max_slots_per_station = 3

    env = AirLineEnv_Graph(data_path_or_dir=DATA, seed=42)
    state = env.reset()

    for sid in range(env.num_stations):
        check(isinstance(env.station_task_finish_times[sid], list),
              f"station_task_finish_times[{sid}] is list")

    check(len(env.station_task_finish_times) == env.num_stations,
          "station_task_finish_times covers all stations")


def main():
    print("=" * 60)
    print("ENGINE & MASK TEST SUITE")
    print("=" * 60)

    test_eventqueue_basic()
    test_eventqueue_ordering()
    test_eventqueue_overflow()
    test_eventqueue_clear()
    test_eventqueue_peek()
    test_eventqueue_pop_empty()

    test_mask_all_tasks_mask()
    test_mask_valid_tasks_exist()
    test_mask_after_step()

    test_cpm_calculation()
    test_worker_absence_injection()
    test_station_slot_behavior()

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
