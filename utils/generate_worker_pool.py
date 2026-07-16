import sys
import numpy as np
import pandas as pd
import random
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs import configs

def build_worker_pool_frame() -> pd.DataFrame:
    """构建技能覆盖尽可能均衡、每人具备 2~4 类技能的候选工人池。"""
    seed = int(getattr(configs, "seed", 42))
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    n_w_max = int(getattr(configs, "worker_pool_size", 1000))
    assert n_w_max >= int(configs.n_w), "候选工人池容量不得小于单回合最大工人数"

    efficiencies = np_rng.uniform(0.8, 1.2, n_w_max)
    num_skill_types = int(configs.num_skill_types)
    assert num_skill_types > 0, "工种数量必须大于 0"
    skill_matrix = np.zeros((n_w_max, num_skill_types), dtype=int)
    skill_counts = np.zeros(num_skill_types, dtype=int)

    # 主工种轮转保证五类工种的基础人数完全均衡。
    for worker_id in range(n_w_max):
        primary_skill = worker_id % num_skill_types
        skill_matrix[worker_id, primary_skill] = 1
        skill_counts[primary_skill] += 1

    # 补充技能始终优先分配给当前覆盖人数最少的工种。
    for worker_id in range(n_w_max):
        target_skills = py_rng.randint(2, min(4, num_skill_types))
        while int(skill_matrix[worker_id].sum()) < target_skills:
            available = np.where(skill_matrix[worker_id] == 0)[0]
            min_count = int(skill_counts[available].min())
            candidates = available[skill_counts[available] == min_count]
            selected = int(py_rng.choice(candidates.tolist()))
            skill_matrix[worker_id, selected] = 1
            skill_counts[selected] += 1

    workers = []
    for worker_id in range(n_w_max):
        row = {'worker_id': worker_id, 'efficiency': efficiencies[worker_id]}
        for skill_id in range(num_skill_types):
            row[f'skill_{skill_id}'] = skill_matrix[worker_id, skill_id]
        workers.append(row)

    frame = pd.DataFrame(workers)
    assert int(frame["worker_id"].nunique()) == n_w_max, "工人编号必须唯一"
    assert int(skill_counts.max() - skill_counts.min()) <= 1, "五类技能覆盖人数必须近似相等"
    return frame


def generate_worker_pool() -> Path:
    """将候选工人池写入配置指定位置并返回绝对路径。"""
    output_path = Path(getattr(configs, 'worker_pool_path', 'data/worker_pool_fixed.csv'))
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = build_worker_pool_frame()
    frame.to_csv(output_path, index=False)
    print(f"成功生成动态采样候选工人池（共 {len(frame)} 名工人），保存于：{output_path}")
    return output_path

if __name__ == "__main__":
    generate_worker_pool()
