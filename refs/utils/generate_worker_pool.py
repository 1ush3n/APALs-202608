import sys
import numpy as np
import pandas as pd
import random
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs import configs

def generate_worker_pool():
    np.random.seed(2026)
    random.seed(2026)
    
    # 根据用户请求，为了满足3000级别超大订单集的鲁棒性，直接生成 1000 名全技能库超级备用池！
    n_w_max = 1000
    output_path = Path(getattr(configs, 'worker_pool_path', 'data/worker_pool_fixed.csv'))
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    workers = []
    
    # 1. 效率因子在 0.8 到 1.2 之间
    efficiencies = np.random.uniform(0.8, 1.2, n_w_max)
    
    # 2. 技能矩阵 (假设最多 10 种技能)
    skill_matrix = np.zeros((n_w_max, 10), dtype=int)
    
    # 保底机制：强制每种技能至少有 5 名顶尖专家
    for s_idx in range(10):
        experts = random.sample(range(n_w_max), 5)
        for e in experts:
            skill_matrix[e, s_idx] = 1
            
    # 其余随机分配 (每人 1~4 个技能)
    for w in range(n_w_max):
        current_skills = np.sum(skill_matrix[w])
        target_skills = random.randint(2, 4)  # 每人至少 2 技能，缓解大数据集死锁
        
        if current_skills < target_skills:
            num_to_add = target_skills - current_skills
            available_skills = np.where(skill_matrix[w] == 0)[0]
            if len(available_skills) > 0:
                selected = np.random.choice(available_skills, size=min(num_to_add, len(available_skills)), replace=False)
                skill_matrix[w, selected] = 1
                
    # 3. 构建 DataFrame
    for w in range(n_w_max):
        row = {'worker_id': w, 'efficiency': efficiencies[w]}
        for s in range(10):
            row[f'skill_{s}'] = skill_matrix[w, s]
        workers.append(row)
        
    df = pd.DataFrame(workers)
    df.to_csv(output_path, index=False)
    print(f"✅ 成功生成固定工人技能池 (共 {n_w_max} 名工人)，保存于: {output_path}")

if __name__ == "__main__":
    generate_worker_pool()
