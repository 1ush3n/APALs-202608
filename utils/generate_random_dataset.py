import sys
from pathlib import Path
# 把父目录加入系统路径以便加载 environment
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import argparse
import re
import random
from environment import AirLineEnv_Graph

def parse_args():
    parser = argparse.ArgumentParser(description="生成基于原工序拓扑结构的随机扰动数据集 (支持拓扑剪枝和安全加点防死锁)")
    parser.add_argument('--output_dir', type=str, default='data/random_datasets', help="输出目录")
    parser.add_argument('--num_samples', type=int, default=10, help="批量生成的样本数量")
    parser.add_argument('--min_length', type=int, default=200, help="生成的最小数据集节点数量")
    parser.add_argument('--max_length', type=int, default=500, help="生成的最大数据集节点数量")
    parser.add_argument('--time_var', type=float, default=0.2, help="工时高斯波动的标准差系数 (如 0.2 代表上下浮动约 20%)")
    parser.add_argument('--seed', type=int, default=None, help="随机种子")
    return parser.parse_args()

def get_active_ancestors(node, drop_set, pred_map, memo, visited):
    """
    递归寻找依赖：如果直接前驱被删了，就越过它去找前驱的前驱。
    """
    if node in visited:
        return set()
    visited.add(node)
    
    if node in memo:
        return memo[node]
        
    if node not in pred_map or not pred_map[node]:
        return {node} if node not in drop_set else set()
        
    active_preds = set()
    for p in pred_map[node]:
        if p in drop_set:
            active_preds.update(get_active_ancestors(p, drop_set, pred_map, memo, visited.copy()))
        else:
            active_preds.add(p)
            
    memo[node] = active_preds
    return active_preds

def find_best_template(target_length):
    """
    动态寻找 data 目录下最适合作为模板的 CSV 文件。
    优先寻找 >= target_length 且差距最小的。
    """
    data_dir = PROJECT_ROOT / "data"
    candidates = []
    for path in data_dir.iterdir():
        if path.suffix == ".csv" and path.name not in ["worker_pool_fixed.csv"]:
            try:
                df = pd.read_csv(path, dtype=str)
                if 'AO号' in df.columns and '类型' in df.columns:
                    type2_count = len(df[df['类型'].astype(str) == '2'])
                    candidates.append((path, type2_count))
            except:
                pass
                
    if not candidates:
        raise ValueError("在 data 目录下未找到合格的基底模板 CSV！")
        
    # 找到 >= target_length 且差距最小的
    valid_cands = [c for c in candidates if c[1] >= target_length]
    if valid_cands:
        valid_cands.sort(key=lambda x: x[1] - target_length)
        return valid_cands[0][0]
    else:
        # 如果都比 target 小，选最大的那个
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

def generate_random_dataset(template_path, output_path, target_length, time_var):
    df = pd.read_csv(template_path, dtype=str)
    
    df['类型'] = df['类型'].astype(int)
    df['加工时间/h'] = df['加工时间/h'].astype(float)
    
    # 解析所有任务的依赖图
    pred_map = {}
    edges = [] # 记录所有的有效边 (前驱 -> 当前节点)
    
    for idx, row in df.iterrows():
        node_id = str(row['AO号']).strip()
        preds_str = str(row.get('紧前工序AO号', ''))
        
        preds_list = []
        if pd.notna(preds_str) and preds_str.lower() not in ['nan', 'none', '', '0']:
            for p in re.split(r'[,，]', preds_str):
                p = p.strip()
                if p: 
                    preds_list.append(p)
                    edges.append((p, node_id))
        pred_map[node_id] = preds_list

    type2_indices = df[df['类型'] == 2].index.tolist()
    current_type2_len = len(type2_indices)
    
    drop_set = set()
    
    # 计算我们需要增删的数量
    # 为了增加多样性，我们如果原本需要删掉 N 个，我们可以超额删 (N + M) 个，然后再通过 Edge Subdivision 增加 M 个。
    if current_type2_len >= target_length:
        base_drop = current_type2_len - target_length
        extra_drop_and_add = random.randint(0, int(target_length * 0.1)) # 额外乱序 0~10%
        num_to_drop = base_drop + extra_drop_and_add
        num_to_add = extra_drop_and_add
    else:
        # 如果模板长度反而不够，就只加不减（或者微减再多加）
        base_add = target_length - current_type2_len
        extra_drop_and_add = random.randint(0, int(current_type2_len * 0.05))
        num_to_drop = extra_drop_and_add
        num_to_add = base_add + extra_drop_and_add
        
    num_to_drop = min(num_to_drop, len(type2_indices))
    
    if num_to_drop > 0:
        drop_indices = np.random.choice(type2_indices, num_to_drop, replace=False)
        drop_set = set(df.loc[drop_indices, 'AO号'].str.strip())
        df = df.drop(drop_indices)
        
    # 重连因删减断裂的边 (Bypassing)
    memo = {}
    for idx, row in df.iterrows():
        node_id = str(row['AO号']).strip()
        new_preds = set()
        for p in pred_map.get(node_id, []):
            if p in drop_set:
                new_preds.update(get_active_ancestors(p, drop_set, pred_map, memo, set()))
            else:
                new_preds.add(p)
                
        if new_preds:
            df.at[idx, '紧前工序AO号'] = ','.join(sorted(list(new_preds)))
        else:
            df.at[idx, '紧前工序AO号'] = ''
            
    # 执行 Edge Subdivision (增加无环节点)
    if num_to_add > 0:
        # 重新扫描现有的合法边
        current_edges = []
        for idx, row in df.iterrows():
            node_id = str(row['AO号']).strip()
            preds_str = str(row.get('紧前工序AO号', ''))
            if pd.notna(preds_str) and preds_str.lower() not in ['nan', 'none', '', '0']:
                for p in re.split(r'[,，]', preds_str):
                    p = p.strip()
                    if p: current_edges.append((p, node_id, idx)) # 包含 idx 方便修改 df
                    
        # 获取现有特征分布以供采样
        existing_demands = df.loc[df['类型'] == 2, '需求人数'].dropna().tolist()
        if not existing_demands: existing_demands = ['1', '2']
        
        added_count = 0
        
        # 只要还有边可以分，就进行划分
        while added_count < num_to_add and len(current_edges) > 0:
            # 随机挑选一条现有的依赖边 A -> B
            edge_idx = random.randint(0, len(current_edges) - 1)
            node_A, node_B, df_idx_B = current_edges.pop(edge_idx)
            
            # 创建新节点 N
            node_N = f"RAND-N{random.randint(1000, 99999)}-{added_count}"
            
            # 找到现有的 A 的特征，作为参考
            A_rows = df[df['AO号'].str.strip() == node_A]
            mean_duration = A_rows['加工时间/h'].values[0] if not A_rows.empty and float(A_rows['加工时间/h'].values[0]) > 0 else 1.0
            
            # 给新节点赋予随机属性
            n_dur = max(0.1, np.random.normal(float(mean_duration), float(mean_duration) * time_var))
            n_demand = random.choice(existing_demands)
            
            new_row = {
                'AO号': node_N,
                '类型': 2,
                '紧前工序AO号': node_A,
                '需求人数': n_demand,
                '加工时间/h': round(n_dur, 2),
                '限定站位': '',
                '部位容量': ''
            }
            
            # 修改节点 B 的紧前工序：将 A 替换为 N
            b_preds_str = str(df.at[df_idx_B, '紧前工序AO号'])
            b_preds_list = [p.strip() for p in re.split(r'[,，]', b_preds_str) if p.strip()]
            
            if node_A in b_preds_list:
                b_preds_list.remove(node_A)
                b_preds_list.append(node_N)
                df.at[df_idx_B, '紧前工序AO号'] = ','.join(b_preds_list)
                
            # 【核心修复】直接在 CSV 的物理位置上，将 N 插入到 B 的紧邻上方
            # 这样 N 就能完美继承 B 所在的隐式 Subgroup，绝对避免产生前后逆流的循环依赖 (Cyclic Dependency)
            df.loc[df_idx_B - 0.001 - (added_count * 0.0001)] = new_row
            
            added_count += 1
            
        if added_count > 0:
            # 将按小数索引插入的新行重新排序，恢复整数顺序
            df = df.sort_index().reset_index(drop=True)
            
    # 工时随机扰动 (仅对剩下的 Type 2，且新节点刚才已经随过了，其实这里也可以统一步骤)
    surviving_type2_mask = df['类型'] == 2
    mu = df.loc[surviving_type2_mask, '加工时间/h'].astype(float).values
    sigma = mu * time_var
    new_durations = np.random.normal(loc=mu, scale=sigma)
    new_durations = np.maximum(0.1, new_durations)
    df.loc[surviving_type2_mask, '加工时间/h'] = np.round(new_durations, 2)
    
    # 确保保存时原本空的地方还是空的
    df = df.fillna('')
    
    # 特别注意：AO号 可能需要重排以满足顺序性 (但 APAL 支持字符串 ID)
    # 不过重新编号“序号”是必要的，防止存在空洞
    # 但我们为了不打乱原来的相对先后逻辑，我们最好保持原来的物理依赖顺序，将新增的节点放在最后或者重新拓扑排序。
    # pandas concat 放在了最后，通常是没问题的。
    df = df.reset_index(drop=True)
    if '序号' in df.columns:
        df['序号'] = df.index + 1
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return num_to_drop, num_to_add

def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始生成随机数据集...")
    print(f"   - 预期生成数量: {args.num_samples}")
    print(f"   - 长度随机区间: [{args.min_length}, {args.max_length}]")
    print(f"   - 工时波动系数: {args.time_var * 100:.1f}%")
    print("-" * 50)
    
    success_count = 0
    for i in range(1, args.num_samples + 1):
        target_len = random.randint(args.min_length, args.max_length)
        template_path = find_best_template(target_len)
        base_name = Path(template_path).stem
        
        out_name = f"Mix_L{target_len}_T{base_name}_s{i}.csv"
        out_path = output_dir / out_name
        
        try:
            drop_cnt, add_cnt = generate_random_dataset(template_path, out_path, target_len, args.time_var)
            
            # --- 合法性验证 ---
            # 通过尝试实例化环境，如果能顺利通过拓扑排序，就证明 DAG 完全合法，无死锁环
            env = AirLineEnv_Graph(data_path_or_dir=out_path, seed=42)
            actual_tasks = env.num_tasks
            
            print(f"[SUCCESS] [{i}/{args.num_samples}] 已生成且验证合法: {out_name}")
            print(f"      (基底: {base_name}.csv | 目标: {target_len} | 删减: {drop_cnt} | 细分新增: {add_cnt} | 最终图节点数: {actual_tasks})")
            success_count += 1
            
        except Exception as e:
            print(f"[FAIL] [{i}/{args.num_samples}] 生成 {out_name} 失败或验证不通过: {e}")
            if out_path.exists():
                out_path.unlink() # 清理废弃数据
        
    print("-" * 50)
    print(f"全部完成！成功生成并通过验证的合法数据集: {success_count}/{args.num_samples}")

if __name__ == "__main__":
    main()
