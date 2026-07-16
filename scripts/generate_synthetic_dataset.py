"""
合成数据集生成工具。
基于模板数据集（如 680.csv），通过随机删除和插入节点的方式生成变体数据集。
用于测试数据加载管线在非标准拓扑下的鲁棒性。
运行方式: python scripts/generate_synthetic_dataset.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
import random
import re
import traceback
import networkx as nx

def get_active_ancestors(node, drop_set, pred_map, memo, visited):
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

def generate(
    template_path: str | Path,
    target_length: int,
    seed: int,
) -> pd.DataFrame | None:
    np.random.seed(seed)
    random.seed(seed)
    df = pd.read_csv(template_path, dtype=str)
    df['类型'] = df['类型'].astype(int)
    df['加工时间/h'] = df['加工时间/h'].astype(float)
    
    pred_map = {}
    for idx, row in df.iterrows():
        node_id = str(row['AO号']).strip()
        preds_str = str(row.get('紧前工序AO号', ''))
        preds_list = []
        if pd.notna(preds_str) and preds_str.lower() not in ['nan', 'none', '', '0']:
            for p in re.split(r'[,，]', preds_str):
                p = p.strip()
                if p: preds_list.append(p)
        pred_map[node_id] = preds_list

    type2_indices = df[df['类型'] == 2].index.tolist()
    current_type2_len = len(type2_indices)
    
    base_drop = current_type2_len - target_length
    num_to_drop = base_drop + random.randint(5, 20)
    num_to_add = num_to_drop - base_drop
    
    drop_indices = np.random.choice(type2_indices, num_to_drop, replace=False)
    drop_set = set(df.loc[drop_indices, 'AO号'].str.strip())
    df = df.drop(drop_indices)
    
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
            
    current_edges = []
    physical_task_ids = set(
        df.loc[df['类型'] == 2, 'AO号'].astype(str).str.strip()
    )
    for idx, row in df.iterrows():
        node_id = str(row['AO号']).strip()
        preds_str = str(row.get('紧前工序AO号', ''))
        if pd.notna(preds_str) and preds_str.lower() not in ['nan', 'none', '', '0']:
            for p in re.split(r'[,，]', preds_str):
                p = p.strip()
                # 新物理工序只允许插入两道物理工序之间，不能继承虚拟节点的工种 -1。
                if p and p in physical_task_ids and node_id in physical_task_ids:
                    current_edges.append((p, node_id, idx))
                
    new_rows = []
    added_count = 0
    insertions = {idx: [] for idx in df.index}
    
    while added_count < num_to_add and len(current_edges) > 0:
        edge_idx = random.randint(0, len(current_edges) - 1)
        node_A, node_B, df_idx_B = current_edges.pop(edge_idx)
        source_rows = df[df['AO号'].astype(str).str.strip() == node_A]
        if source_rows.empty:
            continue
        source_row = source_rows.iloc[0]
        profession_code = str(source_row.get('专业编码', '')).strip().upper()
        skill_type = int(source_row.get('工种', -1))
        if not profession_code or skill_type < 0:
            continue

        same_skill_rows = df[
            (df['类型'] == 2)
            & (pd.to_numeric(df['工种'], errors='coerce') == skill_type)
        ]
        donor_row = same_skill_rows.iloc[random.randrange(len(same_skill_rows))]
        donor_duration = max(0.1, float(donor_row['加工时间/h']))
        sampled_duration = max(0.1, float(np.random.normal(donor_duration, donor_duration * 0.2)))
        sampled_demand = max(1, int(float(donor_row['需求人数'])))

        # AO号第二个字符继续编码原始专业，使专业编码可由 AO号独立复算。
        node_N = f"R{profession_code}ND-N{random.randint(1000, 99999)}-{added_count}"
        new_row = {column: '' for column in df.columns}
        new_row.update({
            'AO号': node_N,
            '类型': 2,
            '专业编码': profession_code,
            '工种': skill_type,
            '紧前工序AO号': node_A,
            '需求人数': sampled_demand,
            '加工时间/h': round(sampled_duration, 2),
            '限定站位': '',
            '部位容量': '',
        })
        new_rows.append(new_row)
        
        b_preds_str = str(df.at[df_idx_B, '紧前工序AO号'])
        b_preds_list = [p.strip() for p in re.split(r'[,，]', b_preds_str) if p.strip()]
        
        if node_A in b_preds_list:
            b_preds_list.remove(node_A)
            b_preds_list.append(node_N)
            df.at[df_idx_B, '紧前工序AO号'] = ','.join(b_preds_list)
            insertions[df_idx_B].append(new_row)
            added_count += 1
        
    new_df_data = []
    for idx, row in df.iterrows():
        if idx in insertions:
            new_df_data.extend(insertions[idx])
        new_df_data.append(row.to_dict())
        
    df = pd.DataFrame(new_df_data)
    if '序号' in df.columns:
        df['序号'] = np.arange(1, len(df) + 1)
        
    G2 = nx.DiGraph()
    for idx, row in df.iterrows():
        u = str(row['AO号']).strip()
        G2.add_node(u)
        preds_str = str(row.get('紧前工序AO号', ''))
        for p in re.split(r'[,，]', preds_str):
            p = p.strip()
            if p:
                G2.add_edge(p, u)
    import tempfile
    from data_loader import load_data
    tmp_path = None
    try:
        # 使用 data_loader 的严格校验
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            tmp_path = Path(f.name)
        df.to_csv(tmp_path, index=False, encoding='utf-8-sig')
        load_data(tmp_path)
        tmp_path.unlink()
        return df
    except Exception as e:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        print(f"Validation failed for seed {seed}: {e}")
        return None

if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "data" / "random_datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generated = 0
    seed = 0
    print(f"开始生成 50 个合法的合成数据集到 {out_dir} ...")
    
    while generated < 50:
        target_len = random.randint(200, 500)
        df = generate(PROJECT_ROOT / "data" / "680.csv", target_len, seed)
        if df is not None:
            # 文件名包含节点数量和随机种子
            filename = f"syn_{len(df)}_{seed}.csv"
            out_path = out_dir / filename
            df.to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"[{generated+1}/50] 成功生成: {filename} (包含 {len(df)} 个工序)")
            generated += 1
        seed += 1
        
    print("生成完成！")
