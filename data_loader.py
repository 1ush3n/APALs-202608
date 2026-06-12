import pandas as pd
import torch
import numpy as np
import re
import networkx as nx
from pathlib import Path

def load_data(file_path):
    """
    加载并解析 Excel/CSV 数据，构建层级拓扑图。
    
    逻辑说明:
    1. 读取数据文件 (支持 .csv 和 .xlsx)。
    2. 映射列名 (统一为 internal_id, duration, predecessors 等)。
    3. 基于行号生成 internal_id，确保与 Excel 顺序一致。
    4. 解析层级结构 (Hierarchy Parsing):
       - Root (Level 1): 如 "A", "B"
       - Sub (Level 2): 如 "A-1", "B-2"
       - Task (Level 3): 具体工序
    5. 构建图的边 (Edges):
       - Rule A: Sub -> Task (子组包含工序)
       - Rule B: Task -> Next Sub (工序完成后流向下一个子组)
       - Rule C: Last Sub -> Next Root (子组完成后流向下一个根节点)
       - Rule D: Explicit Predecessors (CSV中指定的紧前工序)
       - Rule E: Root -> First Sub (根节点指向其第一个子组)
       
    Args:
        file_path (str): 数据文件路径
        
    Returns:
        dict: {
            'task_df': DataFrame (包含所有任务信息的宽表),
            'precedence_edges': Tensor [2, E] (所有边的连接关系),
            'num_tasks': int (任务总数),
            'id_map': dict (原始ID -> 内部ID 的映射)
        }
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
        
    # ------------------
    # 1. 读取数据
    # ------------------
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    
    # ------------------
    # 2. 列名标准化
    # ------------------
    col_candidates = {
        'task_id': ['工序号', 'TaskID', 'id', 'Task_ID', 'ID', 'AO号'],
        'duration': ['装配时间', 'Duration', '工时', 'Time', 'Duration_Time', '加工时间/h'],
        'predecessors': ['紧前工序', 'Predecessors', 'Preds', 'Predecessor_IDs', '紧前工序AO号'],
        'skill_type': ['工种', 'Skill', 'Skill_Type', 'Type','类型'],
        'fixed_station': ['限定站位', 'Fixed_Station', 'Station_Constraint'],
        'demand_workers': ['需求人数', 'Demand_Workers', 'Workers_Required', 'Req_Workers'],
    }
    
    mapping = {}
    for target, candidates in col_candidates.items():
        found = False
        for c in candidates:
            if c in df.columns:
                mapping[c] = target
                found = True
                break
    
    df = df.rename(columns=mapping)
    
    # 填充默认值
    if 'skill_type' not in df.columns: df['skill_type'] = 0
    if 'demand_workers' not in df.columns: df['demand_workers'] = 1
    if 'fixed_station' not in df.columns: df['fixed_station'] = np.nan
    
    df['demand_workers'] = df['demand_workers'].fillna(1).astype(int)
    df['task_id'] = df['task_id'].astype(str).str.strip()
    
    # 增加工时校验：必须非负
    if (df['duration'] < 0).any():
        invalid_tasks = df[df['duration'] < 0]['task_id'].tolist()
        raise ValueError(f"数据错误：以下工序的工时为负数，请检查数据集！{invalid_tasks}")
    
    # [关键] 3. ID 映射 (基于行号)
    # 使用 DataFrame 的 index 作为 internal_id，保证与 Excel 行号严格一致 (0-based)
    df['internal_id'] = df.index
    id_map = {row['task_id']: row['internal_id'] for idx, row in df.iterrows()}
    
    # ------------------
    # 4. 状态机解析 (State Machine Parsing)
    # ------------------
    
    edges = [] # 存储边 (src, dst)
    
    current_root = None # 当前 Root 节点的 internal_id
    current_sub = None # 当前 Sub 节点的 internal_id
    
    root_groups = [] # 存储所有 Root 节点
    sub_groups_in_root = {} # {root_id: [sub_ids...]}
    tasks_in_sub = {} # {sub_id: [task_ids...]}
    
    # Pass 1: 扫描所有行，识别节点层级
    for idx, row in df.iterrows():
        tid = row['task_id']
        iid = row['internal_id']
        
        # 启发式规则 (根据 AO号 格式判断层级)
        # Root: 纯字母 (A, B) 或不带横杠
        # Sub: Root + "-N" (A-1, B-2)
        
        if '-' not in tid and tid.isalpha():
            # e.g., "A", "B" -> Root
            current_root = iid
            root_groups.append(iid)
            sub_groups_in_root[iid] = []
            current_sub = None # Reset
            
        elif '-' in tid and tid.split('-')[0].isalpha() and tid.split('-')[1].isdigit():
            # e.g., "A-1", "B-2" -> Sub
            current_sub = iid
            if current_root is not None:
                sub_groups_in_root[current_root].append(iid)
            tasks_in_sub[iid] = []
            
        else:
            # Ordinary Task (普通工序)
            if current_sub is not None:
                tasks_in_sub[current_sub].append(iid)
    
    # Pass 2: 构建隐式边 (Implicit Edges)
    
    # Rule E: Root -> First Sub (根节点指向第一个子组)
    for r_id in root_groups:
        subs = sub_groups_in_root.get(r_id, [])
        if subs:
            first_sub = subs[0]
            edges.append((r_id, first_sub))
            
    # Rule A: Sub -> Tasks (子组指向其包含的任务)
    for s_id, t_ids in tasks_in_sub.items():
        for t_id in t_ids:
            edges.append((s_id, t_id))
            
    # =====================================================================
    # Rule B & C: 基于 Sub 显式紧前关系构建工序流转边
    # =====================================================================
    # Sub 行同样拥有 "紧前工序AO号" 字段（例如 A-5 的紧前为 "A-4,A-3"），
    # 意味着 A-3 和 A-4 的任务可以并行推进，A-5 必须等二者都完成后才能开始。
    # 算法：
    #   1. 解析所有 Sub 的紧前字段，构建 Sub 后继映射表
    #   2. 每个 Sub 中的 Task 流向该 Sub 的所有后继 Sub
    #   3. 终端子组（无 Sub 后继）的 Task 流向下一个 Root
    # =====================================================================
    
    # Step 1: 构建 Sub 显式后继映射
    all_sub_ids = set(tasks_in_sub.keys())
    sub_successors = {s: [] for s in all_sub_ids}
    
    for idx, row in df.iterrows():
        succ_id = row['internal_id']
        if succ_id not in all_sub_ids:
            continue
        preds_str = str(row.get('predecessors', ''))
        if pd.isna(preds_str) or preds_str.lower() in ['nan', 'none', '', '0']:
            continue
        preds_list = preds_str.replace('，', ',').replace(';', ',').split(',')
        for p_str in preds_list:
            p_str = p_str.strip()
            if p_str.endswith('.0'): p_str = p_str[:-2]
            if p_str and p_str in id_map:
                pred_id = id_map[p_str]
                if pred_id in sub_successors and succ_id not in sub_successors[pred_id]:
                    sub_successors[pred_id].append(succ_id)
    
    # Step 2: 对每个 Root 按显式 Sub 紧前关系构建边
    for r_idx, r_id in enumerate(root_groups):
        subs = sub_groups_in_root.get(r_id, [])
        for s_id in subs:
            t_ids = tasks_in_sub.get(s_id, [])
            successors = sub_successors.get(s_id, [])
            
            if successors:
                # Sub 有显式 Sub 后继 → Task 并行流向所有后继 Sub
                if not t_ids:
                    for succ_id in successors:
                        edges.append((s_id, succ_id))
                else:
                    for t_id in t_ids:
                        for succ_id in successors:
                            edges.append((t_id, succ_id))
            else:
                # 无显式 Sub 后继 → 终端子组，流向下一 Root
                if r_idx < len(root_groups) - 1:
                    target_node = root_groups[r_idx + 1]
                    if not t_ids:
                        edges.append((s_id, target_node))
                    else:
                        for t_id in t_ids:
                            edges.append((t_id, target_node))

    # Pass 3: 显式紧前工序 (Rule D)
    for idx, row in df.iterrows():
        succ_id = row['internal_id']
        preds_str = str(row.get('predecessors', ''))
        
        if pd.isna(preds_str) or preds_str.lower() in ['nan', 'none', '', '0']:
            continue
            
        # 分割并处理
        preds_list = preds_str.replace('，', ',').replace(';', ',').split(',')
        for p_str in preds_list:
            p_str = p_str.strip()
            if p_str.endswith('.0'): p_str = p_str[:-2]
            
            if p_str and p_str in id_map:
                pred_id = id_map[p_str]
                edges.append((pred_id, succ_id))
    
    # 去重
    edges = sorted(list(set(edges)))
    
    # 拓扑环路校验 (Cyclic Dependency Check)
    G = nx.DiGraph()
    G.add_nodes_from(range(len(df)))
    G.add_edges_from(edges)
    
    try:
        cycles = list(nx.find_cycle(G, orientation="original"))
        if cycles:
            raise ValueError(f"数据错误：发现工艺路线存在循环依赖 (Cyclic Dependency)！闭环链路: {cycles}")
    except nx.NetworkXNoCycle:
        pass # 图是 DAG，正常
    
    print(f"[DataLoader] 已成功加载数据集：{len(df)} 个工序，{len(root_groups)} 个根节点，{len(tasks_in_sub)} 个子组，{len(edges)} 条边")
    
    return {
        'task_df': df,
        'precedence_edges': torch.tensor(edges, dtype=torch.long).t(),
        'num_tasks': len(df),
        'id_map': id_map
    }

if __name__ == "__main__":
    # 测试代码
    path = Path("data") / "3182.csv"
    if path.exists():
        data = load_data(path)
        print("Edges Shape:", data['precedence_edges'].shape)
    else:
        print(f"未找到测试文件 {path}")
