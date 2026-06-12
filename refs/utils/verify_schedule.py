import sys
import pandas as pd
import numpy as np
import argparse
import ast
from pathlib import Path

# 将项目根目录加入导入路径，便于直接运行验证脚本
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from environment import AirLineEnv_Graph
from configs import configs

# 为终端输出添彩色
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def verify_schedule(data_path, schedule_path):
    print(f"正在加载环境数据: {data_path} ...")
    # 初始化环境仅为了提取 Ground Truth 物理信息，杜绝污染
    env = AirLineEnv_Graph(data_path_or_dir=data_path, seed=2026)
    
    print(f"正在加载排程结果: {schedule_path} ...")
    try:
        df = pd.read_csv(schedule_path)
    except Exception as e:
         print(f"{bcolors.FAIL}无法读取排程文件: {e}{bcolors.ENDC}")
         return False

    all_passed = True
    
    # 构建从真实 TaskID (可能是 Excel 中的 '序号') 到内部 0-based 索引 tid 的映射
    real_id_to_internal = {}
    
    # [NEW] Autodetect CSV Index Type
    task_ids_in_csv = df['TaskID'].unique()
    is_zero_based = (0 in task_ids_in_csv and env.num_tasks not in task_ids_in_csv)
    
    if is_zero_based:
        print(f"{bcolors.OKGREEN}[Verifier] 检测到 TaskID 为内部 0-based 索引，直接映射。{bcolors.ENDC}")
        for tid in range(env.num_tasks):
            real_id_to_internal[tid] = tid
            real_id_to_internal[str(tid)] = tid
            real_id_to_internal[float(tid)] = tid
    else:
        print(f"{bcolors.OKGREEN}[Verifier] 检测到 TaskID 为非内部索引 (如 序号)，执行转换映射。{bcolors.ENDC}")
        for tid in range(env.num_tasks):
            if 'task_df' in env.raw_data and '序号' in env.raw_data['task_df'].columns:
                real_id = env.raw_data['task_df']['序号'].iloc[tid]
                real_id_to_internal[real_id] = tid
                # 兼容读取为字符串或浮点数的情况
                real_id_to_internal[str(real_id)] = tid
                try:
                    real_id_to_internal[int(float(real_id))] = tid
                except: pass
            else:
                real_id_to_internal[tid] = tid
                real_id_to_internal[str(tid)] = tid

    # 构建一个查表用的列表，因为我们要在整个时间轴上审查
    # item: {'task_id': internal_id, 'sid': StationID - 1, 'team': [w1, w2...], 'start': float, 'end': float}
    scheduled_tasks = {}
    time_events = [] 

    for idx, row in df.iterrows():
        try:
             # CSV 中 StationID 是 +1 后的结果（如果是正常的站位则是 1..N）
             # 提取出内部使用的站位编号(0..N-1), Virtual (0.0 工时通常标为 0 或者不受影响)
             s_id = int(row['StationID']) - 1 
             real_task_id = row['TaskID']
             
             if real_task_id not in real_id_to_internal and str(real_task_id) not in real_id_to_internal:
                  print(f"{bcolors.WARNING}警告: 找不到任务 {real_task_id} 的内部映射。跳过。{bcolors.ENDC}")
                  continue
             
             internal_tid = real_id_to_internal.get(real_task_id, real_id_to_internal.get(str(real_task_id)))
             team_str = str(row['Team']).strip()
             if team_str.startswith('[') and team_str.endswith(']'):
                 # 处理像 [0 35] 或 [0, 35] 以及多空格的情况
                 content = team_str[1:-1].replace(',', ' ')
                 team = [int(x) for x in content.split() if x.strip()]
             else:
                 team = []
                 
             start = float(row['Start'])
             end = float(row['End'])
             duration = float(row['Duration'])
             
             scheduled_tasks[internal_tid] = {
                 'task_id': internal_tid,
                 'raw_row': row,
                 'sid': s_id,
                 'team': team,
                 'start': start,
                 'end': end,
                 'duration': duration
             }
             
             # 为了 Sweep line 加入事件
             time_events.append({'time': start, 'type': 'start', 'task_id': internal_tid, 'sid': s_id, 'team': team})
             time_events.append({'time': end, 'type': 'end', 'task_id': internal_tid, 'sid': s_id, 'team': team})
             
        except Exception as e:
             print(f"{bcolors.FAIL}解析 CSV 第 {idx+2} 行时出错: {e}{bcolors.ENDC}")
             all_passed = False
             
    print(f"\n{bcolors.BOLD}================== 审查报告 =================={bcolors.ENDC}")
    
    # -------------------------------------------------------------
    # 审查项一: 物理工时严格计算 (Duration check)
    # -------------------------------------------------------------
    failed_durations = 0
    for tid, info in scheduled_tasks.items():
         static_feat = env.task_static_feat[tid].numpy()
         t_std = static_feat[0] if static_feat[0] > 1e-3 else 0.0
         req_demand = max(1, int(static_feat[2]))
         
         if t_std < 1e-5:
             if info['duration'] > 1e-5:
                 print(f"{bcolors.FAIL}[耗时错误] 0工时任务 {info['task_id']} 却占用了 {info['duration']} 小时{bcolors.ENDC}")
                 failed_durations += 1
         else:
             n_act = len(info['team'])
             if n_act == 0:
                 continue # 因为下文人数不足会报错，不要重复报这里
                 
             eff_sum = sum([env.worker_efficiency[w] for w in info['team']])
             synergy = pow(0.95, n_act - 1)
             expected_dur = (t_std * req_demand) / (eff_sum * synergy)
             
             if abs(expected_dur - info['duration']) > 1e-4:
                 print(f"{bcolors.FAIL}[耗时违规] 任务 {info['task_id']} 理论应耗时 {expected_dur:.4f}，但实际为 {info['duration']:.4f}{bcolors.ENDC}")
                 failed_durations += 1
                 
    if failed_durations == 0:
         print(f"{bcolors.OKGREEN}[PASS] [工期耗时公式核验]{bcolors.ENDC}")
    else:
         all_passed = False

    # -------------------------------------------------------------
    # 审查项二: 拓扑先后依赖 (Precedence Constraints)
    # -------------------------------------------------------------
    failed_precedence = 0
    for tid, info in scheduled_tasks.items():
         start_time = info['start']
         preds = env.predecessors.get(tid, [])
         for p_id in preds:
              if p_id not in scheduled_tasks:
                   print(f"{bcolors.FAIL}[拓扑前驱丢失] 任务 {tid} 需要先做 {p_id}，但排程中没有 {p_id}{bcolors.ENDC}")
                   failed_precedence += 1
                   continue
              p_end_time = scheduled_tasks[p_id]['end']
              if start_time < p_end_time - 1e-4: # 增加 1e-4 的容差，防止浮点数微小误差误报
                   print(f"{bcolors.FAIL}[拓扑时空错乱] 任务 {tid} 在 {start_time:.6f} 开始，但它的前置任务 {p_id} 在 {p_end_time:.6f} 才完成!{bcolors.ENDC}")
                   failed_precedence += 1
                   
    if failed_precedence == 0:
         print(f"{bcolors.OKGREEN}[PASS] [严格图拓扑先后顺序]{bcolors.ENDC}")
    else:
         all_passed = False

    # -------------------------------------------------------------
    # 审查项三 & 四 & 五: Sweep line 检测并发重叠
    # -------------------------------------------------------------
    # 事件按时间排序
    # 事件按时间排序 (并对时间进行 6 位小数约简，消除 CSV 读写带来的浮点误差)
    time_events.sort(key=lambda x: (round(x['time'], 6), 0 if x['type'] == 'end' else 1))
    
    current_station_slots = {s: set() for s in range(env.num_stations)}
    current_worker_tasks = {w: set() for w in range(env.num_workers)}
    worker_forever_locks = {} # w -> s_id
    
    max_slots = getattr(configs, 'max_slots_per_station', 3)
    
    failed_slots = 0
    failed_worker_overlap = 0
    failed_worker_lock = 0
    failed_worker_skills = 0
    
    for ev in time_events:
        tid = ev['task_id']
        s_id = ev['sid']
        team = ev['team']
        
        if ev['type'] == 'start':
            # 获取静态特征以辨别是否为虚拟节点 (0工时)
            static_feat = env.task_static_feat[tid].numpy()
            duration_raw = static_feat[0]
            
            # 1. 工位并行槽位审查
            if s_id >= 0 and duration_raw > 1e-5:
                 current_station_slots[s_id].add(tid)
                 if len(current_station_slots[s_id]) > max_slots:
                      print(f"{bcolors.FAIL}[站位过度拥挤] 时刻 {ev['time']:.2f} 站位 {s_id+1} 竟然有 {len(current_station_slots[s_id])} 个任务并发！上限为 {max_slots}{bcolors.ENDC}")
                      failed_slots += 1
                      
                 # 结合检查该站位是否合法 (Fixed Station)
                 fixed_s = env.fixed_stations[tid]
                 if fixed_s != -1 and s_id != fixed_s:
                      print(f"{bcolors.FAIL}[强制站位违规] 任务 {tid} 必须呆在站位 {fixed_s+1}，但它跑到了 {s_id+1}{bcolors.ENDC}")
                      all_passed = False
                      
            # 2. 工人状态审查
            static_feat = env.task_static_feat[tid].numpy()
            req_skill = int(static_feat[1])
            req_demand = max(1, int(static_feat[2]))
            
            # 这里特殊处理 0 工时任务
            duration_raw = static_feat[0]
            if duration_raw > 1e-5 and len(team) < req_demand:
                 print(f"{bcolors.FAIL}[派工人数违约] 任务 {tid} 需 {req_demand} 人，实配 {len(team)} 人{bcolors.ENDC}")
                 failed_worker_skills += 1
                 
            for w in team:
                # 重叠检查
                if duration_raw > 1e-5 and len(current_worker_tasks[w]) > 0:
                     print(f"{bcolors.FAIL}[工人分身术] 工人 {w} 在 {ev['time']:.2f} 时已经被派往做{current_worker_tasks[w]}，现在又被安排接手了 {tid}！{bcolors.ENDC}")
                     failed_worker_overlap += 1
                     
                # 防止由于 0 耗时引起的瞬间进出卡死，只向集合记录有实体耗时的
                if duration_raw > 1e-5:
                     current_worker_tasks[w].add(tid)
                
                # 技能检查
                if env.worker_skill_matrix[w, req_skill] < 0.5:
                     print(f"{bcolors.FAIL}[滥竽充数] 人员 {w} 无技能 {req_skill}，但被派去干 任务 {tid}！{bcolors.ENDC}")
                     failed_worker_skills += 1
                     
                # Locking 绑定检查
                if s_id >= 0:
                    if w in worker_forever_locks:
                        if worker_forever_locks[w] != s_id:
                             print(f"{bcolors.FAIL}[跨岛偷渡] 工人 {w} 早就绑定在了站位 {worker_forever_locks[w]+1}，现在居然跑去做站位 {s_id+1} 的任务 {tid}！死锁判定！{bcolors.ENDC}")
                             failed_worker_lock += 1
                    else:
                        worker_forever_locks[w] = s_id
                        
        elif ev['type'] == 'end':
            # 清理
            if s_id >= 0:
                if tid in current_station_slots[s_id]:
                    current_station_slots[s_id].remove(tid)
            for w in team:
                if tid in current_worker_tasks[w]:
                    current_worker_tasks[w].remove(tid)

    if failed_slots == 0:
         print(f"{bcolors.OKGREEN}[PASS] [站位最大物理并发槽位(Max={max_slots})]{bcolors.ENDC}")
    else:
         all_passed = False
         
    if failed_worker_overlap == 0:
         print(f"{bcolors.OKGREEN}[PASS] [无工人同一时刻多线分身]{bcolors.ENDC}")
    else:
         all_passed = False
         
    if failed_worker_lock == 0:
         print(f"{bcolors.OKGREEN}[PASS] [工人绑定跨岛隔离约束]{bcolors.ENDC}")
    else:
         all_passed = False

    if failed_worker_skills == 0:
         print(f"{bcolors.OKGREEN}[PASS] [无技能滥竽充数 & 人数达标]{bcolors.ENDC}")
    else:
         all_passed = False


    print(f"\n==============================================")
    if all_passed:
         print(f"{bcolors.OKGREEN}{bcolors.BOLD}[最终判定] 本调度表 100% 合法，完美无瑕！{bcolors.ENDC}")
    else:
         print(f"{bcolors.FAIL}{bcolors.BOLD}[最终判定] 发现黑幕调度！请查看上报的各类 Error 判定。{bcolors.ENDC}")
    return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立的排程合法性检察官")
    parser.add_argument('--data_path', type=str, required=True, help="环境的基础知识参数数据 (例如 data/283.csv)")
    parser.add_argument('--schedule_path', type=str, required=True, help="需要被调查的排程结果 (.csv)")
    args = parser.parse_args()
    
    verify_schedule(args.data_path, args.schedule_path)
