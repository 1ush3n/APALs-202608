import datetime
from pathlib import Path

class TrainingReporter:
    def __init__(self, log_dir="results/reports"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
            
        self.history = []
        
    def add_record(self, ep, makespan, balance, w_util, s_util, best_sch, eval_reward):
        self.history.append({
            'ep': ep,
            'makespan': makespan,
            'balance': balance,
            'w_util': w_util,
            's_util': s_util,
            'best_sch': best_sch,
            'eval_reward': eval_reward
        })
        
    def generate_report(self, current_ep, metrics_dict=None):
        if not self.history:
            return
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = self.log_dir / f"report_ep{current_ep}_{timestamp}.md"
        
        recent_records = self.history[-10:] # last 10 evals
        best_record = min(self.history, key=lambda x: x['makespan'])
        
        recent_avg_makespan = sum(r['makespan'] for r in recent_records) / len(recent_records)
        recent_avg_reward = sum(r['eval_reward'] for r in recent_records) / len(recent_records)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# 飞机脉动装配线 (APAL) 强化学习自动诊断报告 (Episode {current_ep})\n\n")
            f.write(f"生成时间: {timestamp}\n\n")
            
            f.write("## 📊 1. 装配线物理性能与收敛情况\n")
            f.write("| 指标类型 | 历史最佳 (Best) | 近期均值 (Recent Avg) | 状态评估 |\n")
            f.write("|---|---|---|---|\n")
            
            # Makespan 状态评估
            m_status = "🟢 优秀" if best_record['makespan'] < 140 else ("🟡 良好" if best_record['makespan'] < 180 else "🔴 待优化")
            f.write(f"| **完工时间 (Makespan)** | `{best_record['makespan']:.1f}h` (Ep {best_record['ep']}) | `{recent_avg_makespan:.1f}h` | {m_status} |\n")
            
            # 负载均衡度评估
            b_status = "🟢 平衡" if best_record['balance'] < 35.0 else "🟡 存在局部瓶颈"
            f.write(f"| **工位负载方差 (Balance Std)** | `{best_record['balance']:.2f}` | - | {b_status} |\n")
            
            # 利用率评估
            f.write(f"| **平均工位利用率 (Station Util)** | `{best_record['s_util']*100:.1f}%` | - | - |\n")
            f.write(f"| **平均工人利用率 (Worker Util)** | `{best_record['w_util']*100:.1f}%` | - | - |\n")
            f.write(f"| **累计奖励回报 (Eval Return)** | `{best_record['eval_reward']:.3f}` | `{recent_avg_reward:.3f}` | - |\n\n")
            
            if metrics_dict:
                f.write("## 🧠 2. 神经网络与训练算法健康度诊断\n")
                
                exp_var = metrics_dict.get('Critic/Explained_Variance', 0)
                kl = metrics_dict.get('Policy/ApproxKL', 0)
                meltdowns = metrics_dict.get('Policy/Meltdown_Count', 0)
                val_loss = metrics_dict.get('Loss/Value', 0)
                
                f.write("### 2.1 价值网络 (Critic) 预测效能\n")
                f.write(f"- **解释方差 (Explained Variance)**: `{exp_var:.4f}`\n")
                f.write(f"- **价值估计损失 (Value Loss)**: `{val_loss:.4f}`\n")
                
                # Critic 拟合诊断
                if exp_var < 0.1:
                    f.write("  > [!CAUTION]\n")
                    f.write("  > **Critic 学习发生严重停滞**。价值网络无法辨识当前装配状态的好坏，这通常是由死锁奖励惩罚过大掩盖正常 Makespan 梯度或学习率过小引起的。\n")
                elif exp_var > 0.7:
                    if val_loss > 0.8:
                        f.write("  > [!WARNING]\n")
                        f.write("  > **Critic 相对拟合极佳，但绝对误差偏大（EV 高且 Value Loss 高）**。这通常是因为探索阶段 Return 的分布跨度变宽，或者 Smooth L1 损失中 `beta` 设置过大压低了梯度。建议改用标准均方误差（MSE Loss）以加快绝对尺度对齐。\n")
                    else:
                        f.write("  > [!NOTE]\n")
                        f.write("  > **Critic 拟合状态极其健康**。价值网络能够非常精准地估计装配状态的长期回报，策略梯度的方向非常可靠。\n")
                else:
                    f.write("  > [!NOTE]\n")
                    f.write("  > **Critic 拟合状态正常**。价值估计处于合理的收敛梯度轨道上。\n")
                
                f.write("\n### 2.2 策略空间 (Actor) 更新稳定性\n")
                f.write(f"- **更新前后 KL 散度 (Approx KL)**: `{kl:.4f}`\n")
                f.write(f"- **本轮触发 KL 熔断次数**: `{meltdowns}`\n")
                if meltdowns > 0:
                    f.write("  > [!WARNING]\n")
                    f.write("  > **策略更新步伐过大，已频繁触发 KL Meltdown 熔断保护**。为了防止模型发生雪崩退化，建议降低 PPO 的初始学习率 `lr`，或者将 `clip_v_grad_norm` 设为更保守的数值（如 0.05）。\n")
                else:
                    f.write("  > [!NOTE]\n")
                    f.write("  > **策略更新平稳**。更新步长处于 PPO Trust Region 安全区间。\n")
                
                f.write("\n### 2.3 动作空间分项探索熵 (Action Exploration Entropy)\n")
                task_ent = metrics_dict.get('Entropy/Task', 0)
                station_ent = metrics_dict.get('Entropy/Station', 0)
                worker_ent = metrics_dict.get('Entropy/WorkerTeam', 0)
                
                f.write(f"- **任务指派熵 (Task)**: `{task_ent:.4f}`\n")
                f.write(f"- **站位选择熵 (Station)**: `{station_ent:.4f}`\n")
                f.write(f"- **工人调度熵 (Worker)**: `{worker_ent:.4f}`\n")
                
                # 动作熵收敛诊断
                if station_ent < 0.3:
                    f.write("  > [!WARNING]\n")
                    f.write("  > **站位动作分支发生提前收敛（Station Entropy 过低）**。智能体极度偏好某特定站位（例如工位0），这会造成排程局部拥堵，请检查 configs 中 `c_entropy` 探索惩罚系数是否设置合理。\n")
                if task_ent < 0.4:
                    f.write("  > [!WARNING]\n")
                    f.write("  > **工序选择动作分支发生提前收敛（Task Entropy 过低）**。智能体在工艺路线选择中丧失探索度。\n")
                if worker_ent < 0.4:
                    f.write("  > [!WARNING]\n")
                    f.write("  > **工人分派动作分支发生提前收敛（Worker Entropy 过低）**。工人组队可能陷入固定搭配，无法找到应对技能缺损或缺勤的最佳冗余解。\n")
                    
            f.write("\n## 🛠️ 3. 生产调度排程质量评估 (Scheduling Optimization Diagnostics)\n")
            
            # 工人与工位匹配平衡性分析
            s_util = best_record['s_util']
            w_util = best_record['w_util']
            if s_util > 0.4 and w_util < 0.1:
                f.write("> [!IMPORTANT]\n")
                f.write("> **检测到严重的“工位繁忙，工人闲置”现象**（Station Util 显着高于 Worker Util）。这表明：\n")
                f.write("> 1. 站位空间虽然一直被占用，但往往指派的是需要极少工人的小任务，或者工位内存在大面积的并发重叠等待。\n")
                f.write("> 2. 存在“技能瓶颈”，多数工人因不符合 Ready 任务的技能要求，或者受困于工位绑定锁定约束，只能在场外闲置等待。\n")
            elif s_util < 0.15 and w_util > 0.05:
                f.write("> [!IMPORTANT]\n")
                f.write("> **检测到明显的“工人超载，工位空置”现象**。这表明当前装配线工位空余槽位十分充足，瓶颈主要在于工人资源匮乏或流动受限，应优先减少动作对工人长距离流转的阻碍。\n")
            else:
                f.write("> **利用率匹配状态正常**。工位物理容量与工人技能流动基本处于良性循环状态。\n")
                
            f.write("\n## 📅 4. 历史最佳排单甘特图序列摘要\n")
            f.write(f"此列表展示了 Episode {best_record['ep']} 产生的历史最佳 Makespan (`{best_record['makespan']:.1f}h`) 动作执行序列:\n\n")
            f.write("```text\n")
            
            sch = best_record['best_sch']
            if sch:
                count = 0
                for (tid, sid, team, start, end) in sch:
                    if count >= 35:
                        f.write("... (出于报告长度限制，以下部分已省略)\n")
                        break
                    f.write(f"Step {count+1:02d} | Station {sid} | Task {tid:3d} | Workers {str(team):15s} | Time: [{start:5.1f} - {end:5.1f}]\n")
                    count += 1
            else:
                f.write("暂无有效调度序列。\n")
            f.write("```\n")
            
        print(f"📄 自动生成训练诊断报告 -> {report_path}")
