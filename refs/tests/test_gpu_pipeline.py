# -*- coding: utf-8 -*-
"""
APAL 调度算法 GPU 极小规模流水线集成验证脚本
用于强制在 CUDA 上测试所有性能优化特性的正确性与兼容性，防范本地 OOM。
"""

from pathlib import Path
import sys
import os
import shutil
import time

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import pandas as pd

# 导入核心组件与配置
from configs import configs
from train import train

class MockArgs:
    def __init__(self):
        self.resume = False
        self.data_path = "data/mini_test.csv"
        self.seed = 42
        self.max_episodes = 2

def run_gpu_integration_test():
    print("=" * 60)
    print("🚀 启动 APAL GPU 集成流水线测试 (极小 30 工序集)")
    print("=" * 60)

    # 1. 强制 CUDA 检测
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 测试失败：当前环境未检测到 CUDA，或 PyTorch 未编译 GPU 支持！本测试必须运行在 GPU 上。")
    
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ 检测到 GPU 设备: {device_name}")

    # 2. 截取前 33 行作为极小测试集 (包含约 30 个有效工序)
    source_csv = PROJECT_ROOT / "data" / "283.csv"
    temp_csv = PROJECT_ROOT / "data" / "mini_test.csv"
    
    if not source_csv.exists():
        raise FileNotFoundError(f"未找到源数据集 283.csv: {source_csv.resolve()}")

    print("📦 正在从 283.csv 提取前 33 行构建极小测试集...")
    try:
        df = pd.read_csv(source_csv, encoding="utf-8")
        df_mini = df.head(33)
        df_mini.to_csv(temp_csv, index=False, encoding="utf-8")
        print(f"✅ 临时测试集已保存 -> {temp_csv.resolve()}")
    except Exception as e:
        print(f"❌ 提取测试集 CSV 失败: {e}")
        return

    # 3. 配置全局参数，强行激活所有新增性能特性
    print("⚙️ 配置全局超参数以覆盖运行：")
    configs.data_file_path = str(temp_csv)
    configs.train_data_path_or_dir = str(temp_csv)
    
    # 开启核心优化特性进行集成测试
    configs.use_shared_trunk = True
    configs.use_gradient_checkpointing = True
    configs.use_compile = False  # Windows 不支持 Triton，跳过图编译
    configs.enable_shadow_mask_verification = True  # 开启双路影子校验以测试向量化精度
    
    # 极小规模设置以防爆显存 (RTX 4060 Laptop 8GB)
    configs.max_episodes = 2
    configs.eval_freq = 1
    configs.num_envs = 2  # 双环境验证 DPPO 并行流程
    configs.batch_size = 4
    configs.seed = 42
    
    # 分流 Tensorboard 与报告日志以防覆盖原有正常训练
    configs.log_dir = str(PROJECT_ROOT / "tf-logs" / "gpu_test")
    configs.report_dir = str(PROJECT_ROOT / "results" / "reports" / "gpu_test")

    print(f"   [Config] use_shared_trunk = {configs.use_shared_trunk}")
    print(f"   [Config] use_gradient_checkpointing = {configs.use_gradient_checkpointing}")
    print(f"   [Config] use_compile = {configs.use_compile}")
    print(f"   [Config] enable_shadow_mask_verification = {configs.enable_shadow_mask_verification}")
    print(f"   [Config] num_envs = {configs.num_envs}")
    print(f"   [Config] max_episodes = {configs.max_episodes}")

    # 清理 CUDA 缓存
    torch.cuda.empty_cache()

    args = MockArgs()
    start_time = time.time()
    
    test_success = False
    try:
        print("\n🔥 开始执行训练流程...")
        train(args)
        test_success = True
        print(f"\n🎉 恭喜！GPU 流水线集成测试成功完成！耗时: {time.time() - start_time:.2f} 秒。")
        print("这证明向量化掩码、Shared-Trunk GNN、梯度检查点、ScheduleFree 优化器可以无缝协调工作。")
    except Exception as e:
        print("\n❌ GPU 流水线集成测试运行失败！")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时生成的 CSV 文件
        if temp_csv.exists():
            try:
                os.remove(temp_csv)
                print("🧹 已成功清理临时测试集 mini_test.csv。")
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {e}")
                
    print("=" * 60)
    if not test_success:
        sys.exit(1)

if __name__ == "__main__":
    run_gpu_integration_test()
