"""
测试环境加载：验证 AirLineEnv_Graph 能否正确加载 680.csv 和 283.csv 数据集。
用于在代码修改后快速验证环境初始化未被破坏。
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import AirLineEnv_Graph
try:
    env = AirLineEnv_Graph(data_path_or_dir='data/680.csv', seed=42)
    print("Success 715")
except Exception as e:
    print("Fail 715:", e)

try:
    env = AirLineEnv_Graph(data_path_or_dir='data/283.csv', seed=42)
    print("Success 290")
except Exception as e:
    print("Fail 290:", e)
