import unittest
import os
import sys
import torch
import numpy as np

# 将项目根目录添加到路径，以便能够导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import configs
from environment import AirLineEnv_Graph

class TestAuditFixes(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/283.csv"
        # 确保有一个可以用来测试的文件
        if not os.path.exists(cls.data_path):
            raise FileNotFoundError(f"Test data {cls.data_path} not found.")

    def test_config_num_envs(self):
        """测试 configs.py 中是否正确添加了 num_envs 并且具有默认值"""
        self.assertTrue(hasattr(configs, 'num_envs'))
        self.assertEqual(configs.num_envs, 4)

    def test_config_reward_scale(self):
        """测试 configs.py 中 reward_scale 和 c_policy 是否已对齐 SMC 旧版经验值"""
        self.assertAlmostEqual(configs.reward_scale, 0.005)
        self.assertAlmostEqual(configs.c_policy, 1.0)
        self.assertAlmostEqual(configs.c_value, 0.5)

    def test_max_time_initialization(self):
        """测试 environment.py 中 max_time 的初始化"""
        env = AirLineEnv_Graph(data_path_or_dir=self.data_path, seed=42)
        self.assertTrue(hasattr(env, 'max_time'))
        self.assertEqual(env.max_time, 1e6)
        
    def test_random_seed_isolation(self):
        """
        测试两个具有不同 seed 的并行环境实例，
        其 Domain Randomization 行为不应完全一致。
        """
        env1 = AirLineEnv_Graph(data_path_or_dir=self.data_path, seed=42)
        env2 = AirLineEnv_Graph(data_path_or_dir=self.data_path, seed=43)
        
        # 强制开启 Domain Randomization
        configs.enable_dynamic_events = True
        
        obs1 = env1.reset(randomize_workers=True, randomize_duration=True)
        obs2 = env2.reset(randomize_workers=True, randomize_duration=True)
        
        # 两次独立的 reset，由于随机数种子被隔离，抽取的工人数量或者持续时间矩阵应存在差异
        # 也有极小概率它们恰巧相同，但在大部分情况下应该是不同的
        w1_count = env1.num_workers
        w2_count = env2.num_workers
        
        task_x1 = obs1['task'].x[:, 0]
        task_x2 = obs2['task'].x[:, 0]
        
        # 只要工人数或时间有一个不同即可证明隔离
        is_isolated = (w1_count != w2_count) or not torch.allclose(task_x1, task_x2)
        self.assertTrue(is_isolated, "Random seeds across environments are not isolated.")
        
    def test_zero_duration_infinite_loop_prevention(self):
        """
        测试 0 工时死循环的防断路机制。
        我们人为地篡改环境数据，构造一个全是 0 工时任务并死锁的情况
        """
        env = AirLineEnv_Graph(data_path_or_dir=self.data_path, seed=42)
        env.reset()
        
        # 人为把所有任务置为 0 工时并置为 Ready，以模拟大量 0 工时穿透
        env.task_static_feat[:, 0] = 0.0
        env.task_status.fill(1) # 全部 Ready
        
        # 这将触发 _advance_time 里的 0工时逻辑，并且由于一直能够继续，
        # 我们模拟一个人工制造的“环”，通过人为增加 task_status 来诱导
        # 但既然我们只是测试其保护断路器是否生效，我们可以强行注入一些能触发循环的状态
        
        try:
            # 正常情况下如果所有任务都是 0 且 ready，advance_time 将一口气把它们处理完 (zero_run_count > 0)。
            # 由于没有真正的环，它将安全退出。我们可以人为地把已经完成的再置为 1，或者
            # 直接检查如果存在无限重置时的抛错机制。
            # 这里我们通过 mock 的方式，强制设置 total_zero_runs 逼近阈值
            env.total_zero_runs = env.num_tasks * 2
            
            # 手动触发下一次 advance_time (在有 ready task=0 的情况下)
            env.task_status[0] = 1 
            env.task_static_feat[0, 0] = 0.0
            
            with self.assertRaises(RuntimeError) as context:
                env._advance_time()
            
            self.assertIn("Infinite loop detected", str(context.exception))
        except Exception as e:
            self.fail(f"Test failed due to unexpected exception: {e}")

if __name__ == '__main__':
    unittest.main()
