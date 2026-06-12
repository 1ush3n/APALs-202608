import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import math
import copy
import numpy as np
import gc
from contextlib import nullcontext
from typing import Tuple, List, Dict, Optional, Any
from torch_geometric.data import HeteroData
from configs import configs
try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None

class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 鏅鸿兘浣撱€?
    璐熻矗涓?Environment 浜や簰锛屾敹闆嗚建杩癸紝骞舵洿鏂?Strategy Network銆?
    """
    def __init__(self, model, lr, gamma, k_epochs, eps_clip, device, batch_size=4, total_timesteps=0):
        self.policy = model.to(device)
        
        from utils.gpu_graph_manager import GPUBatchGraphManager
        self.gpu_graph_manager = GPUBatchGraphManager(device)
        
        self.use_schedule_free = getattr(configs, 'use_schedule_free', False)
        
        # [SF Enhancement] 鑻ユ湭瀹夎 schedulefree 搴擄紝寮鸿闄嶇骇涓烘櫘閫?AdamW 浼樺寲妯″紡锛岄槻姝㈠悗缁彂鐢?train()/eval() 璋冪敤宕╂簝
        if self.use_schedule_free and AdamWScheduleFree is None:
            print("WARNING: ScheduleFree is requested but the 'schedulefree' package is not installed. Falling back to default AdamW.")
            self.use_schedule_free = False
        
        actor_lr_multiplier = float(getattr(configs, 'actor_lr_multiplier', 1.0))
        critic_lr_multiplier = float(getattr(configs, 'critic_lr_multiplier', 1.0))
        actor_params = []
        critic_params = []
        for name, param in self.policy.named_parameters():
            if not param.requires_grad:
                continue
            if 'critic' in name or 'attn' in name:
                critic_params.append(param)
            else:
                actor_params.append(param)
        optimizer_params = [
            {'params': actor_params, 'lr': lr * actor_lr_multiplier, 'name': 'actor'},
            {'params': critic_params, 'lr': lr * critic_lr_multiplier, 'name': 'critic'},
        ]

        if self.use_schedule_free and AdamWScheduleFree is not None:
            # [SF Enhancement] 鍔ㄦ€佽皟鏁撮鐑湡锛岃瀹氫负鎬绘洿鏂版鏁扮殑 5% (鏈€灏?100)
            warmup = getattr(configs, 'sf_warmup_steps', max(100, int(max(1, total_timesteps) * 0.05)))
            self.optimizer = AdamWScheduleFree(optimizer_params, lr=lr, weight_decay=1e-4, warmup_steps=warmup)
        else:
            self.optimizer = torch.optim.AdamW(optimizer_params, lr=lr, weight_decay=1e-4)
            
        self.use_ema = getattr(configs, 'use_ema', False)
        self.ema_decay = getattr(configs, 'ema_decay', 0.995)
        
        # [SF Enhancement] 鑷姩瑕嗙洊 EMA 闃叉鎺у埗鏉冨啿绐佷笌鍐椾綑璁＄畻
        if self.use_schedule_free and self.use_ema:
            print("INFO: ScheduleFree optimizer enabled; disabling EMA to avoid duplicate parameter averaging.")
            self.use_ema = False
            
        if self.use_ema:
            self.ema_policy = copy.deepcopy(self.policy).to(device)
            # 鍐荤粨 EMA 妯″瀷鐨勬搴﹁繍绠楋紝瀹冨彧浣滀负鏃佽鑰呮帴鏀舵寚娲?
            for param in self.ema_policy.parameters():
                param.requires_grad = False
                
        self.lr = lr
        self.gamma = gamma          # 鎶樻墸鍥犲瓙
        self.k_epochs = k_epochs    # 姣忔 Update 鐨勮凯浠ｈ疆鏁?
        self.eps_clip = eps_clip    # PPO Clip鍙傛暟 (e.g., 0.2)
        self.device = device
        self.batch_size = batch_size
        self.accumulation_steps = configs.accumulation_steps
        self.gae_lambda = configs.gae_lambda
        
        self.MseLoss = nn.MSELoss() 
        
        self.kl_early_stop = configs.kl_early_stop
        
        self.initial_lr = lr
        
        self.total_timesteps = max(1, total_timesteps)
        self.current_step = 0
        self.amp_device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
        self.amp_enabled = self.amp_device_type == "cuda"
        self.scaler = torch.amp.GradScaler(self.amp_device_type, enabled=self.amp_enabled)
        
        # 鑷€傚簲璇勪及鏂版棫绛栫暐宸窛 (KL鏁ｅ害) 鐨勬柟娉曞湪 update 灏鹃儴鍙樺姩 LR銆?

    def autocast_context(self):
        """返回与当前设备匹配的 AMP 上下文，CPU 路径默认禁用混合精度。"""
        if self.amp_enabled:
            return torch.amp.autocast(device_type=self.amp_device_type)
        return nullcontext()

    def clear_device_cache(self) -> None:
        """按当前设备清理缓存，降低连续 PPO 更新后的显存或内存残留风险。"""
        gc.collect()
        if self.amp_device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def validate_snapshot_homogeneity(states: List[Any]) -> None:
        """确保一次 PPO update 只包含同一窄池图和固定工人数。"""
        snapshot_states = [state for state in states if isinstance(state, dict)]
        if not snapshot_states:
            return
        if len(snapshot_states) != len(states):
            raise RuntimeError("PPO memory 混合了 snapshot 与 HeteroData 状态")
        dataset_ids = {int(state.get("dataset_idx", 0)) for state in snapshot_states}
        worker_counts = {len(state["worker_free_time"]) for state in snapshot_states}
        if len(dataset_ids) != 1 or len(worker_counts) != 1:
            raise RuntimeError(
                "PPO update 要求同质窄池轨迹，"
                f"dataset_ids={sorted(dataset_ids)}, worker_counts={sorted(worker_counts)}"
            )

    @staticmethod
    def get_task_demand(task_x: torch.Tensor, task_idx: int) -> int:
        """从 task 特征第 16 列读取 APAL 工序硬性需求人数。"""
        assert task_x.dim() == 2 and task_x.size(1) > 16, f"task_x 形状异常: {tuple(task_x.shape)}"
        demand = int(task_x[task_idx, 16].item())
        return max(1, demand)

    def get_memory_snapshot(self) -> Dict[str, float]:
        """返回当前设备显存快照，单位为 GB；CPU 环境返回 0。"""
        if self.amp_device_type != "cuda" or not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }

    def ensure_hetero_batch_vectors(self, batch: HeteroData, batch_size: Optional[int] = None) -> int:
        """确保 task/station/worker 都带有 PyG batch 向量。"""
        if batch_size is None:
            if hasattr(batch, 'y_task'):
                batch_size = int(batch.y_task.view(-1).numel())
            elif hasattr(batch['task'], 'ptr'):
                batch_size = int(batch['task'].ptr.numel() - 1)
            else:
                batch_size = 1
        if batch_size <= 0:
            raise ValueError("HeteroData batch_size 必须大于 0")

        repaired = 0
        for node_type in ('task', 'station', 'worker'):
            storage = batch[node_type]
            if not hasattr(storage, 'x') or storage.x is None:
                continue
            total_nodes = int(storage.x.size(0))
            if hasattr(storage, 'batch') and storage.batch is not None and storage.batch.numel() == total_nodes:
                storage.batch = storage.batch.to(self.device, dtype=torch.long)
                continue
            if hasattr(storage, 'ptr') and storage.ptr is not None and storage.ptr.numel() == batch_size + 1:
                counts = (storage.ptr[1:] - storage.ptr[:-1]).to(self.device, dtype=torch.long)
                storage.batch = torch.repeat_interleave(
                    torch.arange(batch_size, device=self.device, dtype=torch.long),
                    counts,
                )
            else:
                if total_nodes % batch_size != 0:
                    raise ValueError(
                        f"无法为 {node_type} 推断 batch 向量: total_nodes={total_nodes}, batch_size={batch_size}"
                    )
                nodes_per_graph = total_nodes // batch_size
                storage.batch = torch.arange(batch_size, device=self.device, dtype=torch.long).repeat_interleave(nodes_per_graph)
            repaired += 1
        return repaired

    @staticmethod
    def compute_gae_returns(
        rewards: List[float],
        terminals: List[bool],
        values: List[Any],
        gamma: float,
        gae_lambda: float,
        truncated: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        璁＄畻 GAE 浼樺娍鍜?Critic 鐩爣鍥炴姤銆?

        truncated 琛ㄧず rollout 杈圭晫锛涢亣鍒?terminal 鎴?truncated 閮介噸缃€掓帹锛?
        闃叉澶氫釜 APAL 鐜杞ㄨ抗鎷兼帴鍚庝紭鍔夸及璁¤法杈圭晫娉勬紡銆?
        """
        if truncated is None or len(truncated) != len(rewards):
            truncated = [False] * len(rewards)

        value_tensor = torch.as_tensor(values, dtype=torch.float32)
        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0

        for step in reversed(range(len(rewards))):
            if terminals[step] or truncated[step]:
                next_value = 0.0
                gae = 0.0

            delta = float(rewards[step]) + gamma * float(next_value) - float(value_tensor[step])
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            next_value = float(value_tensor[step])

        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        return_tensor = advantage_tensor + value_tensor
        return advantage_tensor, return_tensor

    @staticmethod
    def compute_stable_log_ratio_and_ratio(
        current_logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        clamp_abs: float = 20.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        璁＄畻 PPO log_ratio 涓庢暟鍊煎畨鍏?ratio銆?

        涓嶄慨鏀圭湡瀹?current_logprob锛屽彧鍦?exp 鍓嶈鍓?log_ratio锛岄伩鍏嶆瀬绔鐜囨瘮婧㈠嚭銆?
        """
        log_ratio = current_logprob - old_logprob
        safe_log_ratio = torch.clamp(log_ratio, min=-clamp_abs, max=clamp_abs)
        ratio = torch.exp(safe_log_ratio)
        return log_ratio, safe_log_ratio, ratio

    @staticmethod
    def compute_value_loss(
        state_values: torch.Tensor,
        returns: torch.Tensor,
        old_values: Optional[torch.Tensor] = None,
        clip_range: float = 0.2,
    ) -> torch.Tensor:
        """
        璁＄畻 PPO value clipping loss銆?

        褰?old_values 缂哄け鏃跺洖閫€涓烘櫘閫?MSE锛屼繚鎸佸巻鍙?Memory 鍏煎銆?
        """
        value_losses = (state_values - returns).pow(2)
        if old_values is None:
            return value_losses.mean()

        old_values = old_values.to(device=state_values.device, dtype=state_values.dtype)
        clipped_values = old_values + torch.clamp(state_values - old_values, -clip_range, clip_range)
        clipped_losses = (clipped_values - returns).pow(2)
        return torch.max(value_losses, clipped_losses).mean()

    def select_action(self, obs: HeteroData, mask_task: Optional[torch.Tensor] = None, mask_station_matrix: Optional[torch.Tensor] = None, mask_worker: Optional[torch.Tensor] = None, deterministic: bool = False, temperature: float = 1.0, is_eval: bool = False) -> Tuple[Tuple[int, int, List[int]], float, float, Optional[torch.Tensor]]:
        """
        閫夋嫨鍔ㄤ綔 (Select Action)銆?
        
        Args:
            obs: 寮傛瀯鍥捐娴嬫暟鎹?(HeteroData)
            mask_task: [N] Bool Tensor, True=Invalid
            mask_station_matrix: [N, S] Bool Tensor, True=Invalid
            mask_worker: [W] Bool Tensor, True=Invalid (Global)
            deterministic: 鏄惁纭畾鎬ч€夋嫨 (ArgMax vs Sampling)
            temperature: 閲囨牱娓╁害锛孴瓒婂皬瓒婅椽濠紝T瓒婂ぇ瓒婇殢鏈猴紝蹇界暐褰?deterministic=True 鏃?
            
        Returns:
            action_tuple: (task_id, station_id, team_indices_list)
            action_logprob: float
            state_value: float
            specific_station_mask: 鐢ㄤ簬 Memory 璁板綍
        """
        no_mask = configs.ablation_no_mask
        
        if self.use_schedule_free:
            if is_eval:
                self.optimizer.eval()
            else:
                self.optimizer.train()
        
        # 鍐冲畾褰撳墠婵€娲荤殑澶ц剳 (Eval+EMA 鏃朵娇鐢ㄥ奖瀛愮綉缁?
        if is_eval and getattr(self, 'use_ema', False) and hasattr(self, 'ema_policy'):
            active_policy = self.ema_policy
        else:
            active_policy = self.policy

        with torch.no_grad():
            with self.autocast_context():
                x_dict, global_context = active_policy(obs)
                
                # 浣跨敤瀹夊叏鐨勫父閲?-1e4锛堥槻姝?FP16 涓?-1e9/finfo 鏋佸皬鍊煎湪 autocast 鏈熼棿婧㈠嚭锛?
                mask_value = -1e4
                
                # ------------------
                # 1. 閫夋嫨宸ュ簭 (Select Task)
                # ------------------
                task_logits = active_policy.task_head(x_dict['task'], global_context, mask=mask_task if not no_mask else None)
            
            # [Robustness] 妫€鏌ュ苟澶勭悊 NaN
            if torch.isnan(task_logits).any():
                task_logits = torch.nan_to_num(task_logits, nan=mask_value)
            
            if deterministic:
                if mask_task is not None and not no_mask:
                    task_logits = task_logits.masked_fill(mask_task, mask_value)
                task_action = torch.argmax(task_logits)
                task_logprob = torch.tensor(0.0).to(self.device)
            else:
                if mask_task is not None and not no_mask:
                     task_logits = task_logits.masked_fill(mask_task, mask_value)
                
                # Check for all -inf
                if (task_logits <= mask_value * 0.99).all():
                     print("WARNING: All Task Logits -inf in select_action. Force picking 0.")
                     task_action = torch.tensor(0).to(self.device)
                     task_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        task_logits = task_logits / max(temperature, 1e-5)
                    task_dist = Categorical(logits=task_logits.float())
                    task_action = task_dist.sample()
                    task_logprob = task_dist.log_prob(task_action)
            
            t_idx = task_action.item()
            selected_task_emb = x_dict['task'][t_idx].unsqueeze(0) # [1, H]
            
            # 鑾峰彇浠诲姟鐨勪汉鏁伴渶姹?
            demand = self.get_task_demand(obs['task'].x, t_idx)
            
            # ------------------
            # 2. 閫夋嫨绔欎綅 (Select Station)
            # ------------------
            specific_station_mask = None
            if mask_station_matrix is not None:
                # [N, S] -> [1, S]
                specific_station_mask = mask_station_matrix[t_idx].unsqueeze(0)
            
            with self.autocast_context():
                station_embs = x_dict['station'].unsqueeze(0)
                station_logits = active_policy.station_head(selected_task_emb, station_embs, mask=specific_station_mask if not no_mask else None)
            
            if torch.isnan(station_logits).any():
                station_logits = torch.nan_to_num(station_logits, nan=mask_value)
            
            if deterministic:
                if specific_station_mask is not None and not no_mask:
                     station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                station_action = torch.argmax(station_logits)
                station_logprob = torch.tensor(0.0).to(self.device)
            else:
                if specific_station_mask is not None and not no_mask:
                     station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                
                if (station_logits <= mask_value * 0.99).all():
                     print("WARNING: All Station Logits -inf. Force picking 0.")
                     station_action = torch.tensor(0).to(self.device)
                     station_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if temperature != 1.0:
                        station_logits = station_logits / max(temperature, 1e-5)
                    station_dist = Categorical(logits=station_logits.float())
                    station_action = station_dist.sample()
                    station_logprob = station_dist.log_prob(station_action)
                
            # ------------------
            # 3. 閫夋嫨宸ヤ汉 (Select Workers) - 鑷洖褰?
            # ------------------
            team_indices = []
            worker_logprobs = []
            
            # 鍔ㄦ€?Mask: 鍒濆 Mask + 鎶€鑳?Mask
            current_worker_mask = mask_worker.clone() if mask_worker is not None else torch.zeros(obs['worker'].num_nodes, dtype=torch.bool).to(self.device)
            
            worker_feats = obs['worker'].x
            worker_skills = worker_feats[:, 1:11] # 10 dim
            
            task_type_idx = torch.argmax(obs['task'].x[t_idx, 5:15]).item() 
            
            has_skill = worker_skills[:, task_type_idx] > 0.5
            skill_mask = ~has_skill 
            
            s_act = station_action.item() + 1
            worker_locks = torch.argmax(worker_feats[:, 13:21], dim=1)
            lock_mask = (worker_locks != 0) & (worker_locks != s_act)
            
            if no_mask:
                current_worker_mask = skill_mask.to(self.device)
            else:
                current_worker_mask = current_worker_mask | skill_mask.to(self.device) | lock_mask.to(self.device)

            worker_embs = x_dict['worker'].unsqueeze(0)
            
            # 鍔犲叆杩唬闃堝€煎拰 Fallback 闃叉鍥犳帺鐮佽繃搴﹂噸鍙犲彂鐢熸寰幆
            max_iter = demand * 2
            iter_cnt = 0
            
            # 鍒濆鍖栧洟闃熻蹇?
            current_team_emb = None 
            
            while len(team_indices) < demand and iter_cnt < max_iter:
                iter_cnt += 1
                
                # 杩樻湁鍙€夊伐浜哄悧?
                if current_worker_mask.all():
                    break
                
                with self.autocast_context():
                    worker_logits = active_policy.worker_head.forward_choice(selected_task_emb, worker_embs, mask=current_worker_mask, current_team_emb=current_team_emb)
                
                if torch.isnan(worker_logits).any():
                    worker_logits = torch.nan_to_num(worker_logits, nan=mask_value)
                
                if deterministic:
                     if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                     if (worker_logits <= mask_value * 0.99).all(): break
                     
                     w_action = torch.argmax(worker_logits)
                     w_lp = torch.tensor(0.0).to(self.device)
                else:
                     if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                     
                     if (worker_logits <= mask_value * 0.99).all():
                         break # 鏃犳硶缁х画閫変汉
                     
                     if temperature != 1.0:
                         worker_logits = worker_logits / max(temperature, 1e-5)
                         
                     w_dist = Categorical(logits=worker_logits)
                     w_action = w_dist.sample()
                     w_lp = w_dist.log_prob(w_action)
                
                w_idx = w_action.item()
                team_indices.append(w_idx)
                worker_logprobs.append(w_lp)
                
                # 鍒锋柊宸查€夊洟闃熻〃寰佽蹇?
                selected_worker_feats = worker_embs[0, team_indices, :]
                current_team_emb = selected_worker_feats.mean(dim=0, keepdim=True) # [1, H]
                
                # 鏇存柊 Mask (閫夎繃鐨勪汉涓嶈兘鍐嶉€?
                current_worker_mask = current_worker_mask.clone() # 纭繚涓?鍘熷湴淇敼 褰卞搷涓嬩竴杞?
                current_worker_mask[w_idx] = True
            
            # [鍏滃簳閫昏緫] 鑻ュ洜杩囧害绔炰簤鎴栨閿侀€変笉澶熶汉閫?
            if len(team_indices) < demand:
                if is_eval:
                    # [Evaluation Strict Mode] 楠岃瘉鏈熼棿缁濆涓嶅厑璁稿厹搴曚綔寮婏紒
                    # 濡傛灉閫変笉澶熶汉锛岃鏄庣瓥鐣ュ嚭鐜版柇灞傛閿侊紝鐩存帴灏嗗け璐ヤ笂浼犱互鏂藉姞鐪熷疄鐨勯獙璇侀泦鎯╃綒銆?
                    return None, 0.0, 0.0, None, True
                    
                # [Zero-Fallback Enforcement] 鍘熸湁鐨勫厹搴曟満鍒跺凡琚交搴曠Щ闄ゃ€?
                # 鐢变簬鐜鐨?get_masks() 宸茬粡鍦ㄧ墿鐞嗗拰鎷撴墤灞傞潰涓婁繚璇佷簡鍙湁褰撴弧瓒?demand 浜烘暟锛堜笖鎶€鑳姐€佸伐浣嶉攣瀹氱姸鎬侀兘绗﹀悎瑕佹眰锛夋椂锛?
                # 绔欎綅鍜屼换鍔℃墠鏄悎娉曠殑銆傚鏋滃湪杩欓噷閫変笉鍑鸿冻澶熺殑浜猴紝璇存槑鍓嶇疆鎺╃爜涓庡唴灞傞€変汉鎺╃爜瀛樺湪閫昏緫鑴辫妭锛屾垨鍑虹幇浜嗘湭鐭ョ殑璁＄畻婕忔礊銆?
                # 姝ゆ椂缁濅笉鍙啀鍑戞暟濉炲叆鍋囦汉鎴栧瓨鍏ュ亣姒傜巼锛岃繖浼氬鑷村悗鏈?update 浜х敓鐖嗙偢鐨勮櫄鍋?KL 骞惰鍙戜竴杩炰覆鐨勫穿婧冿紒
                raise RuntimeError(
                    f"FATAL DEADLOCK: Failed to select enough valid workers (needed {demand}, got {len(team_indices)}).\n"
                    f"The masking logic in environment get_masks() strictly guarantees worker sufficiency.\n"
                    f"No manual fallback is ever allowed to preserve the KL purity. Please inspect the mask consistency!"
                )
            
            
            total_worker_logprob = sum(worker_logprobs) if worker_logprobs else torch.tensor(0.0).to(self.device)
            
            action_logprob = task_logprob + station_logprob + total_worker_logprob
            # 鐗╃悊闅旂 Critic 闃叉鍏跺法澶х殑 Value Error 姊害鎹ｆ瘉搴曞眰鍏变韩 GAT 鎷撴墤鐗瑰緛
            # 浼犲叆瀹屾暣鐨?state (batch_data)锛岀敱浜庡浜?with torch.no_grad() 涓嬶紝姝ゅ鏃犻渶 detach锛岀洿鎺ュ墠鍚戞彁鍙栦环鍊笺€?
            with self.autocast_context():
                state_value = active_policy.get_value(obs, actor_x_dict_encoded=x_dict)
            
            action_tuple = (t_idx, station_action.item(), team_indices)
            
            # 妫€鏌?action 瀵逛簬 soft penalty 鐨勬湁鏁堟€?
            is_invalid_action = False
            if mask_task is not None and mask_task[t_idx].item():
                is_invalid_action = True
            if specific_station_mask is not None and specific_station_mask[0, station_action.item()].item():
                is_invalid_action = True
            if mask_worker is not None:
                for w_idx in team_indices:
                    if mask_worker[w_idx].item():
                        is_invalid_action = True
            
        return action_tuple, action_logprob.item(), state_value.item(), specific_station_mask, is_invalid_action

    def select_actions_batch(
        self,
        obs_list: List[HeteroData],
        mask_task_list: List[Optional[torch.Tensor]],
        mask_station_matrix_list: List[Optional[torch.Tensor]],
        mask_worker_list: List[Optional[torch.Tensor]],
        deterministic: bool = False,
        temperature: float = 1.0,
        is_eval: bool = False
    ) -> List[Tuple[Tuple[int, int, List[int]], float, float, Optional[torch.Tensor], bool]]:
        """
        鎵归噺閫夋嫨鍔ㄤ綔 (Batch Select Action)銆?
        
        Args:
            obs_list: 寮傛瀯鍥捐娴嬫暟鎹垪琛?(List[HeteroData])
            mask_task_list: 姣忎釜鐜瀵瑰簲鐨?[N] Bool Tensor, True=Invalid
            mask_station_matrix_list: 姣忎釜鐜瀵瑰簲鐨?[N, S] Bool Tensor, True=Invalid
            mask_worker_list: 姣忎釜鐜瀵瑰簲鐨?[W] Bool Tensor, True=Invalid (Global)
            deterministic: 鏄惁纭畾鎬ч€夋嫨
            temperature: 閲囨牱娓╁害
            is_eval: 鏄惁鏄瘎浼版ā寮?
            
        Returns:
            results: List of Tuples containing (action_tuple, action_logprob, state_value, specific_station_mask, is_invalid_action)
        """
        no_mask = configs.ablation_no_mask
        mask_value = -1e4
        
        if self.use_schedule_free:
            if is_eval:
                self.optimizer.eval()
            else:
                self.optimizer.train()
        
        # 鍐冲畾褰撳墠婵€娲荤殑澶ц剳 (Eval+EMA 鏃朵娇鐢ㄥ奖瀛愮綉缁?
        if is_eval and getattr(self, 'use_ema', False) and hasattr(self, 'ema_policy'):
            active_policy = self.ema_policy
        else:
            active_policy = self.policy

        batch_size = len(obs_list)
        results = []
        state_value_tensors = []
        _decoded_actions = []
        _m_task_refs = []
        _m_worker_refs = []
        _eval_fail_flags = []

        with torch.no_grad():
            # 1. 鎵归噺鎵撳寘寮傛瀯鍥捐娴嬫暟鎹苟閫佸叆 GPU锛屼互 O(1) 澶嶆潅搴﹁繍琛?GNN 缂栫爜鍜?Critic 浠峰€肩綉缁?
            batch_obs = Batch.from_data_list(obs_list).to(self.device)
            
            with self.autocast_context():
                x_dict_batch, global_context_batch = active_policy(batch_obs)
                state_values_batch = active_policy.get_value(batch_obs, actor_x_dict_encoded=x_dict_batch)

            # 2. 閫愪釜鎻愬彇鍚勭幆澧冪殑灞€閮ㄥ瓙鍥剧壒寰侊紝鍦ㄤ富杩涚▼鎵ц杞婚噺鐨?Pointer Head 鑷洖褰掑姩浣滈€夋嫨
            for i in range(batch_size):
                # 渚濋潬 PyG 鐨?.batch 绱㈠紩瀵瑰瓙鍥剧壒寰佽繘琛屽眬閮ㄥ垏鍒?
                task_mask = (batch_obs['task'].batch == i)
                station_mask = (batch_obs['station'].batch == i)
                worker_mask = (batch_obs['worker'].batch == i)

                task_embs = x_dict_batch['task'][task_mask]
                station_embs = x_dict_batch['station'][station_mask]
                worker_embs = x_dict_batch['worker'][worker_mask]
                
                global_context_i = global_context_batch[i].unsqueeze(0)
                
                # ------------------
                # 2.1 閫夋嫨宸ュ簭 (Select Task)
                # ------------------
                m_task = mask_task_list[i].to(self.device) if mask_task_list[i] is not None else None
                with self.autocast_context():
                    task_logits = active_policy.task_head(task_embs, global_context_i, mask=m_task if not no_mask else None)
                
                if torch.isnan(task_logits).any():
                    task_logits = torch.nan_to_num(task_logits, nan=mask_value)
                
                if deterministic:
                    if m_task is not None and not no_mask:
                        task_logits = task_logits.masked_fill(m_task, mask_value)
                    task_action = torch.argmax(task_logits)
                    task_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if m_task is not None and not no_mask:
                         task_logits = task_logits.masked_fill(m_task, mask_value)
                    
                    if (task_logits <= mask_value * 0.99).all():
                         print(f"WARNING: All Task Logits -inf in select_actions_batch for env {i}. Force picking 0.")
                         task_action = torch.tensor(0).to(self.device)
                         task_logprob = torch.tensor(0.0).to(self.device)
                    else:
                        if temperature != 1.0:
                            task_logits = task_logits / max(temperature, 1e-5)
                        task_dist = Categorical(logits=task_logits.float())
                        task_action = task_dist.sample()
                        task_logprob = task_dist.log_prob(task_action)
                
                t_idx = task_action.item()
                selected_task_emb = task_embs[t_idx].unsqueeze(0) # [1, H]
                
                # 鑾峰彇璇ヤ换鍔＄殑宸ヤ汉浜烘暟闇€姹?
                demand = self.get_task_demand(obs_list[i]['task'].x, t_idx)
                
                # ------------------
                # 2.2 閫夋嫨绔欎綅 (Select Station)
                # ------------------
                m_station = mask_station_matrix_list[i]
                specific_station_mask = None
                if m_station is not None:
                    specific_station_mask = m_station[t_idx].unsqueeze(0).to(self.device)
                
                with self.autocast_context():
                    station_embs_i = station_embs.unsqueeze(0) # [1, S, H]
                    station_logits = active_policy.station_head(selected_task_emb, station_embs_i, mask=specific_station_mask if not no_mask else None)
                
                if torch.isnan(station_logits).any():
                    station_logits = torch.nan_to_num(station_logits, nan=mask_value)
                
                if deterministic:
                    if specific_station_mask is not None and not no_mask:
                         station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                    station_action = torch.argmax(station_logits)
                    station_logprob = torch.tensor(0.0).to(self.device)
                else:
                    if specific_station_mask is not None and not no_mask:
                         station_logits = station_logits.masked_fill(specific_station_mask, mask_value)
                    
                    if (station_logits <= mask_value * 0.99).all():
                         print(f"WARNING: All Station Logits -inf in select_actions_batch for env {i}. Force picking 0.")
                         station_action = torch.tensor(0).to(self.device)
                         station_logprob = torch.tensor(0.0).to(self.device)
                    else:
                        if temperature != 1.0:
                            station_logits = station_logits / max(temperature, 1e-5)
                        station_dist = Categorical(logits=station_logits.float())
                        station_action = station_dist.sample()
                        station_logprob = station_dist.log_prob(station_action)
                
                # ------------------
                # 2.3 閫夋嫨宸ヤ汉 (Select Workers) - 鑷洖褰?
                # ------------------
                team_indices = []
                worker_logprobs = []
                
                m_worker = mask_worker_list[i]
                obs_worker_num_nodes = obs_list[i]['worker'].num_nodes
                current_worker_mask = m_worker.clone().to(self.device) if m_worker is not None else torch.zeros(obs_worker_num_nodes, dtype=torch.bool).to(self.device)
                
                worker_feats = obs_list[i]['worker'].x
                worker_skills = worker_feats[:, 1:11]
                
                task_type_idx = torch.argmax(obs_list[i]['task'].x[t_idx, 5:15]).item() 
                has_skill = worker_skills[:, task_type_idx] > 0.5
                skill_mask = ~has_skill 
                
                s_act = station_action.item() + 1
                worker_locks = torch.argmax(worker_feats[:, 13:21], dim=1)
                lock_mask = (worker_locks != 0) & (worker_locks != s_act)
                
                if no_mask:
                    current_worker_mask = skill_mask.to(self.device)
                else:
                    current_worker_mask = current_worker_mask | skill_mask.to(self.device) | lock_mask.to(self.device)
    
                worker_embs_i = worker_embs.unsqueeze(0) # [1, W, H]
                
                max_iter = demand * 2
                iter_cnt = 0
                current_team_emb = None 
                
                while len(team_indices) < demand and iter_cnt < max_iter:
                    iter_cnt += 1
                    
                    if current_worker_mask.all():
                        break
                    
                    with self.autocast_context():
                        worker_logits = active_policy.worker_head.forward_choice(selected_task_emb, worker_embs_i, mask=current_worker_mask, current_team_emb=current_team_emb)
                    
                    if torch.isnan(worker_logits).any():
                        worker_logits = torch.nan_to_num(worker_logits, nan=mask_value)
                    
                    if deterministic:
                         if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                         if (worker_logits <= mask_value * 0.99).all(): break
                         
                         w_action = torch.argmax(worker_logits)
                         w_lp = torch.tensor(0.0).to(self.device)
                    else:
                         if not no_mask: worker_logits = worker_logits.masked_fill(current_worker_mask, mask_value)
                         if (worker_logits <= mask_value * 0.99).all():
                             break
                         
                         if temperature != 1.0:
                             worker_logits = worker_logits / max(temperature, 1e-5)
                             
                         w_dist = Categorical(logits=worker_logits)
                         w_action = w_dist.sample()
                         w_lp = w_dist.log_prob(w_action)
                    
                    w_idx = w_action.item()
                    team_indices.append(w_idx)
                    worker_logprobs.append(w_lp)
                    
                    selected_worker_feats = worker_embs_i[0, team_indices, :]
                    current_team_emb = selected_worker_feats.mean(dim=0, keepdim=True)
                    
                    current_worker_mask = current_worker_mask.clone()
                    current_worker_mask[w_idx] = True
                
                # 鑻ュ洜杩囧害绔炰簤鎴栨閿侀€変笉澶熷伐浜?
                if len(team_indices) < demand:
                    if is_eval:
                        state_value_tensors.append(torch.tensor(0.0))
                        _decoded_actions.append((0, 0, [], torch.tensor(0.0), None))
                        _m_task_refs.append(None)
                        _m_worker_refs.append(None)
                        _eval_fail_flags.append(True)
                        continue
                    raise RuntimeError(
                        f"FATAL DEADLOCK: Failed to select enough valid workers (needed {demand}, got {len(team_indices)}).\n"
                        f"Please inspect the mask consistency!"
                    )
                
                total_worker_logprob = sum(worker_logprobs) if worker_logprobs else torch.tensor(0.0).to(self.device)
                state_value_tensors.append(state_values_batch[i])
                _decoded_actions.append((t_idx, s_act, team_indices,
                                         task_logprob + station_logprob + total_worker_logprob,
                                         specific_station_mask))
                _m_task_refs.append(m_task)
                _m_worker_refs.append(m_worker)
                _eval_fail_flags.append(False)
        state_values_list = [v.item() for v in state_value_tensors]
        for i in range(len(_decoded_actions)):
            if _eval_fail_flags[i]:
                results.append((None, 0.0, 0.0, None, True))
                continue
            t_idx, s_act, team_indices, action_logprob, sp_mask = _decoded_actions[i]
            action_lp = action_logprob.item()
            state_v = state_values_list[i]

            is_invalid_action = False
            m_task = _m_task_refs[i]
            m_worker = _m_worker_refs[i]
            if m_task is not None and m_task[t_idx].item():
                is_invalid_action = True
            if sp_mask is not None and sp_mask[0, s_act - 1].item():
                is_invalid_action = True
            if m_worker is not None:
                for w_idx in team_indices:
                    if m_worker[w_idx].item():
                        is_invalid_action = True

            results.append(((t_idx, s_act - 1, team_indices), action_lp, state_v, sp_mask, is_invalid_action))

        return results

    def update(self, memory: Any, env: Any = None, current_ep: int = 1) -> Dict[str, float]:
        """
        PPO 鏇存柊閫昏緫銆?
        
        Args:
            memory: 瀛樺偍杞ㄨ抗鐨?Buffer
            
        Returns:
            metrics: dict, 鐢ㄤ簬 TensorBoard 璁板綍
        """
        # 1. 璁＄畻骞夸箟浼樺娍浼拌 (GAE - Generalized Advantage Estimation)
        # 灏?rewards 涓?values 寮犻噺鍖栦互杩涜 GAE 璁＄畻
        mem_rewards = memory.rewards
        mem_is_terminals = memory.is_terminals
        mem_is_truncated = getattr(memory, 'is_truncated', [False] * len(mem_rewards))
        
        # 鎻愬彇瀛樺偍鍦?states 涓殑 state_values
        # (杩欓渶瑕佸湪 select_action 涔嬪悗琚褰曚笅鏉ワ紝濡傛灉娌℃湁璁板綍锛屽洖閫€涓烘櫘閫氱殑 MC 鍥炴姤鍔犲熀绾?
        if hasattr(memory, 'values') and len(memory.values) == len(mem_rewards):
            advantages, rewards = self.compute_gae_returns(
                rewards=mem_rewards,
                terminals=mem_is_terminals,
                values=memory.values,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                truncated=mem_is_truncated,
            )
        else:
            # Fallback 鍒?Monte-Carlo + Advantage (濡傛灉缂哄皯 Value 璁板綍)
            rewards = []
            discounted_reward = 0
            for reward, is_terminal, is_trunc in zip(reversed(mem_rewards), reversed(mem_is_terminals), reversed(mem_is_truncated)):
                if is_terminal or is_trunc:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
                
            rewards = torch.tensor(rewards, dtype=torch.float32)
            # 鍏煎澶勭悊
            advantages = rewards.clone()
            
        # 褰掍竴鍖?Advantages 涓?Returns (鏈夊姪浜庨暱鏈熻礋鍙嶉鐜鐨勮缁冪ǔ瀹氭€?
        if advantages.std() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        else:
            advantages = advantages - advantages.mean()
            
        # [CRITICAL FIX: Removed Return Normalization]
        # 缁濅笉搴斿 Critic 鐨?Target Returns 杩涜鍔ㄦ€佹壒娆℃爣鍑嗗寲锛?
        # 鍚﹀垯姣忎竴杞?Update 鐨勫潎鍊煎拰鏂瑰樊閮藉湪鍙橈紙绉诲姩闈讹級锛屽鑷?Critic 姘歌繙鏃犳硶鏀舵暃锛屼骇鐢熷法澶х殑姊害闇囪崱銆?
        # 鎴戜滑鏀圭敤閰嶇疆涓殑闈欐€佺郴鏁扮缉灏忓叏灞€ reward銆?
        
        # 2. 鍑嗗 Batch 鏁版嵁
        old_actions = memory.actions 
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        
        # Pad Team List (鍙橀暱 -> 瀹氶暱 Tensor)
        max_team_size = max(len(a[2]) for a in old_actions) if old_actions else 1
        
        b_task = torch.tensor([a[0] for a in old_actions], dtype=torch.long)
        b_station = torch.tensor([a[1] for a in old_actions], dtype=torch.long)
        
        team_list = []
        for a in old_actions:
            t = a[2]
            pad = [-1] * (max_team_size - len(t))
            team_list.append(t + pad)
        b_team = torch.tensor(team_list, dtype=torch.long)
        
        # Attach targets to Data objects for Batching
        enable_gpu_batch = getattr(configs, 'enable_gpu_batch_rebuild', False) and env is not None
        N_samples = len(memory.states)
        self.validate_snapshot_homogeneity(memory.states)
        update_batch_size = max(1, int(self.batch_size))
        gpu_rebuild_fallback_count = 0
        gpu_rebuild_fallback_messages = []

        def _attach_update_targets(state, idx: int):
            """为单个重建图绑定 PPO 更新所需字段。"""
            state.y_task = b_task[idx].unsqueeze(0)
            state.y_station = b_station[idx].unsqueeze(0)
            state.y_team = b_team[idx].unsqueeze(0)
            state.y_logprob = old_logprobs[idx].unsqueeze(0)
            state.y_reward = rewards[idx].unsqueeze(0)
            state.y_advantage = advantages[idx].unsqueeze(0)
            if len(memory.values) > idx:
                state.y_value = torch.tensor([memory.values[idx]], dtype=torch.float32)
            if idx < len(memory.masks):
                t_mask, s_mask, w_mask = memory.masks[idx]
                state.y_task_mask = t_mask
                state.y_station_mask = s_mask
                state.y_worker_mask = w_mask
            return state

        def _build_cpu_rebuild_batch(batch_indices):
            """GPU 快速重建不可用时，回退到逐 snapshot 的标准 CPU rebuild。"""
            assert env is not None, "CPU rebuild fallback 需要可用的 APAL 环境实例"
            batch_data_list = []
            for raw_idx in batch_indices:
                idx = int(raw_idx)
                state = env.rebuild_state_from_snapshot(memory.states[idx])
                batch_data_list.append(_attach_update_targets(state, idx))
            return Batch.from_data_list(batch_data_list).to(self.device)

        def _bind_gpu_batch_targets(batch, batch_indices):
            """为 GPU 原地重建后的 Batch 绑定同一批 PPO 标签与 mask。"""
            batch.y_task = b_task[batch_indices].to(self.device)
            batch.y_station = b_station[batch_indices].to(self.device)
            batch.y_team = b_team[batch_indices].to(self.device)
            batch.y_logprob = old_logprobs[batch_indices].to(self.device)
            batch.y_reward = rewards[batch_indices].to(self.device)
            batch.y_advantage = advantages[batch_indices].to(self.device)
            if len(memory.values) > 0:
                b_vals = [memory.values[int(idx)] for idx in batch_indices]
                batch.y_value = torch.tensor(b_vals, dtype=torch.float32, device=self.device)
            if len(memory.masks) >= len(memory.states):
                b_masks = [memory.masks[int(idx)] for idx in batch_indices]
                batch.y_task_mask = torch.cat([m[0] for m in b_masks], dim=0).to(self.device)
                batch.y_station_mask = torch.cat([m[1] for m in b_masks], dim=0).to(self.device)
                batch.y_worker_mask = torch.cat([m[2] for m in b_masks], dim=0).to(self.device)
            return batch

        def _gpu_rebuild_or_cpu_fallback(snapshots_batch, batch_indices):
            """GPU fast rebuild 失败时自动降级，避免训练在 batch.y_task 处崩溃。"""
            nonlocal gpu_rebuild_fallback_count
            try:
                batch = self.gpu_graph_manager.batched_rebuild_on_gpu(snapshots_batch, env)
                if batch is None:
                    raise RuntimeError("GPUBatchGraphManager.batched_rebuild_on_gpu 杩斿洖 None")
                return _bind_gpu_batch_targets(batch, batch_indices)
            except Exception as exc:
                gpu_rebuild_fallback_count += 1
                if len(gpu_rebuild_fallback_messages) < 3:
                    gpu_rebuild_fallback_messages.append(f"{type(exc).__name__}: {exc}")
                if gpu_rebuild_fallback_count <= 3:
                    print(f"WARNING: GPU batch rebuild failed; falling back to CPU rebuild for this PPO batch. Reason: {type(exc).__name__}: {exc}")
                return _build_cpu_rebuild_batch(batch_indices)
        
        if not enable_gpu_batch:
            rebuilt_states = []
            if env is not None:
                for snap in memory.states:
                    rebuilt_states.append(env.rebuild_state_from_snapshot(snap))
            else:
                rebuilt_states = memory.states
                
            for i, state in enumerate(rebuilt_states):
                _attach_update_targets(state, i)
            
            import os
            num_workers = 0 if os.name == 'nt' else 4
            loader = DataLoader(rebuilt_states, batch_size=update_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            num_batches = len(loader)
            print(f"PPO Update: BatchSize={update_batch_size}, Total Batches={num_batches} (CPU DataLoader)")
        else:
            num_batches = (N_samples + update_batch_size - 1) // update_batch_size
            print(f"PPO Update: BatchSize={update_batch_size}, Total Batches={num_batches} (GPU In-place Rebuild)")
        
        # 3. PPO Optimization Loop
        avg_loss = 0
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy_loss = 0
        avg_task_entropy = 0
        avg_station_entropy = 0
        avg_team_entropy = 0
        update_counts = 0
        approx_kls = []
        explained_vars = []
        ratio_means = []
        ratio_stds = []
        clip_fractions = []
        batch_vector_repair_count = 0
        
        # 鏀堕泦 batch 绾?Critic 棰勬祴鍋忓樊涓庝紭鍔垮垎甯?
        batch_pred_vals = []
        batch_target_rets = []
        batch_abs_errors = []
        batch_adv_means = []
        batch_adv_stds = []
        
        if self.use_schedule_free:
             self.optimizer.train()
        
        self.optimizer.zero_grad()
            
        final_epoch = self.k_epochs - 1
        kl_meltdown_occurred = False 
        total_batches_diagnosed = 0 
        kl_exceeded_count = 0
        for i_epoch in range(self.k_epochs):
            epoch_kls = []
            
            if enable_gpu_batch:
                shuffled_indices = np.random.permutation(N_samples)
                step_items = range(num_batches)
            else:
                step_items = loader
                
            for step_idx, item in enumerate(step_items):
                if enable_gpu_batch:
                    start_idx = item * update_batch_size
                    end_idx = min(start_idx + update_batch_size, N_samples)
                    batch_idx = shuffled_indices[start_idx:end_idx]
                    snapshots_batch = [memory.states[idx] for idx in batch_idx]
                    
                    batch = _gpu_rebuild_or_cpu_fallback(snapshots_batch, batch_idx)

                else:
                    batch = item.to(self.device)
                    # 棰嗗煙闅忔満鍖栧悗 HeteroData 鐨?worker 鏁板彲鑳戒笉鍚岋紝
                    # PyG DataLoader 浜у嚭瑁?HeteroData 鏃剁己灏?.batch 灞炴€э紝
                    # 姝ゆ椂寮哄埗鍖呰涓?Batch 浠ョ‘淇?to_dense_batch 鍙敤
                    if not hasattr(batch['task'], 'batch'):
                        batch = Batch.from_data_list([batch]).to(self.device)
                batch_vector_repair_count += self.ensure_hetero_batch_vectors(
                    batch,
                    batch_size=int(batch.y_task.view(-1).numel()) if hasattr(batch, 'y_task') else None,
                )
                
                with self.autocast_context():
                    # 褰撳墠绛栫暐鐨勫墠鍚戜紶鎾?
                    x_dict, global_context = self.policy(batch)
                    
                    # 鐙珛楠ㄥ共璇勪及 state_values
                    state_values = self.policy.get_value(batch, actor_x_dict_encoded=x_dict).view(-1)
                    
                    # --- Re-evaluate LogProbs ---
                    # A. Task LogProb
                    from torch_geometric.utils import to_dense_batch
                    task_x, p_mask = to_dense_batch(x_dict['task'], batch['task'].batch)
                    
                    # 鎭㈠ Mask
                    if hasattr(batch, 'y_task_mask'):
                        logical_task_mask, _ = to_dense_batch(batch.y_task_mask, batch['task'].batch)
                        combined_task_mask = logical_task_mask | (~p_mask)
                    else:
                        combined_task_mask = ~p_mask
                        
                    task_logits = self.policy.task_head(task_x, global_context, mask=combined_task_mask)
                    if torch.isnan(task_logits).any(): task_logits = torch.nan_to_num(task_logits, nan=(torch.finfo(task_logits.dtype).min / 2.0))
                    
                    task_dist = Categorical(logits=task_logits.float())
                    task_lp = task_dist.log_prob(batch.y_task)
                    task_entropy = task_dist.entropy()
                    
                    # B. Station LogProb
                    batch_indices = torch.arange(batch.y_task.size(0)).to(self.device)
                    sel_task_emb = task_x[batch_indices, batch.y_task] 
                    
                    station_x, s_p_mask = to_dense_batch(x_dict['station'], batch['station'].batch)
                    
                    if hasattr(batch, 'y_station_mask'):
                        dense_s_mask, _ = to_dense_batch(batch.y_station_mask, batch['task'].batch)
                        specific_station_mask = dense_s_mask[batch_indices, batch.y_task]
                        curr_s_mask = specific_station_mask | (~s_p_mask)
                    else:
                        curr_s_mask = ~s_p_mask
                    
                    station_logits = self.policy.station_head(sel_task_emb, station_x, mask=curr_s_mask)
                    if torch.isnan(station_logits).any(): station_logits = torch.nan_to_num(station_logits, nan=(torch.finfo(station_logits.dtype).min / 2.0))
                    
                    station_dist = Categorical(logits=station_logits.float())
                    station_lp = station_dist.log_prob(batch.y_station)
                    station_entropy = station_dist.entropy()
                    
                    # C. Worker Team LogProb
                    worker_x, w_p_mask = to_dense_batch(x_dict['worker'], batch['worker'].batch)
                    team_lp = torch.zeros_like(task_lp)
                    team_entropy = torch.zeros_like(task_entropy)
                    
                    if hasattr(batch, 'y_worker_mask') and not getattr(configs, 'ablation_no_mask', False):
                         d_w_mask, _ = to_dense_batch(batch.y_worker_mask.float(), batch['worker'].batch)
                         curr_mask = (d_w_mask > 0.5) | (~w_p_mask)
                    else:
                         curr_mask = (~w_p_mask)
                    
                    # Add Skill Mask based on the selected task
                    task_raw, _ = to_dense_batch(batch['task'].x, batch['task'].batch)
                    sel_task_raw = task_raw[batch_indices, batch.y_task]
                    task_type_idx = torch.argmax(sel_task_raw[:, 5:15], dim=1) # [B]
                    
                    worker_raw, _ = to_dense_batch(batch['worker'].x, batch['worker'].batch)
                    worker_skills = worker_raw[:, :, 1:11] # [B, Max_W, 10]
                    
                    B_size, Max_W_size = worker_skills.shape[0], worker_skills.shape[1]
                    b_indices_expanded = torch.arange(B_size).view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                    w_indices_expanded = torch.arange(Max_W_size).view(1, -1).expand(B_size, -1).reshape(-1)
                    t_indices_expanded = task_type_idx.view(-1, 1).expand(-1, Max_W_size).reshape(-1)
                    
                    has_skill_flat = worker_skills[b_indices_expanded, w_indices_expanded, t_indices_expanded] > 0.5
                    skill_mask = (~has_skill_flat).view(B_size, Max_W_size).to(self.device)
                    
                    s_act = batch.y_station + 1 # [B]
                    worker_locks = torch.argmax(worker_raw[:, :, 13:21], dim=2) # [B, Max_W]
                    s_act_expanded = s_act.view(B_size, 1).expand(B_size, Max_W_size).to(self.device)
                    lock_mask = (worker_locks != 0) & (worker_locks != s_act_expanded)
                    
                    curr_mask = curr_mask | skill_mask | lock_mask.to(self.device)
                    
                    current_team_emb = None # [B, H]
                    team_emb_sum = torch.zeros(B_size, worker_x.size(-1)).to(self.device)
                    team_cnt = torch.zeros(B_size, 1).to(self.device)
                    
                    for k in range(batch.y_team.size(1)):
                        target = batch.y_team[:, k] 
                        valid_step = (target != -1)
                        if not valid_step.any(): continue
                        
                        logits = self.policy.worker_head.forward_choice(sel_task_emb, worker_x, mask=curr_mask, current_team_emb=current_team_emb)
                        if torch.isnan(logits).any(): logits = torch.nan_to_num(logits, nan=(torch.finfo(logits.dtype).min / 2.0))
                        
                        dist = Categorical(logits=logits.float())
                        step_lp = dist.log_prob(torch.clamp(target, min=0)) 
                        team_lp[valid_step] += step_lp[valid_step]
                        team_entropy[valid_step] += dist.entropy()[valid_step]
                        
                        # Update current_team_emb
                        valid_b_indices = torch.nonzero(valid_step).squeeze(-1)
                        valid_targets = target[valid_step]
                        
                        selected_feats = worker_x[valid_b_indices, valid_targets]
                        
                        # 浣跨敤 clone() 淇濋殰 PyTorch 鑷姩姹傚鏈哄埗鐨勮繛缁€?(Gradient Preservation)
                        next_team_emb_sum = team_emb_sum.clone()
                        next_team_cnt = team_cnt.clone()
                        
                        next_team_emb_sum[valid_b_indices] += selected_feats
                        next_team_cnt[valid_b_indices] += 1
                        
                        team_emb_sum = next_team_emb_sum
                        team_cnt = next_team_cnt
                        
                        current_team_emb = team_emb_sum / torch.clamp(team_cnt, min=1.0)
                        
                        # Update mask for next worker in team
                        curr_mask = curr_mask.clone()
                        curr_mask[valid_b_indices, target[valid_step]] = True
                                
                    total_lp = task_lp + station_lp + team_lp
                    entropy = task_entropy + station_entropy + team_entropy
                    old_lp = batch.y_logprob.view(-1)
                    log_ratio, safe_log_ratio, ratios = self.compute_stable_log_ratio_and_ratio(total_lp, old_lp)
                    
                    # --- PPO Loss Calculation ---
                    with torch.no_grad():
                        approx_kl = ((ratios - 1) - safe_log_ratio).mean()
                        epoch_kls.append(approx_kl.item())
    
                    # 鏋佺畝 KL 鐔旀柇鏈哄埗 (Meltdown Protection)
                    loss_scale = 1.0
                    hard_limit = self.kl_early_stop
                    
                    if approx_kl.item() > hard_limit:
                        kl_exceeded_count += 1
                        loss_scale = 0.01
    
                    # Use GAE advantages if available, else batch.y_reward - state_values (MC fallback)
                    b_reward = batch.y_reward.view(-1)
                    b_adv = batch.y_advantage.view(-1) if hasattr(batch, 'y_advantage') else (b_reward - state_values.detach())
                    
                    # Calculate Explained Variance
                    var_y = torch.var(b_reward, correction=0)
                    if b_reward.numel() <= 1 or var_y <= 1e-8:
                        exp_var = torch.tensor(0.0, device=b_reward.device)
                    else:
                        exp_var = 1.0 - torch.var(b_reward - state_values.detach(), correction=0) / (var_y + 1e-8)
                    explained_vars.append(exp_var.item())
                    
                    # 鍔ㄦ€佽“鍑忔帰绱笂闄?
                    progress = min(1.0, self.current_step / max(1, self.total_timesteps))
                    eps_clip_end = float(getattr(configs, 'eps_clip_end', 0.05))
                    curr_eps_clip = self.eps_clip - progress * (self.eps_clip - eps_clip_end)
                    
                    surr1 = ratios * b_adv
                    surr2 = torch.clamp(ratios, 1-curr_eps_clip, 1+curr_eps_clip) * b_adv
                    with torch.no_grad():
                        ratio_means.append(ratios.mean().item())
                        ratio_stds.append(ratios.std(unbiased=False).item())
                        clip_mask = torch.abs(ratios - 1.0) > curr_eps_clip
                        clip_fractions.append(clip_mask.float().mean().item())
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    c_val = configs.c_value
                    decay_eps = max(1, configs.entropy_decay_episodes)
                    
                    ent_progress = min(1.0, current_ep / decay_eps)
                    
                    c_ent_base = configs.c_entropy
                    c_ent_end = configs.c_entropy_end
                    import math
                    c_ent = c_ent_end + (c_ent_base - c_ent_end) * math.exp(-3.0 * ent_progress)
                    
                    c_pol = configs.c_policy
                    
                    old_values = batch.y_value.view(-1) if hasattr(batch, 'y_value') else None
                    value_loss_raw = self.compute_value_loss(
                        state_values=state_values,
                        returns=b_reward,
                        old_values=old_values,
                        clip_range=curr_eps_clip,
                    )
                    value_loss = c_val * value_loss_raw
                    
                    # 鍔ㄦ€佽幏鍙栧綋鍓?batch 涓悇鍐崇瓥鍒嗘敮鐨勬渶澶у姩浣滅淮搴︿互杩涜鐔靛綊涓€鍖?
                    max_task_dim = float(task_logits.size(-1))
                    max_station_dim = float(station_logits.size(-1))
                    max_worker_dim = float(worker_x.size(1))
                    
                    # 瀵瑰悇鍒嗘敮鐨勭喌鍒嗗埆鍦ㄥ悇鑷渶澶у姩浣滅淮搴︿笅閲囩敤瀵规暟灏哄害杩涜 [0, 1] 褰掍竴鍖?
                    norm_task_entropy = task_entropy / math.log(max(2.0, max_task_dim))
                    norm_station_entropy = station_entropy / math.log(max(2.0, max_station_dim))
                    # 鍥㈤槦宸ヤ汉鐨勬渶澶х喌浠ラ€夋嫨 2 涓伐浜轰负鍩哄簳绠椾綔 2.0 鍊嶇殑鏈€澶у伐浜虹淮鏁板鏁?
                    norm_team_entropy = team_entropy / (math.log(max(2.0, max_worker_dim)) * 2.0)
                    
                    # 璁剧珛鍒嗘敮瑙ｈ€︾郴鏁颁互鐙珛鎺у埗鍜屽钩琛″悇鍒嗘敮鎺㈢储鍔涘害
                    c_ent_task = 1.0
                    c_ent_station = 1.5
                    c_ent_team = 0.5
                    
                    avg_normalized_entropy = (
                        c_ent_task * norm_task_entropy.mean() +
                        c_ent_station * norm_station_entropy.mean() +
                        c_ent_team * norm_team_entropy.mean()
                    )
                    entropy_loss = -c_ent * avg_normalized_entropy
                    
                    loss = c_pol * policy_loss + value_loss + entropy_loss
                    
                    # 搴旂敤杞啍鏂缉鏀?
                    loss = loss * loss_scale
                    
                    # Backprop
                    loss = loss / self.accumulation_steps # 褰掍竴鍖?Gradient
                
                # Scaled Backprop
                self.scaler.scale(loss).backward()
                
                if ((step_idx + 1) % self.accumulation_steps == 0) or (step_idx + 1 == num_batches):
                    self.scaler.unscale_(self.optimizer)
                    
                    # 鐙珛鍙傛暟姊害瑁佸壀
                    actor_params = [p for n, p in self.policy.named_parameters() if 'critic' not in n and 'attn' not in n]
                    critic_params = [p for n, p in self.policy.named_parameters() if 'critic' in n or 'attn' in n]
                    
                    torch.nn.utils.clip_grad_norm_(actor_params, max_norm=0.5)
                    # 缁?Critic 鎸傝杩滄瘮 Actor 鏇磋杽寮辩殑瑁呯敳锛岄槻姝㈠眬閮ㄨ剦鍐插甫宕╁叏鐩?
                    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=configs.clip_v_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    update_counts += 1
                
                # Log Stats (璁板綍鍘熷鏈缉鏀剧殑 loss 鐢ㄤ簬璇婃柇)
                avg_loss += (loss.item() / max(1e-8, loss_scale)) * self.accumulation_steps
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss_raw.item()
                avg_entropy_loss += entropy.mean().item()
                avg_task_entropy += task_entropy.mean().item()
                avg_station_entropy += station_entropy.mean().item()
                avg_team_entropy += team_entropy.mean().item()
                total_batches_diagnosed += 1
                
                # 璁板綍 batch 绾х殑棰勬祴鍋忓樊涓庝紭鍔垮垎甯冿紝杈呭姪缁嗗寲 TensorBoard 璇婃柇
                with torch.no_grad():
                    batch_pred_vals.append(state_values.mean().item())
                    batch_target_rets.append(b_reward.mean().item())
                    batch_abs_errors.append(torch.abs(state_values - b_reward).mean().item())
                    batch_adv_means.append(b_adv.mean().item())
                    batch_adv_stds.append(b_adv.std(unbiased=False).item())
            
            # 灏芥棭瑙﹀彂 early stopping锛岄槻姝㈤€€鍖?
            # 璁＄畻褰撳墠 epoch 鐨勫钩鍧?KL
            curr_epoch_kl = sum(epoch_kls) / len(epoch_kls) if epoch_kls else 0.0
            
            # 鎴戜滑濮嬬粓璁板綍鏈€鍚庝竴杞湭鎺愭柇鐨?KL 浣滀负鑷€傚簲寮曟搸鐨勫弬鑰?
            approx_kls = epoch_kls
            
            # 濡傛灉鍋忕瓒呰繃纭槇鍊硷紝鎻愬墠缁堟鏈 Update 寰幆浠ヤ繚鎶ゆā鍨?
            if curr_epoch_kl > self.kl_early_stop:
                print(f"      -> Early stopping at epoch {i_epoch+1} due to reaching max KL: {curr_epoch_kl:.4f}")
                break
                
        if kl_exceeded_count > 0:
            print(f"      [KL Warning] {kl_exceeded_count}/{total_batches_diagnosed} batches exceeded KL threshold {self.kl_early_stop}. (Extreme Braking Applied)")
            
        # (宸茬Щ闄ゅ啑鏉傜殑瀛︿範鐜囦笅闄嶉€昏緫锛屽畬鍏ㄤ氦缁?Schedule-Free 鎴栨亽瀹?LR)
        mean_kl = sum(approx_kls) / len(approx_kls) if approx_kls else 0.0
        mean_ratio = sum(ratio_means) / len(ratio_means) if ratio_means else 1.0
        std_ratio = sum(ratio_stds) / len(ratio_stds) if ratio_stds else 0.0
        mean_clip_fraction = sum(clip_fractions) / len(clip_fractions) if clip_fractions else 0.0
        
        self.current_step += 1
        
        # [EMA 鏇存柊] 姣忎竴杞閮?Update 缁撴潫锛堝寘鎷唴閮?k_epochs锛夊悗锛岀敱涓绘ā鍨嬪悜褰卞瓙妯″瀷杩涜涓€娆?Exponential Moving Averaging 鍚屾
        if getattr(self, 'use_ema', False) and hasattr(self, 'ema_policy'):
            alpha = self.ema_decay
            with torch.no_grad():
                for ema_param, param in zip(self.ema_policy.parameters(), self.policy.parameters()):
                    ema_param.data.copy_(alpha * ema_param.data + (1.0 - alpha) * param.data)
                
        mean_exp_var = sum(explained_vars) / len(explained_vars) if explained_vars else 0.0
        
        mean_pred_val = sum(batch_pred_vals) / len(batch_pred_vals) if batch_pred_vals else 0.0
        mean_target_ret = sum(batch_target_rets) / len(batch_target_rets) if batch_target_rets else 0.0
        mean_abs_err = sum(batch_abs_errors) / len(batch_abs_errors) if batch_abs_errors else 0.0
        mean_adv = sum(batch_adv_means) / len(batch_adv_means) if batch_adv_means else 0.0
        std_adv = sum(batch_adv_stds) / len(batch_adv_stds) if batch_adv_stds else 0.0
        memory_snapshot = self.get_memory_snapshot()
        
        metrics = {
            'Loss/Total': avg_loss / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Loss/Policy': avg_policy_loss / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Loss/Value': avg_value_loss / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Loss/Entropy': avg_entropy_loss / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Entropy/Task': avg_task_entropy / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Entropy/Station': avg_station_entropy / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Entropy/WorkerTeam': avg_team_entropy / total_batches_diagnosed if total_batches_diagnosed > 0 else 0,
            'Critic/Explained_Variance': mean_exp_var,
            'Critic/Value_Predictions_Mean': mean_pred_val,
            'Critic/Target_Returns_Mean': mean_target_ret,
            'Critic/Absolute_Error_Mean': mean_abs_err,
            'Critic/Advantage_Mean': mean_adv,
            'Critic/Advantage_Std': std_adv,
            'Policy/ApproxKL': mean_kl,
            'Policy/ClipFraction': mean_clip_fraction,
            'Policy/RatioMean': mean_ratio,
            'Policy/RatioStd': std_ratio,
            'Policy/Meltdown_Count': kl_exceeded_count,
            'PPO/GPURebuildFallbackCount': gpu_rebuild_fallback_count,
            'PPO/BatchVectorRepairCount': batch_vector_repair_count,
            'Train/LearningRate': self.optimizer.param_groups[0]['lr'],
            'Train/ActorLearningRate': self.optimizer.param_groups[0]['lr'],
            'Train/CriticLearningRate': self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else self.optimizer.param_groups[0]['lr'],
            'Train/ScheduleFreeEnabled': 1.0 if self.use_schedule_free else 0.0,
            'Train/UpdateBatchSize': float(update_batch_size),
            'Memory/Allocated_GB': memory_snapshot['allocated_gb'],
            'Memory/Reserved_GB': memory_snapshot['reserved_gb'],
        }
        return metrics


