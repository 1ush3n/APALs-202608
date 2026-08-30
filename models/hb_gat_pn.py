import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GATv2Conv,
    HeteroConv,
    Linear,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)
from configs import configs
from models.worker_pointer_context import (
    WorkerPointerV2DecodeCache,
    WorkerPointerV2State,
    WorkerPressureContext,
    build_v2_marginal_reserve_scarcity,
)
from runtime.modes import is_worker_pointer_v2_mode

# ---------------------------------------------------------------------------
# 辅助函数: 动态获取激活函数 (防止 ReLU 死亡)
# ---------------------------------------------------------------------------
def get_activation():
    if getattr(configs, 'use_leaky_relu', True):
        return nn.LeakyReLU(0.1)
    return nn.ReLU()

def apply_activation(x):
    if getattr(configs, 'use_leaky_relu', True):
        return F.leaky_relu(x, negative_slope=0.1)
    return F.relu(x)

def get_layer_norm(dim, enabled=None):
    if enabled is None:
        enabled = getattr(configs, 'use_layer_norm', True)
    if enabled:
        return nn.LayerNorm(dim)
    return nn.Identity()

def get_input_layer_norm(dim):
    return get_layer_norm(dim, getattr(configs, 'use_input_layer_norm', True))

def get_gat_layer_norm(dim):
    return get_layer_norm(dim, getattr(configs, 'use_gat_layer_norm', getattr(configs, 'use_layer_norm', True)))

def get_head_layer_norm(dim):
    return get_layer_norm(dim, getattr(configs, 'use_head_layer_norm', True))

# ---------------------------------------------------------------------------
# 特征嵌入模块 (Feature Embedder)
# 作用: 将原始异构节点特征投影到统一的隐藏层维度
# ---------------------------------------------------------------------------
class FeatureEmbedder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 为每种节点类型定义一个 MLP 
        self.task_emb = nn.Sequential(
            nn.Linear(config.task_feat_dim, config.hidden_dim),
            get_input_layer_norm(config.hidden_dim),
            get_activation()
        )
        self.worker_emb = nn.Sequential(
            nn.Linear(config.worker_feat_dim, config.hidden_dim),
            get_input_layer_norm(config.hidden_dim),
            get_activation()
        )
        self.station_emb = nn.Sequential(
            nn.Linear(config.station_feat_dim, config.hidden_dim),
            get_input_layer_norm(config.hidden_dim),
            get_activation()
        )
        self.skill_emb = None
        if getattr(config, 'use_skill_hub', False):
            self.skill_emb = nn.Sequential(
                nn.Linear(config.skill_feat_dim, config.hidden_dim),
                get_input_layer_norm(config.hidden_dim),
                get_activation(),
            )

    def forward(self, x_dict):
        """
        x_dict: PyG HeteroData.x_dict 字典
        返回: 嵌入后的字典 (Key -> [N, HiddenDim])
        """
        out = {}
        if 'task' in x_dict:
            out['task'] = self.task_emb(x_dict['task'])
        if 'worker' in x_dict:
            out['worker'] = self.worker_emb(x_dict['worker'])
        if 'station' in x_dict:
            out['station'] = self.station_emb(x_dict['station'])
        if 'skill' in x_dict:
            assert self.skill_emb is not None, "输入包含 Skill 节点，但模型未启用 use_skill_hub"
            out['skill'] = self.skill_emb(x_dict['skill'])
        return out

# ---------------------------------------------------------------------------
# 异构图注意力编码器 (Hetero GAT Encoder)
# 作用: 通过消息传递捕获节点间的拓扑依赖和资源约束
# ---------------------------------------------------------------------------
class HeteroGATEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(config.num_gat_layers):
            conv = HeteroConv({
                # 1. 拓扑流：任务间的优先关系 (Precedence Constraint)
                ('task', 'precedes', 'task'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                
                # 2. 归属流：任务与站位的动态绑定
                ('task', 'assigned_to', 'station'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                ('station', 'has_task', 'task'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                
                # 3. 资源流：工人与任务的能力匹配/执行关系
                **({
                    ('worker', 'has_skill', 'skill'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                    ('skill', 'required_by', 'task'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                    **({
                        ('task', 'requires', 'skill'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                        ('skill', 'provided_by', 'worker'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                    } if getattr(config, 'skill_hub_bidirectional', False) else {}),
                } if getattr(config, 'use_skill_hub', False) else {
                    ('worker', 'can_do', 'task'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                }),
                ('task', 'done_by', 'worker'): GATv2Conv(config.hidden_dim, config.hidden_dim, heads=config.num_heads, concat=False, add_self_loops=False),
                
            }, aggr='sum')
            self.layers.append(conv)
            
            # [新增] 图卷积层的稳压器 (LayerNorm)
            self.norms.append(nn.ModuleDict({
                'task': get_gat_layer_norm(config.hidden_dim),
                'worker': get_gat_layer_norm(config.hidden_dim),
                'station': get_gat_layer_norm(config.hidden_dim),
                **({'skill': get_gat_layer_norm(config.hidden_dim)} if getattr(config, 'use_skill_hub', False) else {}),
            }))
            
    def forward(self, x_dict, edge_index_dict):
        from configs import configs
        use_ckpt = getattr(configs, 'use_gradient_checkpointing', False)
        
        for i, conv in enumerate(self.layers):
            def run_layer(x_in, edge_idx_in):
                return conv(x_in, edge_idx_in)
                
            any_requires_grad = any(v.requires_grad for v in x_dict.values() if isinstance(v, torch.Tensor))
            if use_ckpt and any_requires_grad:
                x_dict_out = checkpoint(run_layer, x_dict, edge_index_dict, use_reentrant=False)
            else:
                x_dict_out = conv(x_dict, edge_index_dict)
                
            norm_dict = self.norms[i]
            
            # HeteroConv 只返回作为 Edge 终点的节点更新。
            # 必须手动保留未更新的节点（残差连接 + 身份映射）。
            x_dict_new = {k: v for k, v in x_dict.items()}
            
            for key, x in x_dict_out.items():
                if key in norm_dict:
                    x = norm_dict[key](x) # [新增] Post-Norm 归一化防暴击
                
                x = apply_activation(x) # [修改] 支持 LeakyReLU
                
                if key in x_dict:
                    # 残差连接 (Residual Connection)
                    x = x + x_dict[key] 
                x_dict_new[key] = x
            x_dict = x_dict_new
            
        return x_dict


class HomogeneousGraphSAGEEncoder(nn.Module):
    """合并全部关系后使用共享 GraphSAGE 参数的关系无关编码器。"""

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = int(config.hidden_dim)
        self.node_types = ("task", "worker", "station", "skill")
        self.type_embedding = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(1, self.hidden_dim))
                for name in self.node_types
            }
        )
        self.layers = nn.ModuleList(
            SAGEConv(self.hidden_dim, self.hidden_dim)
            for _ in range(int(config.num_gat_layers))
        )
        self.norms = nn.ModuleList(
            get_gat_layer_norm(self.hidden_dim)
            for _ in range(int(config.num_gat_layers))
        )

    def forward(self, x_dict, edge_index_dict):
        present_types = [name for name in self.node_types if name in x_dict]
        assert present_types, "GraphSAGE 编码器至少需要一种节点类型"
        offsets: dict[str, int] = {}
        sizes: dict[str, int] = {}
        chunks = []
        cursor = 0
        for node_type in present_types:
            node_x = x_dict[node_type]
            assert node_x.ndim == 2 and node_x.size(1) == self.hidden_dim
            offsets[node_type] = cursor
            sizes[node_type] = int(node_x.size(0))
            chunks.append(node_x + self.type_embedding[node_type])
            cursor += int(node_x.size(0))
        # [N_type, H] -> [N_all, H]
        x = torch.cat(chunks, dim=0)

        merged_edges = []
        for (src_type, _relation, dst_type), edge_index in edge_index_dict.items():
            if src_type not in offsets or dst_type not in offsets or edge_index.numel() == 0:
                continue
            shifted = edge_index.clone()
            shifted[0] += offsets[src_type]
            shifted[1] += offsets[dst_type]
            merged_edges.append(shifted)
        edge_index = (
            torch.cat(merged_edges, dim=1)
            if merged_edges
            else torch.empty((2, 0), dtype=torch.long, device=x.device)
        )
        assert edge_index.ndim == 2 and edge_index.size(0) == 2

        for conv, norm in zip(self.layers, self.norms, strict=True):
            updated = conv(x, edge_index)
            x = x + apply_activation(norm(updated))

        result = {}
        for node_type in present_types:
            start = offsets[node_type]
            end = start + sizes[node_type]
            result[node_type] = x[start:end]
        return result


def build_graph_encoder(config) -> nn.Module | None:
    mode = str(getattr(config, "graph_encoder_mode", "hetero_gat"))
    if mode == "hetero_gat":
        return HeteroGATEncoder(config)
    if mode == "homogeneous_graphsage":
        return HomogeneousGraphSAGEEncoder(config)
    if mode == "none":
        return None
    raise ValueError(f"未知 graph_encoder_mode: {mode}")

# ---------------------------------------------------------------------------
# 决策一：工序选择 (Task Pointer)
# 机制: 指针网络 (Pointer Network) 从候选集中选择一个工序
# ---------------------------------------------------------------------------
class TaskPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_mode = str(getattr(config, "actor_context_mode", "attention"))
        c_dim = config.hidden_dim * (6 if self.context_mode == "mean_max" else 3)
        self.context_proj = (
            None if self.context_mode == "local_only"
            else nn.Linear(c_dim, config.hidden_dim)
        )
        self.task_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn = nn.Linear(config.hidden_dim, 1)
        self.local_score = nn.Linear(config.hidden_dim, 1)
        
        # Ablation Fallback
        self.ablation_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, task_emb, global_context, mask=None):
        """
        task_emb: [N, H] 所有任务的 Embedding
        global_context: [B, H] 全局上下文（通常是 Station 的均值池化）
        mask: [B, N] True 表示 Invalid (不可选)
        """
        if task_emb.dim() == 2:
             task_emb = task_emb.unsqueeze(0) # [1, N, H]
        tsk = self.task_proj(task_emb)      

        if self.context_mode == "local_only":
            scores = self.local_score(torch.tanh(tsk)).squeeze(-1)
            if mask is not None:
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                scores = scores.masked_fill(mask, -1e4)
            return scores

        assert self.context_proj is not None
        ctx = self.context_proj(global_context).unsqueeze(1) # [B, 1, H]
        
        from configs import configs
        if getattr(configs, 'ablation_no_pointer', False):
            # Ablation: Simple Dense Network over Concatenated Features
            B, _, H = ctx.shape
            if tsk.dim() == 3:
                _, N, _ = tsk.shape
                ctx_expand = ctx.expand(B, N, H)
                tsk_expand = tsk.expand(B, N, H)
            else:
                N = tsk.shape[0]
                ctx_expand = ctx.expand(B, N, H)
                tsk_expand = tsk.unsqueeze(0).expand(B, N, H)
                
            cat_feat = torch.cat([ctx_expand, tsk_expand], dim=-1)
            scores = self.ablation_mlp(cat_feat).squeeze(-1)
        else:
            features = torch.tanh(ctx + tsk) 
            scores = self.attn(features).squeeze(-1) # [B, N]
        
        if mask is not None:
             if mask.dim() == 1: mask = mask.unsqueeze(0)
             # 将无效动作的 Logit 设为负无穷 (使用 -1e4 防止 FP16 下 -1e9 溢出)
             scores = scores.masked_fill(mask, -1e4)
            
        return scores 

# ---------------------------------------------------------------------------
# 决策二：站位选择 (Station Pointer / Selector)
# 机制: 站位指针网络 (Pointer Network)，输入 (SelectedTask, Station) 引入全局站位竞争
# ---------------------------------------------------------------------------
class StationSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.task_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.station_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn = nn.Linear(config.hidden_dim, 1)
        
        # Ablation Fallback
        self.ablation_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, selected_task_emb, station_embs, mask=None):
        B, S, H = station_embs.size()
        
        t_proj = self.task_proj(selected_task_emb).unsqueeze(1) # [B, 1, H]
        s_proj = self.station_proj(station_embs)                # [B, S, H]
        
        from configs import configs
        if getattr(configs, 'ablation_no_pointer', False):
            task_repeat = selected_task_emb.unsqueeze(1).expand(-1, S, -1) 
            cat_feat = torch.cat([task_repeat, station_embs], dim=2)
            scores = self.ablation_mlp(cat_feat).squeeze(-1)
        else:
            features = torch.tanh(t_proj + s_proj)
            scores = self.attn(features).squeeze(-1) # [B, S]
        
        if mask is not None:
            # 使用 -1e4 防止 FP16 下 -1e9 溢出
            scores = scores.masked_fill(mask, -1e4)
            
        return scores

# ---------------------------------------------------------------------------
# 决策三：工人选择 (Worker Pointer)
# 机制: 自回归指针网络 (Autoregressive Pointer)
#       循环选择工人，直到选择 "Stop Action" 或无法继续
# ---------------------------------------------------------------------------
class WorkerPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._last_v2_marginal_total: torch.Tensor | None = None
        self._last_v2_marginal_extra: torch.Tensor | None = None
        self.use_explicit_team_state = bool(
            getattr(config, "worker_pointer_v2_explicit_team_state", False)
        )
        self.use_marginal_scarcity = bool(
            getattr(config, "worker_pointer_v2_marginal_scarcity", False)
        )
        self.use_interaction_residual = bool(
            getattr(config, "worker_pointer_v2_interaction_residual", False)
        )
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ar_query_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim) # Autoregressive Optimization A
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn = nn.Linear(config.hidden_dim, 1)
        
        # Ablation Fallback
        self.ablation_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Stop Head: 预测是否停止选人 [Logit_Continue, Logit_Stop]
        self.stop_head = nn.Linear(config.hidden_dim * 2, 2) 

        if is_worker_pointer_v2_mode(config):
            # v2 专属参数使用局部确定性种子；退出后恢复全局 RNG，避免改变共享模块初始化序列。
            local_seed = int(getattr(config, "seed", 42)) + int(
                getattr(config, "worker_pointer_v2_init_seed_offset", 1009)
            )
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.manual_seed(local_seed)
                hidden_dim = int(config.hidden_dim)
                self.v2_member_proj = nn.Linear(hidden_dim, hidden_dim)
                self.v2_team_proj = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    get_activation(),
                )
                # query shape: task H + station H + global 3H + team H + pressure/team/progress 17
                self.v2_query_proj = nn.Linear(hidden_dim * 6 + 17, hidden_dim)
                # key shape: worker H + long/near exposure 10 + two maximum exposures
                self.v2_key_proj = nn.Linear(hidden_dim + 12, hidden_dim)
                self.v2_attn = nn.Linear(hidden_dim, 1)
                if bool(getattr(config, "worker_pointer_v2_dynamic_eft_features", False)):
                    self.v2_eft_proj = nn.Linear(2, hidden_dim, bias=False)
                    nn.init.zeros_(self.v2_eft_proj.weight)
                if self.use_marginal_scarcity:
                    self.v2_marginal_proj = nn.Linear(1, hidden_dim, bias=False)
                    nn.init.zeros_(self.v2_marginal_proj.weight)
                if self.use_interaction_residual:
                    self.v2_interaction_mlp = nn.Sequential(
                        nn.Linear(hidden_dim * 4, hidden_dim),
                        get_activation(),
                        nn.Linear(hidden_dim, 1),
                    )
                    nn.init.zeros_(self.v2_interaction_mlp[-1].weight)
                    nn.init.zeros_(self.v2_interaction_mlp[-1].bias)
                if self.use_explicit_team_state:
                    base_proj = self.v2_query_proj
                    with torch.random.fork_rng(devices=[], enabled=True):
                        torch.manual_seed(local_seed + 1)
                        expanded_proj = nn.Linear(hidden_dim * 6 + 19, hidden_dim)
                    with torch.no_grad():
                        expanded_proj.weight[:, : hidden_dim * 6 + 17].copy_(
                            base_proj.weight
                        )
                        expanded_proj.weight[:, hidden_dim * 6 + 17 :].zero_()
                        expanded_proj.bias.copy_(base_proj.bias)
                    self.v2_query_proj = expanded_proj

    def forward_choice(self, task_emb, worker_embs, mask=None, current_team_emb=None):
        """选择下一个工人"""
        if (
            str(getattr(self.config, "team_selection_mode", "autoregressive")) == "autoregressive"
            and current_team_emb is not None
        ):
            # 拼接历史被选人的联合特征
            cat_feat_q = torch.cat([task_emb, current_team_emb], dim=-1) # [B, H*2]
            query = self.ar_query_proj(cat_feat_q).unsqueeze(1)
        else:
            query = self.query_proj(task_emb).unsqueeze(1) 
            
        keys = self.key_proj(worker_embs)
        
        from configs import configs
        if getattr(configs, 'ablation_no_pointer', False):
            B, _, H = query.shape
            if keys.dim() == 3:
                _, N, _ = keys.shape
                q_expand = query.expand(B, N, H)
                k_expand = keys.expand(B, N, H)
            else:
                N = keys.shape[0]
                q_expand = query.expand(B, N, H)
                k_expand = keys.unsqueeze(0).expand(B, N, H)
            cat_feat = torch.cat([q_expand, k_expand], dim=-1)
            scores = self.ablation_mlp(cat_feat).squeeze(-1)
        else:
            features = torch.tanh(query + keys)
            scores = self.attn(features).squeeze(-1) 
        
        if mask is not None:
            # 使用 -1e4 防止 FP16 下 -1e9 溢出
            scores = scores.masked_fill(mask, -1e4)
        return scores

    def forward_stop(self, task_emb, current_team_emb):
        """决定是否因为人够了/协同成本过高而停止"""
        cat_feat = torch.cat([task_emb, current_team_emb], dim=1)
        logits = self.stop_head(cat_feat) 
        return logits

    def initialize_v2_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> WorkerPointerV2State:
        """创建空团队状态；仅允许 v2 模式调用。"""

        if not hasattr(self, "v2_member_proj"):
            raise RuntimeError("legacy WorkerPointer 不支持 v2 团队状态")
        hidden_dim = int(self.config.hidden_dim)
        dtype = self.v2_member_proj.weight.dtype
        return WorkerPointerV2State(
            mapped_sum=torch.zeros((batch_size, hidden_dim), device=device, dtype=dtype),
            mapped_max=torch.zeros((batch_size, hidden_dim), device=device, dtype=dtype),
            count=torch.zeros((batch_size, 1), device=device, dtype=dtype),
            selected_skill_sum=torch.zeros(
                (batch_size, 5), device=device, dtype=torch.float32
            ),
            selected_max_wait=torch.zeros((batch_size, 1), device=device, dtype=torch.float32),
            selected_capacity_sum=torch.zeros((batch_size, 1), device=device, dtype=torch.float32),
        )

    def advance_v2_state(
        self,
        state: WorkerPointerV2State,
        selected_worker_emb: torch.Tensor,
        selected_worker_skills: torch.Tensor,
        valid: torch.Tensor | None = None,
        selected_wait: torch.Tensor | None = None,
        selected_capacity: torch.Tensor | None = None,
    ) -> WorkerPointerV2State:
        """以一个自回归选择增量更新 sum/max/count 和技能消耗。"""

        assert selected_worker_emb.ndim == 2
        assert selected_worker_skills.ndim == 2 and selected_worker_skills.size(-1) == 5
        batch_size = selected_worker_emb.size(0)
        assert state.count.shape == (batch_size, 1)
        valid_mask = (
            torch.ones((batch_size, 1), dtype=torch.bool, device=selected_worker_emb.device)
            if valid is None
            else valid.to(device=selected_worker_emb.device, dtype=torch.bool).reshape(batch_size, 1)
        )
        with torch.autocast(device_type=selected_worker_emb.device.type, enabled=False):
            mapped = self.v2_member_proj(selected_worker_emb.float())  # [B,H] -> [B,H]
        mapped_for_state = mapped.to(state.mapped_sum.dtype)
        next_sum = state.mapped_sum + mapped_for_state * valid_mask.to(mapped_for_state.dtype)
        first_member = state.count <= 0
        candidate_max = torch.where(
            first_member,
            mapped_for_state,
            torch.maximum(state.mapped_max, mapped_for_state),
        )
        next_max = torch.where(valid_mask, candidate_max, state.mapped_max)
        next_count = state.count + valid_mask.to(state.count.dtype)
        next_skills = state.selected_skill_sum + (
            selected_worker_skills.float() * valid_mask.float()
        )
        wait = (
            torch.zeros((batch_size, 1), device=selected_worker_emb.device, dtype=torch.float32)
            if selected_wait is None
            else selected_wait.to(device=selected_worker_emb.device, dtype=torch.float32).reshape(batch_size, 1).clamp_min(0.0)
        )
        capacity = (
            torch.zeros((batch_size, 1), device=selected_worker_emb.device, dtype=torch.float32)
            if selected_capacity is None
            else selected_capacity.to(device=selected_worker_emb.device, dtype=torch.float32).reshape(batch_size, 1).clamp_min(1.0e-6)
        )
        next_wait = torch.where(valid_mask, torch.maximum(state.selected_max_wait, wait), state.selected_max_wait)
        next_capacity = state.selected_capacity_sum + capacity * valid_mask.float()
        return WorkerPointerV2State(
            next_sum,
            next_max,
            next_count,
            next_skills,
            next_wait,
            next_capacity,
        )

    def v2_team_representation(self, state: WorkerPointerV2State) -> torch.Tensor:
        """返回顺序不敏感的 sum+mean+max 团队表示。"""

        with torch.autocast(device_type=state.mapped_sum.device.type, enabled=False):
            mapped_sum = state.mapped_sum.float()
            count = state.count.float()
            nonempty = count > 0
            mean = mapped_sum / count.clamp_min(1.0)
            maximum = torch.where(
                nonempty, state.mapped_max.float(), torch.zeros_like(mapped_sum)
            )
            # input shape: [B,H] * 3 -> [B,3H] -> [B,H]
            return self.v2_team_proj(torch.cat([mapped_sum, mean, maximum], dim=-1))

    @staticmethod
    def _v2_worker_global_context(
        global_context: torch.Tensor,
        *,
        batch_size: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        """规范化 Worker v2 的全局上下文，只保留双头上下文的首头。"""

        assert global_context.ndim == 2
        assert global_context.size(0) == batch_size
        assert global_context.size(1) in (3 * hidden_dim, 6 * hidden_dim), (
            "global_context 必须为 [B,3H] 或 [B,6H]"
        )
        if global_context.size(1) == 6 * hidden_dim:
            # [B,6H] -> [B,3H]，Worker v2 只消费双头上下文的第一头。
            return global_context[:, : 3 * hidden_dim]
        return global_context

    def build_v2_decode_cache(
        self,
        *,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        global_context: torch.Tensor,
        worker_embs: torch.Tensor,
        pressure_context: WorkerPressureContext,
        demand: torch.Tensor,
        candidate_skills: torch.Tensor | None = None,
        task_required_skills: torch.Tensor | None = None,
    ) -> WorkerPointerV2DecodeCache:
        """构造单队复用的静态 key 与 query 特征，不跨团队或 update 保存。"""

        batch_size, num_workers, hidden_dim = worker_embs.shape
        assert task_emb.shape == station_emb.shape == (batch_size, hidden_dim)
        worker_context = self._v2_worker_global_context(
            global_context,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        assert pressure_context.pressure_all.shape == (batch_size, 5)
        assert pressure_context.candidate_exposure.shape == (batch_size, num_workers, 10)
        assert demand.reshape(-1).shape == (batch_size,)
        if candidate_skills is not None:
            assert candidate_skills.shape == (batch_size, num_workers, 5)
            assert torch.isfinite(candidate_skills).all()
        if task_required_skills is not None:
            assert task_required_skills.shape == (batch_size, 5)
            assert torch.isfinite(task_required_skills).all()
        if self.use_marginal_scarcity:
            if candidate_skills is None or task_required_skills is None:
                raise ValueError(
                    "A2 marginal scarcity 必须同时提供 candidate_skills 和 task_required_skills"
                )
        with torch.autocast(device_type=task_emb.device.type, enabled=False):
            query_prefix = torch.cat(
                [task_emb.float(), station_emb.float(), worker_context.float()], dim=-1
            )
            pressure_features = torch.cat(
                [
                    pressure_context.pressure_all.to(
                        device=task_emb.device, dtype=torch.float32
                    ),
                    pressure_context.pressure_near.to(
                        device=task_emb.device, dtype=torch.float32
                    ),
                ],
                dim=-1,
            )
            candidate_features = torch.cat(
                [
                    worker_embs.float(),
                    pressure_context.candidate_exposure.to(
                        device=worker_embs.device, dtype=torch.float32
                    ),
                    pressure_context.candidate_max_exposure.to(
                        device=worker_embs.device, dtype=torch.float32
                    ),
                ],
                dim=-1,
            )
            candidate_keys = self.v2_key_proj(candidate_features)
            supply_all = pressure_context.supply_all.to(
                device=task_emb.device, dtype=torch.float32
            )
            demand_f = demand.to(
                device=task_emb.device, dtype=torch.float32
            ).reshape(-1).clamp_min(1.0)
            cached_candidate_skills = (
                None
                if candidate_skills is None
                else candidate_skills.to(device=task_emb.device, dtype=torch.float32)
            )
            cached_task_required_skills = (
                None
                if task_required_skills is None
                else task_required_skills.to(
                    device=task_emb.device, dtype=torch.float32
                )
            )
        return WorkerPointerV2DecodeCache(
            candidate_keys=candidate_keys,
            query_prefix=query_prefix,
            pressure_features=pressure_features,
            supply_all=supply_all,
            demand=demand_f,
            candidate_skills=cached_candidate_skills,
            task_required_skills=cached_task_required_skills,
        )

    def forward_choice_v2(
        self,
        *,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        global_context: torch.Tensor,
        worker_embs: torch.Tensor,
        pressure_context: WorkerPressureContext,
        team_state: WorkerPointerV2State,
        demand: torch.Tensor,
        mask: torch.Tensor | None,
        decode_cache: WorkerPointerV2DecodeCache | None = None,
        dynamic_eft_features: torch.Tensor | None = None,
        candidate_skills: torch.Tensor | None = None,
        task_required_skills: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """以增强状态表达执行原始 ``tanh(query + key)`` pointer 打分。"""

        batch_size, num_workers, hidden_dim = worker_embs.shape
        assert task_emb.shape == station_emb.shape == (batch_size, hidden_dim)
        worker_context = self._v2_worker_global_context(
            global_context,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
        )
        assert pressure_context.pressure_all.shape == (batch_size, 5)
        assert pressure_context.candidate_exposure.shape == (batch_size, num_workers, 10)
        assert demand.reshape(-1).shape == (batch_size,)
        cache = decode_cache or self.build_v2_decode_cache(
            task_emb=task_emb,
            station_emb=station_emb,
            global_context=worker_context,
            worker_embs=worker_embs,
            pressure_context=pressure_context,
            demand=demand,
            candidate_skills=candidate_skills,
            task_required_skills=task_required_skills,
        )
        assert cache.candidate_keys.shape == (batch_size, num_workers, hidden_dim)
        assert cache.query_prefix.shape == (batch_size, hidden_dim * 5)
        assert cache.pressure_features.shape == (batch_size, 10)
        assert cache.supply_all.shape == (batch_size, 5)
        assert cache.demand.shape == (batch_size,)
        if self.use_marginal_scarcity:
            if cache.candidate_skills is None or cache.task_required_skills is None:
                raise ValueError(
                    "A2 marginal scarcity cache 缺少 candidate_skills 或 task_required_skills"
                )
            assert cache.candidate_skills.shape == (batch_size, num_workers, 5)
            assert cache.task_required_skills.shape == (batch_size, 5)
        if dynamic_eft_features is not None:
            assert dynamic_eft_features.shape == (batch_size, num_workers, 2)
            if not hasattr(self, "v2_eft_proj"):
                raise RuntimeError("动态 EFT 特征未在当前 WorkerPointer v2 中启用")
        with torch.amp.autocast(device_type=task_emb.device.type, enabled=False):
            team_repr = self.v2_team_representation(team_state)
            assert team_state.selected_max_wait.shape == (batch_size, 1)
            assert team_state.selected_capacity_sum.shape == (batch_size, 1)
            team_operational_state = torch.cat(
                [
                    torch.log1p(
                        team_state.selected_max_wait.float().clamp_min(0.0)
                    ),
                    torch.log1p(
                        team_state.selected_capacity_sum.float().clamp_min(0.0)
                    ),
                ],
                dim=-1,
            )  # [B,1] + [B,1] -> [B,2]
            assert team_operational_state.shape == (batch_size, 2)
            assert torch.isfinite(team_operational_state).all()
            epsilon = float(getattr(self.config, "worker_pointer_supply_epsilon", 1.0e-6))
            consumption = (
                team_state.selected_skill_sum.float()
                / cache.supply_all.clamp_min(epsilon)
            )
            selected_count = team_state.count.float().reshape(-1)
            progress = torch.stack(
                [
                    selected_count / cache.demand,
                    (cache.demand - selected_count).clamp_min(0.0) / cache.demand,
                ],
                dim=-1,
            )
            query_parts = [
                cache.query_prefix,
                team_repr.float(),
                cache.pressure_features,
                consumption,
                progress,
            ]
            if self.use_explicit_team_state:
                query_parts.append(team_operational_state)
            query_features = torch.cat(query_parts, dim=-1)
            expected_query_width = hidden_dim * 6 + (
                19 if self.use_explicit_team_state else 17
            )
            assert query_features.shape == (batch_size, expected_query_width)
            query = self.v2_query_proj(query_features).unsqueeze(1)  # [B,1,H]
            dynamic_keys = torch.zeros_like(cache.candidate_keys)
            if dynamic_eft_features is not None:
                dynamic_keys = dynamic_keys + self.v2_eft_proj(
                    dynamic_eft_features.float()
                )
            if self.use_marginal_scarcity:
                assert cache.candidate_skills is not None
                assert cache.task_required_skills is not None
                marginal_total, marginal_extra = (
                    build_v2_marginal_reserve_scarcity(
                        demand_all=pressure_context.demand_all,
                        supply_all=cache.supply_all,
                        selected_skill_sum=team_state.selected_skill_sum,
                        candidate_skills=cache.candidate_skills,
                        task_required_skills=cache.task_required_skills,
                        epsilon=epsilon,
                        clip=float(
                            getattr(
                                self.config,
                                "worker_pointer_v2_marginal_scarcity_clip",
                                10.0,
                            )
                        ),
                    )
                )
                self._last_v2_marginal_total = marginal_total.detach()
                self._last_v2_marginal_extra = marginal_extra.detach()
                dynamic_keys = dynamic_keys + self.v2_marginal_proj(
                    marginal_extra.float()
                )
            else:
                self._last_v2_marginal_total = None
                self._last_v2_marginal_extra = None
            candidate_repr = cache.candidate_keys + dynamic_keys
            base_scores = self.v2_attn(torch.tanh(query + candidate_repr)).squeeze(-1)  # [B,N]
            if self.use_interaction_residual:
                query_expanded = query.expand(-1, num_workers, -1)  # [B,1,H] -> [B,N,H]
                interaction_features = torch.cat(
                    [
                        query_expanded,
                        candidate_repr,
                        query_expanded * candidate_repr,
                        (query_expanded - candidate_repr).abs(),
                    ],
                    dim=-1,
                )  # [B,N,H] * 4 -> [B,N,4H]
                assert interaction_features.shape == (
                    batch_size,
                    num_workers,
                    hidden_dim * 4,
                )
                residual_scores = self.v2_interaction_mlp(
                    interaction_features
                ).squeeze(-1)  # [B,N,4H] -> [B,N,1] -> [B,N]
                scores = base_scores + residual_scores
            else:
                scores = base_scores
        # 后续可控消融可比较 query-candidate 交互 MLP 或 legacy score+压力残差；本轮不实现。
        if mask is not None:
            assert mask.shape == (batch_size, num_workers)
            scores = scores.masked_fill(mask, -1.0e4)
        return scores


class ConditionalTeamSelector(nn.Module):
    """在固定工序—工位后，对安全团队候选进行条件式残差重排序。"""

    def __init__(self, config) -> None:
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        self.scoring_mode = str(
            getattr(config, "conditional_team_scoring_mode", "fixed_prior_v1")
        )
        self.nonbaseline_logit = float(config.conditional_team_nonbaseline_logit)
        self.prior_margin = float(
            getattr(config, "conditional_team_prior_margin", 4.0)
        )
        self.prior_weight = float(
            getattr(config, "conditional_team_prior_weight", 1.0)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            get_activation(),
            nn.Linear(hidden_dim, 1),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 5, hidden_dim),
            get_activation(),
            nn.Linear(hidden_dim, 1),
        )
        # 残差零初始化与负门控偏置确保初始确定性动作退化为候选 0。
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.constant_(self.gate[-1].bias, float(config.conditional_team_gate_bias))

    def _compose_logits(
        self,
        residual: torch.Tensor,
        gate: torch.Tensor,
        candidate_prior_costs: torch.Tensor | None,
    ) -> torch.Tensor:
        """按配置合成团队 logits；固定先验模式保持旧实现行为。"""
        assert residual.ndim == 2
        assert gate.shape == (residual.size(0), 1)
        batch_size, candidate_count = residual.shape
        if self.scoring_mode == "fixed_prior_v1":
            base = residual.new_full(
                (batch_size, candidate_count), self.nonbaseline_logit
            )
            base[:, 0] = 0.0
            return base + gate * residual
        if self.scoring_mode != "relative_heuristic_prior_v1":
            raise ValueError(
                f"未知 conditional_team_scoring_mode: {self.scoring_mode!r}"
            )
        if candidate_prior_costs is None:
            raise ValueError("relative_heuristic_prior_v1 必须提供候选相对完工代价")
        assert candidate_prior_costs.shape == residual.shape
        assert bool(torch.isfinite(candidate_prior_costs).all())
        assert bool(torch.all(candidate_prior_costs[:, 0] == 0.0))
        assert bool(torch.all(candidate_prior_costs[:, 1:] >= 0.0))
        # 仅保留候选相对修正，移除共同平移这一不可辨识自由度。
        relative_residual = residual - residual[:, :1]
        prior = -self.prior_margin - self.prior_weight * candidate_prior_costs
        prior[:, 0] = 0.0
        return prior + gate * relative_residual

    def forward(
        self,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        candidate_team_emb: torch.Tensor,
        gate_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        candidate_prior_costs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回团队 logits 与状态门控值。

        task/station: [B, H]；candidate_team_emb: [B, K, H]；
        gate_features: [B, 5]；candidate_mask: [B, K]，True 表示填充候选。
        """
        assert task_emb.ndim == station_emb.ndim == gate_features.ndim == 2
        assert candidate_team_emb.ndim == 3
        batch_size, candidate_count, hidden_dim = candidate_team_emb.shape
        assert task_emb.shape == station_emb.shape == (batch_size, hidden_dim)
        assert gate_features.shape == (batch_size, 5)

        state_features = torch.cat([task_emb, station_emb, gate_features], dim=-1)
        gate = torch.sigmoid(self.gate(state_features))
        expanded_state = state_features.unsqueeze(1).expand(-1, candidate_count, -1)
        residual_input = torch.cat([expanded_state, candidate_team_emb], dim=-1)
        residual = self.residual(residual_input).squeeze(-1)
        if candidate_prior_costs is not None:
            candidate_prior_costs = candidate_prior_costs.to(
                device=candidate_team_emb.device, dtype=candidate_team_emb.dtype
            )
        logits = self._compose_logits(residual, gate, candidate_prior_costs)
        if candidate_mask is not None:
            assert candidate_mask.shape == logits.shape
            logits = logits.masked_fill(candidate_mask, -1.0e4)
        return logits, gate


class AnchorConditionedTeamPointer(nn.Module):
    """锚点条件完整团队提议器（APCF full_team_v1）。

    自回归生成人数恰为工序需求 d 的完整合法团队。每步查询显式包含：
      h_o（工序嵌入）、h_s（已选工位嵌入）、h̄_H（锚点团队均值嵌入）、
      h̄_{w<j}（已生成成员均值嵌入）。
    相比原 WorkerPointer（查询仅 [h_o, h̄_{w<j}]），本头以锚点为反事实参考，
    并对工位显式条件化。合法性（技能/锁定/空闲/去重/与锚点差异）由调用方掩码保证。
    """

    def __init__(self, config) -> None:
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        self.query_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward_choice(
        self,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        anchor_emb: torch.Tensor,
        worker_embs: torch.Tensor,
        mask: torch.Tensor | None = None,
        current_team_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回 [B, N] 的合法工人打分（调用方负责掩码）。"""
        assert task_emb.ndim == station_emb.ndim == anchor_emb.ndim == 2
        assert worker_embs.ndim == 3
        context = (
            current_team_emb
            if current_team_emb is not None
            else torch.zeros_like(task_emb)
        )
        query = self.query_proj(
            torch.cat([task_emb, station_emb, anchor_emb, context], dim=-1)
        ).unsqueeze(1)  # [B, 1, H]
        keys = self.key_proj(worker_embs)  # [B, N, H]
        features = torch.tanh(query + keys)
        scores = self.attn(features).squeeze(-1)  # [B, N]
        if mask is not None:
            scores = scores.masked_fill(mask, -1.0e4)
        return scores


class AnchorProposalGate(nn.Module):
    """锚点 vs 神经提议分支门控（APCF）。

    分支 logits：
      ℓ_H = 0
      ℓ_P = −ρ + g · 6·tanh(ΔÂ / 0.01)
    其中 ΔÂ = A_ψ(x,o,s,P) − A_ψ(x,o,s,H) 是反事实相对价值差，
    g ∈ [0,1] 为状态条件门控。价值头末层零初始化 + 负门控偏置保证
    未预训练时温度 0 严格选择锚点。
    """

    def __init__(self, config) -> None:
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        self.prior_margin = float(config.anchor_proposal_prior_margin)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            get_activation(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),
            get_activation(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.constant_(self.gate[-1].bias, float(config.anchor_proposal_gate_bias))

    def forward(
        self,
        task_emb: torch.Tensor,
        station_emb: torch.Tensor,
        anchor_emb: torch.Tensor,
        proposal_emb: torch.Tensor,
        gate_features: torch.Tensor,
        hamming_distance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (branch_logits [B,2], delta_A [B,1], gate_value [B,1])。"""
        assert task_emb.ndim == station_emb.ndim == anchor_emb.ndim == proposal_emb.ndim == 2
        assert gate_features.ndim == 2 and gate_features.size(1) == 5
        assert hamming_distance.shape == (task_emb.size(0), 1)
        a_h = self.value_head(
            torch.cat([task_emb, station_emb, anchor_emb], dim=-1)
        )
        a_p = self.value_head(
            torch.cat([task_emb, station_emb, proposal_emb], dim=-1)
        )
        delta_a = a_p - a_h  # [B, 1]
        gate_input = torch.cat(
            [task_emb, station_emb, gate_features, hamming_distance], dim=-1
        )
        g = torch.sigmoid(self.gate(gate_input))  # [B, 1]
        proposal_logit = -self.prior_margin + g * 6.0 * torch.tanh(delta_a / 0.01)
        anchor_logit = torch.zeros_like(proposal_logit)
        branch_logits = torch.cat([anchor_logit, proposal_logit], dim=-1)  # [B, 2]
        return branch_logits, delta_a, g


def _build_conditional_value_head(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        get_head_layer_norm(64),
        get_activation(),
        nn.Linear(64, 1),
    )

# ---------------------------------------------------------------------------
# 完整模型: HB-GAT-PN (Heterogeneous Graph Attention Pointer Network)
# ---------------------------------------------------------------------------
class HBGATPN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 嵌入与编码 (Feature Extraction)
        self.embedder = FeatureEmbedder(config)
        self.encoder = build_graph_encoder(config)
        
        # 2. 解码器 (Policy Heads)
        self.task_head = TaskPointer(config)
        self.station_head = StationSelector(config)
        self.worker_head = WorkerPointer(config)
        self.conditional_team_head = None
        self.anchor_team_head = None
        self.anchor_proposal_gate = None

        action_scope = str(
            getattr(config, "policy_action_scope", "operation_station_worker")
        )
        if action_scope == "operation":
            self.station_head.requires_grad_(False)
            self.worker_head.requires_grad_(False)
        elif action_scope == "operation_station":
            self.worker_head.requires_grad_(False)
        elif action_scope == "operation_station_gated_team":
            self.worker_head.requires_grad_(False)
            self.conditional_team_head = ConditionalTeamSelector(config)
        elif action_scope == "operation_station_anchor_proposal_team":
            self.worker_head.requires_grad_(False)
            self.anchor_team_head = AnchorConditionedTeamPointer(config)
            self.anchor_proposal_gate = AnchorProposalGate(config)
        if str(getattr(config, "team_selection_mode", "autoregressive")) == "static_topq":
            self.worker_head.ar_query_proj.requires_grad_(False)
        
        # 3. 价值网络 (Critic) 
        # 独立骨干网络维度
        if not getattr(config, 'use_shared_trunk', False):
            self.critic_embedder = FeatureEmbedder(config)
            self.critic_encoder = build_graph_encoder(config)
        else:
            self.critic_embedder = None
            self.critic_encoder = None
        
        # Attention Pooling Optimization (加装防暴击稳压器)
        self.actor_station_attn = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            get_head_layer_norm(32),
            get_activation(),
            nn.Linear(32, 1)
        )
        self.actor_task_worker_attn = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            get_head_layer_norm(32),
            get_activation(),
            nn.Linear(32, 1)
        )
        if str(getattr(config, "actor_context_mode", "attention")) != "attention":
            self.actor_station_attn.requires_grad_(False)
            self.actor_task_worker_attn.requires_grad_(False)
        
        self.critic_station_attn = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            get_head_layer_norm(32),
            get_activation(),
            nn.Linear(32, 1),
        )
        self.critic_task_worker_attn = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            get_head_layer_norm(32),
            get_activation(),
            nn.Linear(32, 1),
        )

        c_dim = config.hidden_dim * 3
        
        self.critic = nn.Sequential(
            nn.Linear(c_dim, 64),
            get_head_layer_norm(64),
            get_activation(),
            nn.Linear(64, 1)
        )

        self.conditional_head_baseline_mode = str(
            getattr(config, "conditional_head_baseline_mode", "off")
        )
        if self.conditional_head_baseline_mode not in {
            "off",
            "diagnostic",
            "factorized",
        }:
            raise ValueError(
                "conditional_head_baseline_mode 必须是 off、diagnostic 或 factorized"
            )
        self.critic_task_cond = None
        self.critic_station_cond = None
        self.critic_worker_cond = None
        if self.conditional_head_baseline_mode != "off":
            conditional_seed = int(getattr(config, "seed", 42)) + 2009
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.manual_seed(conditional_seed)
                self.critic_task_cond = _build_conditional_value_head(c_dim)
                self.critic_station_cond = _build_conditional_value_head(
                    c_dim + config.hidden_dim
                )
                self.critic_worker_cond = _build_conditional_value_head(
                    c_dim + config.hidden_dim * 2
                )
        
        self.last_s_weights = None 
        self.last_s_var = 0.0      # [新增] 站位关注度方差，用于衡量 Critic 是否定位到瓶颈

    def compute_conditional_values(
        self,
        *,
        critic_context: torch.Tensor,
        critic_task_emb: torch.Tensor,
        critic_station_emb: torch.Tensor,
        virtual_station: torch.Tensor | None = None,
        detach_inputs: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """基于同一 Critic context 计算 task/station/worker 条件基线。"""

        if self.conditional_head_baseline_mode == "off":
            raise RuntimeError("conditional critic heads 未启用")
        hidden_dim = int(self.config.hidden_dim)
        batch_size = critic_context.size(0)
        assert critic_context.shape == (batch_size, hidden_dim * 3)
        assert critic_task_emb.shape == (batch_size, hidden_dim)
        assert critic_station_emb.shape == (batch_size, hidden_dim)
        if virtual_station is not None:
            assert virtual_station.shape == (batch_size,)
            assert virtual_station.dtype == torch.bool
            critic_station_emb = torch.where(
                virtual_station.unsqueeze(-1),
                torch.zeros_like(critic_station_emb),
                critic_station_emb,
            )
        if detach_inputs is None:
            detach_inputs = self.conditional_head_baseline_mode == "diagnostic"
        if detach_inputs:
            critic_context = critic_context.detach()
            critic_task_emb = critic_task_emb.detach()
            critic_station_emb = critic_station_emb.detach()
        assert self.critic_task_cond is not None
        assert self.critic_station_cond is not None
        assert self.critic_worker_cond is not None
        with torch.amp.autocast(
            device_type=critic_context.device.type, enabled=False
        ):
            critic_context = critic_context.float()
            critic_task_emb = critic_task_emb.float()
            critic_station_emb = critic_station_emb.float()
            task_value = self.critic_task_cond(critic_context)
            station_input = torch.cat(
                [critic_context, critic_task_emb], dim=-1
            )  # [B,3H] + [B,H] -> [B,4H]
            station_value = self.critic_station_cond(station_input)
            worker_input = torch.cat(
                [critic_context, critic_task_emb, critic_station_emb], dim=-1
            )  # [B,3H] + [B,H] + [B,H] -> [B,5H]
            worker_value = self.critic_worker_cond(worker_input)
        values = {
            "task": task_value,
            "station": station_value,
            "worker": worker_value,
        }
        assert all(value.shape == (batch_size, 1) for value in values.values())
        assert all(torch.isfinite(value).all() for value in values.values())
        return values

    def _policy_node_types(self) -> set[str]:
        scope = str(getattr(self.config, "policy_observation_scope", "full"))
        if scope == "task":
            return {"task"}
        if scope == "task_station":
            return {"task", "station"}
        if scope == "full":
            return {"task", "station", "worker", "skill"}
        raise ValueError(f"未知 policy_observation_scope: {scope!r}")

    def _encode_policy_graph(
        self,
        batch_data: HeteroData,
        embedder: FeatureEmbedder,
        encoder: nn.Module | None,
    ) -> dict[str, torch.Tensor]:
        allowed = self._policy_node_types()
        raw_x_dict = {
            name: value for name, value in batch_data.x_dict.items() if name in allowed
        }
        x_dict = embedder(raw_x_dict)
        edge_index_dict = {
            edge_type: edge_index
            for edge_type, edge_index in batch_data.edge_index_dict.items()
            if edge_type[0] in allowed and edge_type[2] in allowed
        }
        if encoder is None:
            return x_dict
        return encoder(x_dict, edge_index_dict)

    def _compute_global_context(
        self,
        x_dict_encoded,
        batch_data,
        *,
        mode: str,
        station_attn: nn.Module,
        task_worker_attn: nn.Module,
    ):
        from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
        from torch_geometric.utils import softmax

        task_x = x_dict_encoded["task"]
        task_batch = getattr(batch_data["task"], "batch", None)
        batch_size = (
            int(task_batch.max().item()) + 1
            if task_batch is not None
            else 1
        )

        def zeros() -> torch.Tensor:
            return torch.zeros(
                (batch_size, self.config.hidden_dim),
                dtype=task_x.dtype,
                device=task_x.device,
            )

        if mode == "local_only":
            return torch.zeros(
                (batch_size, self.config.hidden_dim * 3),
                dtype=task_x.dtype,
                device=task_x.device,
            )

        def attention_context(
            node_type: str,
            scorer: nn.Module,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            node_x = x_dict_encoded.get(node_type)
            if node_x is None:
                return zeros(), None
            node_batch = getattr(batch_data[node_type], "batch", None)
            weights = scorer(node_x)
            if node_batch is not None:
                alphas = softmax(weights, node_batch)
                context = global_add_pool(node_x * alphas, node_batch)
            else:
                alphas = F.softmax(weights, dim=0)
                context = torch.sum(node_x * alphas, dim=0, keepdim=True)
            return context, alphas

        if mode == "attention":
            station_ctx, station_weights = attention_context("station", station_attn)
            task_ctx, _ = attention_context("task", task_worker_attn)
            worker_ctx, _ = attention_context("worker", task_worker_attn)
            self.last_s_weights = (
                station_weights.detach() if station_weights is not None else None
            )
            self.last_s_var = (
                float(torch.var(station_weights).item())
                if station_weights is not None
                else 0.0
            )
            return torch.cat([station_ctx, task_ctx, worker_ctx], dim=1)

        def pooled_context(node_type: str) -> tuple[torch.Tensor, torch.Tensor]:
            node_x = x_dict_encoded.get(node_type)
            if node_x is None:
                return zeros(), zeros()
            node_batch = getattr(batch_data[node_type], "batch", None)
            if node_batch is not None:
                return (
                    global_mean_pool(node_x, node_batch),
                    global_max_pool(node_x, node_batch),
                )
            return (
                torch.mean(node_x, dim=0, keepdim=True),
                torch.max(node_x, dim=0, keepdim=True)[0],
            )

        station_mean, station_max = pooled_context("station")
        task_mean, task_max = pooled_context("task")
        worker_mean, worker_max = pooled_context("worker")
        return torch.cat(
            [station_mean, task_mean, worker_mean, station_max, task_max, worker_max],
            dim=1,
        )

    def forward(self, batch_data):
        """
        前向传播: 仅用于计算 Encoder 的输出和 Global Context。
        具体的 Action Logits 计算在 Agent 中分步调用各个 Head。
        """
        # --- Step 1: 编码 ---
        x_dict_encoded = self._encode_policy_graph(
            batch_data,
            self.embedder,
            self.encoder,
        )
        
        global_context = self._compute_global_context(
            x_dict_encoded,
            batch_data,
            mode=str(getattr(self.config, "actor_context_mode", "attention")),
            station_attn=self.actor_station_attn,
            task_worker_attn=self.actor_task_worker_attn,
        )
             
        return x_dict_encoded, global_context

    @staticmethod
    def _select_critic_node_embeddings(
        encoded: dict[str, torch.Tensor],
        batch_data: HeteroData,
        node_type: str,
        local_indices: torch.Tensor,
        graph_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_embeddings = encoded[node_type]
        assert local_indices.ndim == 1
        node_store = batch_data[node_type]
        ptr = getattr(node_store, "ptr", None)
        node_batch = getattr(node_store, "batch", None)
        if ptr is not None:
            ptr = ptr.to(device=local_indices.device, dtype=torch.long)
            if graph_indices is None:
                graph_indices = torch.arange(
                    local_indices.numel(), device=local_indices.device
                )
            graph_indices = graph_indices.to(
                device=local_indices.device, dtype=torch.long
            )
            assert graph_indices.shape == local_indices.shape
            global_indices = ptr[graph_indices] + local_indices
            assert torch.all(
                (graph_indices >= 0) & (graph_indices + 1 < ptr.numel())
            )
        else:
            assert node_batch is None
            global_indices = local_indices
        assert torch.all(
            (global_indices >= 0) & (global_indices < node_embeddings.size(0))
        )
        return node_embeddings[global_indices]

    def get_conditional_values(
        self,
        batch_data: HeteroData,
        *,
        selected_task: torch.Tensor,
        selected_station: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        actor_x_dict_encoded: dict[str, torch.Tensor] | None = None,
        critic_x_dict_encoded: dict[str, torch.Tensor] | None = None,
        critic_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """从同一 Critic context 和 Critic 节点 embedding 得到三项基线。"""

        if self.conditional_head_baseline_mode == "off":
            raise RuntimeError("conditional critic heads 未启用")
        selected_task = selected_task.reshape(-1).to(dtype=torch.long)
        selected_station = selected_station.reshape(-1).to(dtype=torch.long)
        assert selected_task.shape == selected_station.shape
        if batch_indices is None:
            batch_indices = torch.arange(
                selected_task.numel(), device=selected_task.device
            )
        batch_indices = batch_indices.reshape(-1).to(
            device=selected_task.device, dtype=torch.long
        )
        assert batch_indices.shape == selected_task.shape
        if critic_x_dict_encoded is None or critic_context is None:
            critic_x_dict_encoded, critic_context = self.get_critic_context(
                batch_data,
                actor_x_dict_encoded=actor_x_dict_encoded,
            )
        task_emb = self._select_critic_node_embeddings(
            critic_x_dict_encoded,
            batch_data,
            "task",
            selected_task,
            batch_indices,
        )
        virtual_station = selected_station < 0
        assert torch.all(
            virtual_station
            | (selected_station < int(batch_data["station"].num_nodes))
        )
        station_emb = self._select_critic_node_embeddings(
            critic_x_dict_encoded,
            batch_data,
            "station",
            selected_station.clamp_min(0),
            batch_indices,
        )
        return self.compute_conditional_values(
            critic_context=critic_context,
            critic_task_emb=task_emb,
            critic_station_emb=station_emb,
            virtual_station=virtual_station,
        )

    def get_critic_context(self, batch_data, actor_x_dict_encoded=None):
        """计算一次 Critic 编码与 context，供 state value 和条件 heads 复用。"""
        if getattr(self.config, "use_shared_trunk", False):
            if actor_x_dict_encoded is None:
                actor_x_dict_encoded = self._encode_policy_graph(
                    batch_data,
                    self.embedder,
                    self.encoder,
                )
            c_x_dict_encoded = actor_x_dict_encoded
        else:
            c_x_dict_encoded = self._encode_policy_graph(
                batch_data,
                self.critic_embedder,
                self.critic_encoder,
            )
        c_global_context = self._compute_global_context(
            c_x_dict_encoded,
            batch_data,
            mode="attention",
            station_attn=self.critic_station_attn,
            task_worker_attn=self.critic_task_worker_attn,
        )
        return c_x_dict_encoded, c_global_context

    def get_value(self, batch_data, actor_x_dict_encoded=None):
        """
        Dual-Stream / Shared-Trunk Critic
        Critic 支持与 Actor 的骨干共享
        """
        if getattr(self.config, 'use_shared_trunk', False):
            if actor_x_dict_encoded is None:
                actor_x_dict_encoded = self._encode_policy_graph(
                    batch_data,
                    self.embedder,
                    self.encoder,
                )
            c_x_dict_encoded = actor_x_dict_encoded
        else:
            # 1. 独立编码
            c_x_dict_encoded = self._encode_policy_graph(
                batch_data,
                self.critic_embedder,
                self.critic_encoder,
            )
            
        # 2. 独立池化 (Attention or Mean+Max)
        c_global_context = self._compute_global_context(
            c_x_dict_encoded,
            batch_data,
            mode="attention",
            station_attn=self.critic_station_attn,
            task_worker_attn=self.critic_task_worker_attn,
        )
             
        # 3. 输出价值
        return self.critic(c_global_context)
