import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, global_mean_pool, global_max_pool
from configs import configs

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

# ---------------------------------------------------------------------------
# 决策一：工序选择 (Task Pointer)
# 机制: 指针网络 (Pointer Network) 从候选集中选择一个工序
# ---------------------------------------------------------------------------
class TaskPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        from configs import configs
        c_dim = config.hidden_dim * 3 if getattr(configs, 'use_attention_critic', True) else config.hidden_dim * 6
        self.context_proj = nn.Linear(c_dim, config.hidden_dim) # 动态支持 Attention Pooling 或 Full Max+Mean
        self.task_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn = nn.Linear(config.hidden_dim, 1)
        
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
        ctx = self.context_proj(global_context).unsqueeze(1) # [B, 1, H]
        
        if task_emb.dim() == 2:
             task_emb = task_emb.unsqueeze(0) # [1, N, H]
        
        tsk = self.task_proj(task_emb)      
        
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

    def forward_choice(self, task_emb, worker_embs, mask=None, current_team_emb=None):
        """选择下一个工人"""
        from configs import configs
        if getattr(configs, 'use_autoregressive_worker', True) and current_team_emb is not None:
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

# ---------------------------------------------------------------------------
# 完整模型: HB-GAT-PN (Heterogeneous Graph Attention Pointer Network)
# ---------------------------------------------------------------------------
class HBGATPN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 嵌入与编码 (Feature Extraction)
        self.embedder = FeatureEmbedder(config)
        self.encoder = HeteroGATEncoder(config)
        
        # 2. 解码器 (Policy Heads)
        self.task_head = TaskPointer(config)
        self.station_head = StationSelector(config)
        self.worker_head = WorkerPointer(config)
        
        # 3. 价值网络 (Critic) 
        # 独立骨干网络维度
        from configs import configs
        if not getattr(configs, 'use_shared_trunk', False):
            self.critic_embedder = FeatureEmbedder(config)
            self.critic_encoder = HeteroGATEncoder(config)
        else:
            self.critic_embedder = None
            self.critic_encoder = None
        
        from configs import configs
        # Attention Pooling Optimization (加装防暴击稳压器)
        self.station_attn = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            get_head_layer_norm(32),
            get_activation(),
            nn.Linear(32, 1)
        )
        self.task_worker_attn = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            get_head_layer_norm(32),
            get_activation(),
            nn.Linear(32, 1)
        )
        
        c_dim = config.hidden_dim * 3 if getattr(configs, 'use_attention_critic', True) else config.hidden_dim * 6
        
        self.critic = nn.Sequential(
            nn.Linear(c_dim, 64),
            get_head_layer_norm(64),
            get_activation(),
            nn.Linear(64, 1)
        )
        
        self.last_s_weights = None 
        self.last_s_var = 0.0      # [新增] 站位关注度方差，用于衡量 Critic 是否定位到瓶颈

    def _compute_global_context(self, x_dict_encoded, batch_data):
        from torch_geometric.utils import softmax
        from configs import configs
        from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
        
        if getattr(configs, 'use_attention_critic', True) and hasattr(batch_data['station'], 'batch') and batch_data['station'].batch is not None:
             s_batch = batch_data['station'].batch
             t_batch = batch_data['task'].batch
             w_batch = batch_data['worker'].batch
             
             s_weights = self.station_attn(x_dict_encoded['station'])
             s_alphas = softmax(s_weights, s_batch)
             station_ctx = global_add_pool(x_dict_encoded['station'] * s_alphas, s_batch)
             
             t_weights = self.task_worker_attn(x_dict_encoded['task'])
             t_alphas = softmax(t_weights, t_batch)
             task_ctx = global_add_pool(x_dict_encoded['task'] * t_alphas, t_batch)
             
             w_weights = self.task_worker_attn(x_dict_encoded['worker'])
             w_alphas = softmax(w_weights, w_batch)
             worker_ctx = global_add_pool(x_dict_encoded['worker'] * w_alphas, w_batch)
             
             global_context = torch.cat([station_ctx, task_ctx, worker_ctx], dim=1)
             self.last_s_weights = s_alphas.detach()
             self.last_s_var = torch.var(s_alphas.view(global_context.size(0), -1), dim=1).mean().item()
             
        elif getattr(configs, 'use_attention_critic', True):
             s_weights = self.station_attn(x_dict_encoded['station'])
             s_alphas = F.softmax(s_weights, dim=0)
             station_ctx = torch.sum(x_dict_encoded['station'] * s_alphas, dim=0, keepdim=True)
             
             t_weights = self.task_worker_attn(x_dict_encoded['task'])
             t_alphas = F.softmax(t_weights, dim=0)
             task_ctx = torch.sum(x_dict_encoded['task'] * t_alphas, dim=0, keepdim=True)
             
             w_weights = self.task_worker_attn(x_dict_encoded['worker'])
             w_alphas = F.softmax(w_weights, dim=0)
             worker_ctx = torch.sum(x_dict_encoded['worker'] * w_alphas, dim=0, keepdim=True)
             
             global_context = torch.cat([station_ctx, task_ctx, worker_ctx], dim=1)
             self.last_s_weights = s_alphas.detach()
             self.last_s_var = torch.var(s_alphas).item()
             
        elif hasattr(batch_data['station'], 'batch') and batch_data['station'].batch is not None:
             station_mean = global_mean_pool(x_dict_encoded['station'], batch_data['station'].batch)
             task_mean = global_mean_pool(x_dict_encoded['task'], batch_data['task'].batch)
             worker_mean = global_mean_pool(x_dict_encoded['worker'], batch_data['worker'].batch)
             
             station_max = global_max_pool(x_dict_encoded['station'], batch_data['station'].batch)
             task_max = global_max_pool(x_dict_encoded['task'], batch_data['task'].batch)
             worker_max = global_max_pool(x_dict_encoded['worker'], batch_data['worker'].batch)
             
             global_context = torch.cat([station_mean, task_mean, worker_mean, station_max, task_max, worker_max], dim=1)
        else:
             station_mean = torch.mean(x_dict_encoded['station'], dim=0, keepdim=True)
             task_mean = torch.mean(x_dict_encoded['task'], dim=0, keepdim=True)
             worker_mean = torch.mean(x_dict_encoded['worker'], dim=0, keepdim=True)
             
             station_max = torch.max(x_dict_encoded['station'], dim=0, keepdim=True)[0]
             task_max = torch.max(x_dict_encoded['task'], dim=0, keepdim=True)[0]
             worker_max = torch.max(x_dict_encoded['worker'], dim=0, keepdim=True)[0]
             
             global_context = torch.cat([station_mean, task_mean, worker_mean, station_max, task_max, worker_max], dim=1)
             
        return global_context

    def forward(self, batch_data):
        """
        前向传播: 仅用于计算 Encoder 的输出和 Global Context。
        具体的 Action Logits 计算在 Agent 中分步调用各个 Head。
        """
        # --- Step 1: 编码 ---
        x_dict = self.embedder(batch_data.x_dict)
        
        from configs import configs
        if getattr(configs, 'ablation_no_gat', False):
            # Ablation Logic: 跳过所有图卷积，直接使用投影后特征
            x_dict_encoded = x_dict
        else:
            x_dict_encoded = self.encoder(x_dict, batch_data.edge_index_dict)
        
        global_context = self._compute_global_context(x_dict_encoded, batch_data)
             
        return x_dict_encoded, global_context

    def get_value(self, batch_data, actor_x_dict_encoded=None):
        """
        Dual-Stream / Shared-Trunk Critic
        Critic 支持与 Actor 的骨干共享
        """
        from configs import configs
        if getattr(configs, 'use_shared_trunk', False):
            if actor_x_dict_encoded is None:
                x_dict = self.embedder(batch_data.x_dict)
                if getattr(configs, 'ablation_no_gat', False):
                    actor_x_dict_encoded = x_dict
                else:
                    actor_x_dict_encoded = self.encoder(x_dict, batch_data.edge_index_dict)
            c_x_dict_encoded = actor_x_dict_encoded
        else:
            # 1. 独立编码
            c_x_dict = self.critic_embedder(batch_data.x_dict)
            
            if getattr(configs, 'ablation_no_gat', False):
                c_x_dict_encoded = c_x_dict
            else:
                c_x_dict_encoded = self.critic_encoder(c_x_dict, batch_data.edge_index_dict)
            
        # 2. 独立池化 (Attention or Mean+Max)
        c_global_context = self._compute_global_context(c_x_dict_encoded, batch_data)
             
        # 3. 输出价值
        return self.critic(c_global_context)
