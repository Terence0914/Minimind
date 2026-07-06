from transformers import PretrainedConfig

class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


#RMSNorm
class RMSNorm(nn.Module):
    def __init__(self,dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        #-1是取最后一个维度也就是沿着特征维度进行归一化
        #rsqrt 是取反平方根（1/sqrt)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

#RoPE & YaRN    
def precompute_freqs(
        #每个注意力头的特征维度
        dim:int, 
        #预计算的最大序列长度
        end:int = int(32*1024), 
        #RoPE的底数（LLaMA 1/2 默认10000， LLaMA 3提高到了500000/1e6）
        rope_base:float = 1e6, 
        #是否启用上下文扩展
        rope_scaling : Optional[dict] = None,
):
    
    #计算基础的旋转频率θ，原公式为 θ = 10000^(-2i/d), 其中2i 是维度的偶数索引 (0, 2, 4, 6)，
    # [:(dim // 2)]是切片， 因为RoPE 旋转位置编码必须在“偶数维度”上成对操作
    # 1/...是为了取倒数，也就是那个-号
    freqs, attn_factor = (1.0 / rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim) , 1.0)

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), #模型最初训练时的最大长度
            rope_scaling.get("factor", 16), #你想拉长的倍数，2048 * 16 = 32k 
            rope_scaling.get("beta_fast", 32.0), #这个和下面的那个都是“波长” 用来界定哪些是高频，哪些是低频
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0), #注意力温度补偿系数，用来在变长后调整softmax的分布
            )
        if end / orig_max > 1.0:
            #反推lambda = orig_max / b 求i
            inv_dim = lambda b : (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0), #floor向下取整，尽量保住更多的高频细节不被修改，同时确保计算出的索引不会小于 0
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1), #ceil向上取整,为了让过渡区宽一点，减 1 是因为代码里的索引是从 0 开始的，同时确保索引不会超过 RoPE 编码的最大维度范围
            )

            #ramp = i - low / high - low
            #dim // 2 是因为旋转位置编码（RoPE）是把维度两两分组来做cos, sin
            # torch.clamp(计算结果, 0, 1)，意思是“强制截断”！ 如果算出来的结果小于 0，强制变成 0，如果算出来的结果大于 1，强制变成 1
            ramp = torch.clamp(
            (torch.arange(dim // 2, device = freqs.device).float() - low) / max(high - low, 0.001), 0, 1,
            )

            #对于高频部分（局部细节）： 这里的 ramp 值为 0 原封不动
            #对于低频部分（全局长文）： 这里的 ramp 值为 1
            #对于中频部分（过渡区）： 这里的 ramp 是 0 到 1 之间的小数（比如 0.5）。
            freqs = freqs * (1 - ramp + ramp / factor)

        #获取每一个词的位置坐标
        t = torch.arange(end, device = freqs.device)
        #计算每个词，每个维度的“绝对旋转角度” 角度 = 位置 * 频率，注意outer 方法是把第一个数组的每一个元素，拿去跟第二个数组里的所有元素分别相乘
        freqs = torch.outer(t, freqs).float()
        #在最后一个维度拼接，因为前面把隐藏维度进行了两两分组，为了旋转，后续要给他拼回去
        freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim = -1) * attn_factor
        freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1) * attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
    def rotate_half(x):
        #设我们有一个隐藏层向量 x，里面有 4 个数字：[A, B, C, D]。
        # x.shape[-1] // 2：就是找到向量的中点（4的一半是 2）
        # x[..., x.shape[-1] // 2:]：拿走后半部分  [C, D]
        # x[..., : x.shape[-1] // 2]：拿走前半部分  [A, B]
        # -x[...] (对右半边加负号)：右半边变成了 [-C, -D]
        # torch.cat(...) (把它们重新拼起来)：把加了负号的右半边放在前面，左半边放在后面
        # [A, B, C, D] 经过 rotate_half 处理后，变成了 [-C, -D, A, B]
        return torch.cat(
            (-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim = -1
        )
    
    # 应用二维平面旋转公式：
    # x' = x * cos(θ) - y * sin(θ)
    # y' = x * sin(θ) + y * cos(θ)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q)) * sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k)) * sin.unsqueeze(unsqueeze_dim)
    return q_embed, k_embed


#repeat_kv是因为后续GQA中，4个query向量匹配一个kv值，在后续计算attention的时候，维度对不齐，所以需要repeat_kv来和query向量对其
def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    #bs: batch_size
    #slen:sequence Length
    #n_rep: 重复的倍数
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    return(
        x[:,:,:,None,:] #在第3维度插入一个新的维度，形状变为(bs, slen, num_key_value_heads, 1, head_dim)
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim) # 广播扩张后形状为(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim) #最终形状为(be, slen, num_key_value_heads * n_rep, head_dim)
    )

#GQA - 4个query向量匹配一个KV值
class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads #配置中给了8个
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        
        #检查num_attention_heads是否是num_key_value_heads的整数倍
        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads #8个
        self.n_local_kv_heads = self.num_key_value_heads #2个
        #计算重复倍数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads #4次
        #计算单头维度
        self.head_dim = args.hidden_size // args.num_attention_heads  #64

        #线性变换矩阵wx + b, 但是在大模型预训练中，不太需要加偏置了
        #nn.Linear(输入维度，输出维度)
        #Q投影，输入512，输出512
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias = False
        )
        #512压缩4倍为128，也就是说在这里，Q的头数是KV头数的4倍
        #K投影，输入512，输出128
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False
        )
        #V投影，输入512，输出128
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias = False
        )
        #Output 投影，后续进行结果拼接，输入512， 输出512
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias = False
        )
        #随机失活防止过拟合，虽然在大模型预训练中防止过拟合是直接投入海量数据，但也要定义一下
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        #检查当前pytorch版本是否支持原生功能
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )
    
    #前向传播
    def forward(
            self,
            x:torch.Tensor, #输入数据，形状为batch_size, sequence_length, hidden_size
            position_embeddings : Tuple[torch.Tensor, torch.Tensor], #位置编码，定义成了一个包含两个Tensor的元组（因为我们用的是旋转位置编码）
            past_key_value : Optional[Tuple[torch.Tensor, torch.Tensor]] = None, #KV Cache 键值缓存
            use_cache = False, #在预训练阶段，不需要缓存历史
            attention_mask: Optional[torch.Tensor] = None,
    ):
        #读取批次大小，序列长度
        bsz,seq_len,_ = x.shape
        #线性投影
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim) #batch_size, seq_len, 8, 64
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim) #batch_size, seq_len, 2, 64
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim) #batch_size, seq_len, 2, 64

        #旋转位置编码RoPE实现
        cos, sin = position_embeddings #这里的cos, sin就是上面的freqs_cos, freqs_sin
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        #KV_cache 实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1) #past_key_value[0]：索引拿到的就是历史的 K, 在这里是沿着序列长度维度拼接
            xv = torch.cat([past_key_value[1], xv], dim = 1) #past_key_value[1]：索引拿到的就是历史的 V，在这里是沿着序列长度维度拼接
        past_kv = (xk, xv) if use_cache else None

        # 重组多头阵型
        xq, xk, xv = (
            xq.transpose(1, 2), #转置后batch_size, num_heads, seq_len, head_dim
            repeat_kv(xk, self.n_rep).transpose(1, 2), #转置后batch_size, num_heads, seq_len, head_dim
            repeat_kv(xv, self.n_rep).transpose(1, 2), #转置后batch_size, num_heads, seq_len, head_dim
        )

        if (
            self.flash #调用Flash Attention 但是得满足下面3个条件
            and (seq_len > 1) #如果 == 1，证明模型还在生成阶段，调用Flash Attention 算子反而会因为调度开销变慢，这个更适合一次性吃进长篇大论
            and (past_key_value is None) #没有使用KV Cache,说明模型现在处于训练阶段，或者正在处理用户的初始输入提示词
            and (attention_mask is None or torch.all(attention_mask == 1)) #要求没有传入极其特殊形状的矩阵掩码
        ):
            #如果上述条件全部满足，一键包办计算带掩码的注意力公式
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p = self.dropout if self.training else 0.0,
                if_causal = True, #使用下三角掩码
            )

        #如果上述条件没有满足，就纯手工
        else:
            #attention 公式计算 Q*K^T（转置) / sqrt(dk),转置是为了矩阵乘法 -> seq_len, head_dim * head_dim, seq_len = seq_len, seq_len
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #因果掩码，防止模型看到后面的词
            #Batch_Size, n_heads, Q_seq_len, K_seq_len：，注意K_seq_len后面有个：,意味着只需要K序列的最后seq_len个
            #-seq_len 代表的是当前新加入的这批新词（因为kv cache中只保存了最新的seq_len个词）。
            # 我们讲的“因果掩码（下三角矩阵）”，仅仅且必须只作用在这批新词内部
            #triu 的全称是 Triangle Upper（上三角）, 保留矩阵右上方的元素，把左下方的元素强行变成 0
            #diagonal = 1：triu 的切割线会向上平移一格, 默认也是1
            scores [:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device = scores.device), #这里创建了一个大小为 [seq_len, seq_len] 的全负无穷 (-inf) 矩阵
                diagonal = 1,
            )

            #填充掩码，一个batch里的句子长短不一，为了补齐张量，需要填充大量的填充符<pad>
            if attention_mask is not None:
                #两个unsqueeze是因为掩码通常是一个二维，但是在计算注意力分数时，需要把它扩展成四维，才能和 scores 的形状对齐
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
                #如果原本是真实词，为1, (1.0 - 1) * -1e9 = 0 对分数没有影响
                #如果原本是填充词，为0, (1.0 - 0) * -1e9 = -1e9, 分数模型不会将注意力放在这里
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9 
                # 一票否决的作用，分辨哪些是有用的，哪些是填充的
                scores = scores + extended_attention_mask
            
            #归一化
            scores = F.softmax(scores.float(), dim = -1).type_as(xq)
            scores = self.attn_dropout(scores)
            #把算好、清理过、且转化为概率分布的注意力权重矩阵，去乘以包含实际内容特征 value，至此完成我们的attention公式 Q*K^T/sqrt(dk) -> softmax -> * V
            output = scores @ xv

        #转置回原型，然后把n_heads, head_dim 压缩成一个维度 -> batch_size, seq_len, 512
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        #把不同的注意力头融合和信息交互，过一次dropout防止过拟合
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

#前馈神经网络 - Dense Model
class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super(). __init__()
        if config.intermediate_size is None:
            #乘以8/3是保持模型参数量和计算量的不变
            intermediate_size = int(config.hidden_size * 8 / 3) # -> 512 * （8/3） = 1365
            #硬件对齐 - 硬件在处理 64 的倍数的矩阵时，效率最高
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64) # intermediate_size = 1408

        #注意x进来后，是同时进入了两个线性层gate & up
        #门控投影： 输入512， 输出1408， 把512 维放大到1408维
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias = False
        )
         #升维：同样输入维度512，输出维度1408，提取丰富的特征
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias = False
        )
        #降维：输入维度1408，输出维度512，等前两步在 1408 维的空间里相乘筛选完之后，再把这 1408 维的结果，重新压缩回 512 维，方便传递给下一层 Transformer
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias = False
        )
        self.dropout = nn.Dropout(config.dropout)
        #激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 经过 gate_proj 放大并穿过激活函数，生成了一个“过滤器”（值通常在 0 到 1 之间徘徊）->用作门控开关
        # 经过 up_proj 放大，提取出丰富的特征，两者相乘，哪些特征是重要的（保留），哪些是不重要的（被削弱）-> 用作提取丰富特征
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x) #1408维度
        #返回压缩降维- 512 维
        return self.dropout(self.down_proj(gated))

#Moe Gate 实现
class MoEGate(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        #稀疏激活参数：2， 模型选出得分最高的2个专家进行前向传播
        self.top_k = config.num_experts_per_tok 
        #参与动态路由的专家总数：4
        self.n_routed_experts = config.n_routed_experts

        #定义了计算专家得分的函数
        self.scoring_func = config.scoring_func
        #用于控制辅助损失（Auxiliary Loss） - 确保所有的底层硬件算力（所有专家）都能得到充分且均匀的利用
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        #概率分布处理，决定在选出top-k个专家后，是否将这几个专家的路由概率重新归一化处理，使得被选中专家的权重分布之和为 1
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        #将权重声明为可学习张量，初始化为空张量，形状为4 x 隐藏层维度
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        #赋予合理的初始值分布
        self.reset_parameters()

    #调用kaiming初始化 - 为了在深层网络中维持信号方差的稳定
    def reset_parameters(self) -> None:
        #传入sqrt(5)是为了符合U（-bound, bound)的均匀分布
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        #维度展平， 因为在MoE中，路由决策是token级别的，它不关心这个 Token 属于哪一句话或哪一个批次
        #因此，代码使用 .view(-1, h) 将前两个维度合并，把所有的 Token 排成一列长队，形状变成了 [总Token数, 隐藏层维度]
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        #原始打分 总token数，隐藏层维度 * 隐藏层维度，专家总数（公式自带转置） —>  总token数，专家总数
        logits = F.linear(hidden_states, self.weight, None)
        
        #为了将其转化为标准的概率分布，代码在最后一个维度（dim=-1，即专家维度）上应用了 Softmax 函数（概率加起来严格等于1.0）
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        #torch.topk 函数会遍历每一个 Token 的专家概率分布，只保留概率最大的前 K 个（这里是2个）
        #topk_weight 记录被选中的这K个专家的具体概率值
        #topk_idx 记录了被选中的那 K 个专家的编号
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        #如果只选1个专家，那么它占据了所有被激活的份额，不需要归一化。
        #只有当选了 2 个或更多专家时，才需要把它们各自残缺的概率加起来放大到 1.0
        if self.top_k > 1 and self.norm_topk_prob:
            #keepdim=True 防止维度坍缩
            #1e-20 防止除零灾难
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            #执行归一化除法
            topk_weight = topk_weight / denominator

        #辅助损失函数的计算 - 类似绩效考核
        #辅助损失只在训练阶段（Training）计算
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            #数据塑性 -> batch_size, seq_len * Top_K
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                #ce 记账本- 记录在这个批次中，每个专家被每条序列调用了多少次
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                #scatter_add 根据 topk_idx_for_aux_loss（选中的专家编号），把 torch.ones（每次算1票）累加到 ce 这个记账本对应的位置上。
                # 执行完后，ce 里面装的就是每个专家实际分到的 Token 数量
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                #算出 总任务量 / 公平配额， 数值越大，说明这个专家越被过度使用
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                #ce 和 scores_for_seq_aux相乘，两项如果都很高，乘积就会极大，产生的 aux_loss 就越大，从而在反向传播时狠狠惩罚路由器，逼迫它把 Token 分给其他专家
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                #独热编码
                #num_classes=self.n_routed_experts是为了固定生成n_routed_experts（这里是4）列
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                #在所有 Token 维度上求平均。得出的 ce 是一个长度为专家总数的向量，代表实际分配给每个专家的 Token 比例
                ce = mask_ce.float().mean(0)
                #路由器对每个专家的平均预测概率
                Pi = scores_for_aux.mean(0)
                #Loss = alpha * sum (P_i * f_i * N)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

        #如果是推理阶段，直接生成一个数值为 0 的张量作为 aux_loss
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss

class MoEFeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        #门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )
    
    def forward(self, x):
        identity = x
        #获得维度大小（3维）
        orig_shape = x.shape
        #赋值给这三个变量
        bsz, seq_len, h = orig_shape

        #使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        #展开x以便处理
        #x.shape[-1]保持最后一个维度不变，前面-1是自动计算前两个维度相乘的总数
        #维度大小：bsz * seq_len, h
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        #是否在训练模式
        if self.training:
            #按照定义的num_experts_per_tok重复输入token
            #每个token安排num_experts_per_tok个专家处理
            #沿着第0维- token 数量
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim = 0)
            #y是空张量，和x形状相同
            y = torch.empty_like(x, dtype = x.dtype)
            #遍历所有专家
            for i, expert in enumerate(self.experts):
                #找到所有指向专家i的token
                #然后将这些token输入专家i进行处理
                #最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    #放回y那个空的箱里
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 *sum(
                        p.sum() for p in expert.parameters()
                    )
            #加权求和
            #y.view(*topk_weight.shape, -1)变回原来维度（3维）
            #* topk_weight.unsqueeze(-1)，算出的权重（Top_K 权重）乘到对应的专家输出上，.sum(dim=1)沿着Top_k求和
            #最后的y意义是每个token经过专家处理后的加权结果，恢复成最开始传入该层时的三维形状
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1).sum(dim=1))
            y = y.view(*orig_shape)
        #如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
            if self.config.n_shared_experts > 0:
                for expert in self.shared_experts:
                    y = y + expert(identity)
            self.aux_loss = aux_loss
            return y
    
    @torch.no_grad()
    #MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        #使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            #计算当前包的初始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i -1]
            if start_idx == end_idx:
                continue
            
            #取出当前包对应的专家
            expert = self.experts[i]
            #取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            #取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens). to(expert_cache.dtype)
            #加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            #将结果散点加在缓存中对应位置
            #scatter_add_(dim, index, src) 根据行/列，进行index指定的地址投递，src是投递的具体内容
            #view(-1,1)变成一列，假设x.shape[-1] 隐藏层维度是4，repeat(1,4)就是横向复印4次
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )
        return expert_cache
    
class MokioMindBlock(nn.Module):
    def __init__(self, layer_id:int, config:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        #计算每一个注意力“头”处理的向量维度大小
        self.head_dim = config.hidden_size // config.num_attention_heads #512 // 8 = 64
        #调用自注意GQA
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        #自注意力模块之前使用的层归一化模块
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        #前馈神经网络（MLP）之前使用的层归一化模块
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps = config.rms_norm_eps
        )
        #调用Dense - Model FFN
        self.mlp = (
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)
        )
    
    def forward(
            self,
            hidden_states,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache = False,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        res = hidden_states

        hidden_states, present_key_value = self.self_attention(
            #进入注意力模块之前过一次归一化
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        #残差连接
        hidden_states = res + hidden_states
        
        #在数据进入MLP之前，再次进行层归一化，传入self.mlp()进行非线性变换
        #然后再进行一次残差连接
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value

class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [MokioMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None, #输入的token序列
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, #KV Cache
        use_cache: bool = False, #布尔值，指示当前前向传播是否需要返回新的 KV Cache 供下一步使用
        **kwargs,
    ):
        # input_ids: [bsz, seq_len]
        batch_size, seq_length = input_ids.shape

        #检查是否具有某个属性
        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [bsz, seq_len, hidden]

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        # 为什么加括号？
        # 因为 enumerate(zip) 产生的数据是嵌套结构：(索引, (当前层, 历史缓存))
        # 加括号是为了让变量结构与数据结构对齐，进行精确的嵌套元组解包(Tuple Unpacking)
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )
            ],
            hidden_states.new_zeros(1).squeeze(),
        )

        return hidden_states, presents, aux_loss
    
class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        super().__init__(config)
        self.model = MokioMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output