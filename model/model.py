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
def precompute_freqs_cis(
        #每个注意力头的特征维度
        dim:int, 
        #预计算的最大序列长度
        end:int = int(32*1024), 
        #RoPE的底数（LLaMA 1/2 默认10000， LLaMA 3提高到了500000/1e6）
        rope_base:float = 1e6, 
        #是否启用上下文扩展
        rope_scaling : Optional[dict] = None):
    
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
                max(math.floor(inv_dim(beta_fast)), 0), #floor向下取整，尽量保住更多的高频细节不被修改
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1), #ceil向上取整,为了让过渡区宽一点，减 1 是因为代码里的索引是从 0 开始的
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

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    #bs: batch_size
    #slen:sequence Length
    #n_rep: 重复的倍数
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    
    return(
        x[:,:,:,None,:]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim) # 广播扩张后形状为(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim) #最终形状为(be, slen, num_key_value_heads * n_rep, head_dim)
    )

#GQA - 4个query向量匹配一个KV值
class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads #配置中给了8个
            if args.self.num_key_value_heads is None
            else args.num_key_value_heads
        )

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
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        #KV_cache 实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim = 1) #past_key_value[0]：索引拿到的就是历史的 K
            xv = torch.cat([past_key_value[1], xv], dim = 1) #past_key_value[1]：索引拿到的就是历史的 V
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
            #attention 公式计算 Q*K^T / sqrt(dk),转置是为了矩阵乘法 -> seq_len, head_dim * head_dim, seq_len = seq_len, seq_len
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            #因果掩码，防止模型看到后面的词
            #Batch_Size, n_heads, Q_seq_len, K_seq_len：，注意K_seq_len后面有个：,意味着只需要K序列的最后seq_len个
            #-seq_len 只取最后一条维度的最后 seq_len 个元素
            #triu 的全称是 Triangle Upper（上三角）, 保留矩阵右上方的元素，把左下方的元素强行变成 0
            #diagonal = 1：triu 的切割线会向上平移一格, 默认也是1
            scores [:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device = scores.device),
                diagonal = 1,
            )

            #填充掩码，一个batch里的句子长短不一，为了补齐张量，需要填充大量的填充符<pad>
            if attention_mask is not None:
                #两个unsqueeze是因为掩码通常是一个二维，需要和多头注意力4维张量补齐
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
                #如果原本是真实词，为1, (1.0 - 1) * -1e9 = 0 对分数没有影响
                #如果原本是填充词，为0, (1.0 - 0) * -1e9 = -1e9, 分数模型不会将注意力放在这里
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9 
                # 一票否决的作用，分辨哪些是有用的，哪些是填充的
                scores = scores + extended_attention_mask
            
            #归一化
            scores = F.softmax(scores.float(), dim = -1).type_as(xq)
            scores = self.attn_dropout(scores)
            #把算好、清理过、且转化为概率分布的注意力权重矩阵，去乘以包含实际内容特征 value
            output = scores @ xv

        #转置回原型，然后把n_heads, head_dim 压缩成一个维度 -> batch_size, seq_len, 512
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        #把不同的注意力头融合和信息交互，过一次dropout防止过拟合
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


