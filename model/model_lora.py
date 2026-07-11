import torch
from torch import optim, nn

# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # LoRA的秩，控制低秩矩阵的大小
        self.rank = rank
        # 低秩矩阵A
        self.A = nn.Linear(in_features, rank, bias = False)
        # 低秩矩阵B
        self.B = nn.Linear(rank, out_features, bias=False)
        # 矩阵A 高斯初始化
        self.A.weight.data.normal_(mean = 0.0, std = 0.02)
        # 矩阵B 全0初始化
        self.B.weight.data.zero_()
    
    def forward(self, x):
        return self.B(self.A(x))
