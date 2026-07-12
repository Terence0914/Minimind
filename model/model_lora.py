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
        # 矩阵A 高斯初始化, 标准差为0.02
        self.A.weight.data.normal_(mean = 0.0, std = 0.02)
        # 矩阵B 全0初始化
        self.B.weight.data.zero_()
    
    def forward(self, x):
        return self.B(self.A(x))

# 外挂一个lora模块在原模型外
def apply_lora(model, rank = 8): # 秩为8
    # 获取当前模型所在的设备
    device = next(model.parameters()).device
    # 递归遍历该模型中所有的子模块
    for name, module in model.named_modules():
        # 必须是全连接线性层
        # 权重矩阵必须是方阵，即输入特征维度等于输出特征维度
        if(
            isinstance(module, nn.Linear)
            and module.weight.shape[0] == module.weight.shape[1]
        ):
            # 根据当前找到的目标层的输入输出维度，实例化一个 LoRA 模块(低秩矩阵 A 和 B)
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank = rank).to(device)
            setattr(module, "lora", lora)
            # 保存原始层的前向传播函数的引用（即冻结的原始权重）
            original_forward = module.forward

            # 定义一个新的前向传播
            def forward_with_lora(x, layer1 = original_forward, layer2 = lora):
                # 原始模型的输出 + LoRA 模块的输出(Wx + BAx)
                return layer1(x) + layer2(x)
            
            # 使用刚刚定义的新函数替换掉原有模块的 forward 方法
            module.forward = forward_with_lora

# 打开硬盘上的大文件，把长名字减掉，把数据集塞入对应的盒子里
def load_lora(model, path):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location = device)
    # 清理数据名字（因为这之前的数据集是在多张显卡上运行的，而作者的显卡只有一张，所以需要删掉module.这前7个字符）
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }
 
    for name, module in model.named_modules():
        # 检查模块身上有没有我们之前外挂的那个"lora"
        if hasattr(module, "lora"):
            lora_state = {
                # 改名，因为内部只认识A.weight 和 B.weight
                # replace(想要被替换的旧词, 替换成的新词)
                k.replace(f"{name}.lora", ""): v
                for k, v in state_dict.items()
                # 只把名字带有layer1.lora.的数据挑出来
                if f"{name}.lora." in k
            }
            # 把处理好的数据，塞入lora模块里
            module.lora.load_state_dict(lora_state)

# 把各个层小盒子里的数据倒出来，拼上长名字标签，统一打包存回硬盘
def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            # 清理数据名字（因为这之前的数据集是在多张显卡上运行的，而作者的显卡只有一张，所以需要删掉module.这前7个字符）
            clean_name = name[7:] if name.startswith("model.") else name
            # 给A.weight 和 B.weight添加一个标签，不然太多一样的
            # 这样使其和其他的不一样（eg.layer1.lora.A.weight)
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            # 打包进最后的那个集合
            state_dict.update(lora_state)
    torch.save(state_dict, path)