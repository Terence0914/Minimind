from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ──────────────────────────────────────────────────────────────────────────────
# 全局预处理 / 后处理工具函数
# ──────────────────────────────────────────────────────────────────────────────

# 百分之80概率什么都不做，百分之20概率从system_prompt随机选中一条进行插入
# 这有效提升泛化能力（适应有/无 System Prompt 的情况）
def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。

    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    if conversations and conversations[0].get("role") != "system":
        if random.random() < add_system_ratio:
            return [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
            ] + conversations
    return conversations


# 大概率情况下是直接删掉，小概率，也就是到了百分之5附近，才保留
def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        # 如果文本里真的有这个空块，就掷一个 0 到 1 之间的随机小数（random.random()）
        # 只有当这个随机数大于 0.05 的时候（也就是有 95% 的概率），条件才算完全成立
        and random.random() > empty_think_ratio
    ):
        # 只有上面满足了，全部替换成空字符
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content


# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Step 1：tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample["text"]),
            # 暂时不用加特殊的开头/结尾符号
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            # 如果遇到特别长、翻译成 Token 后数量超过了 max_length 的文本，请直接把超出的尾巴一刀切掉
            truncation=True,
        ).input_ids

        # Step 2：拼接 BOS + token序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3：右侧用 PAD 补齐到 max_length，保证 batch 内等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        # 把这个 Python 列表转换成 PyTorch 的 tensor（张量），数据类型指定为 64 位整数 torch.long
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4：labels 与 input_ids 完全相同，但 PAD 位置置 -100，
        #         CrossEntropyLoss 会自动忽略 -100，不计入 loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # 返回 attention_mask，使 attention 层能屏蔽 padding token，最后为64位整数类型
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask

