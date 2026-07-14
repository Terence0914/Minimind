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

# SFTDataset —— 有监督微调（Supervised Fine-Tuning）数据集
class SFTDataset(Dataset):
    def __init__ (self, jsonl_path, tokenizer, max_length = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files = jsonl_path, split = "train")
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n ", add_special_tokens = False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens = False
        ).input_ids

    def __len__(self):
        return len(self.samples)
    
    # 把“字典格式的对话”，转换成“模型专属格式的长字符串”。
    # eg: {role: user, content: 你好} -> <|im_start|>user\n你好<|im_end|>
    def create_chat_prompt(self, conversations):
        # 复制原始conversations, 防止修改原始数据
        messages = conversations.copy()
        tools = (
            conversations[0]["functions"]
            if(
                # conversations 不为空
                conversations
                # 对话的第一句话[0]的角色[role]是系统提示词[system]
                and conversations[0]["role"] == 'system'
                # 检查里面包含了function 字样，function calling 场景
                and conversations[0].get("functions")
            )
            else None
        )
        # add_generation_prompt=False：不在末尾追加"请模型续写"的 prompt
        return self.tokenizer.apply_chat_template(
            messages, tokenize = False, add_generation_prompt = False, tools = tools
        )
    
    def generate_labels(self, input_ids):
        # 将全部的label都设置为 -100
        labels = [-100] * len(input_ids)
        i = 0
        # 逐位扫描 input_ids，检测是否匹配 bos_id（assistant 回复起始）, 没找到就else: i += 1
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id: # list[start : stop]
                # 找到bos_id就记做start
                start = i + len(self.bos_id)
                end = start
                #  匹配到 bos_id 后，向后扫描直到找到 eos_id（回复结束）
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 [start, end+len(eos_id)) 区间内的 label 设为对应的 input_ids 值，min防止数组越界
                # 即这段 assistant 回复参与 loss 计算
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                # 跳过已处理区间，继续扫描下一段 assistant 回复（支持多轮对话）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # 随机决定是否插入 system prompt（数据增强）
        conversations = pre_processing_chat(sample['conversations'])
        # 用 chat template 渲染完整对话字符串
        prompt = self.create_chat_prompt(conversations)
        # 清理可能出现的空 <think> 块
        prompt = post_processing_chat(prompt)
        # tokenize 并截断到 max_length [:self.max_length]就是做截断
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 不足则右侧 PAD 补齐
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 生成稀疏标签，只有 assistant 回复部分有有效 label
        labels = self.generate_labels(input_ids)
        # 生成一个注意力掩码，是真实文字的位置标记为1，padding填充部分为0
        attention_mask = (
            torch.tensor(input_ids, dtype = torch.long) != self.tokenizer.pad_token_id).long()
        return (
            torch.tensor(input_ids, dtype= torch.long),
            torch.tensor(labels, dtype = torch.long),
            attention_mask
        )

# DPODataset —— 直接偏好优化（Direct Preference Optimization）数据集
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length = 4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens = False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens = False
        ).input_ids
        self.samples = load_dataset("json", data_files = file_path, split = "train")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # # 优质回答对话列表，格式：[{role, content}, ...]
        chosen = sample["chosen"]
        # 劣质回答对话列表，格式同上
        rejected = sample["rejected"]

        # Step 1：将 chosen / rejected 对话分别渲染为字符串
        chosen_prompt = self.tokenizer.apply_chat_templete(
            chosen, tokenize = False, add_generation_prompt = False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_templete(
            rejected, tokenize = False, add_generation_prompt = False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # Step 2：tokenize 并 padding 到 max_length（统一序列长度，方便 batch）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation = True,
            max_length = self.max_length,
            padding = "max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation = True,
            max_length = self.max_length,
            padding = "max_length"
        )
        
        # Step 3：生成 loss mask，只有 assistant 回复部分为 1
        chosen_input_ids = chosen_encoding["input_ids"]
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # Step 4：构造自回归训练对，x=[:-1] 作为输入，y=[1:] 作为目标
        # mask=[1:] 与 y 对齐，决定哪些位置的 loss 计入梯度
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype = torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype = torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype = torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype = torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype = torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype = torch.long)

        attention_mask_chosen = (
            torch.tensor(chosen_input_ids[:-1], dtype = torch.long) != self.padding
        ).long()
        attention_mask_rejected = (
            torch.tensor(rejected_input_ids[:-1], dtype = torch.long) != self.padding
        ).long()

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
            "attention_mask_chosen": attention_mask_chosen,
            "attention_mask_rejected": attention_mask_rejected,
        }
    
    def generate_loss_mask(self, input_ids):
        # 上来先造一个跟输入序列 input_ids 一模一样长的列表，里面全塞满 0
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # min(..., self.max_length) 防止越界报错
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    # 把这个区间里的 0 全部翻转成 1
                    loss_mask[j] = 1

                # 既然已经处理完这一段 Assistant 的回复了，主指针 i 就可以直接“瞬移”到这句话的结尾, 继续扫码下一段
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

# RLAIFDataset —— 基于 AI 反馈的强化学习数据集（用于 PPO / GRPO）
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files = jsonl_path, split = "train")
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        messages = []
        answer = ""
        # 如果是偶数就是user发言，奇数的话就是assistant发言
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            # 将当前轮次的内容提取出来，带上身份标签，打包成标准字典格式存入列表
            messages.append({"role" : role, "content" : turn["content"]})
            # 不断覆盖更新 answer。当循环结束时，这里保留的就是最后一条（即最新的）回复
            answer = turn['content']
        # 添加标签
        # messages[:-1] 表示取 messages 列表中除了最后一条之外的所有消息，因为最后一条已经被作为 answer 剥离了
        # add_generation_prompt = True, 在字符串的末尾自动加上让助手开始发言的引导符
        # 这样模型接到这段 prompt 后就知道接下来该它生成回复了
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize = False,
            add_generation_prompt = True,
        )
        # 清理多余的空格/特殊字符
        prompt = post_processing_chat(prompt)
        return prompt, answer
    
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample["conversations"])
        return {"prompt": prompt, "answer" : answer}
    
if __name__ == "__main__":
    pass