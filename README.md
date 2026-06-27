# Minimind 个人手撕

原项目地址：https://github.com/jingyaogong/minimind

### 需要安装的包：
* **Python**: 3.12
* **Pytorch**: 2.9.1
* **CUDA**: 12.8
* **datasets**: `pip install datasets`
* **swanlab**: `pip install swanlab`

配置：4090 24GB

## 数据准备 (Data Preparation)

本项目预训练阶段需要使用高质量的中文语料库。

### 1. 数据下载
请从以下地址下载预训练数据集 `pretrain_hq.jsonl`：

- [点击下载：Minimind 预训练数据集 (Hugging Face)](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/6b952cc50427c84eac543d0b38a8066208433847/pretrain_hq.jsonl)

### 2. 存放路径
下载完成后，请将该文件移动至项目根目录下的 `dataset/` 文件夹中，确保路径如下：
`dataset/pretrain_hq.jsonl`

---

## 模型架构 (Model Architecture)

### 基础版架构
![Minimind LLM Structure](image/Minimind-LLM-structure.jpg)

### MoE (混合专家) 版架构
![Minimind LLM MoE Structure](image/Minimind-LLM-structure-moe.jpg)


