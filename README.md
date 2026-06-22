# Minimind 全流程解析

## layerNorm vs RMSNorm
---
| 对比维度 | LayerNorm (经典方案) | RMSNorm (现代大模型标配) |
| :--- | :--- | :--- |
| **核心思想** | 平移 + 缩放 (Mean-centering & Scaling) | **仅缩放** (Scaling only) |
| **数学公式** | $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | $RMS(x) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}$<br><br>$y = \frac{x}{RMS(x) + \epsilon} \cdot \gamma$ |
| **操作步骤** | 算均值 $\rightarrow$ 减均值 $\rightarrow$ 算方差 $\rightarrow$ 缩放 | 算平方和平均 $\rightarrow$ 开根号 $\rightarrow$ 缩放 |
| **偏置项 ($\beta$)** | 通常有 (整体平移特征) | **无** (保持原点不动) |
| **计算速度** | 相对较慢 (需多次内存读写和同步) | **快 10% ~ 40%** (省去了计算和减去均值的步骤) |
| **显存消耗** | 较高 (反向传播时需保存均值 $\mu$ 等中间态) | **极低** (只需保存 RMS 标量) |
| **最终准确率** | 行业基准线 | **几乎与 LayerNorm 完全一致** |
| **代表模型** | BERT, GPT-2, 传统的 Vision Transformer (ViT) | **LLaMA, Qwen, Mistral** |