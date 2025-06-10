# Informer: Long Sequence Time-Series Forecasting 论文笔记

## 📘 背景与研究动机

在现实场景中，许多应用场合（如电力负载、气象预测）需要 **长序列时间序列预测（LSTF）**。传统方法（如 ARIMA、Prophet、LSTM）在预测短序列时表现尚可，但在长序列下性能急剧下降，尤其存在推理速度慢、误差累积等问题。

Transformer架构因其全局建模能力被引入时间序列预测中，但也有以下问题：

- 时间复杂度 O(L2)O(L^2)O(L2)
- 高内存占用
- 解码器推理慢（step-by-step）

## 🧠 Informer的三大创新

1. **ProbSparse Self-Attention**：从所有 Query 中筛选出重要的一小部分（25个）参与 attention，大幅减少计算开销；
2. **Self-Attention Distilling**：使用 `Conv1D + MaxPool` 的方式对特征图降维，提升速度与压缩内存；
3. **Generative Decoder**：一次性生成所有预测值，避免传统Decoder的逐步预测误差传播问题。

![image-20250610145954266](img\image-20250610145954266.png)

------

## 🧱 模型架构概览

- **Encoder** 使用 ProbSparse Attention + Distilling 技术堆叠多层；
- **Decoder** 输入真实标签部分与0填充的预测部分，通过 Masked Attention 进行一次性预测。

------

## 🔍 模块详解

### 1. ProbSparse Attention

> Q太多了？我们只选最“有用”的Q！

- 对每一个Query，计算它与所有Key(抽样后的所有)的注意力分布，与**均匀分布的KL散度**衡量其稀疏程度；
- 只保留 KL 散度最大的 Top-u 个Query，形成一个稀疏矩阵参与 attention 计算；
- 时间复杂度从 O(L2)  O(L^2)  O(L2) 降至 O(Llog⁡L)   O(L\log L)   O(LlogL)。

![image-20250610150408887](img\image-20250610150408887.png)

![image-20250610150441502](img\image-20250610150441502.png)

### 2. Self-Attention Distilling

- 加入 Conv1D + MaxPooling 模块对特征进行降采样；
- 多层 Encoder 架构中，每层逐步压缩输入长度；
- 通过减少重复无效的信息，提升长序列建模效率。

![image-20250610150632799](img\image-20250610150632799.png)

### 3. Generative Decoder

- 输入：[真实标签段 + 0填充预测段]
- 使用 Masked Attention 保证不会“看未来”；
- 一步生成完整预测序列，推理更快，无需动态解码。

### 4.位置编码信息

​		位置编码包含了绝对位置，和时间相关的各种编码

------

## 📊 实验验证

- 数据集：ETTh1/2、ETTm1、ECL、Weather（单/多变量预测）
- 与 Reformer、LogSparse、LSTM、ARIMA、Prophet、DeepAR 等方法比较
- **Informer 在长序列预测准确率与速度上均显著优于其他模型**

------

## 🔩 敏感性分析与消融实验

- 输入长度越长 -> 长序列预测效果越好；
- Sampling Factor 设置为 5 效果最佳；
- 自注意力提取 + 生成式解码器组合提升明显；
- 没有Distilling模块的Informer在输入>720时出现OOM问题。

------

## 🧾 总结

Informer 在长序列预测问题中，通过引入 **稀疏注意力机制** 和 **高效编码解码结构**，在提升性能的同时也显著减少了资源消耗，为LSTF任务提供了强有力的解决方案。