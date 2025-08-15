# Pytorch

## 经典网络训练流程 

[地址]: https://github.com/yw5579358/pytorch/blob/main/study/%E7%BB%8F%E5%85%B8%E7%BD%91%E7%BB%9C%E6%9E%B6%E6%9E%84%E8%AE%AD%E7%BB%83%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB.ipynb	"study/经典网络架构训练图像分类.ipynb"

### 1.数据加载与预处理

- **数据目录**：`datasets.ImageFolder` 按文件夹类别加载。
- **数据增强（train）**：
  - `Resize` 统一尺寸
  - `RandomRotation` 随机旋转
  - `CenterCrop` 中心裁剪
  - `RandomHorizontalFlip` / `RandomVerticalFlip` 翻转
  - `ColorJitter` 调节亮度、对比度、饱和度
  - `RandomGrayscale` 灰度化
  - `ToTensor` 转张量并归一化到 `[0,1]`
  - `Normalize(mean, std)` 归一化
- **验证集（valid）**：只做 `Resize` + `ToTensor` + `Normalize`
- **DataLoader**：`DataLoader(dataset, batch_size, shuffle)`

### 2.模型构建

- **迁移学习**：
  - 载入预训练模型：`models.resnet34(pretrained=True)`
  - 冻结特征层参数：`for param in model.parameters(): param.requires_grad = False`
  - 替换全连接层：`model.fc = nn.Sequential(...)`
- **自定义网络**：用 `nn.Sequential` 或 `nn.Module` 定义

### 3. 损失函数 & 优化器

- **损失函数**：`nn.CrossEntropyLoss()`（多分类）
- **优化器**：`optim.Adam(model.parameters(), lr=...)`
- **学习率调整**：`optim.lr_scheduler.StepLR`

### 4. 训练流程

- 循环 `epoch`：
  1. 切换模式：`model.train()` / `model.eval()`
  2. 遍历 `DataLoader`：
     - 前向传播 `outputs = model(inputs)`
     - 计算损失 `loss = criterion(outputs, labels)`
     - 反向传播 `loss.backward()`
     - 更新权重 `optimizer.step()`
  3. 验证集评估：计算准确率、保存最佳模型。
- 使用 `torch.no_grad()` 禁用梯度计算（验证阶段）。

### 5. 模型保存 & 加载

- 保存：`torch.save(model.state_dict(), 'model.pth')`

- 加载：

  ```python
  model.load_state_dict(torch.load('model.pth'))
  model.eval()
  ```

### 6. 推理与可视化

- **单张图片预测**：`Image.open()` → `transform()` → `unsqueeze(0)` → `model(img)`
- **可视化**：`matplotlib` 显示图片与预测类别。

```
数据准备 → 数据增强 → DataLoader
       ↓
选择模型（预训练 / 自定义）
       ↓
冻结 or 解冻部分参数
       ↓
替换输出层适配任务
       ↓
定义损失函数 & 优化器
       ↓
训练循环（train + valid）
       ↓
保存最佳模型
       ↓
推理 & 可视化

```

```
速记
Data: ImageFolder + transforms
Model: resnet34 + freeze + replace_fc
Loss: CrossEntropy
Opt: Adam + StepLR
Train: forward → loss → backward → step
Eval: no_grad + acc
Save: state_dict
```

### 总结

**数据准备** → ImageFolder 按文件夹加载

**数据增强** → transforms 处理（train / valid 不同策略）

**DataLoader** → 批量加载

**选择模型** → 预训练 ResNet34

**冻结参数 + 替换 FC 层** → 适配任务

**定义损失函数** → CrossEntropyLoss

**优化器** → Adam + 学习率调度

**训练循环** → train + valid

**保存最佳模型** → state_dict

**推理预测** → no_grad 模式

## **PyTorch 核心知识点总结**

### 1. 基础

- **张量**（`torch.Tensor`）
  - 类似 NumPy 数组，但支持 GPU 运算 (`.to(device)` 或 `.cuda()` / `.cpu()` )
  - 创建方式：`torch.tensor`, `torch.zeros`, `torch.ones`, `torch.randn`
  - 数据类型：`float32`, `int64` 等，通过 `dtype` 指定
  - 自动求导属性：`requires_grad=True`
- **自动求导**（`torch.autograd`）
  - 动态计算图：每次前向传播都会构建新的计算图
  - `loss.backward()` 反向传播
  - `with torch.no_grad()` 在推理时关闭梯度计算

------

### 2. 数据处理

- **Dataset**

  - 自定义数据集需继承 `torch.utils.data.Dataset` 并实现：

    ```
    def __getitem__(self, idx): ...
    def __len__(self): ...
    ```

- **DataLoader**

  - 提供批量加载、打乱、并行读取
  - 常用参数：`batch_size`, `shuffle`, `num_workers`

- **数据增强**（`torchvision.transforms`）

  - `Resize`, `RandomCrop`, `RandomHorizontalFlip`, `Normalize`, `ToTensor`

------

### 3. 模型构建

- **两种方式**：
  1. `nn.Sequential`：简单层堆叠
  2. 自定义类继承 `nn.Module`：适合复杂网络
     - 需实现 `__init__` 和 `forward`
- **参数管理**：
  - `model.parameters()` 获取可训练参数
  - 冻结参数：`for p in model.parameters(): p.requires_grad = False`

------

### 4. 损失函数（`torch.nn`）

- 回归：`nn.MSELoss`, `nn.L1Loss`
- 分类：`nn.CrossEntropyLoss`（包含 softmax）
- 其他：`nn.BCELoss`, `nn.BCEWithLogitsLoss`

------

### 5. 优化器（`torch.optim`）

- 常见优化器：`SGD`, `Adam`, `RMSprop`

- 参数更新流程：

  ```
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  ```

- 学习率调度：

  - `StepLR`, `ReduceLROnPlateau`, `CosineAnnealingLR`

------

### 6. 训练流程

```
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        ...
    model.eval()
    with torch.no_grad():
        ...
```

------

### 7. 模型保存与加载

- 保存参数：

  ```
  torch.save(model.state_dict(), 'model.pth')
  ```

- 加载：

  ```
  model.load_state_dict(torch.load('model.pth'))
  model.eval()
  ```

------

### 8. GPU/CPU 切换

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```

------

## **二、PyTorch 高频面试题及答案**

### **2. `requires_grad` 有什么作用？**

- 控制张量是否参与**自动求导**计算
- 训练中需计算梯度的参数必须 `**requires_grad=True**`
- 冻结层时可设置 `requires_grad=False`，减少计算开销

### **3. `torch.no_grad()` 的作用**

**答**：

- 关闭梯度计算，减少内存和计算量
- 常用于推理阶段，不需要反向传播

### **4. `model.train()` 和 `model.eval()` 的区别**

**答**：

- `model.train()`：启用 dropout、batchnorm 的训练模式
- `model.eval()`：关闭 dropout、batchnorm 的随机性，使用推理模式

### **5. `CrossEntropyLoss` 是否需要 softmax？**

**答**：

- 不需要。`CrossEntropyLoss` 内部已包含 `log_softmax` 操作
- 输入应是 **未经过 softmax 的 logits**

### **6. 如何冻结预训练模型的参数？**

**答**：

```python
for param in model.parameters():
    param.requires_grad = False
```

然后替换输出层，让新层参与训练

### **7. DataLoader 中 `num_workers` 作用**

**答**：

- 控制加载数据的子进程数量
- 增大可以提高数据读取速度，但受 CPU 核心数和内存限制

### **8. `optimizer.zero_grad()` 为什么要放在 `loss.backward()` 前？**

**答**：

- PyTorch 梯度是累加的，不清零会导致梯度叠加
- 每次反向传播前需清空梯度

### **9. 如何保存整个模型而不是 state_dict？**

```python
torch.save(model, 'model.pth')
model = torch.load('model.pth')
```

但更推荐保存 `state_dict`，因为保存整个模型会绑定代码结构

**state_dict 更灵活、轻量，整个模型保存依赖代码结构**

### 1**0. 常见的 GPU 内存不足解决方法**

- 减小 batch_size
- 使用 `torch.cuda.empty_cache()`
- 半精度训练（Mixed Precision）
- 冻结部分层

## PyTorch 常用优化器 / 激活函数 / 损失函数 对比

#### 1. 优化器对比

| 优化器         | 公式特性             | 优点                 | 缺点                 | 常用场景          |
| -------------- | -------------------- | -------------------- | -------------------- | ----------------- |
| **SGD**        | 纯梯度下降           | 简单、稳定、可控     | 收敛慢，对学习率敏感 | 基础任务          |
| SGD + Momentum | 加动量项             | 抑制震荡，加快收敛   | 参数调节多           | 深度网络          |
| **Adam**       | 一阶动量 + 二阶动量  | 自适应学习率，收敛快 | 对泛化能力可能差     | NLP、CV           |
| RMSprop        | 指数加权平均二阶梯度 | 适合非平稳目标       | 不一定优于 Adam      | RNN 任务          |
| **AdamW**      | Adam + 权重衰减解耦  | 改进 Adam 泛化能力   | 较新，需要更多验证   | Transformer、BERT |

---

#### 2. 激活函数对比

| 激活函数       | 公式/范围                  | 优点                       | 缺点             | 适用场景     |
| -------------- | -------------------------- | -------------------------- | ---------------- | ------------ |
| **Sigmoid**    | (0,1)                      | 平滑，输出概率             | 梯度消失，计算慢 | 二分类输出层 |
| Tanh           | (-1,1)                     | 中心对称，收敛快于 Sigmoid | 梯度消失         | 隐藏层旧网络 |
| **ReLU**       | [0,∞)                      | 收敛快，计算简单           | 死亡 ReLU 问题   | CNN/MLP 主流 |
| **Leaky ReLU** | (-∞,∞)                     | 解决死亡 ReLU              | 参数需调         | 深层网络     |
| GELU           | 平滑版 ReLU（近似正态CDF） | 更平滑，表现优于 ReLU      | 计算稍慢         | Transformer  |
| **Softmax**    | (0,1)，且总和为1           | 多分类概率输出             | 对大数值敏感     | 多分类输出层 |

---

#### 3. 损失函数对比

| 损失函数             | 公式特性              | 优点                 | 缺点                         | 常用场景       |
| -------------------- | --------------------- | -------------------- | ---------------------------- | -------------- |
| **MSELoss**          | 均方误差              | 平滑、凸函数         | 对异常值敏感                 | **回归任务**   |
| L1Loss               | 绝对误差              | 对异常值不敏感       | 不可导点多                   | 稀疏回归       |
| **CrossEntropyLoss** | log_softmax + NLLLoss | 多分类常用，稳定     | 仅适用分类任务               | **多分类**     |
| BCEWithLogitsLoss    | Sigmoid + BCELoss     | 数值稳定，二分类常用 | 仅适用二分类                 | 二分类         |
| HingeLoss            | max(0, 1 - y·f(x))    | 适合 SVM 风格分类    | 不适合概率输出               | 大间隔分类     |
| KLDivLoss            | KL 散度               | 适合分布对齐任务     | 需要配合 softmax/log_softmax | 蒸馏、概率匹配 |