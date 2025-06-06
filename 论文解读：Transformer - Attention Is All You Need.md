## 【大模型系列篇】论文解读：Transformer - Attention Is All You Need

 ![](https://i-blog.csdnimg.cn/blog_migrate/f910e63582c3c11f25ed5be81d7805ce.gif)

  _Attention Is All You Need (Transformer)_ 是当今大模型初学者必读的一篇论文，已经有不少业内大佬都翻译解读过这篇论文，此处仅作为自己学习的记录。该论文是由谷歌机器翻译团队于2017年发表在NIPS ，提出了一个只基于attention的结构来处理序列模型相关的问题，比如机器翻译。相比传统的CNN与RNN来作为Encoder-Decoder的模型，谷歌这个模型摒弃了固有的方式，并没有使用任何的CNN或者RNN的结构，该模型可以高度并行的工作，相比以前串行并且无法叠加多层、效率低的问题。那么Transorformer可以高度并行的工作，所以在提升翻译性能的同时训练速度也特别快。同时因为Attention的超强记忆能力_，_克服了RNN因为输入太长而导致的丢失信息、记忆不够精准的问题。

 ![](https://i-blog.csdnimg.cn/direct/9aadbfed11fe4743bff3236f18e21a6c.png)

 论文：Attention Is All You Need

 下载：https://arxiv.org/abs/1706.03762

 代码：https://github.com/tensorflow/tensor2tensor

### `摘要`

 现有的`Seq2Seq`模型主要基于复杂的循环或卷积神经网络，这些模型包括一个编码器和解码器，在一些表现好的模型中，还会在编码器和解码器之间引入注意力机制。我们提出了一种新的简单网络架构`Transformer`，该结构不再使用循环结构`（RNN）`和卷积结构`（CNN）`，而是完全依赖注意力机制运行。并且作者在一些典型的机器翻译任务上做了实验，实验结果显示基于 Transformer 结构的模型效果非常好。我们在两个机器翻译任务上进行实验，发现这种模型在质量上具有优势，`一是可以并行计算，提高模型计算效率；二是，相较于当时的模型可以有效减少训练时间。`

 在WMT 2014英德翻译任务中，Transformer模型实现了28.4的BLEU分数，比现有的最佳结果（包括集合模型）提高了2个BLEU分数。在WMT 2014英法翻译任务中，该模型在单一模型的BLEU得分上达到了41.8，树立了新的单模型性能标杆，且只用了8个GPU进行3.5天的训练，而文献中的最佳模型训练成本更高。我们还证明了Transformer在其他任务中的良好泛化性，例如在使用大数据和小数据集进行的英语构成解析任务中取得成功。

### `一、引言`

 `循环神经网络（RNN），尤其是长短期记忆（LSTM）和门控循环神经网络（GRU）`，已经被广泛应用于序列建模和转换问题，如语言建模和机器翻译。多个研究努力不断推动循环语言模型和编码器-解码器架构的发展。

 递归模型通常会将计算过程因子化为输入和输出序列的符号位置。将位置与计算时间步骤对齐，它们会根据前一个隐状态 ![h_{t-1}](https://latex.csdn.net/eq?h_%7Bt-1%7D)和位置 ![t](https://latex.csdn.net/eq?t)的输入生成隐状态序列![h_{t}](https://latex.csdn.net/eq?h_%7Bt%7D) 。这种隐含的顺序性使得在训练示例中无法实现并行化，而在较长的序列长度下这一点变得至关重要，因为内存限制限制了跨示例的批处理(batching across examples)。最近的研究通过因子化技巧(factorization tricks) [21]和条件计算(conditional computation ) [32]在计算效率方面取得了显著进展，同时在后者的情况下也改进了模型性能。然而，顺序计算的基本限制仍然存在。

 `注意力机制`已经成为引人注目的序列建模和转换模型的重要组成部分，适用于各种任务，可以模拟输入或输出序列中的依赖关系，而不考虑它们的距离[2，19]。然而，在除了一些特殊情况[27]之外，这种`注意机制`通常与`RNN`一起使用。在这项工作中，我们提出了`Transformer`，`这是一种模型架构，完全避免使用循环，并且完全依赖于注意力机制来在输入和输出之间建立全局依赖关系。` Transformer可以`实现更高程度的并行化`，并且在仅在8个P100 GPU上训练12小时后，能够达到最先进的翻译质量水平。

*    `RNN 无法完成并行计算`

*    ```
     RNN 无法解决长文本输入导致信息丢失的问题
        
    ```

 Transformer 架构是对于传统基于RNN的编码器-解码器架构的一次重大创新。

### `二、背景`

 减少顺序计算的目标也是扩展神经

```
GPU（Extended Neural GPU）
```

[16]、`ByteNet`[18]和`ConvS2S`[9]的基础，它们都使用卷积神经网络作为基本构建模块，对所有输入和输出位置并行计算隐状态表示。在这些模型中，将两个任意输入或输出位置之间的信号相关联所需的操作次数，随着位置之间的距离增加而增加，对于ConvS2S是线性增加，对于ByteNet是对数增加。这使得学习远距离位置之间的依赖关系更加困难[12]。在Transformer中，这种增长被减少为一定数量的操作，尽管通过平均注意力加权位置来降低了有效分辨率的代价，我们通过3.2节中描述的多头注意力来抵消这种影响。

 自注意力，有时也称为内部注意力，是一种注意力机制，用于关联单个序列的不同位置，以便计算序列的表示。自注意力已成功应用于各种任务，包括阅读理解、摘要生成、文本推理和学习与任务无关的句子表示[4, 27, 28, 22]。

 端到端记忆网络是基于循环注意力机制（

```
recurrent attention mechanism
```

）而不是基于序列对齐的循环（

```
sequence aligned recurrence
```

），并且已经显示出在简单语言问答和语言建模任务上表现良好[34]。

 然而，据我们所知，Transformer 是第一个完全依赖自注意力来计算其输入和输出表示的转换模型，而不使用序列对齐的循环神经网络或卷积神经网络。在接下来的几节中，我们将描述Transformer，解释自注意力的动机，并讨论它相对于诸如[17, 18]和[9]模型的优势。

### 三、模型架构

 大多数有竞争力的神经序列转换模型都采用编码器-解码器结构 [5,2,35] 。编码器将符号表示的输入序列 ![(x_1,...,x_n)](https://latex.csdn.net/eq?%28x_1%2C...%2Cx_n%29) 映射到连续的序列 ![z=(z_1,...,z_n)](https://latex.csdn.net/eq?z%3D%28z_1%2C...%2Cz_n%29) 。给定 z ，解码器随之生成一个符号输出序列 ![(y_1,...,y_m)](https://latex.csdn.net/eq?%28y_1%2C...%2Cy_m%29) ，一次生成一个元素。每一步中，模型都是自回归（auto-regressive）[10] 的，生成下一个元素时，将先前生成的符号用作附加输入。

 Transformer 遵循这个整体架构，对编码器和解码器使用多层堆叠的自注意力层，以及逐点（point-wise）的全连接层，分别如图 1 的左右两部分所示。

![图 1：Transformer 模型架构](https://i-blog.csdnimg.cn/direct/a7c7a704877b42c3b620deca35cc475f.png)

图 1：Transformer 模型架构

#### 3.1 编码器和解码器栈

 `编码器`：编码器由 N=6 个相同层组成的栈构成。每一层有两个子层，其一是多头自注意力（multi-head self-attention）机制，其二是简单的位置全连接前馈网络。我们的两个子层都采用残差连接（residual connection）[11]，随之进行层归一化（layer normalization）[1]。换言之，每个子层的输出为 LayerNorm(x+Sublayer(x)) ，其中 Sublayer(x) 是子层本身实现的函数。为方便残差连接，模型中所有子层及嵌入（embedding）层都生成![d_{model}=512](https://latex.csdn.net/eq?d_%7Bmodel%7D%3D512)维的输出。

 `解码器`：解码器也由 N=6 个相同层组成的栈构成。除了编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器栈的输出执行多头注意力。与编码器类似，我们对每个子层采用残差连接，随之进行层归一化。我们还修改了解码器栈中的自注意力子层，以防止以当前位置信息中被添加进后续的位置信息。这种掩码（mask）与偏移一个位置输出嵌入的相结合，保证位置 i 的预测只能依赖于位置小于 i 的已知输出。

#### 

#### 3.2 注意力

 注意力函数的作用是：将查询（query）和一组键值对（key-value pairs）映射到输出，其中 query、keys、values 和输出都是向量。输出是 values 的加权和，其中分配给每个 value 的权重是由 query 与相应 key 的兼容函数（compatibility function）计算。

![](https://i-blog.csdnimg.cn/direct/b30d829fc1574c1f86f851f1902a70ed.png)

图 2：（左）缩放点积注意力；（右）多头注意力由多个并行运行的注意力层组成。

##### 3.2.1 缩放点积注意力（Scaled Dot-Product Attention）

 我们的特别注意力机制称作缩放点积注意力（图 2 左）。输入由 dk 维的 queries 和 keys 以及 dv 维的 values 组成。我们使用计算 query 和所有 keys 的点积，随之除以![\sqrt{d_k}](https://latex.csdn.net/eq?%5Csqrt%7Bd_k%7D)，再应用 softmax 函数来获取 values 的权重。

 实际应用中，我们将一组 queries 转换成一个矩阵 Q ，同时应用注意力函数。keys 和 values 也同样被转换成矩阵 K 和 V 。按照如下方式计算输出矩阵：

 ![](https://i-blog.csdnimg.cn/direct/22249877523e4dbf9043ac3fa446b7de.png)

 两种最常用的注意力函数是加性注意力（additive attention）[2]和点积（乘法）注意力。点积注意力与我们的算法相同，只是没有![\frac{1}{\sqrt{{d_k}}}](https://latex.csdn.net/eq?%5Cfrac%7B1%7D%7B%5Csqrt%7B%7Bd_k%7D%7D%7D)的缩放因子。加性注意力使用有单个隐藏层的前馈网络来计算兼容函数。虽然二者在理论复杂性上相似，但在实践中点积注意力更快、更节省空间，因其可以使用高度优化的矩阵乘法代码来实现。

 对于较小的![d_k](https://latex.csdn.net/eq?d_k)值，两种机制的表现相似，但对于较大的![d_k](https://latex.csdn.net/eq?d_k)值，加性注意力优于点积注意力，且无需进行缩放[3]。我们认为，对于较大的![d_k](https://latex.csdn.net/eq?d_k)值，点积的数量级会变大，从而会将 softmax 函数推入梯度极小的区域 【注：为说明点积变大的原因，假设 q 和 k 的分量是独立随机变量，均值为 0 ，方差为 1 。那么它们点积![q\cdot k = \sum _{i=1}^{d^k}q_ik_i](https://latex.csdn.net/eq?q%5Ccdot%20k%20%3D%20%5Csum%20_%7Bi%3D1%7D%5E%7Bd%5Ek%7Dq_ik_i)的均值为 0 ，方差为 ![d_k](https://latex.csdn.net/eq?d_k)】。为了抵消这种影响，我们将点积缩放![\frac{1}{\sqrt{{d_k}}}](https://latex.csdn.net/eq?%5Cfrac%7B1%7D%7B%5Csqrt%7B%7Bd_k%7D%7D%7D)倍。

##### `

```
2 多头注意力（Multi-Head Attention）
```

`

 我们发现，与其使用 dmodel 维的 keys、values 和 queries 执行单个注意力函数，使用学习到的不同线性映射分 h 次将 queries、keys 和 values 线性投影到![d_k](https://latex.csdn.net/eq?d_k)、![d_k](https://latex.csdn.net/eq?d_k) 和 ![d_v](https://latex.csdn.net/eq?d_v) 维则更有裨益。随后，在 queries、keys 和 values 的每个投影上，我们并行地执行注意力函数，产生![d_v](https://latex.csdn.net/eq?d_v) 维输出值，其被连接起来再次进行投影，产生最终值，如图 2 所示。

 多头注意力使得模型同时关注来自不同位置的、不同表示子空间的信息。对于单一注意力头，均值运算反而会抑制之。

 ![](https://i-blog.csdnimg.cn/direct/b19a1268114c47c8a51c1fa82e1e3bd4.png)

 其中投影操作为参数矩阵![W_i^Q \epsilon \mathbb{R}^{d_{model}×d_k}](https://latex.csdn.net/eq?W_i%5EQ%20%5Cepsilon%20%5Cmathbb%7BR%7D%5E%7Bd_%7Bmodel%7D%D7d_k%7D)、![W_i^K \epsilon \mathbb{R}^{d_{model}×d_k}](https://latex.csdn.net/eq?W_i%5EK%20%5Cepsilon%20%5Cmathbb%7BR%7D%5E%7Bd_%7Bmodel%7D%D7d_k%7D)、![W_i^V \epsilon \mathbb{R}^{d_{model}×d_v}](https://latex.csdn.net/eq?W_i%5EV%20%5Cepsilon%20%5Cmathbb%7BR%7D%5E%7Bd_%7Bmodel%7D%D7d_v%7D)和![W^O \epsilon \mathbb{R}^{hd_v × d_{model}}](https://latex.csdn.net/eq?W%5EO%20%5Cepsilon%20%5Cmathbb%7BR%7D%5E%7Bhd_v%20%D7%20d_%7Bmodel%7D%7D)。

 在这项工作中，我们采用 h=8 个并行注意力层或头（head）。每个头都采用![d_k=d_v=d_{model}/h=64](https://latex.csdn.net/eq?d_k%3Dd_v%3Dd_%7Bmodel%7D/h%3D64)。由于每个头的维度减少，总计算成本与全维度的单头注意力相似。

##### `3.2.3 注意力在我们模型中的应用`

 多头注意力在 Transformer 中有三种不同的使用方式：

*    在编码器-解码器注意力（encoder-decoder attention）层中，queries 来自先前的解码器层，而 keys 和 values 来自编码器的输出。这使得解码器中的每个位置都可关联到输入序列中的所有位置。这是在模仿序列到序列（seq2seq）模型中典型的编码器-解码器注意机制，例如[38,2,9]。

*    编码器包含了自注意力层。在自注意力层中，所有 keys、values 和 queries 都来自同一位置，在本例中是编码器中前一层的输出。编码器中的每个位置可以关注到编码器上一层中的所有位置。

*    类似地，解码器中也包含自注意力层，这使得解码器中的每个位置都关注到解码器之前的所有位置（并包括当前位置）。为了保持解码器的自回归特性，需要防止解码器中的信息向左流动。在缩放点积注意力的内部，我们通过屏蔽（设置为−∞）softmax 输入中所有非法连接对应的值，从而实现了这一点。见图 2。

#### 

#### 3.3 逐位置的前馈神经网络

 除了注意力子层之外，我们编码器与解码器中的每个层中都包含一个全连接前馈网络，该网络单独且相同地应用于每个位置。其由两个线性变换组成，中间有一个 ReLU 激活。

 ![FFN(x)=max(0,xW_1+b_1)W_2+b_2](https://latex.csdn.net/eq?FFN%28x%29%3Dmax%280%2CxW_1&plus;b_1%29W_2&plus;b_2)

 虽然不同位置的线性变换是相同的，但它们在层与层之间采用不同的参数。另一种描述方式是两个核大小为 1 的卷积。输入和输出的维度为![d_{model}=512](https://latex.csdn.net/eq?d_%7Bmodel%7D%3D512)，内层的维度为![d_{ff}=2048](https://latex.csdn.net/eq?d_%7Bff%7D%3D2048)。

#### 

#### 3.4 词嵌入和 softmax

 与其他序列转换模型类似，我们使用学习到的嵌入将输入和输出 tokens 转换为维度 ![d_{model}](https://latex.csdn.net/eq?d_%7Bmodel%7D)的向量。我们还使用常用的线性变换与 softmax 函数，将解码器输出转换为预测下一个 token 的概率。在我们的模型中，两个嵌入层和 pre-softmax 线性变换之间共享相同的权重矩阵，类似于 [30]。在嵌入层中，我们将这些权重乘以![\sqrt{d_{model}}](https://latex.csdn.net/eq?%5Csqrt%7Bd_%7Bmodel%7D%7D)。

#### 3.5 位置编码

 由于我们的模型不包含循环和卷积，所以为了使模型能够利用序列的顺序信息，我们必须注入一些有关序列中 tokens 的相对或绝对位置的信息。为此，我们将位置编码（positional encodings）添加到编码器和解码器栈底部的输入嵌入中。位置编码与嵌入具有相同的 ![d_{model}](https://latex.csdn.net/eq?d_%7Bmodel%7D) 维度，因此可以将两者相加。位置编码有多种选择，既可以学习得到，也可以将其固定[9]。

 在本工作中，我们使用不同频率的正弦和余弦函数：

 ![PE_{(pos,2i)}=sin(\frac{pos}{10000^{2i/d_{model}}})](https://latex.csdn.net/eq?PE_%7B%28pos%2C2i%29%7D%3Dsin%28%5Cfrac%7Bpos%7D%7B10000%5E%7B2i/d_%7Bmodel%7D%7D%7D%29)

 ![PE_{(pos,2i+1)}=cos(\frac{pos}{10000^{2i/d_{model}}})](https://latex.csdn.net/eq?PE_%7B%28pos%2C2i&plus;1%29%7D%3Dcos%28%5Cfrac%7Bpos%7D%7B10000%5E%7B2i/d_%7Bmodel%7D%7D%7D%29)

 其中 pos 是位置， i 是维度。换言之，位置编码的每个维度都对应于一个正弦曲线。波长呈从 2π 到 10000⋅2π 的几何级数。之所以选择此函数，是因为我们假设它可以让模型很容易地关注相对位置进行学习，因为对于任何固定偏移 k ，![PE_{pos+k}](https://latex.csdn.net/eq?PE_%7Bpos&plus;k%7D) 可以表示为![PE_{pos}](https://latex.csdn.net/eq?PE_%7Bpos%7D)的线性函数。

 我们还尝试使用可学习的位置嵌入 [9]，发现这两种方法结果几乎相同（参见表 3 第 (E) 行）。我们选择正弦函数，因其可以令模型推断出的序列长度比训练期间遇到的序列更长。

### 

### 四、为什么使用自注意力

 在本节中，我们将从各个方面将自注意力层与循环层和卷积层进行比较，这些层都通常用于将用符号表示的一个可变长度序列![(x_1,...,x_n)](https://latex.csdn.net/eq?%28x_1%2C...%2Cx_n%29)映射到另一个等长序列![(z_1,...,z_n)](https://latex.csdn.net/eq?%28z_1%2C...%2Cz_n%29)，其中 ![x_i,z_i\epsilon \mathbb{R}^d](https://latex.csdn.net/eq?x_i%2Cz_i%5Cepsilon%20%5Cmathbb%7BR%7D%5Ed)，比如用于典型的序列转换编码器或解码器中的隐藏层。主要有三个方面促使我们使用自注意力。

 其一，是关于每层的总计算复杂度。其二是可以并行化的计算量，以所需的最小顺序操作数来衡量。

 其三，是网络中长距离依赖之间的路径长度。长距离依赖（long-range dependencies）的学习是许多序列转换任务中的一个关键挑战。有一个关键因素会对这种依赖性的学习能力产生影响：前向和后向信号在网络中必须经过的路径的长度。输入和输出序列中的任意位置组合之间的路径越短，学习长距离依赖关系就越容易 [12]。因此，我们还比较了由不同层类型组成的网络中、任意两个输入和输出位置之间的最大路径长度。

![](https://i-blog.csdnimg.cn/direct/2ed006dcc52d43a89a2987c8e4f6a376.png)

表 1：不同层类型的最大路径长度、每层复杂度和最小顺序操作数。  
n 是序列长度，d 是表示维度，k 为卷积核大小，r 是受限自注意力中邻域的大小。

 如表 1 中所示，自注意力层将所有位置与常数个顺序执行操作相连，而循环层需要 O(n) 次顺序操作。就计算复杂度而言，当序列长度 n 小于表示维度 d 时，自注意力层比循环层更快，使用 sota 的机器翻译模型表示句子（sentence representations）时，这是常见情况，例如 word-piece [38] 和 byte-pair [31] 表示。

 对于涉及很长序列的任务，为了提高计算性能，可以对自注意力进行限制，仅考虑输入序列中以相应输出位置为中心的大小为 r 的邻域。这会将最大路径长度增加到 O(n/r) 。我们计划在未来的工作中对该方法进一步研究。

 核宽度为 k<n 的单个卷积层不会连接所有输入和输出位置对。要实现这点，需要在卷积核连续（contiguous kernels）的情况下堆叠 O(n/k) 个卷积层，或者在空洞卷积（dilated convolutions）[18] 的情况下需要 O(logk(n)) ，这增加了网络中任意两个位置之间的最长路径的长度。卷积层通常比循环层开销贵 k 倍。然而，可分离卷积（Separable convolutions）[6] 大大降低了复杂性，可至 O(k⋅n⋅d+n⋅d2) 。然而，即使 k=n ，可分离卷积的复杂性也等于自注意力层和逐点前馈层的组合，即我们模型采用的方法。

 一个附加好处是，自注意力可以产生更多可解释的模型。我们可以对模型中的注意力分布进行检查，相关展示和讨论例子见附录。单个注意力头不仅可以清楚地学习并执行不同的任务，而且多个注意力头似乎表现出与句子的句法和语义结构相关的行为。

### 五、训练

#### 5.1 训练数据和批处理

 我们使用标准 WMT 2014 英语-德语数据集进行训练，该数据集包含约 450 万对句子。句子编码采用 byte-pair 编码（byte-pair encoding）[3] ，源语句和目标语句共享约 37000 个 tokens 的词汇表。对于英语-法语翻译，我们使用了更大的 WMT 2014 英语-法语数据集，其由 3600 万个句子组成，并将 tokens 拆分为 32000 个 word-piece 词汇表 [38] 。序列长度大体相近的句子分入同一批。每个训练批次（batch）包含一组句子对，其中包含大约 25000 个源 tokens 和 25000 个目标 tokens。

#### 5.2 硬件和时间调度

 我们在一台配备 8 个 NVIDIA P100 GPU 的机器上训练模型。我们使用论文中描述的超参数作为基础模型（base model），每个训练步骤大约需要 0.4 秒。我们对基础模型进行了总计 100,000 步即 12 小时的训练。对于大模型（big model）（见表 3 的最下列），单步时间为 1.0 秒。大模型进行了 300,000 步（3.5 天）的训练。

#### 5.3 优化器

 我们使用 Adam 优化器[20] ，参数为 ![\beta_1=0.9](https://latex.csdn.net/eq?%5Cbeta_1%3D0.9) 、 ![\beta_2=0.98](https://latex.csdn.net/eq?%5Cbeta_2%3D0.98) 、![\varepsilon = 10^{-9}](https://latex.csdn.net/eq?%5Cvarepsilon%20%3D%2010%5E%7B-9%7D)。在训练过程中，我们根据下述公式改变学习率：

 ![lrate=d_{model}^{-0.5}\cdot min(step\_num^{-0.5},step\_num\cdot warmup\_steps^{-1.5})](https://latex.csdn.net/eq?lrate%3Dd_%7Bmodel%7D%5E%7B-0.5%7D%5Ccdot%20min%28step%5C_num%5E%7B-0.5%7D%2Cstep%5C_num%5Ccdot%20warmup%5C_steps%5E%7B-1.5%7D%29)

 这对应于第一次 warmup_steps 训练步骤中线性地增加学习速率，随之将其与步骤数的平方根成比例地减小。我们使用 warmup_steps=4000 。

#### 5.4 正则化

 训练期间，我们使用了三种类型的正则化（Regularization）：

 残差丢弃（Residual Dropout）：我们将 dropout[33]应用于每个子层的输出，随即将其加到子层输入，并进行归一化。此外，在编码器和解码器栈中，我们将 dropout 应用于嵌入和位置编码的加和。对于基础模型，比例为![P_{drop}=0.1](https://latex.csdn.net/eq?P_%7Bdrop%7D%3D0.1)。

 标签平滑（Label Smoothing）：在训练期间，我们采用了值为![\varepsilon _{ls}=0.1](https://latex.csdn.net/eq?%5Cvarepsilon%20_%7Bls%7D%3D0.1)[36]的标签平滑。这会影响模型的困惑度（perplexity），因为模型会变得更加不确定，但会提高准确性和 BLEU 分数。

### 六、结果

#### 6.1 机器翻译

 在 WMT 2014 英德翻译任务中，大 Transformer 模型（表 2 中的 Transformer (big)）比之前报道的最佳模型（包括集成）的性能高出超过 2.0 的 BLEU 分数 ，达到了新的 sota 的 BLEU 分数 28.4。该模型的配置列在 Table 3 的最后一行中。模型在 8 个 P100 GPU 上进行训练了 3.5 天。甚至我们的基础模型也超越了之前发布的所有模型和集成模型（ensembles），而训练成本只是这些竞争模型的一小部分。在 WMT 2014 英法翻译任务中，我们的大模型获得了 41.0 的 BLEU 分数，优于之前发布的所有单个模型，且训练成本不到先前 sota 模型的 1/4 。针对英译法训练的（大） Transformer 模型使用 dropout 比例为![P_{drop}=0.1](https://latex.csdn.net/eq?P_%7Bdrop%7D%3D0.1)，而非 0.3。

 对于基础模型，我们使用的单个模型来自最后 5 个 checkpoints 的均值，这些 checkpoints 每十分钟保存一次。对于大模型，我们对最后 20 个 checkpoints 进行了平均。我们使用了束搜索（beam search），束宽（beam size）为 4，长度惩罚 α=0.6[38]。这些超参数是在开发集（development set）上进行实验后选择的。推理期间的最大输出长度设为输入长度+50，但尽可能提前终止[38]。

 表 2 对我们的结果进行了总结，并就的翻译质量和训练成本将我们的模型与文献中其他模型架构进行了比较。我们将训练时间、使用的 GPU 数量以及每个 GPU 持续单精度浮点能力的估计相乘，用来估计用于训练模型的浮点运算数量。

 【注：对于 K80、K40、M40 和 P100 ，我们使用的 TFLOPS 值为分别 2.8、3.7、6.0 和 9.5 】

![](https://i-blog.csdnimg.cn/direct/83a7e4ca6717459c9129b93c5e17b42d.png)

表 2：在英译德和英译法 newstest2014 测试中，Transformer 比之前最先进的模型取得了更好的 BLEU 分数，而训练成本只是其一小部分。

#### 6.2 模型变体

 为了估计 Transformer 不同组件的重要性，我们以不同的方式对基本模型进行了修改，并观测了开发集 newstest2013 上英译德性能的变化。我们使用了上一节中描述的束搜索，但没有平均 checkpoints。这些结果见表 3。

 在表 3 的 (A) 行中，我们改变了注意力头的数量以及注意力 keys 和 values 维度，但保持计算量不变，如第 3.2.2 节所述。单头注意力比最佳设置差 0.9 BLEU，但头数过多质量也会下降。

 在表 3 的 (B) 行中，我们观察到：减少注意力 keys 的![d_k](https://latex.csdn.net/eq?d_k)大小会影响模型质量。这表明确定兼容性并不容易，并且比点积更复杂的兼容函数可能是有益的。我们在 (C) 和 (D) 行中进一步观察到，一如预期，模型越大越好，并且 dropout 对于避免过拟合非常有帮助。在 (E) 行中，我们用可学习的位置嵌入替换正弦位置编码[9]，并观察到其结果与基本模型几乎相同。

![](https://i-blog.csdnimg.cn/direct/4e3c9e58be6e4340b686fdb8d76cfa97.png)

表 3：Transformer 架构的变体。未列出的值与基础模型的值相同。所有指标均来自英译德开发集 newstest2013。  
根据我们的 byte-pair 编码，列出的困惑度是每个单词的困惑度，不应与每个单词的困惑度进行比较。

#### 6.3 英语成分句法分析

 为了评估 Transformer 是否可以泛化到其他任务，我们对英语成分句法分析（Constituency Parsing）任务进行了实验。该任务有特殊挑战：输出受很强的结构约束，并且明显长于输入。此外，RNN 序列到序列模型尚未能够在小数据情况下获得最 sota 的结果[37]。

 我们在 Penn Treebank 数据集[25]的《华尔街日报（Wall Street Journal, WSJ）》部分训练了一个![d_{model}=1024](https://latex.csdn.net/eq?d_%7Bmodel%7D%3D1024)的 4 层 Transformer，大约 40K 训练句子。我们还在半监督环境中对其进行了训练，使用了更大的高置信度（high-confidence）以及 BerkleyParser 语料库，其中包含大约 1700 万个句子 [37]。我们在仅用 WSJ 的设置下使用了 16K 个 tokens 的词汇表，在半监督设置下使用了 32K 个 tokens 的词汇表。

 我们只在开发集的Section 22 上进行了少量的实验来选择 dropout、注意力和残差（第 5.4 节）、学习率和束宽，所有其他参数与英译德的基础模型保持相同。在推理过程中，我们将最大输出长度增加到输入长度+300。对于仅 WSJ 和半监督设置，我们使用的束宽为 21， α=0.3 。

 表 4 中的结果表明，尽管缺乏针对特定任务的调整，但我们的模型表现非常好，其结果比之前报告的所有模型都要好，仅有 循环神经网络语法（Recurrent Neural Network Grammar）[8]除外。

 与 RNN 序列到序列模型[37]相比，即使仅在 WSJ 40K 句子训练集上进行训练，Transformer 的性能也优于 BerkeleyParser[29]。

![](https://i-blog.csdnimg.cn/direct/b87cf08a31d84d819961dba83ff87e07.png)

表4：Transformer英语成分句法分析对比 (Results are on Section 23 of WSJ)

### `七、结论`

 这项工作中，我们提出了 Transformer，这是首个完全基于注意力的序列转换模型，用多头自注意力取代了编码器-解码器架构中最常用的循环层。对于翻译任务，Transformer 的训练速度明显快于基于循环层或卷积层的架构。在 WMT 2014 英移德和 WMT 2014 英译法任务中，我们都达到了新的 sota 水平。在前一项任务中，我们最好的模型甚至优于所有先前报告的集成模型。

 我们对基于注意力的模型之未来感到格外兴奋，并计划着手将其应用于其他任务。我们计划将 Transformer 扩展到文本以外的输入和输出模式的问题，并研究局部的、受限的注意力机制，以有效地处理图像、音频和视频等大型输入和输出。让生成具有更少的顺序性，是我们的另一个研究目标。

  ```
  文中标注的文献请查阅原论文的References

  ```

### `总结`

 传统的序列转录模型（你可以理解为讲一个序列转换为另一个序列的 Seq2Seq模型）都是基于Encoder-Decoder架构。Encoder也叫编码器，Decoder也叫解码器。Transformer 架构的整体仍然是编码器和解码器结构，但是和之前不同的是，编码器和解码器都采用了`注意力层+全连接层的结构。`

 ![](https://i-blog.csdnimg.cn/direct/9082ab9c8c6a47c9ace2085b265059ea.png)

 `编码器`

 编码器由6个相同的层组成，上图左侧的N=6。

 每个层有两个子层组成：第一个子层叫做多头注意力层，第二个子层是一个非常简单的、逐点运算的全连接前馈神经网络层。

 在两个子层中，都使用了残差结构，将输入和输出进行相连（你可以看到左侧的残差相连结构），每个子层残差相连（使用Add运算将输入和子层的输出相加）后，都会通过一个 LayerNorm层，因此，每一个子层都可以表达为: ![LayerNorm(x + Sublayer(x))](https://latex.csdn.net/eq?LayerNorm%28x%20&plus;%20Sublayer%28x%29%29)的形式。因此，对于第一个子层，你可以写成![LayerNorm(x + MHA(x))](https://latex.csdn.net/eq?LayerNorm%28x%20&plus;%20MHA%28x%29%29), MHA 代表多头注意力层，对于第二个子层，你可以写成![LayerNorm(x + FF(x))](https://latex.csdn.net/eq?LayerNorm%28x%20&plus;%20FF%28x%29%29)，FF代表前馈全连接层。

 在这里，我们可以将两个子层组成的神经网络块叫做 ``Transformer Block```，也就是` ``Transformer`` `块，和` ``Resnet`` `结构中的残差块类似`。

 在每一个block中，每个子层中都有残差连接，而残差连接（加法）要求两个输入的数据维度相同，因此作者在本段的最后一句说到，为了满足残差连接的需求，也为了使模型更加简单，所有子层输入的维度（包括Embedding层，因为Embedding的输出是MHA层的输入）和所有子层输出的维度，都是512。也就是说作者为了简化模型运算，将每一个block输入和输出的维度都做了恒等限定。

 `解码器`

 解码器也是由6个完全相等的block组成，区别在于，解码器中每个block有3个子层。

 其中，后面的两个子层和编码器完全相同，而第一个子层，也就是多出来的子层也是一个多头注意力层，唯一不一样的是该子层的多头注意力层中间的 softmax 函数计算方式不一样，因此作者把这一层叫做 Masked-MHA，带掩码的多头注意力层。

 `为什么要多出一层带掩码的注意力呢？`

 解码器是负责输出目标句子的。假设将一个中文句子翻译为英文，解码器在输出某个单词时（假设该单词在目标句子的第 i 的位置），此时计算注意力时应该只关注前 i-1 个单词，而不应该关注到 i 之后的单词。这是因为解码器是输出结果，只能已经输出的内容（过去）对当前输出的单词有影响，i 之后的单词的（未来）不应该对当前的输出产生任何影响。因此作者在其中加了掩码，就是将 i 之后的注意力全部置为零，使其不对当前时刻产生影响。当然这么做也是为了保持训练和推理时的一致性，从而使得推理更加准确。

 ``注意力机制``

 按照论文中这段的描述，注意力机制可以将 query 和 key-value 对映射到输出上，这里的 query、key、value和输出都是向量。

 首先模型处理的都是向量，这个向量可以理解为是某个单词（token）的词嵌入表示。输出是由 value 的加权和得到的，而加权里面的权重，则是通过计算 query 和 key 的相似度得到。计算query 和 key 之间的相似度可以有非常多的相似函数完成，不同的函数计算的结果肯定不一样，但是作用是相似的。

 作者把Transformer 中使用的注意力叫做`

```
Scaled Dot-Product Attention
```

`，翻译过来就是`带缩放`的点乘注意力。点乘很好理解，实际上就是内积运算。在学数学的内积运算时，你可能还记得，两个向量的内积代表了两个向量的相似程度。内积越大，说明两个向量越相似，如果两个向量正交（二维平面垂直）则说明两个向量相关性为零。

 那么`带缩放`是什么意思呢？其实很简单，作者在原本点乘运算的基础上，除以了![\sqrt{d_k}](https://latex.csdn.net/eq?%5Csqrt%7Bd_k%7D)，然后将除之后的结果，输入给softmax计算权重。

 ![](https://i-blog.csdnimg.cn/direct/89f19aac0bb1473daf48c421b404dc98.png)

 Q和K的转置直接相乘，结果是两者的相似度，当然你可以可以把它叫做两者的注意力分数，然后将结果除以![\sqrt{d_k}](https://latex.csdn.net/eq?%5Csqrt%7Bd_k%7D)后经过softmax运算，得到的是归一化后的注意力分数，此时的注意力分数便是权重，和V矩阵进行相乘。下面用一个示意图来表示注意力机制的计算过程。

 ![](https://i-blog.csdnimg.cn/direct/ef489399b2c34e79a1aea0dc33ed06a6.png)

### 

### ``参考文献``

 Attention Is All You Need

$(function() { setTimeout(function () { var mathcodeList = document.querySelectorAll('.htmledit_views img.mathcode'); if (mathcodeList.length  0) { for (let i = 0; i < mathcodeList.length; i++) { if (mathcodeList[i].complete) { if (mathcodeList[i].naturalWidth === 0 || mathcodeList[i].naturalHeight === 0) { var alt = mathcodeList[i].alt; alt = '\\\\(' + alt + '\\\\)'; var curSpan = $('<span class="img-codecogs"</span'); curSpan.text(alt); $(mathcodeList[i]).before(curSpan); $(mathcodeList[i]).remove(); } } else { mathcodeList[i].onerror = function() { var alt = mathcodeList[i].alt; alt = '\\\\(' + alt + '\\\\)'; var curSpan = $('<span class="img-codecogs"</span'); curSpan.text(alt); $(mathcodeList[i]).before(curSpan); $(mathcodeList[i]).remove(); }; } } MathJax.Hub.Queue(["Typeset",MathJax.Hub]); } }, 500) });