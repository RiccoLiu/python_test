

# 深度学习

## 卷积

网络： Conv2d(B, 32, 96, 480) -> BatchNorm2d -> ReLU6 -> MaxPool2d

数据：(B,3,192,960) -> (B, 32, 96, 480) -> (B, 32, 96, 480) -> (B, 32, 96, 480) -> (B, 32, 48, 240)

```
self.inplanes = res_dims[0] # 32

卷积层: 取空间特征，同时压缩数据尺寸（步长 2 实现下采样）

参数：
    in_channels=3:  
        输入图像的通道数 3

    out_channels=self.inplanes
        卷积层输出的通道数 32 

    kernel_size=5：
        卷积核尺寸 5x5
    
    stride=2:
        卷积核移动步长 2 (图像尺寸减少1半)

    padding=2:
        输入图像的四周填充 2 圈像素

    bias=False：
        禁用卷积层偏置项（下一层归一化层已有偏置参数，两者同时存在冗余）

作用：
    提取空间特征，同时压缩数据尺寸        
```
self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)

```
归一化层: 对每个通道的激活值进行归一化处理

参数：
    self.inplanes: 
        归一化通道数量

数学原理：
    对每个通道独立计算一批数据的均值（μ）和方差（σ²），进行标准化：
        x̂ = (x - μ) / √(σ² + ε)

    再通过可学习的缩放（γ）和偏移（β）参数调整：
        y = γ * x̂ + β

作用:
    1. 加速训练收敛：缓解梯度消失/爆炸。
    2. 提供正则化效果：减少对 Dropout 的依赖。
    3. 稳定训练：对参数初始化和学习率不敏感。
```
self.bn1 = nn.BatchNorm2d(self.inplanes)

```
激活函数：将负值置零，正值保留(最大输出值为6), ReLU 的变种，数学公式：ReLU6(x) = min(max(0, x), 6)

参数：
    inplace=True:
        直接修改输入数据，不创建张量

作用：
    在低精度计算（如移动端）中防止数值溢出，提升量化鲁棒性（常用 MobileNet 系列）。
```
self.relu = nn.ReLU6(inplace=True)

```
最大池化层: 下采样
    
参数：
    kernel_size=2:
        池化窗口大小为:2x2, 在2x2区域内取最大值作为代表

    stride=2:
        池化窗口移动步长：2, (图像尺寸减少1半)

    padding=0
        不进行填充

作用：
    1.空间下采样，减少计算量
    2.特征增强，突出局部区域的最强特征
    3.平移不变性，降低微小平移的影响，增强鲁棒性
    4.感受野扩展，下一层神经元可以看到更大的区域
```
self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


# `torch.sigmoid` vs `F.softmax` 对比表

| **特性**               | **`torch.sigmoid`**                                | **`F.softmax`**                                    |
|------------------------|---------------------------------------------------|---------------------------------------------------|
| **数学定义**           | $\sigma(x) = \frac{1}{1+e^{-x}}$                   | $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$ |
| **函数类型**           | 元素级（逐元素计算）                              | 集合级（整个输入向量）                            |
| **输出范围**           | $[0, 1]$                                          | $[0, 1]$                                          |
| **输出总和约束**       | 无约束（各元素独立）                              | 固定为 1（概率分布）                              |
| **元素间关系**         | 相互独立                                          | 相互竞争（相对大小决定概率）                      |
| **核心功能**           | 将实数映射到概率空间                              | 将实数向量转换为概率分布                          |
| **梯度特性**           | 输入值接近 ±∞ 时梯度趋于 0（梯度消失风险）       | 当某个 logit 远大于其他时梯度不稳定               |
| **温度参数支持**       | ❌ 不支持                                          | ✅ 支持（通过 `logits/temperature`）              |
| **PyTorch 模块**       | `torch.sigmoid`                                   | `torch.nn.functional.softmax`                     |
| **主要用途**           | 1. 二分类输出层<br>2. 多标签分类<br>3. 置信度分数 | 1. 单标签多分类<br>2. 注意力权重<br>3. 概率分布   |
| **输入维度要求**       | 任意维度张量                                      | 需要指定 `dim` 参数操作的维度                     |
| **反向传播稳定性**     | 中等                                              | 中等（需数值稳定实现）                            |
| **计算效率**           | 高（简单计算）                                    | 较低（需要指数计算和求和）                        |
| **输出示例**           | `[0.8808, 0.7311, 0.2689]`（总和可能 ≠ 1）       | `[0.8438, 0.1142, 0.0420]`（总和 = 1）            |
| **替代函数**           | `torch.nn.Sigmoid()`                              | `torch.nn.Softmax(dim)`                           |
| **数值稳定实现**       | 通常不需要                                        | 建议使用 `log_softmax` 或稳定实现                 |

## Transfomer 网络

### Positon embedding

```
# 调用方式：
# PositionEmbeddingSine（16, normalize=true）
# pos    = self.position_embedding(p, pmasks) # p:(B, 256, 6, 30), mask:(B, 6, 30)

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 128 # 16
        self.temperature = temperature  # 10000
        self.normalize = normalize   # True

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 2pi

    # x:(B, 256, 6, 30), mask:(B, 6, 30)
    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask  # image 0 -> 0

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        '''
            y_embed = [
                        [0,0,0],
                        [1,1,1],
                        [2,2,2]
                      ]
            x_embed = [
                        [0,1,2], ...
                        [0,1,2], ...
                        [0,1,2]  ...
                    ]
        '''

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        '''
            广播机制扩张维度：
                x_embed[:, :, :, None]
                    x_embed：(B, 6, 30) => (B, 6, 30, 1)
                
                dim_t:
                    (16,)
                
                pos_x: 
                    (B, 6, 30, 16)
        '''

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) 
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        '''
            偶数下标用sin, 奇数下标用cos, 堆叠到4维后，在3维展开

            偶数索引：
               (B,6,30,8)
            
            奇数索引：
               (B,6,30,8)

            堆叠:
                (B,6,30,8,2)

            展开
                (B,6,30,16)
        '''

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)      # pos: (B, 32, 6, 30)
        return pos
```

### TransfomerEncoder

Transformer Encoder Layer 包含两个部分：多头自注意力（Self-Attention）机制和前馈神经网络（FeedForward），还加了两次残差连接（residual）和 LayerNorm。

逻辑流程:

```
+-------------------------+
|        Input src       |
+-------------------------+
            |
        (加上 pos 编码)
            ↓
+--------------------------+
|   MultiHead Attention    |
+--------------------------+
            ↓
    + Residual + Dropout
            ↓
        LayerNorm
            ↓
+--------------------------------+
|     FeedForward (Linear -> ReLU) |
+--------------------------------+
            ↓
    + Residual + Dropout
            ↓
        LayerNorm
            ↓
    Final output + weights

```

代码流程：

```

# 32, 2, 4x*64, 0.1, 'relu',  False
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    '''
        src: (180, B, 256)
        src_key_padding_mask: (B, 180)
        pos: (180, B, 32)
    '''
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        '''
            作用：将位置编码 pos 加到输入 src 上，作为 self-attention 的 Query 和 Key。
            直觉：告诉模型每个 token 处于哪个位置（因为注意力不懂顺序）
            为什么 q = k：因为是自注意力（自己注意自己），Q 和 K 是一样的东西。
        '''         
        q = k = self.with_pos_embed(src, pos)

        '''
            作用：计算多头注意力输出 src2，同时得到注意力权重 weights。
            直觉：对每个位置，去“注意”整个序列的其他位置，提取有用的信息，融合上下文；

            attn_mask：一般用于掩蔽未来信息（在decoder用的多）；
            key_padding_mask：掩盖掉 pad 的位置，不参与注意力。
        '''
        src2, weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                       key_padding_mask=src_key_padding_mask)

        '''
            作用：残差连接 + dropout
            直觉：保留原始输入信息的同时，引入注意力的增强表示；防止过拟合 
        '''
        src = src + self.dropout1(src2)

        '''
            作用：LayerNorm 归一化

            直觉：让网络更稳定、训练更快
        '''
        src = self.norm1(src)

        '''
            作用：前馈神经网络（FFN） 先升维（linear1）→ 激活（如ReLU）→ dropout → 再降维（linear2）

            直觉：像一个“感知器”，对每个位置单独做非线性转换，增加表达能力
        '''
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)))) # ffns （Linear => ReLU => dropout => Linear)

        '''
            作用：再次做残差连接 + dropout
            直觉：融合前馈输出与原始表示，防止信息丢失
        '''
        src = src + self.dropout2(src2)

        '''
            作用：LayerNorm，保持数值稳定

            直觉：像“矫正器”，保证数据不会在层之间发散
        '''
        src = self.norm2(src)

        return src, weights
```

### TransfomerDecoder

TransfomerDecoder 主要包含四个部分：Self-Attention（解码器内部，自我理解），Cross-Attention（关注编码器的输出），Feed Forward（非线性变换），残差连接 + LayerNorm

逻辑流程:

```
    tgt ---------------------------+
                                        |
                                    [Add & Norm]
                                        |
    +---------------------------+       ↓
    |     Self-Attention        | → dropout
    +---------------------------+
                                        |
                                    [Add & Norm]
                                        |
    +---------------------------+       ↓
    |  Cross-Attention (w/ src) | → dropout
    +---------------------------+
                                        |
                                    [Add & Norm]
                                        |
    +---------------------------+       ↓
    |   Feed Forward (FFN)      | → dropout
    +---------------------------+
                                        |
                                    [Add & Norm]
                                        ↓
                                Final decoder output
```

代码流程：
```
# 32, 2, 4x*64, 0.1, 'relu',  False
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                    activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    '''
        tgt: (L, B, C)      # 当前解码器输入（target），比如已经生成的 token 表示
        memory: (S, B, C)   # 编码器的输出（源序列编码结果）
        pos: (S, B, C')     # 编码器的 positional encoding（注意可能需要升维）
        query_pos: (L, B, C') # 解码器当前 token 的位置编码

        L：target 序列长度
        S：source 序列长度
        B：batch size
        C：嵌入维度（d_model）
    '''

    def forward_post(self, tgt, memory,
                        tgt_mask: Optional[Tensor] = None,
                        memory_mask: Optional[Tensor] = None,
                        tgt_key_padding_mask: Optional[Tensor] = None,
                        memory_key_padding_mask: Optional[Tensor] = None,
                        pos: Optional[Tensor] = None,
                        query_pos: Optional[Tensor] = None):

        '''
            作用：
                将位置编码 query_pos 加到 target 的输入 tgt 上，作为 Q 和 K。
        '''
        q = k = self.with_pos_embed(tgt, query_pos)

        '''
            作用：
                自注意力, Q和K是相同，Value是原始decode输出序列

            参数：
                attn_mask：
                    一般用于“掩盖未来”（让 decoder 不能看到将来的 token）。
                
                key_padding_mask:
                    掩盖掉 pad 的位置，不参与注意力。
        '''
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]

        '''
            作用：
                残差连接 + dropout + LayerNorm
            
            直觉：
                保留原始输入 + 当前上下文信息，保证训练稳定。    
        '''
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        '''
            Cross Attention：
                参数：
                    query：
                        解码器当前的 token 表示（tgt）加上位置编码（query_pos）。

                    key:
                        编码器输出（memory）加上位置编码（pos）。

                    value: 
                        真正的“信息内容”来源, attention 最后是从 value 里提取信息。

                作用：
                    对每个 tgt token，用 query 去「查」memory 中的每个 token（由 key 提供索引），然后根据权重从 value 中拿出信息。

        '''
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]

        '''
            作用：
                残差连接 + dropout + LayerNorm
            
            直觉：
                融合来自 source（编码器）的信息，且保持训练稳定。
        '''
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        '''
            作用：
                前馈网络，两层全连接（Linear），中间加激活（ReLU 或 GELU）和 Dropout
                    linear1: 升维，比如从 256 → 2048
                    activation: 加非线性
                    linear2: 再降回 256

            直觉：
                每个 token 单独做特征增强。
        '''
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        '''
            作用：
                最后一次残差连接 + LayerNorm
            
            直觉：
                保持原始信息和新特征的平衡，让网络层之间更稳定。
        '''
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt
```

