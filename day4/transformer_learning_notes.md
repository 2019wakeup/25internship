# Transformer与Vision Transformer (ViT) 学习笔记

## 1. Transformer 基础概念

### 1.1 核心组件
- **Self-Attention**: 自注意力机制，允许模型关注输入序列中的不同部分
- **Multi-Head Attention**: 多头注意力，允许模型同时学习不同的注意力模式
- **Feed Forward Network**: 前馈神经网络，用于特征转换
- **Layer Normalization**: 层归一化，用于稳定训练
- **Residual Connection**: 残差连接，帮助解决深层网络训练问题

### 1.2 注意力机制详解
```python
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        # dim: 输入维度
        # heads: 注意力头数
        # dim_head: 每个注意力头的维度
```
注意力机制的计算步骤：
1. 输入经过线性变换得到Q(Query)、K(Key)、V(Value)
2. 计算注意力权重：`attention = softmax(Q·K^T / sqrt(dim))`
3. 加权求和：`output = attention·V`

## 2. Vision Transformer (ViT)

### 2.1 架构特点
1. **图像分块（Patch Embedding）**
   - 将输入图像分成固定大小的patch
   - 每个patch通过线性投影转换为embedding

2. **位置编码**
   - 使用可学习的位置编码
   - 添加到patch embeddings中以保留位置信息

3. **分类头设计**
   - 使用特殊的[CLS]标记进行分类
   - 或使用所有token的平均池化

### 2.2 关键实现
```python
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        # image_size: 输入图像尺寸
        # patch_size: 每个patch的大小
        # num_classes: 分类数量
        # dim: 模型维度
        # depth: Transformer层数
        # heads: 注意力头数
        # mlp_dim: 前馈网络维度
```

## 3. 实现要点

### 3.1 Patch Embedding
- 使用`Rearrange`操作将图像重组为patches
- 通过线性层将patches映射到所需维度
- 添加位置编码

### 3.2 Transformer Block
- 包含Multi-Head Self-Attention
- Feed Forward Network
- Layer Normalization
- Residual Connections

### 3.3 分类头
- 可选择使用CLS token或mean pooling
- 最后通过线性层映射到类别数

## 4. 优势

### 4.1 优势
- 全局感受野，可以捕获长距离依赖
- 结构简单，易于扩展
- 参数共享，计算效率高
