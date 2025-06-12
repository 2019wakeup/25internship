# Day3 学习笔记

## 项目概述

Day3是本实习项目的第三阶段，主要聚焦于**自定义数据集处理**和**激活函数深入理解**的学习与实践。这一阶段从数据预处理到模型组件的深入学习，构建了完整的数据处理和网络设计技能体系。

### 核心内容
- 📊 **自定义数据集处理**: 数据集划分、路径管理、自定义Dataset类设计
- 🔧 **数据预处理流程**: 从原始数据到训练就绪的完整数据处理pipeline
- ⚡ **激活函数实践**: ReLU、Sigmoid等激活函数的可视化和应用
- 📈 **数据可视化**: TensorBoard在激活函数效果展示中的应用
- 🎯 **CIFAR-100数据集**: 100类图像分类数据集的处理和应用

### 项目结构
```
day3/
├── private_dataset/              # 📁 自定义数据集处理模块
│   ├── deal_with_datasets.py     # 🔄 数据集划分脚本
│   ├── prepare.py                # 📝 路径文件生成脚本  
│   └── dataset.py                # 🎯 自定义Dataset类定义
├── Images/                       # 🖼️ CIFAR-100图像数据集（100个类别）
│   ├── sunflower/               # 🌻 向日葵类别
│   ├── tiger/                   # 🐅 老虎类别
│   ├── butterfly/               # 🦋 蝴蝶类别
│   └── ...                      # 其他97个类别
├── activate_function/            # ⚡ 激活函数学习模块
│   └── function.py              # 📊 激活函数实践与可视化
└── day3_notes.md                # 📋 当日学习笔记
```

## 🎯 核心学习内容

### 1. 自定义数据集处理 📊

#### 数据集划分 (`deal_with_datasets.py`)
**功能**: 将原始数据集按照7:3比例划分为训练集和验证集
**核心特点**:
- 🔄 自动创建train/val目录结构
- 📁 保持类别文件夹组织
- 🎲 随机种子确保可重复性
- 🚀 支持大规模数据集处理

```python
# 核心功能
train_ratio = 0.7  # 训练集比例
train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)
```

#### 路径文件生成 (`prepare.py`)
**功能**: 为训练和验证集生成路径-标签映射文件
**输出**: `train.txt` 和 `val.txt` 文件
**格式**: `图片路径 类别标签`

#### 自定义Dataset类 (`dataset.py`)
**功能**: 实现PyTorch Dataset接口，支持从txt文件加载数据
**核心特点**:
- 🎯 **ImageTxtDataset类**: 灵活的数据加载器
- 🔧 **Transform支持**: 完整的数据增强pipeline
- 📝 **路径管理**: 高效的文件路径处理
- 🖼️ **图像预处理**: RGB转换和标准化

### 2. 激活函数深入学习 ⚡

#### 激活函数实践 (`function.py`)
**学习内容**:
- 🔧 **ReLU激活函数**: 解决梯度消失问题
- 📈 **Sigmoid激活函数**: 概率输出和二分类应用
- 👁️ **可视化对比**: TensorBoard实时展示激活前后效果
- 🎯 **实际应用**: 在CIFAR-10数据上的效果验证

**技术亮点**:
```python
class Chen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input):
        output = self.sigmoid(input)
        return output
```

### 3. CIFAR-100数据集应用 🖼️

#### 数据集特点
- **类别数量**: 100个不同类别
- **数据规模**: 每类包含多张高质量图像
- **类别多样性**: 涵盖动物、植物、物体、场景等多个领域
- **应用场景**: 多分类任务、细粒度分类研究

#### 主要类别示例
```
动物类: tiger, lion, elephant, dolphin, butterfly...
植物类: sunflower, rose, tulip, maple_tree, oak_tree...
物体类: bottle, table, chair, telephone, keyboard...
场景类: forest, mountain, bridge, castle, road...
```

## 🚀 技术收获

### 数据处理技能
- ✅ **数据集组织**: 学会了标准的数据集目录结构设计
- ✅ **数据划分**: 掌握了训练/验证集的科学划分方法
- ✅ **路径管理**: 实现了高效的文件路径处理机制
- ✅ **自定义Dataset**: 能够根据需求设计灵活的数据加载器

### 深度学习组件理解
- ✅ **激活函数原理**: 深入理解不同激活函数的特点和应用场景
- ✅ **可视化技能**: 掌握了TensorBoard在模型调试中的应用
- ✅ **实践验证**: 通过实际代码验证了理论知识

### 工程化能力
- ✅ **代码模块化**: 实现了可复用的数据处理组件
- ✅ **参数配置**: 学会了灵活的配置管理方式
- ✅ **错误处理**: 增强了代码的健壮性和可维护性

## 💡 实践应用

这一天的学习成果可以直接应用于：
- 🎯 **图像分类项目**: 完整的数据预处理pipeline
- 🔧 **自定义数据集**: 处理任意格式的图像数据集
- 📊 **模型可视化**: 深入理解网络组件的工作机制
- 🚀 **项目实战**: 为后续复杂项目奠定坚实基础

*学习日期: Day3*
*核心技能: 数据处理 + 激活函数 + 可视化* 