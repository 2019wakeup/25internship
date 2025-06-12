# 25internship

## 项目概述

这是2025年实习期间的学习记录和项目实践仓库，包含了完整的学习历程和技术笔记。

## Day1 项目概况

Day1是本实习项目的第一阶段，主要聚焦于**遥感图像处理**和**Python/Git基础技能**的学习与实践。这一阶段涵盖了从环境搭建到核心技术应用的完整学习路径。

### 核心内容
- 🛰️ **卫星图像处理**: 哨兵2号（Sentinel-2）多光谱图像处理
- 🐍 **Python编程基础**: 从基础语法到高级特性的全面学习
- 📝 **Git版本控制**: 完整的版本管理工作流程
- 📊 **数据处理技术**: NumPy、Rasterio、PIL等核心库的应用

### 项目结构
```
day1/
├── radar/              # 雷达相关项目文件
├── test/               # 测试代码和示例
├── note_radar.md       # 📋 卫星图像处理完整笔记
└── note_git&py.md      # 📋 Python与Git基础教程
```

## 🎯 重要学习资料

### 📖 核心笔记文档

#### 1. [卫星图像处理知识点总结](day1/note_radar.md) 
**位置**: `day1/note_radar.md` | **大小**: 11KB | **行数**: 433行

这是一份**全面的卫星图像处理技术指南**，包含：
- 🔧 **核心Python库详解**: NumPy、Rasterio、PIL的深入应用
- 📏 **数据归一化技术**: 线性归一化、Z-score标准化、百分位数拉伸
- 🌍 **遥感图像处理基础**: 多光谱图像概念、波段组合、图像增强
- ⚡ **性能优化技巧**: 内存管理、并行处理、大型图像分块处理
- 🛠️ **实用工具函数**: 完整的图像处理工具类和最佳实践

#### 2. [Python与Git基础课程](day1/note_git&py.md)
**位置**: `day1/note_git&py.md` | **大小**: 8.5KB | **行数**: 446行

这是一份**完整的Python和Git学习教程**，涵盖：
- 🏗️ **Python基础**: 变量、类型、作用域、运算符、控制语句
- 🔧 **高级特性**: 函数、类与对象、装饰器、文件操作
- 📦 **模块管理**: 包和模块的创建与使用
- 🌿 **Git版本控制**: 从配置到高级分支管理的完整工作流
- 💡 **实践技巧**: 配置管理、协作开发、最佳实践

## 🚀 快速开始

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd 25internship
   ```

2. **查看Day1学习资料**
   ```bash
   cd day1
   # 阅读卫星图像处理笔记
   cat note_radar.md
   # 阅读Python与Git基础笔记  
   cat note_git&py.md
   ```

3. **环境准备**
   ```bash
   # 创建Python虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   
   # 安装依赖
   pip install numpy rasterio pillow
   ```

## Day2 项目概况

Day2是本实习项目的第二阶段，主要聚焦于**深度学习理论基础**和**PyTorch实践应用**的学习与实践。这一阶段从理论到实践，构建了完整的深度学习开发技能体系。

### 核心内容
- 🧠 **深度学习理论**: 核心操作、网络架构、优化技术全面学习
- 🔥 **PyTorch实践**: 从模型搭建到训练部署的完整工作流
- 🖼️ **图像分类项目**: CIFAR-10数据集上的CNN实现与训练
- 📊 **模型优化**: GPU加速、TensorBoard可视化、模型保存与加载
- 🎯 **目标检测基础**: 损失函数、评价指标、非极大值抑制等核心概念

### 项目结构
```
day2/
├── day2_notes/              # 📚 深度学习理论笔记集合
│   ├── 深度学习核心操作笔记.md    # 卷积、池化、激活函数等核心概念
│   ├── 目标检测损失函数.md        # 目标检测相关理论
│   ├── 常用激活函数及用法.md      # 激活函数详解
│   ├── 多尺度注意力机制.md        # 注意力机制原理
│   └── FPN.md                   # 特征金字塔网络
├── alexnet/                 # AlexNet网络学习
├── model.py                 # 🎯 自定义CNN模型定义
├── train.py                 # 🚀 完整训练脚本
├── train_GPU_1.py          # ⚡ GPU加速训练版本1
├── train_GPU_2.py          # ⚡ GPU加速训练版本2
├── test.py                 # 🧪 模型测试脚本
├── model_save/             # 💾 训练模型保存目录
└── dataset_chen/           # 📁 CIFAR-10数据集
```

## 🎯 重要学习资料

### 📖 核心笔记文档

#### 1. [深度学习核心操作全集](day2/day2_notes/深度学习核心操作笔记.md)
**位置**: `day2/day2_notes/深度学习核心操作笔记.md` | **大小**: 14KB | **行数**: 180行

这是一份**深度学习核心操作的权威指南**，包含：
- 🔧 **基础操作详解**: 卷积、池化、上采样、归一化等10大核心操作
- 📏 **参数配置指南**: 每种操作的关键参数、使用场景、注意事项
- 🎯 **实践应用**: 在不同网络架构中的具体应用和最佳实践
- ⚡ **性能优化**: 计算效率、内存管理、模型优化技巧
- 🔬 **技术深度**: 从基础概念到高级变种的完整覆盖

#### 2. [目标检测理论基础](day2/day2_notes/)
**多个笔记文件** | **总计**: 约30KB

涵盖目标检测的**完整理论体系**：
- 📊 **损失函数**: 分类损失、回归损失、多任务损失设计
- 📈 **评价指标**: mAP、IoU、精确率、召回率等核心指标
- 🎯 **非极大值抑制**: NMS算法原理与实现
- 🔍 **注意力机制**: 多尺度注意力、Outlook Attention等前沿技术

### 🚀 实践项目

#### CIFAR-10图像分类项目
**核心文件**: `model.py`, `train.py`, `train_GPU_*.py`

这是一个**完整的深度学习项目**，实现了：
- 🏗️ **自定义CNN架构**: 3层卷积 + 2层全连接的经典结构
- 📊 **完整训练流程**: 数据加载、模型训练、性能评估
- ⚡ **GPU加速支持**: 多版本GPU训练脚本，显著提升训练效率
- 📈 **可视化监控**: TensorBoard实时监控训练过程
- 💾 **模型管理**: 自动保存训练checkpoints，支持模型恢复

**技术特点**:
```python
# 模型架构示例
class Chen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),    # 卷积层1
            nn.MaxPool2d(kernel_size=2),        # 池化层1
            nn.Conv2d(32, 32, 5, padding=2),   # 卷积层2
            nn.MaxPool2d(kernel_size=2),        # 池化层2  
            nn.Conv2d(32, 64, 5, padding=2),   # 卷积层3
            nn.MaxPool2d(kernel_size=2),        # 池化层3
            nn.Flatten(),                       # 展平
            nn.Linear(1024, 64),               # 全连接层1
            nn.Linear(64, 10)                  # 输出层
        )
```

## 🚀 快速开始

### Day1 - 遥感图像处理

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd 25internship
   ```

2. **查看Day1学习资料**
   ```bash
   cd day1
   # 阅读卫星图像处理笔记
   cat note_radar.md
   # 阅读Python与Git基础笔记  
   cat note_git&py.md
   ```

3. **环境准备**
   ```bash
   # 创建Python虚拟环境
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   
   # 安装依赖
   pip install numpy rasterio pillow
   ```

### Day2 - 深度学习实践

1. **深度学习环境准备**
   ```bash
   cd day2
   # 安装PyTorch和相关依赖
   pip install torch torchvision tensorboard
   ```

2. **运行CIFAR-10分类项目**
   ```bash
   # CPU训练
   python train.py
   
   # GPU训练（如果有GPU）
   python train_GPU_1.py
   
   # 模型测试
   python test.py
   ```

3. **查看训练可视化**
   ```bash
   # 启动TensorBoard
   tensorboard --logdir=logs_train
   # 在浏览器访问 http://localhost:6006
   ```

4. **学习理论笔记**
   ```bash
   # 查看深度学习核心操作笔记
   cat day2_notes/深度学习核心操作笔记.md
   
   # 查看目标检测相关笔记
   ls day2_notes/
   ```

## Day3 项目概况

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

## 🎯 重要学习资料

### 📖 核心笔记文档

#### 1. [Day3学习笔记](day3/day3_notes.md)
**位置**: `day3/day3_notes.md` | **大小**: 约8KB

这是一份**自定义数据集处理和激活函数学习指南**，包含：
- 📊 **数据集处理完整流程**: 从原始数据到训练就绪的全套处理方案
- 🔧 **自定义Dataset类设计**: PyTorch Dataset接口的灵活实现
- ⚡ **激活函数深入理解**: ReLU、Sigmoid等激活函数的原理与应用
- 📈 **可视化技术**: TensorBoard在模型调试中的实际应用
- 🎯 **CIFAR-100数据集**: 100类图像分类任务的完整处理方案

### 🚀 实践项目

#### 自定义数据集处理项目
**核心文件**: `deal_with_datasets.py`, `prepare.py`, `dataset.py`

这是一个**完整的数据预处理项目**，实现了：
- 🔄 **智能数据划分**: 7:3比例的训练/验证集科学划分
- 📝 **路径管理系统**: 自动生成train.txt和val.txt路径文件
- 🎯 **灵活Dataset类**: 支持任意格式数据集的高效加载
- 🖼️ **图像预处理**: RGB转换、标准化等完整预处理pipeline
- 📁 **目录结构管理**: 自动创建和维护数据集目录结构

**技术特点**:
```python
# 数据集划分核心代码
train_images, val_images = train_test_split(images, train_size=0.7, random_state=42)

# 自定义Dataset类
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path: str, folder_name, transform):
        # 从txt文件加载图像路径和标签
        # 支持灵活的数据变换pipeline
```

#### 激活函数可视化项目
**核心文件**: `activate_function/function.py`

这是一个**激活函数学习和可视化项目**，实现了：
- ⚡ **多种激活函数**: ReLU、Sigmoid等主流激活函数实现
- 👁️ **实时可视化**: TensorBoard展示激活前后的图像效果
- 🔧 **网络设计**: 自定义神经网络类的激活函数应用
- 📊 **效果对比**: 不同激活函数在实际数据上的表现分析

### Day3 - 自定义数据集与激活函数

1. **数据集处理实践**
   ```bash
   cd day3/private_dataset
   
   # 数据集划分
   python deal_with_datasets.py
   
   # 生成路径文件
   python prepare.py
   
   # 测试自定义Dataset
   python dataset.py
   ```

2. **激活函数学习**
   ```bash
   cd day3/activate_function
   
   # 运行激活函数可视化
   python function.py
   
   # 查看TensorBoard可视化结果
   tensorboard --logdir=sigmod_logs
   ```

3. **查看学习笔记**
   ```bash
   # 查看Day3完整学习笔记
   cat day3/day3_notes.md
   ```

*更新时间: 2025年6月*

