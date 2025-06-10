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

*更新时间: 2025年6月*

