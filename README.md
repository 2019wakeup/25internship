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

*更新时间: 2025年1月*

