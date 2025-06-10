# AlexNet CIFAR-10 十分类实现

本项目实现了AlexNet网络架构，用于CIFAR-10数据集的十分类任务。

## 项目结构

```
alexnet/
├── model.py          # AlexNet网络架构定义
├── train.py          # 训练脚本
├── test.py           # 测试脚本
├── README.md         # 说明文档
└── alexnet_save/     # 模型保存目录（训练后生成）
```

## 网络架构

AlexNet针对CIFAR-10数据集（32x32像素）进行了适配，包含：

### 特征提取部分（卷积层）
- **第1层**: 3→96通道，5x5卷积核，ReLU激活，LRN标准化，2x2最大池化
- **第2层**: 96→256通道，3x3卷积核，ReLU激活，LRN标准化，2x2最大池化  
- **第3层**: 256→384通道，3x3卷积核，ReLU激活
- **第4层**: 384→384通道，3x3卷积核，ReLU激活
- **第5层**: 384→256通道，3x3卷积核，ReLU激活，2x2最大池化

### 分类器部分（全连接层）
- **FC1**: 4096个神经元，Dropout(0.5)，ReLU激活
- **FC2**: 4096个神经元，Dropout(0.5)，ReLU激活  
- **FC3**: 10个神经元（对应10个类别）

**模型参数总数**: 约3700万个

## 使用方法

### 1. 训练模型

```bash
cd day2/alexnet
python train.py
```

训练特性：
- **数据增强**: 随机水平翻转、旋转、颜色抖动
- **优化器**: SGD (lr=0.01, momentum=0.9, weight_decay=5e-4)
- **学习率调度**: 每30轮衰减0.1倍
- **训练轮数**: 100轮
- **批次大小**: 128
- **自动保存**: 最佳模型和定期检查点

### 2. 测试模型

```bash
cd day2/alexnet  
python test.py
```

测试功能：
- 整体准确率评估
- 各类别准确率统计
- 预测结果可视化
- 单张图像预测示例

### 3. 模型验证

```bash
python model.py
```

验证网络架构是否正确实现。

## 数据集

使用CIFAR-10数据集，包含10个类别：
- plane（飞机）
- car（汽车） 
- bird（鸟）
- cat（猫）
- deer（鹿）
- dog（狗）
- frog（青蛙）
- horse（马）
- ship（船）
- truck（卡车）

数据集会自动下载到 `../../dataset_chen` 目录。

## 训练监控

训练过程使用TensorBoard记录：
- 训练损失
- 测试损失  
- 训练准确率
- 测试准确率
- 学习率变化

查看训练曲线：
```bash
tensorboard --logdir=../../logs_train/alexnet
```

## 模型保存

训练过程中会保存以下文件到 `alexnet_save/` 目录：
- `alexnet_best.pth`: 最佳模型（测试准确率最高）
- `alexnet_epoch_20.pth`: 第20轮检查点
- `alexnet_epoch_40.pth`: 第40轮检查点
- ... （每20轮保存一次）
- `alexnet_final.pth`: 最终模型

## 预期性能

在CIFAR-10测试集上，AlexNet预期可达到：
- **准确率**: 75-85%（取决于训练轮数和超参数调优）
- **训练时间**: GPU约2-4小时，CPU约10-20小时

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- matplotlib（用于可视化）
- tensorboard（用于训练监控）

## 注意事项

1. 建议使用GPU训练以获得合理的训练时间
2. 可根据硬件配置调整批次大小（batch_size）
3. 训练过程中会占用约2-4GB显存
4. 首次运行会自动下载CIFAR-10数据集（约160MB）

## 代码特点

- **规范化**: 代码结构清晰，注释详细
- **可扩展**: 易于修改网络结构和训练参数
- **完整性**: 包含训练、测试、可视化等完整功能
- **监控**: 支持TensorBoard实时监控训练过程
- **容错**: 自动创建目录，处理异常情况 