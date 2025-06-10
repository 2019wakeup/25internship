# AlexNet网络架构实现
import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 第一层卷积：输入3通道，输出96通道，卷积核5x5（适配32x32输入）
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第二层卷积：输入96通道，输出256通道，卷积核3x3
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积：输入256通道，输出384通道，卷积核3x3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四层卷积：输入384通道，输出384通道，卷积核3x3
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # 第五层卷积：输入384通道，输出256通道，卷积核3x3
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            # 计算特征图大小：32->16->8->4，256通道，所以是256*4*4=4096
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 分类
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # 测试网络
    model = AlexNet(num_classes=10)
    input_tensor = torch.randn(64, 3, 32, 32)  # CIFAR-10图像尺寸
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
