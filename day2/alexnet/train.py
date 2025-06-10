# AlexNet训练脚本
import time
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from model import AlexNet

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理和增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root="../../dataset_chen",
    train=True,
    transform=transform_train,
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="../../dataset_chen",
    train=False,
    transform=transform_test,
    download=True
)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度: {train_data_size}")
print(f"测试数据集的长度: {test_data_size}")

# 加载数据集
batch_size = 128
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

# CIFAR-10类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 创建网络模型
model = AlexNet(num_classes=10).to(device)
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 创建损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
initial_lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epochs = 100
best_accuracy = 0.0

# 创建保存目录
os.makedirs("alexnet_save", exist_ok=True)

# 添加tensorboard
writer = SummaryWriter("../../logs_train/alexnet")

print("开始训练...")
start_time = time.time()

for epoch in range(epochs):
    print(f"\n----- 第 {epoch+1} 轮训练开始 -----")
    
    # 训练模式
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        total_train_step += 1
        
        if batch_idx % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    
    # 计算训练准确率
    train_accuracy = 100. * correct / total
    avg_train_loss = running_loss / len(train_loader)
    
    print(f'训练 - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
    
    # 测试模式
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # 计算测试准确率
    test_accuracy = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    
    print(f'测试 - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
    
    # 记录到tensorboard
    writer.add_scalar("test_loss", avg_test_loss, epoch)
    writer.add_scalar("test_accuracy", test_accuracy, epoch)
    writer.add_scalar("train_accuracy", train_accuracy, epoch)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)
    
    # 保存最佳模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
        }, 'alexnet_save/alexnet_best.pth')
        print(f'保存最佳模型，准确率: {best_accuracy:.2f}%')
    
    # 定期保存检查点
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': test_accuracy,
        }, f'alexnet_save/alexnet_epoch_{epoch+1}.pth')
    
    # 更新学习率
    scheduler.step()
    
    total_test_step += 1

end_time = time.time()
total_time = end_time - start_time
print(f"\n训练完成！")
print(f"总训练时间: {total_time/3600:.2f} 小时")
print(f"最佳测试准确率: {best_accuracy:.2f}%")

writer.close()

# 保存最终模型
torch.save(model.state_dict(), 'alexnet_save/alexnet_final.pth')
print("最终模型已保存")
