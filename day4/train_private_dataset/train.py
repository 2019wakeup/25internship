# 完整的模型训练套路(以CIFAR10为例)
import time
import os
import torch.optim
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# 设置项目根目录路径
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 添加模块搜索路径
private_dataset_path = os.path.join(project_root, "day3", "private_dataset")
alexnet_path = os.path.join(project_root, "day2", "alexnet")
sys.path.append(private_dataset_path)
sys.path.append(alexnet_path)

# 导入自定义数据集和模型
sys.path.append(os.path.join(project_root, "day3/private_dataset"))
from dataset import ImageTxtDataset
from model import AlexNet  # 从当前目录导入

# 设置设备
device = torch.device("cpu")
print(f"使用设备: {device}")

# 设置数据增强
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # AlexNet需要224x224的输入
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 准备数据集路径
train_txt = os.path.join(project_root, "day3", "private_dataset", "train.txt")
val_txt = os.path.join(project_root, "day3", "private_dataset", "val.txt")
train_images_dir = os.path.join(project_root, "day3", "Images", "train")
val_images_dir = os.path.join(project_root, "day3", "Images", "val")

# 创建数据集实例
train_data = ImageTxtDataset(train_txt, train_images_dir, transform_train)
test_data = ImageTxtDataset(val_txt, val_images_dir, transform_val)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{train_data_size}")
print(f"测试数据集的长度：{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)  # Mac系统设置num_workers=0
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

# 创建网络模型
model = AlexNet(num_classes=100).to(device)  # 修改100个类别

# 创建损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
learning_rate = 0.0001  # 降低学习率以适应更多的类别
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # 添加权重衰减以防止过拟合

# 设置训练网络的参数
total_train_step = 0
total_test_step = 0
epoch = 50  # 增加训练轮数
best_test_loss = float('inf')  # 初始化最佳测试损失

# 创建保存模型的目录
model_save_dir = os.path.join(current_dir, "model_save")
os.makedirs(model_save_dir, exist_ok=True)

# 添加tensorboard
log_dir = os.path.join(current_dir, "logs_train")
writer = SummaryWriter(log_dir)

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    model.train()
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:  # 更频繁地打印loss
            print(f"训练次数：{total_train_step}, Loss：{loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    print(f"整体测试集上的Loss：{total_test_loss:.4f}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size:.4f}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 保存最好的模型
    if i == 0 or total_test_loss < best_test_loss:
        best_test_loss = total_test_loss
        model_save_path = os.path.join(model_save_dir, "best_model.pth")
        torch.save(model.state_dict(), model_save_path)
    
    # 每5轮保存一次checkpoint
    if (i + 1) % 5 == 0:
        checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{i+1}.pth")
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_test_loss,
        }, checkpoint_path)

end_time = time.time()
print(f"总训练时间：{end_time - start_time:.2f}秒")
writer.close()