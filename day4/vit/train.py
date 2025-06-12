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
prepare_dataset_path = os.path.join(project_root, "day4", "prepare_dataset")
sys.path.append(prepare_dataset_path)

# 导入自定义数据集和模型
from dataset import ImageTxtDataset
from vit import ViT  # 从当前目录导入ViT模型

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设置数据增强
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT需要固定大小的输入
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
train_txt = os.path.join(project_root, "day4", "prepare_dataset", "train.txt")
val_txt = os.path.join(project_root, "day4", "prepare_dataset", "val.txt")
images_dir = os.path.join(project_root, "datasets", "Images")

# 创建数据集实例
train_data = ImageTxtDataset(train_txt, images_dir, transform_train)
test_data = ImageTxtDataset(val_txt, images_dir, transform_val)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{train_data_size}")
print(f"测试数据集的长度：{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

# 创建ViT模型
model = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 100,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# 设置训练网络的参数
total_train_step = 0
total_test_step = 0
epoch = 50
best_test_loss = float('inf')

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
    epoch_loss = 0
    
    # 训练步骤
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        epoch_loss += loss.item()

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if batch_idx % 20 == 0:  # 每20个batch打印一次
            print(f"Epoch: {i+1}/{epoch}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 更新学习率
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar("learning_rate", current_lr, i)
    
    # 计算平均训练损失
    avg_train_loss = epoch_loss / len(train_loader)
    writer.add_scalar("avg_train_loss", avg_train_loss, i)

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

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = total_accuracy / test_data_size
    
    print(f"Epoch {i+1} 测试结果:")
    print(f"平均测试Loss：{avg_test_loss:.4f}")
    print(f"测试准确率：{accuracy:.4f}")
    
    writer.add_scalar("test_loss", avg_test_loss, i)
    writer.add_scalar("test_accuracy", accuracy, i)
    total_test_step += 1

    # 保存最好的模型
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        model_save_path = os.path.join(model_save_dir, "best_model.pth")
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_test_loss,
        }, model_save_path)
        print(f"保存最佳模型，Loss: {best_test_loss:.4f}")
    
    # 每10轮保存一次checkpoint
    if (i + 1) % 10 == 0:
        checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{i+1}.pth")
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_test_loss,
        }, checkpoint_path)

end_time = time.time()
print(f"总训练时间：{end_time - start_time:.2f}秒")
writer.close()