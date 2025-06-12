import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 设置项目根目录路径
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 添加模块搜索路径
prepare_dataset_path = os.path.join(project_root, "day4", "prepare_dataset")
sys.path.append(prepare_dataset_path)

# 导入自定义数据集和模型
from dataset import ImageTxtDataset
from vit import ViT

def setup_cuda():
    """设置CUDA设备和优化"""
    if not torch.cuda.is_available():
        print("错误：未检测到可用的CUDA设备！")
        sys.exit(1)
        
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # 为固定大小输入优化CUDNN
    torch.backends.cudnn.deterministic = False  # 关闭确定性模式以提高性能
    
    # 打印GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # 转换为GB
    print(f"使用GPU: {gpu_name}")
    print(f"GPU总内存: {gpu_memory:.2f}GB")
    
    return device

def create_transforms(image_size=224):
    """创建数据增强转换"""
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform_train, transform_val

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter('Loss', ':.4f')
    accuracy = AverageMeter('Acc', ':.4f')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        # 使用混合精度训练
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        # 计算准确率
        pred = outputs.argmax(dim=1)
        acc = (pred == targets).float().mean()
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        accuracy.update(acc.item(), images.size(0))
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracy.avg:.4f}'
        })
    
    return losses.avg, accuracy.avg

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    losses = AverageMeter('Loss', ':.4f')
    accuracy = AverageMeter('Acc', ':.4f')
    
    for images, targets in tqdm(val_loader, desc='Validating'):
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        pred = outputs.argmax(dim=1)
        acc = (pred == targets).float().mean()
        
        losses.update(loss.item(), images.size(0))
        accuracy.update(acc.item(), images.size(0))
    
    return losses.avg, accuracy.avg

def main():
    # 基础设置
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    
    # 设置设备
    device = setup_cuda()
    
    # 创建数据转换
    transform_train, transform_val = create_transforms(IMAGE_SIZE)
    
    # 准备数据集路径
    train_txt = os.path.join(project_root, "day4", "prepare_dataset", "train.txt")
    val_txt = os.path.join(project_root, "day4", "prepare_dataset", "val.txt")
    images_dir = os.path.join(project_root, "datasets", "Images")
    
    # 创建数据集和数据加载器
    train_dataset = ImageTxtDataset(train_txt, images_dir, transform_train)
    val_dataset = ImageTxtDataset(val_txt, images_dir, transform_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    # 创建模型
    model = ViT(
        image_size=IMAGE_SIZE,
        patch_size=16,
        num_classes=100,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)
    
    # 使用DistributedDataParallel包装模型（如果有多个GPU）
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU训练")
        model = nn.DataParallel(model)
    
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # 创建混合精度训练的scaler
    scaler = GradScaler()
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(os.path.join(current_dir, "logs_train_gpu"))
    
    # 创建模型保存目录
    model_save_dir = os.path.join(current_dir, "model_save_gpu")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    try:
        for epoch in range(NUM_EPOCHS):
            # 训练一个epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch
            )
            
            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 打印训练信息
            print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(model_save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, checkpoint_path)
                print(f'保存最佳模型，验证Loss: {val_loss:.4f}')
            
            # 定期保存checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, checkpoint_path)
            
            # 打印GPU内存使用情况
            if torch.cuda.is_available():
                print(f'GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
    
    except KeyboardInterrupt:
        print('训练被用户中断')
    
    finally:
        # 训练结束，保存最后的模型
        final_checkpoint_path = os.path.join(model_save_dir, 'final_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, final_checkpoint_path)
        
        end_time = time.time()
        print(f'\n训练完成！总用时: {(end_time-start_time)/3600:.2f}小时')
        writer.close()

if __name__ == '__main__':
    main() 