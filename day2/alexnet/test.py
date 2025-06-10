# AlexNet测试脚本
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理（与训练时的测试预处理保持一致）
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 准备测试数据集
test_data = torchvision.datasets.CIFAR10(
    root="../../dataset_chen",
    train=False,
    transform=transform_test,
    download=True
)

test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

# CIFAR-10类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 创建模型并加载训练好的权重
model = AlexNet(num_classes=10).to(device)

# 加载最佳模型
try:
    checkpoint = torch.load('alexnet_save/alexnet_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载最佳模型，准确率: {checkpoint['best_accuracy']:.2f}%")
except FileNotFoundError:
    print("未找到训练好的模型文件，请先运行train.py进行训练")
    exit()

# 评估模型
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 计算每个类别的准确率
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print(f'\n整体测试准确率: {100 * correct / total:.2f}%')
    print('\n各类别准确率:')
    for i in range(10):
        if class_total[i] > 0:
            print(f'{classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')

# 可视化预测结果
def visualize_predictions(num_images=8):
    model.eval()
    
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 反标准化图像用于显示
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    # 显示图像和预测结果
    images = denormalize(images.cpu())
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(num_images):
        ax = axes[i//4, i%4]
        
        # 转换图像格式用于显示
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(f'真实: {classes[labels[i]]}\n预测: {classes[predicted[i]]}', 
                    color='green' if labels[i] == predicted[i] else 'red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('alexnet_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("预测结果已保存为 alexnet_predictions.png")

# 单张图像预测
def predict_single_image(image_index=0):
    model.eval()
    
    # 获取单张图像
    image, label = test_data[image_index]
    image = image.unsqueeze(0).to(device)  # 添加批次维度
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, predicted = torch.max(output, 1)
    
    print(f"\n图像 {image_index} 的预测结果:")
    print(f"真实标签: {classes[label]}")
    print(f"预测标签: {classes[predicted.item()]}")
    print("\n各类别概率:")
    for i in range(10):
        print(f"{classes[i]}: {probabilities[i].item():.4f}")

if __name__ == '__main__':
    print("开始评估AlexNet模型...")
    
    # 整体评估
    evaluate_model()
    
    # 可视化预测结果
    try:
        visualize_predictions()
    except:
        print("无法显示图像，可能是在无图形界面环境中运行")
    
    # 单张图像预测示例
    predict_single_image(0)
    predict_single_image(100)
    predict_single_image(1000) 