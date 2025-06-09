import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class Sentinel2Processor:
    """
    哨兵2号卫星图像处理器
    """
    
    def __init__(self, file_path):
        """
        初始化处理器
        
        Args:
            file_path (str): TIF文件路径
        """
        self.file_path = file_path
        self.data = None
        self.metadata = None
        
    def load_data(self):
        """
        加载TIF文件数据
        """
        try:
            with rasterio.open(self.file_path) as src:
                # 读取所有波段数据
                self.data = src.read()  # 形状: (bands, height, width)
                self.metadata = src.meta
                
                print(f"数据形状: {self.data.shape}")
                print(f"波段数量: {self.data.shape[0]}")
                print(f"图像尺寸: {self.data.shape[1]} x {self.data.shape[2]}")
                print(f"数据类型: {self.data.dtype}")
                print(f"数据范围: {self.data.min()} - {self.data.max()}")
                
        except Exception as e:
            print(f"加载数据失败: {e}")
            
    def normalize_data(self, input_range=(0, 10000), output_range=(0, 255)):
        """
        将数据从输入范围压缩到输出范围
        
        Args:
            input_range (tuple): 输入数据范围，默认(0, 10000)
            output_range (tuple): 输出数据范围，默认(0, 255)
        """
        if self.data is None:
            print("请先加载数据")
            return
            
        # 裁剪数据到输入范围
        normalized_data = np.clip(self.data, input_range[0], input_range[1])
        
        # 线性缩放到输出范围
        normalized_data = (normalized_data - input_range[0]) / (input_range[1] - input_range[0])
        normalized_data = normalized_data * (output_range[1] - output_range[0]) + output_range[0]
        
        # 转换为uint8类型
        self.normalized_data = normalized_data.astype(np.uint8)
        
        print(f"数据归一化完成")
        print(f"归一化后数据范围: {self.normalized_data.min()} - {self.normalized_data.max()}")
        
    def extract_rgb_channels(self, band_indices=[0, 1, 2]):
        """
        提取RGB通道
        
        Args:
            band_indices (list): RGB波段的索引，默认[0, 1, 2]表示前三个波段
        """
        if not hasattr(self, 'normalized_data'):
            print("请先进行数据归一化")
            return
            
        # 提取RGB三个通道
        rgb_data = self.normalized_data[band_indices]
        
        # 转换为标准的RGB格式 (height, width, channels)
        self.rgb_image = np.transpose(rgb_data, (1, 2, 0))
        
        print(f"RGB图像形状: {self.rgb_image.shape}")
        
    def save_rgb_image(self, output_path):
        """
        保存RGB图像
        
        Args:
            output_path (str): 输出文件路径
        """
        if not hasattr(self, 'rgb_image'):
            print("请先提取RGB通道")
            return
            
        # 使用PIL保存图像
        image = Image.fromarray(self.rgb_image)
        image.save(output_path)
        print(f"RGB图像已保存到: {output_path}")
        
    def display_comparison(self):
        """
        显示原始数据和处理后的RGB图像对比
        """
        if not hasattr(self, 'rgb_image'):
            print("请先提取RGB通道")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 显示原始波段
        for i in range(min(5, self.data.shape[0])):
            row = i // 3
            col = i % 3
            axes[row, col].imshow(self.data[i], cmap='gray')
            axes[row, col].set_title(f'原始波段 {i+1}')
            axes[row, col].axis('off')
        
        # 显示RGB合成图像
        if self.data.shape[0] >= 5:
            axes[1, 2].imshow(self.rgb_image)
            axes[1, 2].set_title('RGB合成图像')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def get_band_statistics(self):
        """
        获取各波段的统计信息
        """
        if self.data is None:
            print("请先加载数据")
            return
            
        print("\n=== 波段统计信息 ===")
        band_names = ['Red', 'Green', 'Blue', 'Near Infrared', 'Short Wave Infrared']
        
        for i in range(self.data.shape[0]):
            band_data = self.data[i]
            band_name = band_names[i] if i < len(band_names) else f"Band {i+1}"
            
            print(f"\n{band_name}:")
            print(f"  最小值: {band_data.min()}")
            print(f"  最大值: {band_data.max()}")
            print(f"  平均值: {band_data.mean():.2f}")
            print(f"  标准差: {band_data.std():.2f}")

def main():
    """
    主函数，演示完整的处理流程
    """
    # 文件路径
    tif_file = "2019_1101_nofire_B2348_B12_10m_roi.tif"
    output_file = "sentinel2_rgb_output.png"
    
    print("开始处理哨兵2号卫星图像...")
    
    # 创建处理器实例
    processor = Sentinel2Processor(tif_file)
    
    # 加载数据
    processor.load_data()
    
    # 获取统计信息
    processor.get_band_statistics()
    
    # 归一化数据 (0-10000 -> 0-255)
    processor.normalize_data(input_range=(0, 10000), output_range=(0, 255))
    
    # 提取RGB通道 (假设前三个波段是RGB)
    processor.extract_rgb_channels(band_indices=[0, 1, 2])
    
    # 保存RGB图像
    processor.save_rgb_image(output_file)
    
    # 显示对比图
    processor.display_comparison()
    
    print("\n处理完成！")

if __name__ == "__main__":
    main()
