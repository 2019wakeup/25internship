import numpy as np
import rasterio
from PIL import Image

def output(tif_file):
    """
    处理哨兵2号卫星图像的简化函数
    将5个波段的数据转换为RGB三通道图像，并将数据范围从0-10000压缩到0-255
    
    Args:
        tif_file (str): TIF文件路径
        
    Returns:
        numpy.ndarray: 归一化后的RGB图像数组 (height, width, 3)
    """
    # 打开TIFF文件
    with rasterio.open(tif_file) as src:
        # 读取所有波段（假设波段顺序为B02, B03, B04, B08, B12）
        bands = src.read()  # 形状为 (波段数, 高度, 宽度)，这里是 (5, height, width)
        # profile = src.profile  # 获取元数据

    # 分配波段（假设TIFF中的波段顺序为B02, B03, B04, B08, B12）
    blue = bands[0].astype(float)   # B02 - 蓝
    green = bands[1].astype(float)  # B03 - 绿
    red = bands[2].astype(float)    # B04 - 红
    nir = bands[3].astype(float)    # B08 - 近红外
    swir = bands[4].astype(float)   # B12 - 短波红外

    # 真彩色正则化
    rgb_origin = np.dstack((red, green, blue))
    array_min, array_max = rgb_origin.min(), rgb_origin.max()
    rgb_normalized = ((rgb_origin - array_min) / (array_max - array_min)) * 255
    rgb_normalized = rgb_normalized.astype(np.uint8)
    
    return rgb_normalized

# 使用示例
if __name__ == "__main__":
    # 处理文件
    tif_file = "2019_1101_nofire_B2348_B12_10m_roi.tif"
    
    print("开始处理哨兵2号卫星图像...")
    rgb_result = output(tif_file)
    
    print(f"处理完成！")
    print(f"RGB图像形状: {rgb_result.shape}")
    print(f"数据范围: {rgb_result.min()} - {rgb_result.max()}")
    
    # 保存结果
    output_file = "simple_output_result.png"
    image = Image.fromarray(rgb_result)
    image.save(output_file)
    print(f"图像已保存到: {output_file}") 