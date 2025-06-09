# 卫星图像处理项目知识点总结

## 1. 项目概述

本项目主要处理哨兵2号（Sentinel-2）卫星图像，将多波段的TIFF文件转换为RGB三通道图像，涉及遥感数据处理、图像归一化等核心技术。

---

## 2. 核心Python库

### 2.1 NumPy - 数值计算基础

```python
import numpy as np

# 数组创建和操作
arr = np.array([1, 2, 3, 4])
matrix = np.array([[1, 2], [3, 4]])

# 数据类型转换
float_arr = arr.astype(float)
int_arr = float_arr.astype(np.uint8)

# 数组形状操作
print(arr.shape)        # 获取形状
reshaped = arr.reshape(2, 2)  # 重塑形状

# 数组堆叠
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])
stacked = np.dstack((a, b, c))  # 沿第三个轴堆叠

# 数组统计
print(arr.min())        # 最小值
print(arr.max())        # 最大值
print(arr.mean())       # 平均值
```

### 2.2 Rasterio - 地理空间数据处理

```python
import rasterio

# 读取地理空间文件
with rasterio.open('satellite_image.tif') as src:
    # 读取所有波段 - 返回形状为 (波段数, 高度, 宽度)
    bands = src.read()
    
    # 读取单个波段
    band1 = src.read(1)
    
    # 获取元数据信息
    profile = src.profile
    print(f"波段数: {src.count}")
    print(f"图像尺寸: {src.width} x {src.height}")
    print(f"坐标系统: {src.crs}")
```

### 2.3 PIL (Pillow) - 图像处理

```python
from PIL import Image

# 从numpy数组创建图像
image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
image = Image.fromarray(image_array)

# 保存图像
image.save('output.png')
image.save('output.jpg', quality=95)

# 图像格式转换
image_rgb = image.convert('RGB')
```

---

## 3. 数据归一化技术

### 3.1 线性归一化概念

**归一化**是将数据缩放到特定范围的过程，常用于：
- 消除不同数据范围的影响
- 提高算法收敛速度
- 便于可视化显示

### 3.2 线性归一化公式

```python
# 基本公式: (x - min) / (max - min) * (new_max - new_min) + new_min

def linear_normalize(data, target_min=0, target_max=255):
    """
    线性归一化函数
    
    Args:
        data: 输入数据数组
        target_min: 目标最小值
        target_max: 目标最大值
    
    Returns:
        归一化后的数据
    """
    data_min = data.min()
    data_max = data.max()
    
    # 避免除零错误
    if data_max == data_min:
        return np.full_like(data, target_min)
    
    # 归一化到0-1，再缩放到目标范围
    normalized = (data - data_min) / (data_max - data_min)
    scaled = normalized * (target_max - target_min) + target_min
    
    return scaled.astype(np.uint8)

# 应用示例
original_data = np.array([100, 500, 1000, 5000, 10000])
normalized_data = linear_normalize(original_data, 0, 255)
print(f"原始数据: {original_data}")
print(f"归一化后: {normalized_data}")
```

### 3.3 其他归一化方法

```python
# Z-score标准化 (均值为0，标准差为1)
def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

# 百分位数拉伸 (去除异常值影响)
def percentile_stretch(data, low_percentile=2, high_percentile=98):
    p_low = np.percentile(data, low_percentile)
    p_high = np.percentile(data, high_percentile)
    
    # 裁剪到百分位数范围
    clipped = np.clip(data, p_low, p_high)
    
    # 线性拉伸到0-255
    return linear_normalize(clipped, 0, 255)
```

---

## 4. 遥感图像处理基础

### 4.1 多光谱图像概念

**多光谱图像**包含多个波段，每个波段记录特定波长范围的电磁辐射：

```python
# 哨兵2号主要波段
SENTINEL2_BANDS = {
    'B02': {'name': '蓝光', 'wavelength': '490nm', 'use': 'true color'},
    'B03': {'name': '绿光', 'wavelength': '560nm', 'use': 'true color'},
    'B04': {'name': '红光', 'wavelength': '665nm', 'use': 'true color'},
    'B08': {'name': '近红外', 'wavelength': '842nm', 'use': 'vegetation analysis'},
    'B12': {'name': '短波红外', 'wavelength': '2190nm', 'use': 'moisture, geology'}
}

# 波段组合应用
def create_false_color(red, nir, blue):
    """创建假彩色图像 (植被分析)"""
    return np.dstack((nir, red, blue))

def create_natural_color(red, green, blue):
    """创建真彩色图像"""
    return np.dstack((red, green, blue))
```

### 4.2 图像增强技术

```python
# 对比度拉伸
def contrast_stretch(image, gamma=1.0):
    """伽马校正进行对比度调整"""
    return np.power(image / 255.0, gamma) * 255

# 直方图均衡化
def histogram_equalization(image):
    """改善图像对比度"""
    from skimage import exposure
    return exposure.equalize_hist(image) * 255

# 滤波处理
def apply_filter(image, filter_type='gaussian'):
    """应用滤波器减少噪声"""
    from scipy import ndimage
    
    if filter_type == 'gaussian':
        return ndimage.gaussian_filter(image, sigma=1)
    elif filter_type == 'median':
        return ndimage.median_filter(image, size=3)
```

---

## 5. 数组操作高级技巧

### 5.1 多维数组索引

```python
# 3D数组操作 (波段, 高度, 宽度)
bands = np.random.randint(0, 10000, (5, 1000, 1000))

# 提取特定波段
blue_band = bands[0]      # 第1个波段
green_band = bands[1]     # 第2个波段
red_band = bands[2]       # 第3个波段

# 提取特定区域
roi = bands[:, 100:200, 100:200]  # 所有波段的感兴趣区域

# 波段重排列
rgb_bands = bands[[2, 1, 0]]      # 红绿蓝重新排序
```

### 5.2 数组广播和向量化

```python
# 广播操作 - 对所有波段应用相同操作
normalized_bands = (bands - bands.min()) / (bands.max() - bands.min())

# 向量化函数应用
def process_band(band):
    """处理单个波段"""
    return np.clip(band * 1.2, 0, 10000)

# 对所有波段应用函数
processed = np.array([process_band(band) for band in bands])

# 使用numpy的apply_along_axis
processed_alt = np.apply_along_axis(process_band, 0, bands)
```

---

## 6. 文件处理最佳实践

### 6.1 上下文管理器使用

```python
# 推荐做法 - 自动关闭文件
with rasterio.open(tif_file) as src:
    data = src.read()
    profile = src.profile
# 文件自动关闭

# 处理多个文件
def process_multiple_files(file_list):
    results = []
    for file_path in file_list:
        with rasterio.open(file_path) as src:
            data = src.read()
            # 处理数据
            results.append(data)
    return results
```

### 6.2 错误处理

```python
def safe_image_processing(tif_file):
    """安全的图像处理函数"""
    try:
        with rasterio.open(tif_file) as src:
            if src.count < 5:
                raise ValueError(f"波段数不足: {src.count}, 需要至少5个波段")
            
            bands = src.read()
            
            # 检查数据有效性
            if np.all(bands == 0):
                raise ValueError("图像数据为空")
            
            return process_bands(bands)
            
    except rasterio.errors.RasterioIOError:
        print(f"无法读取文件: {tif_file}")
        return None
    except ValueError as e:
        print(f"数据错误: {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None
```

---

## 7. 性能优化技巧

### 7.1 内存管理

```python
# 处理大型图像时的内存优化
def process_large_image_by_chunks(tif_file, chunk_size=1000):
    """分块处理大型图像"""
    with rasterio.open(tif_file) as src:
        height, width = src.height, src.width
        
        for row in range(0, height, chunk_size):
            for col in range(0, width, chunk_size):
                # 计算实际读取窗口
                window = rasterio.windows.Window(
                    col, row,
                    min(chunk_size, width - col),
                    min(chunk_size, height - row)
                )
                
                # 读取数据块
                chunk = src.read(window=window)
                
                # 处理数据块
                processed_chunk = process_chunk(chunk)
                
                # 保存或累积结果
                yield processed_chunk
```

### 7.2 并行处理

```python
from multiprocessing import Pool
import concurrent.futures

def parallel_band_processing(bands):
    """并行处理多个波段"""
    def process_single_band(band):
        return linear_normalize(band, 0, 255)
    
    # 使用进程池
    with Pool() as pool:
        results = pool.map(process_single_band, bands)
    
    return np.array(results)

# 使用线程池处理I/O密集型任务
def parallel_file_processing(file_list):
    """并行处理多个文件"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(output, file) for file in file_list]
        results = [future.result() for future in futures]
    return results
```

---

## 8. 调试和验证

### 8.1 数据质量检查

```python
def validate_image_data(image_array):
    """验证图像数据质量"""
    checks = {
        'shape_valid': len(image_array.shape) == 3,
        'has_data': not np.all(image_array == 0),
        'no_nan': not np.any(np.isnan(image_array)),
        'value_range': (image_array.min() >= 0) and (image_array.max() <= 255),
        'data_type': image_array.dtype == np.uint8
    }
    
    print("数据质量检查:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}: {passed}")
    
    return all(checks.values())

# 可视化检查
def quick_preview(image_array, title="Image Preview"):
    """快速预览图像"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image_array)
    plt.title(title)
    plt.axis('off')
    
    # 显示统计信息
    print(f"图像形状: {image_array.shape}")
    print(f"数据类型: {image_array.dtype}")
    print(f"数值范围: {image_array.min()} - {image_array.max()}")
    
    plt.show()
```

---

## 9. 常用工具函数总结

```python
# 完整的图像处理工具包
class SatelliteImageProcessor:
    """卫星图像处理工具类"""
    
    @staticmethod
    def load_image(file_path):
        """加载卫星图像"""
        with rasterio.open(file_path) as src:
            return src.read(), src.profile
    
    @staticmethod
    def normalize_bands(bands, method='linear'):
        """归一化波段数据"""
        if method == 'linear':
            return linear_normalize(bands, 0, 255)
        elif method == 'percentile':
            return percentile_stretch(bands)
    
    @staticmethod
    def create_rgb(red, green, blue):
        """创建RGB图像"""
        return np.dstack((red, green, blue)).astype(np.uint8)
    
    @staticmethod
    def save_image(image_array, output_path):
        """保存图像"""
        image = Image.fromarray(image_array)
        image.save(output_path)
        print(f"图像已保存到: {output_path}")

# 使用示例
processor = SatelliteImageProcessor()
bands, profile = processor.load_image("satellite_data.tif")
rgb_image = processor.create_rgb(bands[2], bands[1], bands[0])
processor.save_image(rgb_image, "output.png")
```
