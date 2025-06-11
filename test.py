#!/usr/bin/env python3  
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
import argparse

import cv2
import tqdm
import random
import numpy as np

from PIL import Image
from loguru import logger


# 查看图片通道和分辨率
# python -c "from PIL import Image; image = Image.open('1039.jpeg');  print(f' image.mode: {image.mode}, size:{image.size}')"

def coord_test():
    '''
        坐标操作：
    '''
    raw_points = [1851.0, 692.0, 1421.0, 620.0, 930.0, 560.0, 760.0, 240.0]
    
    # 1、从raw_points 中抽出坐标对 (x,y)
    # [(1851.0, 692.0), (1421.0, 620.0), (930.0, 560.0), (760.0, 240.0)]
    # 0开始，每次移动2个单位，取出 X；1开始每次移动一个单位取出 Y
    points = list(zip(raw_points[::2], raw_points[1::2]))

    # !!! 这样展开会报错
    # raw_points2 = [p[0], p[1] for p in points]
    
    # 套上元组或者list可以修复报错, 但与想要的平面展开结果不一致
    raw_points2 = [(p[0], p[1]) for p in points]
    logger.info(f'raw_points2: {raw_points2}')

    # 2、坐标对 再次展开
    raw_points3 = [coord for p in points for coord in p]
    logger.info(f'raw_points3: {raw_points3}')
    
    # 3、坐标根据条件过滤 -> [(1851.0, 692.0), (1421.0, 620.0), (930.0, 560.0)]
    valid_points = [p for p in points if p[1] > 320]
    
    # 4、坐标按照条件排序
    # 按照Y值降序排列，取出排序后的 Y序列
    sorted_points = sorted(valid_points, key=lambda p: -p[1])
    points_ys = [p[1] for p in sorted_points]

    # 5、按照 Y 坐标插值
    # 生成插值点, ys 是 numpy.ndarray 类型
    start_y = 600
    end_y = 1079
    R = 32
    
    ys = np.linspace(end_y, start_y, R)
    logger.info(f'ys: {ys}, type{type(ys)}')

    # numpy.ndarray 转 list
    ys_list = ys.tolist()
    logger.info(f'ys_list: {ys_list}, type{type(ys_list)}') 

def pathlib_test():
    '''
    1、from pathlib import Path 与 os.path 对比：
    
        | `os.path` 函数               | `pathlib.Path` 等价操作                  | 示例与说明                                                                 |
        |-----------------------------|------------------------------------------|---------------------------------------------------------------------------|
        | `os.path.abspath(path)`      | `Path(path).resolve()`                   | `Path("data/file.txt").resolve()` → 返回绝对路径                          |
        | `os.path.basename(path)`     | `Path(path).name`                        | `Path("/data/file.txt").name` → `"file.txt"`                              |
        | `os.path.dirname(path)`      | `Path(path).parent`                      | `Path("/data/file.txt").parent` → `Path("/data")`                         |
        | `os.path.exists(path)`       | `Path(path).exists()`                    | `Path("file.txt").exists()` → 返回 `True`/`False`                        |
        | `os.path.join(a, b, c...)`   | `Path(a) / b / c...`                     | `Path("data") / "images"` → `Path("data/images")`                         |
        | `os.path.splitext(path)`     | `Path(path).stem` + `Path(path).suffix`  | `Path("file.tar.gz").stem` → `"file.tar"`<br>`Path().suffix` → `".gz"`     |
        | `os.path.getsize(path)`      | `Path(path).stat().st_size`              | `Path("file.txt").stat().st_size` → 文件字节大小                          |
        | `os.path.isdir(path)`        | `Path(path).is_dir()`                    | `Path("data").is_dir()` → 检查是否为目录                                  |
        | `os.path.isfile(path)`       | `Path(path).is_file()`                   | `Path("file.txt").is_file()` → 检查是否为文件                             |
        | `os.path.normpath(path)`     | `Path(path)`（自动规范化）                | `Path("data//docs/../file.txt")` → 自动转为 `data/file.txt`               |
        | `os.path.relpath(path, start)` | `Path(path).relative_to(start)`         | `Path("/data/file.txt").relative_to("/")` → `Path("data/file.txt")`       |
        
    2、文件读写
        传统写法
        with open("data.txt", "r") as f:
            content = f.read()

        pathlib 写法
        content = Path("data.txt").read_text(encoding="utf-8")

    3、查找所有 .txt 文件
        txt_files = list(Path("data").rglob("*.txt"))

    4、路径属性操作
        path = Path("/data/images/file.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建父目录
        path.write_bytes(b"binary_data")               # 直接写入二进制数据

    5. Path.glob(pattern)
        | 通配符      | 说明                                  | 示例                          | 匹配案例                    |
        |-------------|--------------------------------------|-------------------------------|-----------------------------|
        | `*`         | 匹配**0个或多个非路径分隔符字符**     | `*.txt`                       | `file.txt`, `data.txt`      |
        | `?`         | 匹配**1个任意字符**（非路径分隔符）   | `image?.jpg`                  | `image1.jpg`, `imageA.jpg`  |
        | `**`        | 递归匹配**所有子目录**               | `**/*.py` + `recursive=True`  | `src/a.py`, `src/utils/b.py`|
        | `[abc]`     | 匹配**括号内的任意字符**              | `file[123].txt`               | `file1.txt`, `file2.txt`    |
        | `[a-z]`     | 匹配**指定范围的字符**（字母/数字）   | `log_[a-z].txt`               | `log_a.txt`, `log_b.txt`    |
        | `[!abc]`    | 匹配**不在括号内的字符**              | `file[!x].csv`                | `fileA.csv`, `file1.csv`    |
        | `{a,b,c}`   | 匹配**逗号分隔的任一模式**            | `*.{jpg,png}`                 | `cat.jpg`, `dog.png`        |


        递归匹配：
            glob模块中使用 ** 递归时必须显式启用 recursive=True，而pathlib 模块使用**就可以，不需要指定 recursive=True
    '''
    pass


def mask_test():
    '''
        cv2.bitwise_or()
            功能：对两个图像的每个像素执行按位或运算（只要有一个像素为1，结果就为1）
            应用场景：在 Mask 区域叠加两张图像
            实例：合并两个圆形到一张图中
            
            dst = cv2.bitwise_or(src1, src2[, dst[, mask]])
    '''

    # 创建黑色背景
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)

    # 在左侧画白色圆
    left_circle = cv2.circle(canvas.copy(), (150, 150), 100, (255, 255, 255), -1)

    # 在右侧画白色圆
    right_circle = cv2.circle(canvas.copy(), (250, 150), 100, (255, 255, 255), -1)

    # 合并两个圆
    merged = cv2.bitwise_or(left_circle, right_circle)

    cv2.imshow('Merged Circles', merged)
    cv2.waitKey(0)
    
    '''
        cv2.bitwise_and()
            功能：对两个图像的每个像素执行按位与运算（两个像素都为1时结果才为1）
            应用场景：提取 Mask 区域
            实例：从彩色图像中提取红色物体

            dst = cv2.bitwise_and(src1, src2[, dst[, mask]])
    '''
    # 读取图像
    image = cv2.imread('apple.jpg')

    # 将BGR转为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色范围 (低值和高值)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # 红色的第二种范围 (在HSV色轮中红色跨0度)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # 组合两个红色掩码
    mask = cv2.bitwise_or(mask1, mask2)

    # 应用掩码提取红色区域
    result = cv2.bitwise_and(image, image, mask=mask)

    cv2.imshow('Red Objects', result)
    cv2.waitKey(0)

    '''
        cv2.bitwise_not()
            功能：反转图像的每个像素值（0变255，255变0）
            应用场景：在 Mask 区域内反色图片
            实例：创建反转掩码并应用于图像
            
            dst = cv2.bitwise_not(src[, dst[, mask]])
    '''

    mask = np.zeros((300, 400), dtype=np.uint8)
    mask = cv2.rectangle(mask, (100, 50), (300, 250), 255, -1)

    # 反转掩码
    inverted_mask = cv2.bitwise_not(mask)

    # 应用到图像
    image = np.full((300, 400, 3), (0, 150, 255), dtype=np.uint8)  # 橙色背景
    result = cv2.bitwise_and(image, image, mask=inverted_mask)

    cv2.imshow('Inverted Mask', result)
    cv2.waitKey(0)

    '''
        cv2.bitwise_xor() 
            功能：对两个图像的每个像素、逐位执行异或运算（相同为0，不同为1）
            应用场景：在 Mask 区域内 XOR 操作
            实例：检测两个图像的差异        
        
            dst = cv2.bitwise_xor(src1, src2[, dst[, mask]])
    '''
    
    image1 = cv2.imread('scene1.jpg', cv2.IMREAD_GRAYSCALE) # 以单通道灰度图的方式读取图片
    image2 = cv2.imread('scene2.jpg', cv2.IMREAD_GRAYSCALE)

    # 计算差异
    diff = cv2.bitwise_xor(image1, image2)

    # 增强差异可视化
    diff_enhanced = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

    # 在原始图像上标记差异
    image2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    image2_color[diff_enhanced == 255] = [0, 0, 255]  # 将差异标为红色

    cv2.imshow('Differences', image2_color)
    cv2.waitKey(0)

def dict_test():
    # 1、dict与 {} 创建字典是等价的，dict创建可以参数化，动态创建字典时更灵活
    d1 = {}
    d2 = dict()
  
    # 使用关键字参数创建
    d1 = dict(first_name="John", last_name="Doe", age=30)
    # {'first_name': 'John', 'last_name': 'Doe', 'age': 30}
  
    # 从键值対序列中创建
    d2 = dict([("a", 1), ("b", 2), ("c", 3)])
    # d2: {'a': 1, 'b': 2, 'c': 3}
    
    # 从两个序列创建
    keys = ['x', 'y', 'z']
    values = [10, 20, 30]
    d3 = dict(zip(keys, values))
    # d3: {'x': 10, 'y': 20, 'z': 30}

    # 2、可能遇到的错误
    # 报错：KeyError: 'test'
    hist = {}
    hist["test"] +=1        

    # 修复1: 使用 defaultdict(int) 自动添加默认 key 为 0
    from collections import defaultdict
    hist = defaultdict(int)
    hist["test"] += 1       # 没有问题，defaultdict(int) 会自动添加 key, 初始值为0

    # 修复2: 使用 get() 方法，如果 key 不存在则返回默认值
    count = hist.get("test", 0)
    hist["test"] = count + 1
    
def pil_draw_test():
    from PIL import Image, ImageDraw

    # 创建新图像 (1280x720, RGB模式, 白色背景)
    width, height = 1280, 720
    img = Image.new('RGB', (width, height), color='white')

    # 创建绘图对象
    draw = ImageDraw.Draw(img)

    # 1. 绘制点 - 创建星星效果
    for i in range(100):
        # 随机位置和大小
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(1, 3)
        
        # 随机颜色 - 偏暖色调
        color = (random.randint(200, 255), random.randint(150, 200), random.randint(50, 150))
        
        # 绘制矩形表示点
        draw.rectangle([x, y, x+size, y+size], fill=color)

    # 2. 绘制线 - 创建山脉效果
    for i in range(10):
        # 起始点（左边随机高度）
        start_y = random.randint(height//2, height - 50)
        
        # 生成起伏的点
        points = []
        for x in range(0, width, 30):
            variation = random.randint(-20, 20)
            points.append((x, start_y + variation))
            start_y += variation
        
        # 绘制山脉线
        draw.line(points, fill=(150, 75, 0), width=3, joint="curve")

    # 3. 绘制多边形 - 创建房屋
    # 房屋主体
    house_body = [
        (300, 400),  # 左下角
        (300, 250),  # 左上角
        (500, 250),  # 右上角
        (500, 400)   # 右下角
    ]
    draw.polygon(house_body, fill=(200, 150, 100), outline="black")

    # 房屋屋顶
    roof = [
        (275, 250),  # 左延伸
        (525, 250),  # 右延伸
        (400, 175)   # 屋顶顶点
    ]
    draw.polygon(roof, fill=(180, 50, 50), outline="black")

    # 房屋门
    door = [
        (375, 400),  # 左下角
        (375, 325),  # 左上角
        (425, 325),  # 右上角
        (425, 400)   # 右下角
    ]
    draw.polygon(door, fill=(100, 50, 25), outline="black")

    # 房屋窗户
    window = [
        (325, 285),  # 左下角
        (325, 315),  # 左上角
        (375, 315),  # 右上角
        (375, 285)   # 右下角
    ]
    draw.polygon(window, fill=(100, 200, 255), outline="black")
    draw.rectangle([340, 290, 360, 310], fill="black")  # 窗户横梁

    # 绘制标题文字
    draw.text((width//2 - 150, 50), "PIL 绘图示例", 
            fill=(30, 30, 150), 
            font_size=40,
            font=None)  # 使用默认字体

    # 保存图片
    img.save("pil_drawing_example.png")
    img.show()  # 自动打开图片查看

    print("图片已保存为 'pil_drawing_example.png'")

def pil_img_test():
    from PIL import Image
    
    '''
        image.mode:    
            模式	    描述	           位深度	通道	    常见用途
            '1'     1位像素（黑白二值）	      1位	 1	     黑白扫描文档、简单图形
            'L'	    灰度图（Luminance）	    8位	    1	    黑白照片、灰度处理
            'P'	    8位调色板索引	        8位	    1       (带256色调色板)	限制颜色的图像（如GIF）
            'RGB'	真彩色（3×8位）	        24位	3	    彩色照片、网页图片
            'RGBA'	带Alpha通道的真彩色	    32位	 4	    需要透明背景的图像
    '''

    img_path = "/home/lc/work/data/satellite_img/test_data/Mask/image.png"
    img = Image.open(img_path)    

    # 1、打印图像的基本信息
    print("=== 图像基本信息 ===")
    print(f"图像路径: {img_path}")
    print(f"文件名: {os.path.basename(img_path)}")
    print(f"图像格式: {img.format}")  # PNG, JPEG, BMP等
    print(f"图像模式: {img.mode}")    # RGB, RGBA, L等
    print(f"图像尺寸: {img.size} (宽 x 高)")
    print(f"宽度: {img.width} 像素")
    print(f"高度: {img.height} 像素")

    # 查看单通道灰度图像素分布
    if img.mode == "L":
        unique_values, counts = np.unique(np.array(img), return_counts=True)
        print("单通道像素分布:", dict(zip(unique_values, counts)))

    # 图像类型转换
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 裁剪图片
    crpped = img.crop((50, 50, 50 + 256, 50 + 256)) # [左上右下]
    
    # 粘贴图片    
    big_image = Image.new('RGB', (1280, 720)) # 创建空白图片
    big_image.paste(crpped, (0, 256)) #  [粘贴的起始位置]

    # 保存图片
    big_image.save("big_image.png")

def np_test(): 
    import numpy as np
    
    # 1、创建np.array    
    small_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    
    # 打印基础信息
    print(f"\n=== small_arr 详细信息 ====")
    print(f"维度: {small_arr.ndim}D")       # 行+列: 2个维度, 训练模型使用的：B+C+H+W 4个维度
    print(f"形状: {small_arr.shape}")       # 形状：(2, 3)
    print(f"数据类型: {small_arr.dtype}")    # 类型: int32
    print(f"元素总数: {small_arr.size:,}")   # :-格式说明符 ,-表示千位符
    print(f"内存占用: {small_arr.nbytes:,} 字节 ({small_arr.nbytes/1024:.2f} KB)") # :.2f 保留2位小数
    
    print(f"步长 (strides): {small_arr.strides}") # 步长:(12, 4)
    print(f"内存布局: {small_arr.flags}")

    # np 创建图片   
    img = np.zeros((720, 1280), dtype=np.uint8)     # 创建单通道图片(2维) - 'L'    
    img = np.zeros((720, 1280, 3), dtype=np.uint8)  # 创建3通道图片(3维) - 'RGB'
    img = np.zeros_like(img, dtype=np.uint8)
    
    # np 创建随机数组 
    # flat: 返回迭代器，横向迭代器（方向和内存的排列方式有关）
    arr = np.arange(9).reshape(3, 3)  # 三维数组
    # logger.info(f'arr.flat:{arr.flat[-5:].tolist()}')
    
    # np.random.rand(row, col): 随机创建 row 行 col 列的 2维数组，数组中的元素是 [0, 1) 区间内均匀分布
    large_arr = np.random.rand(1000, 1000)  
    
    # randint(low, high, size): 随机生成size行列的数组, 数组中的元素是属于[low, high)范围的随机整数
    # uniform(low, high, size): 随机生成size行列的数组, 数组中的原属是属于[low, high]范围的随机浮点数
    
    int_arr = np.random.randint(5, 10, size=(2, 3))
    float_arr = np.random.uniform(5, 10, size = (2, 3))
    
    # np.random.choice(arr, size, replace=True, p): 从arr数组中随机抽取size行列的元素, replace=True（返回的数组有重复元素）, p指取arr每个元素的概率
    bool_arr = np.random.choice([True, False], size=(3, 3))
    
    img_files = ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]
    val_files = np.random.choice(img_files, size=2, replace=False) 

    # 2、np 多维数组切片
    arr = np.arange(25).reshape(5, 5)

    # 2.1、基本用法： [起始下标:结束下标:步长]
    print("\n第2-3行，第1-3列:")
    print(arr[1:3, 0:3])

    # 每隔一行取数据
    print("\n每隔一行取数据:")
    print(arr[::2, :])

    # 每隔一列取数据
    print("\n步长为2的列:")
    print(arr[:, ::2])

    # 逆序排列
    print("\n逆序排列:")
    print(arr[::-1, ::-1])

    # 2.2、整数索引和布尔索引
    # 整数索引：选择特定行
    print("\n选择第0、2、4行:")
    print(arr[[0, 2, 4], :])

    # 布尔索引
    mask = arr > 10
    print("\n布尔索引（大于10的元素）:")
    print(mask)
    print("\n使用布尔索引获取元素:")
    print(arr[mask])

    # 2.3、多维切片
    arr_3d = np.arange(24).reshape(2, 3, 4)
    print("\n3维数组:")
    print(arr_3d)

    # 切片3维数组
    print("\n第一维的第一个元素:")
    print(arr_3d[0])

    print("\n第一维的第一个元素，第二维的最后两个元素，第三维的所有元素:")
    print(arr_3d[0, 1:, :])

    print("\n所有维度的所有元素，但第三维每隔两个取一个:")
    print(arr_3d[:, :, ::2])

    # 2.4、典型场景
    # 假设有一个样本特征矩阵 (样本数, 特征数)
    X = np.random.rand(1000, 20)

    # 划分训练集和测试集
    X_train = X[:800] # 省略了列的冒号:, 等价与：X_train = X[:800, :]
    X_test = X[800:] # 省略了列的冒号:, 等价与：X_train = X[800:, :]

    # 选择特定的特征
    selected_features = X[:, [0, 3, 5, 8]]

    # 布尔索引 - 选择标签为1的样本
    labels = np.random.randint(0, 2, 1000)
    class1_samples = X[labels == 1]

    '''
    3、堆叠测试-(np.stack, np.concatenate, np.vstack, np.hstack)
        函数	            核心逻辑	                是否要求形状一致	            示例
        np.stack	    沿新轴堆叠，创建更高维度	    ✅ 必须严格一致	            (2,3)+(2,3) → (2,2,3)
        np.concatenate	沿现有轴拼接，不新增维度	    指定轴以外的其他轴必须一致      (2,3)+(2,4) → axis=1 → (2,7)
        np.vstack	    垂直堆叠（沿第0轴）	            列数需一致	                (2,3)+(1,3) → (3,3)
        np.hstack	    水平堆叠（沿第1轴）	            行数需一致	                (2,3)+(2,1) → (2,4)    
    '''

    a = np.array([[1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3]]) # shape = (3, 3)

    b = np.array([[10, 20, 30],
                  [10, 20, 30],
                  [10, 20, 30]]) # shape = (3, 3)

    # 3.1、np.stack((a, b), axis=0)
    c = np.stack((a, b), axis=0) # shape = (2, 3, 3)
    
    '''
        当axis = 0时, 相当于(1, 3, 3) + (1, 3, 3) 堆叠成: (2, 3, 3)
        a = (1, 3, 3)
        [
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]
            ]
        ]
        b = (1, 3, 3)
        [
            [
                [10, 20, 30],
                [10, 20, 30],
                [10, 20, 30]
            ]
        ]
        
        c = (2, 3, 3)
        [
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]
            ],
            [
                [10, 20, 30],
                [10, 20, 30],
                [10, 20, 30]
            ]
        ]
    '''

    c = np.stack((a, b), axis=1) # shape = (3, 2, 3)
    '''
        当axis = 1时, 相当于(3, 1, 3) + (3, 1, 3) 堆叠成: (3, 2, 3)
        
        a = (3, 1, 3)
        [
            [[1, 2, 3]],
            [[1, 2, 3]],
            [[1, 2, 3]]
        ]
        b = (3, 1, 3)
        [
            [[10, 20, 30]],
            [[10, 20, 30]],
            [[10, 20, 30]]
        ]
        c = (3, 2, 3)
        [
            [[1, 2, 3], [10, 20, 30]],
            [[1, 2, 3], [10, 20, 30]],
            [[1, 2, 3], [10, 20, 30]]
        ]
    '''

    c = np.stack((a, b), axis=2) # shape = (3, 3, 2)
    '''
        当axis = 2时, 相当于(3, 3, 1) + (3, 3, 1) 堆叠成: (3, 3, 2)
        
        a = (3, 3., 1)
        [
            [[1], [2], [3]],
            [[1], [2], [3]],
            [[1], [2], [3]],
        ]
        b = (3, 3, 1)
        [
            [[10, 20, 30]],
            [[10, 20, 30]],
            [[10, 20, 30]]
        ]
        c = (3, 3, 2)
        [
            [[1, 10], [2, 20], [3, 30]],
            [[1, 10], [2, 20], [3, 30]],
            [[1, 10], [2, 20], [3, 30]],
        ]
    '''

    # 3.1、np.concatenate((a, b), axis=0)
    a = np.arange(18).reshape(2, 3, 3)
    b = np.arange(12).reshape(2, 2, 3)
   
    c = np.concatenate((a, b), axis=1) # 1轴尺度可以不一样，其他轴轴尺度必须相同
    # logger.info(f"a.shape: {a.shape}, b.shape: {b.shape}, c.shape: {c.shape}")

    a = a.reshape(3, 2, 3)
    c = np.vstack((a, b)) # 相当于 c = np.concatenate((a, b), axis=0), 第0个轴可以不一样，其他必须一样
    
    a = a.reshape(2, 3, 3)
    b = b.reshape(2, 2, 3)
    c = np.hstack((a, b)) # 相当于 c = np.concatenate((a, b), axis=1), 第1个轴可以不一样，其他必须一样

    # 4、np.pad 填充
    arr = np.array([[1, 2], 
                    [3, 4]])

    # 各方向填充1个元素(默认值0)
    arr_pad = np.pad(arr, 1)
    # [[0 0 0 0]
    #  [0 1 2 0]
    #  [0 3 4 0]
    #  [0 0 0 0]]

    # 第0维填充：前填充0单位元素，后填充2单位元素 第1维填充：前填充1单位元素，后填充3单位元素
    arr_pad = np.pad(arr, ((0, 2), (1, 3)), constant_values=0)
    # [[0 1 2 0 0 0]
    #  [0 3 4 0 0 0]
    #  [0 0 0 0 0 0]
    #  [0 0 0 0 0 0]]

    # 5、np.where 条件选择模式
    
    # 5.1、np.where(condition): 返回满足条件的索引，元组，每个元组里是array

    # 在一维数组中查找
    arr = np.array([3, 1, 4, 1, 5, 9, 2])
    # logger.info(f'type(arr):{type(arr)}, arr: {arr}')
    
    indices = np.where(arr > 3)
    logger.info(f'indices: {indices}') 

    # 在二维数组中查找, 返回一个 tuple
    matrix = np.array([[1, 2, 5],
                    [4, 0, 6],
                    [3, 9, 7]])

    indices = np.where(matrix >= 5)
    # indices： ( [0, 1, 2, 2], [2, 2, 1, 2] )

    logger.info(f'matrix.ndim: {matrix.ndim}')
    logger.info(f"满足条件的值:{matrix[indices]}")  # [5, 6, 9, 7]

    # 查找非零元素位置
    arr = np.array([[0, 1, 0], [2, 0, 3]])
    # non_zero_indices = np.where(arr != 0)
    rows, cols = np.where(arr != 0)
    logger.info(f"非零元素位置: {list(zip(rows, cols))}") 
    # [(0, 1), (1, 0), (1, 2)]

    # 5.2、np.where(condition, x, y): 条件满足的元素替换为x，否则替换为y
    # 使用标量
    arr = np.random.randint(0, 10, (5,))
    print("原数组:", arr)  # 如 [3, 8, 0, 6, 2]
    result = np.where(arr > 5, -1, arr)
    print("替换大于5的值为-1:", result)  # [3, -1, 0, -1, 2]

    # 使用数组
    x = np.array([10, 20, 30, 40, 50])
    y = np.array([1, 2, 3, 4, 5])
    cond = np.array([True, False, True, False, True])
    result = np.where(cond, x, y)
    print("条件选择:", result)  # [10, 2, 30, 4, 50]

    # 处理多维数组
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    result = np.where(matrix % 2 == 0, '偶', '奇')
    print("奇偶判断:\n", result)
    """
    [['奇' '偶']
    ['奇' '偶']
    ['奇' '偶']]
    """

    # 广播示例, [:, np.newaxis] 转为列向量， 等价于 cond.reshpale(-1, 1)
    cond = np.array([True, False, True])
    cond = cond[:, np.newaxis]
    """
    cond: [[True],
           [False],
           [True]]
    """
    result = np.where(cond[:, np.newaxis], [[10, 20]], [[0, 0]])
    print("广播条件选择:\n", result)
    """
    [[10 20]
    [ 0  0]
    [10 20]]
    """
    
    # 5.3、典型场景

    # 分段函数计算
    x = np.linspace(-5, 5, 11)
    y = np.where(x < -1, 0,
                np.where(x < 1, x**2,
                        np.where(x < 3, x-1, 2)))
    # print("x:", x)
    # print("y:", y)
    """
    x: [-5. -4. -3. -2. -1.  0.  1.  2.  3.  4.  5.]
    y: [ 0.  0.  0.  0.  0.  0.  1.  1.  2.  2.  2.]
    """
    
    # 机器学习编码    
    y = np.array([2, 0, 1, 2, 1, 0])
    num_classes = 3
    
    cond = np.arange(num_classes) == y[:, np.newaxis]
    '''
    np.arange(num_classes) = [0 1 2]
    y[:, np.newaxis] = [[2]
                        [0]
                        [1]
                        [2]
                        [1]
                        [0]]
    cond:
        [[False False  True]
            [ True False False]
            [False  True False]
            [False False  True]
            [False  True False]
            [ True False False]]
    '''
    y_onehot = np.where(
        cond, 
        1, 
        0
    )
    print("标签:\n", y)
    print("One-hot编码:\n", y_onehot)
    """
    标签: [2 0 1 2 1 0]
    One-hot编码:
    [[0 0 1]
    [1 0 0]
    [0 1 0]
    [0 0 1]
    [0 1 0]
    [1 0 0]]
    """
    
    # 不必要使用np.where
    arr = np.arange(5)
    # 反例
    result = np.where(arr < 3, arr, np.sin(arr))

    # 使用向量化
    result = np.copy(arr)
    mask = arr >= 3
    result[mask] = np.sin(arr[mask])
    

if __name__ == '__main__':    
    # coord_test()
    # pathlib_test()
    # mask_test()
    # dict_test()
    
    # pil_img_test()
    # pil_draw_test()

    np_test()

    # # 读每一行文件，以空格为分割符解析
    # with open("pred.txt", 'r') as f:
    #     for line_idx, line in enumerate(f):
    #         parts = line.strip().split()
    #         if len(parts) < 38:
    #             continue
            
    #         front_cls = int(parts[0])
    #         rear_cls = int(parts[1])
    #         x_coords = list(map(float, parts[6:38]))
            
    # # cv2.polylines 画线, list of numpy.ndarray 或 numpy.ndarray, 
    # gt = dict()
    # points = [(x, y) for x, y in zip(gt['sampled_x'], gt['sampled_y']) if x != -1]
    # if len(points) > 1:
    #     # points : [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] => np.array(points, np.int32) => [np.array(points, np.int32)]
    #     cv2.polylines(image.copy(), [np.array(points, np.int32)], False, (0, 0, 255), 4)
    
    # # cv2.circle 画点，point(tuple)
    # for p in points:
    #     # tuple
    #     cv2.circle(image.copy(), tuple(p), 5, (255, 0, 0), -1)

    # # 文件过滤，只保留json文件
    # file_names = os.listdir(os.getcwd())
    # file_names = [name for name in file_names if name.endswith('.json')]
