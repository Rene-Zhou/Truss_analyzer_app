import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from OpenCV_core import merge_lines, better_cluster_endpoints

def process_image(image_path, threshold=40, min_line_length=40, max_line_gap=30, save_path=None):
    """
    处理图像并测试桁架归一化功能
    
    Args:
        image_path: 输入图像路径
        threshold: 霍夫变换阈值
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 对二值图像取反色
    edges = cv2.bitwise_not(gray)

    # 添加腐蚀操作，细化线条
    kernel = np.ones((3,3), np.uint8)
    eroded_edges = cv2.erode(edges, kernel, iterations=1)

    # 霍夫线变换检测线段
    lines = cv2.HoughLinesP(eroded_edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        print("未检测到线段")
        return
    
    # 提取线段坐标
    detected_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # OpenCV_app.py中使用line[0]获取坐标
        detected_lines.append([x1, y1, x2, y2])
    
    # 合并线段
    merged_lines = merge_lines(detected_lines, angle_threshold=3, parallel_distance_threshold=40)
    
    # 端点聚类
    clustered_lines = better_cluster_endpoints(merged_lines, threshold=30)
    
    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    for line in clustered_lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], 'r-')
    plt.show()

if __name__ == "__main__":

    process_image("test_img/test5.png")
