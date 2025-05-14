import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from OpenCV_core import merge_lines, classify_H_truss_members, classify_V_truss_members, cluster_endpoints, normalize_truss_size

def process_image(image_path, threshold=40, min_line_length=40, max_line_gap=30, save_path=None):
    """
    处理图像并测试桁架归一化功能
    
    Args:
        image_path: 输入图像路径
        threshold: 霍夫变换阈值
        min_line_length: 最小线段长度
        max_line_gap: 最大线段间隙
        save_path: 结果保存路径，如果为None则使用默认路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊减少噪声 (与OpenCV_app.py一致)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # 自适应阈值处理 (与OpenCV_app.py一致)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 对二值图像取反色 (与OpenCV_app.py一致)
    edges = cv2.bitwise_not(thresh)
    
    # 添加腐蚀操作，细化线条 (与OpenCV_app.py一致)
    kernel = np.ones((3,3), np.uint8)
    eroded_edges = cv2.erode(edges, kernel, iterations=1)
    
    # 霍夫线变换检测线段 (与OpenCV_app.py一致的参数)
    lines = cv2.HoughLinesP(eroded_edges, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is None:
        print("未检测到线段")
        return
    
    # 提取线段坐标
    detected_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]  # OpenCV_app.py中使用line[0]获取坐标
        detected_lines.append([x1, y1, x2, y2])
    
    # 合并线段 (与OpenCV_app.py一致的参数)
    merged_lines = merge_lines(detected_lines, angle_threshold=17, parallel_distance_threshold=40)
    
    # 分类桁架构件 (与OpenCV_app.py一致的参数)
    h_truss_result = classify_H_truss_members(merged_lines, threshold=25)
    truss_classification = classify_V_truss_members(h_truss_result, angle_threshold=20)
    
    # 对桁架点进行聚类和对齐 (与OpenCV_app.py一致的参数)
    aligned_truss = cluster_endpoints(truss_classification, threshold=50)
    
    # 桁架归一化处理
    normalized_truss = normalize_truss_size(aligned_truss)
    
    # 创建结果可视化函数
    def visualize_truss(truss_dict, title, subplot_idx):
        plt.subplot(2, 3, subplot_idx)
        plt.title(title)
        
        # 绘制水平桁架构件 (红色，与OpenCV_app.py一致)
        for line in truss_dict.get("H-truss", []):
            plt.plot([line[0], line[2]], [line[1], line[3]], 'r-', linewidth=2, label='H-truss' if subplot_idx == 6 else None)
        
        # 绘制垂直桁架构件 (绿色，与OpenCV_app.py一致)
        for line in truss_dict.get("V_truss", []):
            plt.plot([line[0], line[2]], [line[1], line[3]], 'g-', linewidth=2, label='V-truss' if subplot_idx == 6 else None)
        
        # 绘制斜向桁架构件 (蓝色，与OpenCV_app.py一致)
        for line in truss_dict.get("D_truss", []):
            plt.plot([line[0], line[2]], [line[1], line[3]], 'b-', linewidth=2, label='D-truss' if subplot_idx == 6 else None)
        
        plt.axis('equal')
        plt.grid(True)
        
        # 只在归一化后的图中添加图例
        if subplot_idx == 6 and (truss_dict.get("H-truss", []) or truss_dict.get("V_truss", []) or truss_dict.get("D_truss", [])):
            # 去除重复的图例条目
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    # 创建图形并显示结果
    plt.figure(figsize=(18, 10))
    
    # 显示原始图像
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 显示处理后的边缘
    plt.subplot(2, 3, 2)
    plt.title("Processed Edges")
    plt.imshow(eroded_edges, cmap='gray')
    
    # 显示检测到的线段
    plt.subplot(2, 3, 3)
    plt.title("Detected Lines")
    line_img = np.zeros_like(img)
    for line in detected_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    
    # 显示合并后的线段
    plt.subplot(2, 3, 4)
    plt.title("Merged Lines")
    merged_img = np.zeros_like(img)
    for line in merged_lines:
        x1, y1, x2, y2 = line
        cv2.line(merged_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    plt.imshow(cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB))
    
    # 显示对齐后的桁架
    visualize_truss(aligned_truss, "Aligned Truss", 5)
    
    # 显示归一化后的桁架
    visualize_truss(normalized_truss, "Normalized Truss", 6)
    
    plt.tight_layout()
    
    # 确保test_result目录存在
    test_result_dir = "test_result"
    os.makedirs(test_result_dir, exist_ok=True)
    
    # 保存结果
    if save_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(test_result_dir, f"{base_name}_normalized_result.png")
    else:
        # 如果提供了保存路径，确保它在test_result目录内
        if not os.path.dirname(save_path):  # 如果没有提供目录部分
            save_path = os.path.join(test_result_dir, save_path)
        elif not save_path.startswith(test_result_dir):  # 如果提供的路径不在test_result目录内
            base_name = os.path.basename(save_path)
            save_path = os.path.join(test_result_dir, base_name)
    
    plt.savefig(save_path)
    print(f"结果已保存到: {save_path}")
    
    # 显示结果
    plt.show()
    
    # 打印归一化后的桁架信息
    print("\n归一化后的桁架信息:")
    print(f"水平桁架数量: {len(normalized_truss.get('H-truss', []))}")
    print(f"垂直桁架数量: {len(normalized_truss.get('V_truss', []))}")
    print(f"斜向桁架数量: {len(normalized_truss.get('D_truss', []))}")
    
    # 保存归一化后的桁架数据到文本文件
    data_save_path = f"{os.path.splitext(save_path)[0]}_data.txt"
    with open(data_save_path, 'w', encoding='utf-8') as f:
        f.write("归一化后的桁架数据:\n\n")
        
        f.write("水平桁架坐标:\n")
        for i, line in enumerate(normalized_truss.get('H-truss', [])):
            f.write(f"H{i+1}: ({line[0]},{line[1]}) - ({line[2]},{line[3]})\n")
        
        f.write("\n垂直桁架坐标:\n")
        for i, line in enumerate(normalized_truss.get('V_truss', [])):
            f.write(f"V{i+1}: ({line[0]},{line[1]}) - ({line[2]},{line[3]})\n")
        
        f.write("\n斜向桁架坐标:\n")
        for i, line in enumerate(normalized_truss.get('D_truss', [])):
            f.write(f"D{i+1}: ({line[0]},{line[1]}) - ({line[2]},{line[3]})\n")
    
    print(f"桁架数据已保存到: {data_save_path}")
    
    return normalized_truss

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='桁架结构归一化测试程序')
    parser.add_argument('--image', type=str, default='test4.jpg', help='输入图像路径')
    parser.add_argument('--threshold', type=int, default=40, help='霍夫变换阈值 (默认: 40)')
    parser.add_argument('--min_line_length', type=int, default=40, help='最小线段长度 (默认: 40)')
    parser.add_argument('--max_line_gap', type=int, default=30, help='最大线段间隙 (默认: 30)')
    parser.add_argument('--save_path', type=str, help='结果保存路径')
    
    args = parser.parse_args()
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"文件 {args.image} 不存在，请确保图像文件在正确的路径")
    else:
        # 处理图像
        normalized_truss = process_image(
            args.image, 
            threshold=args.threshold, 
            min_line_length=args.min_line_length, 
            max_line_gap=args.max_line_gap,
            save_path=args.save_path
        ) 