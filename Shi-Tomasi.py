import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_convert_image(image_path):
    """
    读取图像并转换为灰度图
    Args:
        image_path: 图像文件路径
    Returns:
        tuple: (原始图像, 灰度图像)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'未找到 {image_path}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def detect_corners(gray_img, max_corners=100, quality_level=0.1, min_distance=30):
    """
    执行Shi-Tomasi角点检测
    Args:
        gray_img: 灰度图像
        max_corners: 最大角点数
        quality_level: 质量水平
        min_distance: 最小距离
    Returns:
        corners: 检测到的角点
    """
    corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=max_corners, 
                                    qualityLevel=quality_level, 
                                    minDistance=min_distance)
    return corners.astype(int) if corners is not None else None

def reconstruct_lines(img, gray, corners, line_threshold=30, min_line_length=20):
    """
    根据角点重建线段
    Args:
        img: 原始图像
        gray: 灰度图像
        corners: 检测到的角点
        line_threshold: 线检测的阈值参数
        min_line_length: 最小线段长度
    Returns:
        result_img: 带有重建线段的图像
    """
    # 创建一个白色背景的图像
    result_img = np.ones_like(img) * 255
    
    # 使用角点信息优化线段连接
    if corners is not None:
        # 将角点坐标展平为一个列表
        corner_points = [tuple(corner.ravel()) for corner in corners]
        
        # 对每对角点直接进行连线和重叠像素判断
        for i in range(len(corner_points)):
            for j in range(i+1, len(corner_points)):
                pt1 = corner_points[i]
                pt2 = corner_points[j]
                
                # 创建两点之间的线段蒙版
                mask = np.zeros_like(gray)
                cv2.line(mask, pt1, pt2, 255, 1)
                
                # 计算线段蒙版与灰度图的重叠像素数量
                overlap = cv2.bitwise_and(gray, mask)
                overlap_count = np.count_nonzero(overlap)
                
                # 计算两点之间的距离
                dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                # 如果重叠像素数量占线段长度的比例超过阈值，则绘制这条线
                if overlap_count > dist * 0.3:  # 30% 的线段有边缘
                    cv2.line(result_img, pt1, pt2, (0, 0, 0), 2)  # 用黑色线条在白色背景上绘制
                    # 在线段端点绘制红色圆点
                    cv2.circle(result_img, pt1, 5, (0, 0, 255), -1)
                    cv2.circle(result_img, pt2, 5, (0, 0, 255), -1)
    
    return result_img

def main():
    # 读取并转换图像
    img, gray = load_and_convert_image('test_img/test4.png')
    
    # 检测角点
    corners = detect_corners(gray)
    
    # 在图像上标记角点
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    # 重建线段
    img_with_lines = reconstruct_lines(img, gray, corners)
    
    # BGR转RGB用于matplotlib显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_with_lines_rgb = cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB)

    # 显示两张图像
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Shi-Tomasi Corner detection results')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_with_lines_rgb)
    plt.title('Reconstructed lines')
    # 设置坐标轴范围为1000x1000
    plt.xlim(0, 1000)
    plt.ylim(1000, 0)  # y轴反转以匹配图像坐标系
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
