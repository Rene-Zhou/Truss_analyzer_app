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

def reconstruct_lines(img, gray, corners, thickness=3, min_line_length=20, max_line_length=300):
    """
    根据角点重建线段，使用改进的方法来验证线段是否存在
    Args:
        img: 原始图像
        gray: 灰度图像
        corners: 检测到的角点
    Returns:
        result_img: 带有重建线段的图像
    """
    # 创建一个白色背景的图像
    result_img = np.ones_like(img) * 255
    
    # 对灰度图像进行二值化处理
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # 对二值图像进行膨胀，使线条更粗，更容易检测到重叠
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(binary, kernel, iterations=1)
    
    # 遍历所有可能的角点对
    for i in range(len(corners)):
        for j in range(i+1, len(corners)):
            # 获取两个角点的坐标
            pt1 = tuple(corners[i].ravel())
            pt2 = tuple(corners[j].ravel())
            
            # 计算两点间距离
            distance = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
            
            # 检查距离是否在合理范围内
            if min_line_length <= distance <= max_line_length:
                # 检查这条线是否与图像中的边缘重叠
                if is_valid_line(dilated_image, pt1, pt2, 0.4):
                    # 在结果图像上绘制这条线
                    cv2.line(result_img, pt1, pt2, (0, 0, 0), thickness)
    
    return result_img

def is_valid_line(edges, pt1, pt2, threshold=0.4):
    """
    检查线段是否与边缘重合
    Args:
        edges: 边缘图像
        pt1: 第一个点坐标 (x1, y1)
        pt2: 第二个点坐标 (x2, y2)
        threshold: 重合比例阈值
    Returns:
        bool: 是否是有效线段
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # 计算线段上的点数
    length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
    if length == 0:
        return False
    
    # 在线段上采样点
    points_on_line = 0
    edge_points = 0
    
    for i in range(0, length + 1):
        # 线性插值计算线段上的点
        t = i / length if length > 0 else 0
        x = int(x1 * (1 - t) + x2 * t)
        y = int(y1 * (1 - t) + y2 * t)
        
        # 确保点在图像范围内
        if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
            points_on_line += 1
            if edges[y, x] > 0:  # 如果是边缘点
                edge_points += 1
    
    # 计算重合比例
    overlap_ratio = edge_points / points_on_line if points_on_line > 0 else 0
    
    return overlap_ratio >= threshold

def main():
    # 读取并转换图像
    img, gray = load_and_convert_image('test_img/test4.png')
    
    # 由于图像是白底黑线，先进行反色操作
    inverted_gray = cv2.bitwise_not(gray)
    
    # 对反色后的灰度图进行强力腐蚀操作
    kernel = np.ones((3, 3), np.uint8)  # 较大的核用于强腐蚀
    eroded_inverted = cv2.erode(inverted_gray, kernel, iterations=3)  # 多次迭代增强腐蚀效果
    
    # 再次反色，得到原始颜色方案下的"膨胀"效果
    eroded_gray = cv2.bitwise_not(eroded_inverted)
    
    # 检测角点
    corners = detect_corners(eroded_gray)
    
    # 将腐蚀后的灰度图转换为彩色图，以便标记彩色角点
    eroded_img = cv2.cvtColor(eroded_gray, cv2.COLOR_GRAY2BGR)
    
    # 在腐蚀后的图像上标记角点
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(eroded_img, (x, y), 5, (0, 0, 255), -1)

    # 重建线段
    img_with_lines = reconstruct_lines(img, gray, corners)
    
    # BGR转RGB用于matplotlib显示
    eroded_img_rgb = cv2.cvtColor(eroded_img, cv2.COLOR_BGR2RGB)
    img_with_lines_rgb = cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB)

    # 显示两张图像
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(eroded_img_rgb)
    plt.title('Shi-Tomasi Corner detection on thickened lines')
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
