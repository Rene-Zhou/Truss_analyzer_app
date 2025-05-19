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

    # BGR转RGB用于matplotlib显示
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title('Shi-Tomasi Corner detection results')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
