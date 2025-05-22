import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

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
        lines: 线段的坐标列表，每个元素为 ((x1, y1), (x2, y2))
    """
    # 用于存储有效线段的列表
    lines = []
    
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
                    # 将有效线段添加到列表中
                    lines.append((pt1, pt2))
    
    return lines

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

def find_h_truss_members(lines):
    """
    寻找桁架结构中的水平结构
    
    Args:
        lines: 线段列表，每个元素为 ((x1, y1), (x2, y2))
        
    Returns:
        h_list: 水平桁架线段列表
    """
    h_list = []
    endpoint_tolerance = 5  # 端点匹配的容差，单位是像素
    angle_tolerance = 30    # 角度容差，单位是度
    
    for i in range(len(lines)):
        pt1_i, pt2_i = lines[i]
        
        # 计算线段i的斜率和角度
        x1_i, y1_i = pt1_i
        x2_i, y2_i = pt2_i
        
        # 避免除以零
        if x2_i - x1_i == 0:
            angle_i = 90  # 垂直线
        else:
            slope_i = (y2_i - y1_i) / (x2_i - x1_i)
            angle_i = math.degrees(math.atan(slope_i))
        
        # 标记是否找到了至少一条满足条件的线段j
        found_matching_line = False
        
        for j in range(len(lines)):
            if i == j:  # 跳过自身
                continue
                
            pt1_j, pt2_j = lines[j]
            x1_j, y1_j = pt1_j
            x2_j, y2_j = pt2_j
            
            # 计算线段j的斜率和角度
            if x2_j - x1_j == 0:
                angle_j = 90  # 垂直线
            else:
                slope_j = (y2_j - y1_j) / (x2_j - x1_j)
                angle_j = math.degrees(math.atan(slope_j))
            
            # 计算角度差的绝对值
            angle_diff = abs(angle_i - angle_j)
            
            # 检查角度差是否在容差范围内
            if angle_diff <= angle_tolerance:
                # 检查是否共用一个端点
                # 检查i的端点1与j的端点1
                if (abs(x1_i - x1_j) <= endpoint_tolerance and 
                    abs(y1_i - y1_j) <= endpoint_tolerance):
                    found_matching_line = True
                    if lines[i] not in h_list:
                        h_list.append(lines[i])
                    if lines[j] not in h_list:
                        h_list.append(lines[j])
                
                # 检查i的端点1与j的端点2
                elif (abs(x1_i - x2_j) <= endpoint_tolerance and 
                      abs(y1_i - y2_j) <= endpoint_tolerance):
                    found_matching_line = True
                    if lines[i] not in h_list:
                        h_list.append(lines[i])
                    if lines[j] not in h_list:
                        h_list.append(lines[j])
                
                # 检查i的端点2与j的端点1
                elif (abs(x2_i - x1_j) <= endpoint_tolerance and 
                      abs(y2_i - y1_j) <= endpoint_tolerance):
                    found_matching_line = True
                    if lines[i] not in h_list:
                        h_list.append(lines[i])
                    if lines[j] not in h_list:
                        h_list.append(lines[j])
                
                # 检查i的端点2与j的端点2
                elif (abs(x2_i - x2_j) <= endpoint_tolerance and 
                      abs(y2_i - y2_j) <= endpoint_tolerance):
                    found_matching_line = True
                    if lines[i] not in h_list:
                        h_list.append(lines[i])
                    if lines[j] not in h_list:
                        h_list.append(lines[j])
        
    return h_list

def group_h_truss_members(h_lines):
    """
    对水平桁架构件进行分组
    
    Args:
        h_lines: 水平桁架线段列表
        
    Returns:
        dict: 包含'h_truss_top'和'h_truss_bottom'两个键的字典，值为相应的线段列表
    """
    if not h_lines:
        return {'h_truss_top': [], 'h_truss_bottom': []}
    
    # 对线段进行标准化，使每个线段的左端点x坐标小于右端点x坐标
    normalized_lines = []
    for line in h_lines:
        pt1, pt2 = line
        x1, y1 = pt1
        x2, y2 = pt2
        if x1 <= x2:
            normalized_lines.append(((x1, y1), (x2, y2)))
        else:
            normalized_lines.append(((x2, y2), (x1, y1)))
    
    # 设置容差
    endpoint_tolerance = 5  # 端点匹配的容差，单位是像素
    
    # 使用邻接表构建线段之间的连接关系
    connected_lines = {}
    for i, line1 in enumerate(normalized_lines):
        connected_lines[i] = []
        for j, line2 in enumerate(normalized_lines):
            if i == j:  # 跳过自身
                continue
            
            # 提取两条线段的端点
            pt1_1, pt2_1 = line1
            pt1_2, pt2_2 = line2
            
            # 检查任意两个端点是否在容差范围内
            if ((abs(pt1_1[0] - pt1_2[0]) <= endpoint_tolerance and abs(pt1_1[1] - pt1_2[1]) <= endpoint_tolerance) or
                (abs(pt1_1[0] - pt2_2[0]) <= endpoint_tolerance and abs(pt1_1[1] - pt2_2[1]) <= endpoint_tolerance) or
                (abs(pt2_1[0] - pt1_2[0]) <= endpoint_tolerance and abs(pt2_1[1] - pt1_2[1]) <= endpoint_tolerance) or
                (abs(pt2_1[0] - pt2_2[0]) <= endpoint_tolerance and abs(pt2_1[1] - pt2_2[1]) <= endpoint_tolerance)):
                connected_lines[i].append(j)
    
    # 使用BFS算法对线段进行分组
    visited = [False] * len(normalized_lines)
    groups = []
    
    for i in range(len(normalized_lines)):
        if not visited[i]:
            # 开始一个新组
            current_group = []
            queue = [i]
            visited[i] = True
            
            # BFS搜索连接的线段
            while queue:
                current = queue.pop(0)
                current_group.append(normalized_lines[current])
                
                # 查找与当前线段连接的所有线段
                for neighbor in connected_lines[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            # 添加当前组到总分组列表
            groups.append(current_group)
    
    # 检查分组结果
    if len(groups) != 2:
        print(f"警告: 检测到 {len(groups)} 组水平桁架，预期为2组")
    
    # 计算每组桁架的y坐标均值
    group_y_means = []
    for group in groups:
        y_values = []
        for line in group:
            pt1, pt2 = line
            y_values.append(pt1[1])
            y_values.append(pt2[1])
        group_y_means.append(sum(y_values) / len(y_values) if y_values else 0)
    
    # 根据y均值命名桁架组
    named_groups = {'h_truss_top': [], 'h_truss_bottom': []}
    
    if len(groups) == 0:
        return named_groups
    elif len(groups) == 1:
        named_groups['h_truss_bottom'] = groups[0]
    else:
        # 在OpenCV中，y坐标从上到下增加，所以较小的y值对应较上方的桁架
        if group_y_means[0] <= group_y_means[1]:
            named_groups['h_truss_top'] = groups[0]
            named_groups['h_truss_bottom'] = groups[1]
        else:
            named_groups['h_truss_top'] = groups[1]
            named_groups['h_truss_bottom'] = groups[0]
    
    return named_groups

def move_truss(lines, h_truss_groups):
    """
    移动整个桁架结构，使最左下角点位于原点，并将y坐标转换为笛卡尔坐标系（从下到上增加）
    
    Args:
        lines: 所有线段的列表
        h_truss_groups: 水平桁架分组字典
        
    Returns:
        tuple: (移动后的所有线段列表, 移动后的水平桁架分组字典)
    """
    # 如果没有线段，直接返回
    if not lines:
        return lines, h_truss_groups
    
    # 收集所有点的坐标
    all_points = []
    for line in lines:
        pt1, pt2 = line
        all_points.append(pt1)
        all_points.append(pt2)
    
    # 寻找最左下角的点 (最小x, 最大y)
    min_x = min(point[0] for point in all_points)
    max_y = max(point[1] for point in all_points)
    min_y = min(point[1] for point in all_points)
    
    # 移动所有线段，同时将y坐标转换为笛卡尔坐标系（从下到上增加）
    moved_lines = []
    for line in lines:
        (x1, y1), (x2, y2) = line
        # 减去min_x来移动x坐标，转换y坐标（坐标系翻转）
        # 将y坐标范围从[min_y, max_y]映射到[0, max_y-min_y]
        new_pt1 = (x1 - min_x, max_y - y1)
        new_pt2 = (x2 - min_x, max_y - y2)
        moved_lines.append((new_pt1, new_pt2))
    
    # 移动分组中的水平桁架线段
    moved_h_truss_groups = {
        'h_truss_top': [],
        'h_truss_bottom': []
    }
    
    for group_name, group_lines in h_truss_groups.items():
        for line in group_lines:
            (x1, y1), (x2, y2) = line
            # 减去min_x来移动x坐标，转换y坐标（坐标系翻转）
            new_pt1 = (x1 - min_x, max_y - y1)
            new_pt2 = (x2 - min_x, max_y - y2)
            moved_h_truss_groups[group_name].append((new_pt1, new_pt2))
    
    return moved_lines, moved_h_truss_groups

def scale_truss(lines, h_truss_groups):
    """
    缩放桁架结构，基于h_truss_bottom的尺寸
    
    Args:
        lines: 所有线段的列表
        h_truss_groups: 水平桁架分组字典
        
    Returns:
        tuple: (缩放后的所有线段列表, 缩放后的水平桁架分组字典)
    """
    # 如果没有底部水平桁架或线段，直接返回
    if not h_truss_groups['h_truss_bottom'] or not lines:
        return lines, h_truss_groups
    
    # 收集底部水平桁架的所有点
    bottom_points = []
    for line in h_truss_groups['h_truss_bottom']:
        pt1, pt2 = line
        bottom_points.append(pt1)
        bottom_points.append(pt2)
    
    # 计算底部水平桁架的x和y的最大最小值
    min_x = min(point[0] for point in bottom_points)
    max_x = max(point[0] for point in bottom_points)
    min_y = min(point[1] for point in bottom_points)
    max_y = max(point[1] for point in bottom_points)
    
    # 计算缩放因子
    x_scale_factor = (max_x - min_x) / 2.5 if max_x != min_x else 1
    y_scale_factor = (max_y - min_y) / 0.575 if max_y != min_y else 1
    
    if x_scale_factor == 0:
        x_scale_factor = 1
    if y_scale_factor == 0:
        y_scale_factor = 1
    
    # 缩放所有线段
    scaled_lines = []
    for line in lines:
        (x1, y1), (x2, y2) = line
        # 缩放x和y坐标，保留为浮点数
        new_pt1 = (x1 / x_scale_factor, y1 / y_scale_factor)
        new_pt2 = (x2 / x_scale_factor, y2 / y_scale_factor)
        scaled_lines.append((new_pt1, new_pt2))
    
    # 缩放分组中的水平桁架线段
    scaled_h_truss_groups = {
        'h_truss_top': [],
        'h_truss_bottom': []
    }
    
    for group_name, group_lines in h_truss_groups.items():
        for line in group_lines:
            (x1, y1), (x2, y2) = line
            # 缩放x和y坐标，保留为浮点数
            new_pt1 = (x1 / x_scale_factor, y1 / y_scale_factor)
            new_pt2 = (x2 / x_scale_factor, y2 / y_scale_factor)
            scaled_h_truss_groups[group_name].append((new_pt1, new_pt2))
    
    return scaled_lines, scaled_h_truss_groups

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
    
    # 在原图上标记角点
    img_with_corners = img.copy()
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img_with_corners, (x, y), 5, (0, 0, 255), -1)

    # 重建线段
    lines = reconstruct_lines(img, gray, corners)
    
    # 寻找水平桁架构件
    h_truss_members = find_h_truss_members(lines)
    
    # 对水平桁架构件进行分组
    grouped_h_truss = group_h_truss_members(h_truss_members)
    
    # 移动桁架，使最左下角点位于原点
    lines, grouped_h_truss = move_truss(lines, grouped_h_truss)
    
    # 缩放桁架
    lines, grouped_h_truss = scale_truss(lines, grouped_h_truss)
    
    # 创建一个白色背景的图像用于绘制线段
    img_with_lines = np.ones_like(img) * 255
    
    # 在图像上绘制非水平桁架构件（黑色）
    for line in lines:
        if line not in h_truss_members:
            pt1, pt2 = line
            # 转换为整数用于绘图
            pt1_int = (int(pt1[0]), int(pt1[1]))
            pt2_int = (int(pt2[0]), int(pt2[1]))
            cv2.line(img_with_lines, pt1_int, pt2_int, (0, 0, 0), 3)
    
    # 定义不同组的颜色
    truss_colors = {
        'h_truss_top': (255, 0, 0),     # 蓝色
        'h_truss_bottom': (0, 255, 0)   # 绿色
    }
    
    # 在图像上绘制不同组的水平桁架构件
    for group_name, group in grouped_h_truss.items():
        color = truss_colors[group_name]
        for line in group:
            pt1, pt2 = line
            # 转换为整数用于绘图
            pt1_int = (int(pt1[0]), int(pt1[1]))
            pt2_int = (int(pt2[0]), int(pt2[1]))
            cv2.line(img_with_lines, pt1_int, pt2_int, color, 3)
    
    # BGR转RGB用于matplotlib显示
    img_with_corners_rgb = cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB)
    img_with_lines_rgb = cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB)

    # 显示两张图像
    plt.figure(figsize=(16, 9))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_with_corners_rgb)
    plt.title('Shi-Tomasi Corners on Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # 创建一个空白画布，而不是显示图像
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # 计算所有线段中的最大x和y值，用于动态设置坐标轴范围
    all_points = []
    for line in lines:
        pt1, pt2 = line
        all_points.append(pt1)
        all_points.append(pt2)
    
    max_x = max(point[0] for point in all_points) if all_points else 1000
    max_y = max(point[1] for point in all_points) if all_points else 1000
    
    # 添加10%的边距，使图形不会紧贴边缘
    margin = 0.1
    max_x_with_margin = max_x * (1 + margin)
    max_y_with_margin = max_y * (1 + margin)
    
    # 直接在matplotlib上绘制非水平桁架构件（黑色）
    for line in lines:
        if line not in h_truss_members:
            (x1, y1), (x2, y2) = line
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # matplotlib中的颜色映射
    plt_truss_colors = {
        'h_truss_top': 'b',     # 蓝色
        'h_truss_bottom': 'g'   # 绿色
    }
    
    # 在matplotlib上绘制不同组的水平桁架构件
    for group_name, group in grouped_h_truss.items():
        color = plt_truss_colors[group_name]
        for line in group:
            (x1, y1), (x2, y2) = line
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=2)
    
    # 更新图例文本，使用组名称
    color_legend = "Black: Non-horizontal members"
    if grouped_h_truss['h_truss_top']:
        color_legend += ", Blue: Top Horizontal Truss"
    if grouped_h_truss['h_truss_bottom']:
        color_legend += ", Green: Bottom Horizontal Truss"
    
    plt.title(f'Reconstructed truss\n{color_legend}')
    
    # 设置坐标轴范围，使原点位于左下角
    plt.xlim(0, 3)
    plt.ylim(0, 3)  # 正常笛卡尔坐标系，y轴向上为正
    plt.grid(True)
    
    plt.tight_layout()
    # 保存图片到test_result目录，文件名带时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'test_result/reconstructed_truss_{timestamp}.png'
    plt.savefig(output_path)
    print(f"结果图片已保存到: {output_path}")

    plt.show()

if __name__ == '__main__':
    main()
