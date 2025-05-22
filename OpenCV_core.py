import cv2
import numpy as np
import os
from datetime import datetime
import tempfile

def merge_lines(lines, angle_threshold=10, parallel_distance_threshold=30):
    """
    线段合并函数，可以处理重叠/部分重叠的线段，即斜率相近、线段间最近距离接近、线段端点相差较远的线段。
    """
    if lines is None or len(lines) == 0:
        return []
    
    # 提取所有线段
    lines_array = lines
    merged_lines = []
    used_lines = [False] * len(lines_array)
    
    # 计算线段角度
    def get_angle(line):
        x1, y1, x2, y2 = line
        # 避免除零错误
        if x2 - x1 == 0:
            return 90.0
        return np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
    
    # 计算两个线段之间的距离（点到线段的最小距离）
    def line_distance(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # 线段1的方向向量
        v1 = np.array([x2 - x1, y2 - y1])
        # 线段1的单位向量
        len_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        if len_v1 == 0:
            return float('inf')
        unit_v1 = v1 / len_v1
        
        # Pad vectors to 3D for np.cross
        unit_v1_3d = np.array([unit_v1[0], unit_v1[1], 0])
        p3_vec_3d = np.array([x3 - x1, y3 - y1, 0])
        p4_vec_3d = np.array([x4 - x1, y4 - y1, 0])

        # Calculate the magnitude of the cross product (z-component)
        p1_to_line = np.abs(np.cross(unit_v1_3d, p3_vec_3d)[2])
        p2_to_line = np.abs(np.cross(unit_v1_3d, p4_vec_3d)[2])
        
        return min(p1_to_line, p2_to_line)
    
    # 判断两个线段是否可以合并
    def can_merge(line1, line2):
        # 检查角度差异
        angle1 = get_angle(line1)
        angle2 = get_angle(line2)
        angle_diff = min(abs(angle1 - angle2), 180 - abs(angle1 - angle2))
        if angle_diff > angle_threshold:
            return False
        
        # 检查平行距离
        distance = line_distance(line1, line2)
        if distance > parallel_distance_threshold:
            return False
        
        return True
    
    # 合并两个线段
    def merge_two_lines(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # 将所有点按照x坐标排序（如果接近垂直则按y坐标排序）
        angle = get_angle(line1)
        if abs(angle) > 45:
            # 垂直线，按y坐标排序
            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda p: p[1])
        else:
            # 水平线，按x坐标排序
            points = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda p: p[0])
        
        # 取最远的两个点作为新线段的端点
        return [points[0][0], points[0][1], points[3][0], points[3][1]]
    
    # 合并过程
    for i in range(len(lines_array)):
        if used_lines[i]:
            continue
            
        current_line = lines_array[i]
        used_lines[i] = True
        
        # 尝试合并其他线段
        while True:
            merged = False
            for j in range(len(lines_array)):
                if used_lines[j] or i == j:
                    continue
                    
                if can_merge(current_line, lines_array[j]):
                    current_line = merge_two_lines(current_line, lines_array[j])
                    used_lines[j] = True
                    merged = True
                    
            if not merged:
                break
                
        merged_lines.append(current_line)
        
    return merged_lines

def classify_H_truss_members(lines, threshold=20, angle_threshold=5):
    """
    判别水平桁架
    """
    if not lines or len(lines) < 3:
        return {"H-truss": [], "Other": lines.copy() if lines else []}
    
    # 计算线段到点的垂直距离
    def perpendicular_distance(line, point):
        x1, y1, x2, y2 = line
        x0, y0 = point
        
        # 如果线段长度接近零，直接返回点到端点的欧氏距离
        if abs(x2-x1) < 1e-8 and abs(y2-y1) < 1e-8:
            return np.sqrt((x0-x1)**2 + (y0-y1)**2)
        
        # 计算线段长度
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # 计算叉积得到面积，除以长度得到高度（垂直距离）
        area = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
        return area / length
    
    # 统计每条线段相邻的端点数量
    adjacent_counts = []
    for i, line in enumerate(lines):
        count = 0
        for j, other_line in enumerate(lines):
            if i == j:
                continue
                
            # 获取另一条线段的端点
            endpoints = [(other_line[0], other_line[1]), (other_line[2], other_line[3])]
            
            # 检查端点是否在当前线段附近
            for point in endpoints:
                dist = perpendicular_distance(line, point)
                if dist < threshold:
                    count += 1
        
        adjacent_counts.append((i, count, line))
    
    # 按相邻端点数量降序排序
    adjacent_counts.sort(key=lambda x: x[1], reverse=True)
    
    # 检查排名前两的线段是否近似平行
    if len(adjacent_counts) >= 2:
        line1 = adjacent_counts[0][2]
        line2 = adjacent_counts[1][2]
        
        # 计算线段的方向向量
        vec1 = [line1[2] - line1[0], line1[3] - line1[1]]
        vec2 = [line2[2] - line2[0], line2[3] - line2[1]]
        
        # 计算向量的长度
        len1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
        len2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
        
        # 归一化向量
        if len1 > 0 and len2 > 0:
            vec1 = [vec1[0]/len1, vec1[1]/len1]
            vec2 = [vec2[0]/len2, vec2[1]/len2]
            
            # 计算点积，判断是否平行（正或反方向）
            dot_product = abs(vec1[0]*vec2[0] + vec1[1]*vec2[1])
            
            # 计算两线段间的夹角（角度）
            angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
            
            # 如果接近平行（点积接近1），则检查夹角是否在阈值内
            if dot_product > 0.9:  # 余弦值接近1表示角度接近0或180度
                if angle <= angle_threshold:
                    # 计算两条线的斜率
                    x1, y1, x2, y2 = line1
                    x3, y3, x4, y4 = line2
                    
                    # 计算两条线的斜率
                    slope1 = (y2 - y1) / (x2 - x1) if abs(x2 - x1) > 1e-8 else float('inf')
                    slope2 = (y4 - y3) / (x4 - x3) if abs(x4 - x3) > 1e-8 else float('inf')
                    
                    # 判断两条线段是否都接近水平（斜率接近0）
                    slope_threshold = 0.1  # 定义斜率接近0的阈值
                    if abs(slope1) < slope_threshold and abs(slope2) < slope_threshold:
                        # 将斜率统一调整为0（完全水平）
                        avg_slope = 0
                    else:
                        # 处理垂直线的情况
                        if slope1 == float('inf') and slope2 == float('inf'):
                            avg_slope = float('inf')
                        elif slope1 == float('inf'):
                            avg_slope = slope2
                        elif slope2 == float('inf'):
                            avg_slope = slope1
                        else:
                            avg_slope = (slope1 + slope2) / 2
                    
                    # 调整线段，保持中点不变
                    # 线段1的中点
                    mid_x1 = (x1 + x2) / 2
                    mid_y1 = (y1 + y2) / 2
                    # 线段2的中点
                    mid_x2 = (x3 + x4) / 2
                    mid_y2 = (y3 + y4) / 2
                    
                    # 线段长度
                    length1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    length2 = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
                    
                    # 调整线段1的端点
                    if avg_slope == float('inf'):
                        # 垂直线
                        x1_new = mid_x1
                        y1_new = mid_y1 - length1/2
                        x2_new = mid_x1
                        y2_new = mid_y1 + length1/2
                    else:
                        # 计算线段1新端点的x偏移量
                        dx1 = length1 / (2 * np.sqrt(1 + avg_slope**2))
                        x1_new = mid_x1 - dx1
                        y1_new = mid_y1 - avg_slope * dx1
                        x2_new = mid_x1 + dx1
                        y2_new = mid_y1 + avg_slope * dx1
                    
                    # 调整线段2的端点
                    if avg_slope == float('inf'):
                        # 垂直线
                        x3_new = mid_x2
                        y3_new = mid_y2 - length2/2
                        x4_new = mid_x2
                        y4_new = mid_y2 + length2/2
                    else:
                        # 计算线段2新端点的x偏移量
                        dx2 = length2 / (2 * np.sqrt(1 + avg_slope**2))
                        x3_new = mid_x2 - dx2
                        y3_new = mid_y2 - avg_slope * dx2
                        x4_new = mid_x2 + dx2
                        y4_new = mid_y2 + avg_slope * dx2
                    
                    # 更新线段
                    line1 = [x1_new, y1_new, x2_new, y2_new]
                    line2 = [x3_new, y3_new, x4_new, y4_new]
                    
                    # 构建其他线段列表
                    h_truss_indices = {adjacent_counts[0][0], adjacent_counts[1][0]}
                    other_lines = [lines[i] for i in range(len(lines)) if i not in h_truss_indices]
                    
                    return {"H-truss": [line1, line2], "Other": other_lines}
    
    # 如果没有找到合适的平行线段，返回端点最多的两条线段
    if len(adjacent_counts) >= 2:
        line1 = adjacent_counts[0][2]
        line2 = adjacent_counts[1][2]
        
        # 构建其他线段列表
        h_truss_indices = {adjacent_counts[0][0], adjacent_counts[1][0]}
        other_lines = [lines[i] for i in range(len(lines)) if i not in h_truss_indices]
        
        return {"H-truss": [line1, line2], "Other": other_lines}
    elif len(adjacent_counts) == 1:
        line1 = adjacent_counts[0][2]
        
        # 构建其他线段列表
        h_truss_indices = {adjacent_counts[0][0]}
        other_lines = [lines[i] for i in range(len(lines)) if i not in h_truss_indices]
        
        return {"H-truss": [line1], "Other": other_lines}
    else:
        return {"H-truss": [], "Other": []}

def classify_V_truss_members(lines, angle_threshold=20):
    """
    判别垂直桁架和斜撑
    """
    h_truss_lines = lines.get("H-truss", [])
    other_lines = lines.get("Other", [])
    
    v_truss_lines = []
    d_truss_lines = []
    
    # 如果没有水平桁架，则无法判断垂直桁架
    if not h_truss_lines:
        return {"H-truss": h_truss_lines, "V_truss": [], "D_truss": other_lines}
    
    # 计算水平桁架的平均角度
    h_angles = []
    for line in h_truss_lines:
        x1, y1, x2, y2 = line
        # 避免除零错误
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
        h_angles.append(angle)
    
    h_avg_angle = sum(h_angles) / len(h_angles) if h_angles else 0
    
    # 遍历其他线段，判断是垂直桁架还是斜撑
    for line in other_lines:
        x1, y1, x2, y2 = line
        # 避免除零错误
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
        
        # 计算与水平桁架的夹角差
        # 垂直桁架应该与水平桁架成90度角
        angle_diff = abs(abs(angle - h_avg_angle) - 90)
        
        # 如果夹角接近90度（在阈值范围内），则认为是垂直桁架
        if angle_diff < angle_threshold:
            v_truss_lines.append(line)
        else:
            d_truss_lines.append(line)
    
    return {
        "H-truss": h_truss_lines,
        "V_truss": v_truss_lines,
        "D_truss": d_truss_lines
    }

def cluster_endpoints(lines, threshold=6):
    """
    对线段端点进行聚类，并找出线段交点，使线段更加对齐。
    """
    # 提取各类线段
    h_truss_lines = lines.get("H-truss", [])
    v_truss_lines = lines.get("V_truss", [])
    d_truss_lines = lines.get("D_truss", [])
    
    if not h_truss_lines:
        return lines  # 如果没有水平桁架，则无法进行对齐
    
    # 计算点到线段的最短距离及最近点
    def point_to_line_distance(point, line):
        x0, y0 = point
        x1, y1, x2, y2 = line
        
        # 如果线段长度为0，则直接返回到端点的距离
        if abs(x2-x1) < 1e-8 and abs(y2-y1) < 1e-8:
            return np.sqrt((x0-x1)**2 + (y0-y1)**2), (x1, y1)
        
        # 线段的方向向量
        dx, dy = x2-x1, y2-y1
        # 线段长度的平方
        length_squared = dx**2 + dy**2
        
        # 计算投影比例 t
        t = max(0, min(1, ((x0-x1)*dx + (y0-y1)*dy) / length_squared))
        
        # 计算线段上的最近点
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # 计算距离
        distance = np.sqrt((x0-closest_x)**2 + (y0-closest_y)**2)
        
        return distance, (closest_x, closest_y)
    
    # 收集垂直和斜向桁架的端点
    vd_endpoints = []
    for i, line in enumerate(v_truss_lines):
        vd_endpoints.append(("V", i, 0, (line[0], line[1])))  # 起点
        vd_endpoints.append(("V", i, 1, (line[2], line[3])))  # 终点
        
    for i, line in enumerate(d_truss_lines):
        vd_endpoints.append(("D", i, 0, (line[0], line[1])))  # 起点
        vd_endpoints.append(("D", i, 1, (line[2], line[3])))  # 终点
    
    # 步骤1：对垂直和斜向桁架的端点进行合并聚类
    clusters = []
    processed = set()
    
    for i, (type1, idx1, end1, point1) in enumerate(vd_endpoints):
        if i in processed:
            continue
            
        cluster = [(type1, idx1, end1, point1)]
        processed.add(i)
        
        for j, (type2, idx2, end2, point2) in enumerate(vd_endpoints):
            if j in processed or i == j:
                continue
                
            distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            if distance < threshold * 0.5:  # 使用较小的阈值进行端点合并
                cluster.append((type2, idx2, end2, point2))
                processed.add(j)
        
        if len(cluster) > 1:  # 只关注有多个点的聚类
            clusters.append(cluster)
    
    # 合并端点聚类
    for cluster in clusters:
        # 计算平均位置
        avg_x = sum(p[3][0] for p in cluster) / len(cluster)
        avg_y = sum(p[3][1] for p in cluster) / len(cluster)
        
        # 更新每个点所属线段的端点
        for truss_type, line_idx, point_idx, _ in cluster:
            if truss_type == "V":
                if point_idx == 0:
                    v_truss_lines[line_idx][0] = avg_x
                    v_truss_lines[line_idx][1] = avg_y
                else:
                    v_truss_lines[line_idx][2] = avg_x
                    v_truss_lines[line_idx][3] = avg_y
            else:  # D
                if point_idx == 0:
                    d_truss_lines[line_idx][0] = avg_x
                    d_truss_lines[line_idx][1] = avg_y
                else:
                    d_truss_lines[line_idx][2] = avg_x
                    d_truss_lines[line_idx][3] = avg_y
    
    # 步骤2：确保垂直桁架和斜向桁架的两个端点分别贴合在不同的水平桁架上
    for v_idx, v_line in enumerate(v_truss_lines):
        # 处理起点和终点
        for point_idx in [0, 1]:
            x, y = v_line[point_idx*2], v_line[point_idx*2+1]
            
            # 寻找最近的水平桁架
            min_distance = float('inf')
            nearest_point = None
            nearest_h_idx = -1
            
            for h_idx, h_line in enumerate(h_truss_lines):
                distance, closest_point = point_to_line_distance((x, y), h_line)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = closest_point
                    nearest_h_idx = h_idx
            
            # 如果找到了足够近的水平桁架，将端点对齐到该桁架上
            if min_distance < threshold and nearest_point:
                # 更新垂直桁架端点
                v_truss_lines[v_idx][point_idx*2] = nearest_point[0]
                v_truss_lines[v_idx][point_idx*2+1] = nearest_point[1]
    
    for d_idx, d_line in enumerate(d_truss_lines):
        # 处理起点和终点
        for point_idx in [0, 1]:
            x, y = d_line[point_idx*2], d_line[point_idx*2+1]
            
            # 寻找最近的水平桁架
            min_distance = float('inf')
            nearest_point = None
            nearest_h_idx = -1
            
            for h_idx, h_line in enumerate(h_truss_lines):
                distance, closest_point = point_to_line_distance((x, y), h_line)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = closest_point
                    nearest_h_idx = h_idx
            
            # 如果找到了足够近的水平桁架，将端点对齐到该桁架上
            if min_distance < threshold and nearest_point:
                # 更新斜向桁架端点
                d_truss_lines[d_idx][point_idx*2] = nearest_point[0]
                d_truss_lines[d_idx][point_idx*2+1] = nearest_point[1]
    
    # 步骤3：确保水平桁架端点与最近的垂直/斜向桁架端点对齐
    # 收集更新后的垂直和斜向桁架的所有端点
    updated_vd_points = []
    for line in v_truss_lines:
        updated_vd_points.append((line[0], line[1]))
        updated_vd_points.append((line[2], line[3]))
    for line in d_truss_lines:
        updated_vd_points.append((line[0], line[1]))
        updated_vd_points.append((line[2], line[3]))

    if not updated_vd_points:
        print("警告：在步骤3中没有找到垂直或斜向桁架端点，无法对齐水平桁架端点。")
    else:
        # 检查并调整每个水平桁架端点
        for h_idx, h_line in enumerate(h_truss_lines):
            for point_idx in [0, 1]:  # 检查起点 (0) 和终点 (1)
                h_point_x, h_point_y = h_line[point_idx * 2], h_line[point_idx * 2 + 1]
                current_h_point = (h_point_x, h_point_y)

                # 查找距离当前水平桁架端点最近的垂直/斜向桁架端点
                min_distance_sq = float('inf')
                nearest_vd_point = None

                for vd_point in updated_vd_points:
                    # 使用距离的平方进行比较，避免开方运算
                    distance_sq = (current_h_point[0] - vd_point[0])**2 + (current_h_point[1] - vd_point[1])**2
                    if distance_sq < min_distance_sq:
                        min_distance_sq = distance_sq
                        nearest_vd_point = vd_point

                # 计算实际最小距离
                min_distance = np.sqrt(min_distance_sq)

                # 如果最小距离大于阈值，则认为该水平桁架端点需要对齐
                if min_distance > threshold:
                    # 将水平桁架端点移动到最近的垂直/斜向桁架端点的位置
                    if nearest_vd_point:
                        h_truss_lines[h_idx][point_idx * 2] = nearest_vd_point[0]
                        h_truss_lines[h_idx][point_idx * 2 + 1] = nearest_vd_point[1]
    
    # 步骤4：调整垂直桁架构件的斜率，使其与水平构件垂直，沿着中点进行旋转
    if h_truss_lines:
        # 计算水平桁架的平均斜率
        h_slopes = []
        for h_line in h_truss_lines:
            x1, y1, x2, y2 = h_line
            # 避免除零错误
            if abs(x2 - x1) < 1e-8:
                h_slopes.append(float('inf'))
            else:
                h_slopes.append((y2 - y1) / (x2 - x1))
        
        # 过滤掉无限值，计算平均斜率
        valid_h_slopes = [s for s in h_slopes if s != float('inf')]
        if valid_h_slopes:
            avg_h_slope = sum(valid_h_slopes) / len(valid_h_slopes)
        else:
            avg_h_slope = 0  # 如果所有斜率都是无限值，默认为0
        
        # 计算垂直于水平桁架的斜率
        if avg_h_slope == 0:
            perp_slope = float('inf')  # 垂直线
        elif abs(avg_h_slope) == float('inf'):
            perp_slope = 0  # 水平线
        else:
            perp_slope = -1 / avg_h_slope  # 垂直斜率
        
        # 调整每个垂直桁架的斜率
        for v_idx, v_line in enumerate(v_truss_lines):
            x1, y1, x2, y2 = v_line
            
            # 计算中点
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # 计算线段长度
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 根据垂直斜率调整端点
            if perp_slope == float('inf'):
                # 垂直线段
                new_x1 = mid_x
                new_y1 = mid_y - length / 2
                new_x2 = mid_x
                new_y2 = mid_y + length / 2
            else:
                # 计算线段新端点的x偏移量
                dx = length / (2 * np.sqrt(1 + perp_slope**2))
                new_x1 = mid_x - dx
                new_y1 = mid_y - perp_slope * dx
                new_x2 = mid_x + dx
                new_y2 = mid_y + perp_slope * dx
            
            # 更新垂直桁架的端点
            v_truss_lines[v_idx] = [new_x1, new_y1, new_x2, new_y2]
    
    # 步骤5：将斜撑端点对齐到最近的更新后的垂直桁架端点
    if v_truss_lines:  # 只有在存在垂直桁架时才执行
        # 收集所有更新后的垂直桁架端点
        updated_v_endpoints = []
        for v_line in v_truss_lines:
            updated_v_endpoints.append((v_line[0], v_line[1]))
            updated_v_endpoints.append((v_line[2], v_line[3]))

        if updated_v_endpoints: # 确保有端点可以对齐
            for d_idx, d_line in enumerate(d_truss_lines):
                for point_idx in [0, 1]: # 检查起点和终点
                    d_point_x, d_point_y = d_line[point_idx * 2], d_line[point_idx * 2 + 1]
                    current_d_point = (d_point_x, d_point_y)

                    # 查找距离当前斜撑端点最近的垂直桁架端点
                    min_distance_sq = float('inf')
                    nearest_v_point = None

                    for v_point in updated_v_endpoints:
                        distance_sq = (current_d_point[0] - v_point[0])**2 + (current_d_point[1] - v_point[1])**2
                        if distance_sq < min_distance_sq:
                            min_distance_sq = distance_sq
                            nearest_v_point = v_point
                    
                    # 计算实际最小距离
                    min_distance = np.sqrt(min_distance_sq)

                    # 如果最近的垂直桁架端点足够近，则将斜撑端点移动到该位置
                    if min_distance < threshold and nearest_v_point:
                        d_truss_lines[d_idx][point_idx * 2] = nearest_v_point[0]
                        d_truss_lines[d_idx][point_idx * 2 + 1] = nearest_v_point[1]

    # 步骤6：更新水平构件两端的顶点，使其与更新后的垂直桁架端点对齐
    if v_truss_lines:  # 只有在存在垂直桁架时才执行
        # 使用更新后的垂直桁架端点（在步骤5中已收集）
        if not 'updated_v_endpoints' in locals():
            updated_v_endpoints = []
            for v_line in v_truss_lines:
                updated_v_endpoints.append((v_line[0], v_line[1]))
                updated_v_endpoints.append((v_line[2], v_line[3]))
        
        # 也收集更新后的斜撑端点
        updated_d_endpoints = []
        for d_line in d_truss_lines:
            updated_d_endpoints.append((d_line[0], d_line[1]))
            updated_d_endpoints.append((d_line[2], d_line[3]))
        
        # 合并所有可能的连接点
        all_connection_points = updated_v_endpoints + updated_d_endpoints
        
        if all_connection_points:  # 确保有端点可以对齐
            for h_idx, h_line in enumerate(h_truss_lines):
                for point_idx in [0, 1]:  # 检查水平构件的起点和终点
                    h_point_x, h_point_y = h_line[point_idx * 2], h_line[point_idx * 2 + 1]
                    current_h_point = (h_point_x, h_point_y)
                    
                    # 查找距离当前水平桁架端点最近的连接点（垂直或斜撑端点）
                    min_distance_sq = float('inf')
                    nearest_connection_point = None
                    
                    for connection_point in all_connection_points:
                        distance_sq = (current_h_point[0] - connection_point[0])**2 + (current_h_point[1] - connection_point[1])**2
                        if distance_sq < min_distance_sq:
                            min_distance_sq = distance_sq
                            nearest_connection_point = connection_point
                    
                    # 计算实际最小距离
                    min_distance = np.sqrt(min_distance_sq)
                    
                    # 如果最近的连接点足够近，则将水平桁架端点移动到该位置
                    if min_distance < threshold and nearest_connection_point:
                        h_truss_lines[h_idx][point_idx * 2] = nearest_connection_point[0]
                        h_truss_lines[h_idx][point_idx * 2 + 1] = nearest_connection_point[1]

    # 构建更新后的线段字典
    updated_lines = {
        "H-truss": h_truss_lines,
        "V_truss": v_truss_lines,
        "D_truss": d_truss_lines
    }
    
    return updated_lines 

def normalize_truss_size(lines):
    """
    将桁架结构大小(长度)归一化，通过点的重新映射实现
    
    Args:
        lines: 包含 H-truss, V_truss, D_truss 的线段字典
        
    Returns:
        updated_lines: 归一化后的线段字典
        
    注意:
        x坐标: 按从小到大排序后映射到0,1,2,3...的整数序列
        y坐标: 保持二值映射(0表示下边界, 1表示上边界)
    """
    # 提取各类线段
    h_truss_lines = lines.get("H-truss", [])
    v_truss_lines = lines.get("V_truss", [])
    d_truss_lines = lines.get("D_truss", [])
     
    # 翻转所有线段的y坐标（1000-y）以处理cv2坐标系
    for line_list in [h_truss_lines, v_truss_lines, d_truss_lines]:
        for i, line in enumerate(line_list):
            line_list[i] = [
                line[0],  # x1
                1000 - line[1],  # y1
                line[2],  # x2
                1000 - line[3]   # y2
            ]
    
    # 1. 将所有端点收集到一个集合中，确保唯一性
    unique_points = set()
    
    for line in h_truss_lines + v_truss_lines + d_truss_lines:
        unique_points.add((line[0], line[1]))  # 起点
        unique_points.add((line[2], line[3]))  # 终点
    
    # 2. 构建点的字典，给每个点分配名称
    point_dict = {}
    for i, point in enumerate(unique_points):
        point_dict[f"P{i}"] = point
    
    # 3. 创建逆向查找映射（从坐标到点名称）
    point_lookup = {point: name for name, point in point_dict.items()}
    
    # 4. 构建线段字典
    lines_dict = {}
    
    # 添加水平桁架线段
    for i, line in enumerate(h_truss_lines):
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        lines_dict[f"H{i}"] = (point_lookup[p1], point_lookup[p2])
    
    # 添加垂直桁架线段
    for i, line in enumerate(v_truss_lines):
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        lines_dict[f"V{i}"] = (point_lookup[p1], point_lookup[p2])
    
    # 添加斜向桁架线段
    for i, line in enumerate(d_truss_lines):
        p1 = (line[0], line[1])
        p2 = (line[2], line[3])
        lines_dict[f"D{i}"] = (point_lookup[p1], point_lookup[p2])
    
    # 5. X坐标归一化: 对所有点按x坐标排序，将它们映射到0,1,2,3...
    # 首先收集所有不同的 x 坐标
    all_x_coords = sorted(set(p[0] for p in unique_points))
    
    # 合并相近的 x 坐标 (差值小于10的视为同一点)
    merged_x_coords = []
    if all_x_coords:
        current_group = [all_x_coords[0]]
        
        for i in range(1, len(all_x_coords)):
            # 如果当前x与上一组最后一个x的差值小于10，则归入同一组
            if all_x_coords[i] - current_group[-1] < 10:
                current_group.append(all_x_coords[i])
            else:
                # 将当前组的平均值添加到结果中
                merged_x_coords.append(sum(current_group) / len(current_group))
                # 开始新的一组
                current_group = [all_x_coords[i]]
        
        # 添加最后一组的平均值
        merged_x_coords.append(sum(current_group) / len(current_group))
    
    # 为原始x坐标创建到合并后坐标的映射
    x_merge_map = {}
    for orig_x in all_x_coords:
        # 找到最接近的合并后坐标
        closest_merged_x = min(merged_x_coords, key=lambda merged_x: abs(merged_x - orig_x))
        x_merge_map[orig_x] = closest_merged_x
    
    # 创建从合并后的x坐标到索引的映射
    unique_x_coords = merged_x_coords
    x_map = {x: i for i, x in enumerate(unique_x_coords)}
    
    # 6. Y坐标归一化: 收集所有唯一的y坐标
    unique_y_coords = sorted(set(p[1] for p in unique_points))
    
    # 获取水平桁架的y坐标，用于确定上下边界
    h_truss_y_coords = []
    for line in h_truss_lines:
        h_truss_y_coords.append(line[1])  # y1
        h_truss_y_coords.append(line[3])  # y2
    
    # 7. 创建Y坐标的二值映射
    binary_y_map = {}
    if h_truss_y_coords:
        # 找出水平桁架的最小和最大y坐标作为下边界和上边界
        min_h_y = min(h_truss_y_coords)
        max_h_y = max(h_truss_y_coords)
        
        # 创建二值映射：0表示下边界，1表示上边界
        for y in unique_y_coords:
            # 计算与上下边界的距离
            dist_to_min = abs(y - min_h_y)
            dist_to_max = abs(y - max_h_y)
            
            # 将y坐标映射到最近的边界
            if dist_to_min <= dist_to_max:
                binary_y_map[y] = 0  # 下边界
            else:
                binary_y_map[y] = 1  # 上边界
    else:
        # 如果没有水平桁架，则使用最小/最大y坐标
        binary_y_map = {y: (1 if i > 0 else 0) for i, y in enumerate(unique_y_coords)}
    
    # 8. 创建坐标归一化映射
    normalized_coords = {}
    for point in unique_points:
        x, y = point
        # 使用合并后的x坐标值
        merged_x = x_merge_map[x]
        normalized_coords[point] = (x_map[merged_x], binary_y_map[y])
    
    # 9. 更新点字典中的坐标为归一化后的坐标
    normalized_point_dict = {}
    for name, coords in point_dict.items():
        normalized_point_dict[name] = normalized_coords[coords]
    
    # 10. 重建线段
    new_h_truss = []
    new_v_truss = []
    new_d_truss = []
    
    # 从线段字典中重建各类桁架
    for line_name, (p1_name, p2_name) in lines_dict.items():
        p1_coords = normalized_point_dict[p1_name]
        p2_coords = normalized_point_dict[p2_name]
        new_line = [p1_coords[0], p1_coords[1], p2_coords[0], p2_coords[1]]
        
        if line_name.startswith('H'):
            new_h_truss.append(new_line)
        elif line_name.startswith('V'):
            new_v_truss.append(new_line)
        elif line_name.startswith('D'):
            new_d_truss.append(new_line)
    
    # 11. 保持类型信息
    h_types = []
    v_types = []
    d_types = []
    
    for line_name, _ in sorted(lines_dict.items()):
        if line_name.startswith('H'):
            h_types.append(line_name)
        elif line_name.startswith('V'):
            v_types.append(line_name)
        elif line_name.startswith('D'):
            d_types.append(line_name)
    
    # 12. 构建更新后的线段字典
    updated_lines = {
        "H-truss": new_h_truss,
        "V_truss": new_v_truss,
        "D_truss": new_d_truss,
        # 额外信息，可以用于调试或其他用途
        "point_dict": normalized_point_dict,
        "lines_dict": lines_dict,
        "h_types": h_types,
        "v_types": v_types,
        "d_types": d_types,
        "x_map": x_map,  # 添加x坐标映射，方便调试
        "y_map": binary_y_map  # 添加y坐标映射，方便调试
    }
    
    return updated_lines

def better_cluster_endpoints(lines, threshold=6):
    """
    改进的端点聚类算法
    
    Args:
        lines: 线段列表，每个线段是一个包含四个元素的列表 [x1, y1, x2, y2]
        threshold: 端点聚类阈值
        
    Returns:
        updated_lines: 聚类后的线段列表
    """
    # 提取所有线段的端点
    all_endpoints = []
    for line in lines:
        all_endpoints.append((line[0], line[1]))
        all_endpoints.append((line[2], line[3]))
    
    # 对端点进行排序，先按照x坐标排序，再按照y坐标排序
    all_endpoints.sort(key=lambda x: (x[0], x[1]))
    
    # 聚类端点，将距离小于阈值的点合并为一个点
    clustered_points = []
    coords_map = {}  # 建立原始坐标到聚类后坐标的映射
    
    for point in all_endpoints:
        # 检查当前点是否可以归入已有的聚类
        found_cluster = False
        for cluster_point in clustered_points:
            # 计算当前点与聚类点的距离
            dist = np.sqrt((point[0] - cluster_point[0])**2 + (point[1] - cluster_point[1])**2)
            if dist < threshold:
                # 将当前点映射到聚类点
                coords_map[point] = cluster_point
                found_cluster = True
                break
        
        # 如果找不到合适的聚类，创建一个新的聚类
        if not found_cluster:
            clustered_points.append(point)
            coords_map[point] = point
    
    # 使用坐标映射更新线段坐标
    updated_lines = []
    
    for line in lines:
        # 获取端点坐标
        start_point = (line[0], line[1])
        end_point = (line[2], line[3])
        
        # 获取聚类后的坐标
        clustered_start = coords_map[start_point]
        clustered_end = coords_map[end_point]
        
        # 创建更新后的线段
        updated_line = [
            clustered_start[0], clustered_start[1],
            clustered_end[0], clustered_end[1]
        ]
        
        updated_lines.append(updated_line)
    
    return updated_lines

def snap_endpoints_to_lines(lines, threshold=5):
    """
    将端点吸附到附近的线段上
    
    Args:
        lines: 线段列表，每个线段是一个包含四个元素的列表 [x1, y1, x2, y2]
        threshold: 吸附阈值，点到线段距离小于此值时会被吸附
        
    Returns:
        updated_lines: 吸附后的线段列表
    """
    if not lines or len(lines) < 2:
        return lines
    
    # 计算点到线段的垂直距离及投影点
    def point_to_line_distance(point, line):
        """计算点到线段的最短距离及投影点坐标"""
        x0, y0 = point
        x1, y1, x2, y2 = line
        
        # 如果线段长度为0，则直接返回到端点的距离
        if abs(x2-x1) < 1e-8 and abs(y2-y1) < 1e-8:
            return np.sqrt((x0-x1)**2 + (y0-y1)**2), (x1, y1)
        
        # 线段的方向向量
        dx, dy = x2-x1, y2-y1
        # 线段长度的平方
        length_squared = dx**2 + dy**2
        
        # 计算投影比例 t
        t = max(0, min(1, ((x0-x1)*dx + (y0-y1)*dy) / length_squared))
        
        # 计算线段上的最近点
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # 计算距离
        distance = np.sqrt((x0-closest_x)**2 + (y0-closest_y)**2)
        
        return distance, (closest_x, closest_y)
    
    # 查找最匹配的端点键，处理浮点数近似比较问题
    def find_matching_endpoint(point, endpoint_dict, tolerance=1e-8):
        x, y = point
        # 直接尝试精确匹配
        if point in endpoint_dict:
            return point
        
        # 遍历所有端点，寻找最近的匹配
        for endpoint in endpoint_dict.keys():
            ex, ey = endpoint
            if abs(ex - x) < tolerance and abs(ey - y) < tolerance:
                return endpoint
        
        # 如果没找到匹配点，则返回None
        return None
    
    # 复制一份线段列表，以便进行修改
    updated_lines = [line.copy() for line in lines]
    
    # 创建端点到线段索引的映射，用于快速查找使用特定端点的所有线段
    endpoint_to_lines = {}
    
    for i, line in enumerate(updated_lines):
        start_point = (line[0], line[1])
        end_point = (line[2], line[3])
        
        if start_point not in endpoint_to_lines:
            endpoint_to_lines[start_point] = []
        endpoint_to_lines[start_point].append((i, 'start'))
        
        if end_point not in endpoint_to_lines:
            endpoint_to_lines[end_point] = []
        endpoint_to_lines[end_point].append((i, 'end'))
    
    # 对每个端点进行吸附处理
    processed_points = set()  # 用于记录已处理过的点，避免重复处理
    
    for i, line in enumerate(updated_lines):
        # 处理起点
        start_point = (line[0], line[1])
        if start_point not in processed_points:
            min_distance = float('inf')
            best_projection = None
            best_line_idx = -1
            
            # 检查此端点是否靠近其他不相连的线段
            for j, other_line in enumerate(updated_lines):
                # 跳过当前线段和共享此端点的线段
                if i == j or (other_line[0], other_line[1]) == start_point or (other_line[2], other_line[3]) == start_point:
                    continue
                
                # 计算点到线段的距离及投影点
                distance, projection = point_to_line_distance(start_point, other_line)
                
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    best_projection = projection
                    best_line_idx = j
            
            # 如果找到了足够近的线段，则进行吸附
            if best_projection:
                # 查找匹配的端点键
                matching_start_point = find_matching_endpoint(start_point, endpoint_to_lines)
                
                if matching_start_point:
                    # 更新所有使用此端点的线段
                    for line_idx, point_type in endpoint_to_lines[matching_start_point]:
                        if point_type == 'start':
                            updated_lines[line_idx][0] = best_projection[0]
                            updated_lines[line_idx][1] = best_projection[1]
                        else:  # 'end'
                            updated_lines[line_idx][2] = best_projection[0]
                            updated_lines[line_idx][3] = best_projection[1]
                    
                    processed_points.add(start_point)
        
        # 处理终点
        end_point = (line[2], line[3])
        if end_point not in processed_points:
            min_distance = float('inf')
            best_projection = None
            best_line_idx = -1
            
            # 检查此端点是否靠近其他不相连的线段
            for j, other_line in enumerate(updated_lines):
                # 跳过当前线段和共享此端点的线段
                if i == j or (other_line[0], other_line[1]) == end_point or (other_line[2], other_line[3]) == end_point:
                    continue
                
                # 计算点到线段的距离及投影点
                distance, projection = point_to_line_distance(end_point, other_line)
                
                if distance < min_distance and distance < threshold:
                    min_distance = distance
                    best_projection = projection
                    best_line_idx = j
            
            # 如果找到了足够近的线段，则进行吸附
            if best_projection:
                # 查找匹配的端点键
                matching_end_point = find_matching_endpoint(end_point, endpoint_to_lines)
                
                if matching_end_point:
                    # 更新所有使用此端点的线段
                    for line_idx, point_type in endpoint_to_lines[matching_end_point]:
                        if point_type == 'start':
                            updated_lines[line_idx][0] = best_projection[0]
                            updated_lines[line_idx][1] = best_projection[1]
                        else:  # 'end'
                            updated_lines[line_idx][2] = best_projection[0]
                            updated_lines[line_idx][3] = best_projection[1]
                    
                    processed_points.add(end_point)
    
    return updated_lines
