import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile
from OpenCV_core import merge_lines, classify_H_truss_members, classify_V_truss_members, cluster_endpoints, normalize_truss_size

# 设置页面标题和布局
st.set_page_config(page_title="桁架分析工具", layout="wide")
st.title("桁架结构分析工具")

# 文件上传器
uploaded_file = st.file_uploader("请上传一张桁架图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 创建临时文件来保存上传的图片
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        input_image_path = tmp_file.name

    # 读取图片
    image = cv2.imread(input_image_path)
    if image is None:
        st.error("无法读取图片，请确保上传了有效的图片文件。")
    else:
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join('img', timestamp)
        os.makedirs(output_dir, exist_ok=True)

        # 使用expander来包装处理过程的图片展示
        with st.expander("点击展开查看处理过程", expanded=False):
            st.subheader("处理过程")
            # 第一行：基础图像处理
            st.markdown("##### 基础图像处理")
            cols = st.columns(4)
            with cols[0]:
                st.image(image, channels="BGR", caption="1. 原始图片")
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            with cols[1]:
                st.image(gray, caption="2. 灰度图")

            # 使用高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            with cols[2]:
                st.image(blurred, caption="3. 高斯模糊")

            # 自适应阈值处理
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            with cols[3]:
                st.image(thresh, caption="4. 自适应阈值处理")

            # 对二值图像取反色
            edges = cv2.bitwise_not(thresh)

            # 添加腐蚀操作，细化线条
            kernel = np.ones((3,3), np.uint8)
            eroded_edges = cv2.erode(edges, kernel, iterations=1)
            
            # 使用Hough变换检测线段
            lines = cv2.HoughLinesP(eroded_edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=30)

            if lines is not None:
                # 显示原始检测的线段
                original_lines_image = np.zeros_like(image)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(original_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # 合并线段
                lines = [line[0] for line in lines]
                merged_lines = merge_lines(lines, angle_threshold=17, parallel_distance_threshold=40)
                
                # 显示合并后的线段
                merged_lines_image = np.zeros_like(image)
                for line in merged_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(merged_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # 识别水平桁架
                truss_result = classify_H_truss_members(merged_lines, threshold=25)
                h_truss_lines = truss_result["H-truss"]
                other_lines = truss_result["Other"]

                # 显示水平桁架
                h_truss_image = np.zeros_like(image)
                for line in h_truss_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(h_truss_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # 第二行：线段检测和处理
                st.markdown("##### 线段检测和处理")
                cols2 = st.columns(4)
                with cols2[0]:
                    st.image(eroded_edges, caption="5. 边缘检测和腐蚀")
                with cols2[1]:
                    st.image(original_lines_image, caption="6. 原始检测的线段")
                with cols2[2]:
                    st.image(merged_lines_image, caption="7. 合并后的线段")
                with cols2[3]:
                    st.image(h_truss_image, caption="8. 水平桁架识别")

                # 分类垂直桁架和斜撑
                all_truss_result = classify_V_truss_members(truss_result, angle_threshold=20)
                h_truss_lines = all_truss_result["H-truss"]
                v_truss_lines = all_truss_result["V_truss"]
                d_truss_lines = all_truss_result["D_truss"]

                # 创建彩色分类图像
                colored_truss_image = np.zeros_like(image)
                # 绘制水平桁架（红色）
                for line in h_truss_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(colored_truss_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # 绘制垂直桁架（绿色）
                for line in v_truss_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(colored_truss_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 绘制斜撑（蓝色）
                for line in d_truss_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(colored_truss_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # 执行端点聚类
                clustered_truss_result = cluster_endpoints(all_truss_result, threshold=50)
                h_clustered_lines = clustered_truss_result["H-truss"]
                v_clustered_lines = clustered_truss_result["V_truss"]
                d_clustered_lines = clustered_truss_result["D_truss"]

                # 桁架归一化处理
                normalized_truss = normalize_truss_size(clustered_truss_result)

                # 创建端点聚类后的图像
                clustered_truss_image = np.zeros_like(image)
                # 绘制水平桁架（红色）
                for line in h_clustered_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(clustered_truss_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # 绘制垂直桁架（绿色）
                for line in v_clustered_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(clustered_truss_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 绘制斜撑（蓝色）
                for line in d_clustered_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(clustered_truss_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # 在每个端点位置绘制小圆点
                for line in h_clustered_lines + v_clustered_lines + d_clustered_lines:
                    cv2.circle(clustered_truss_image, (int(line[0]), int(line[1])), 3, (255, 255, 255), -1)
                    cv2.circle(clustered_truss_image, (int(line[2]), int(line[3])), 3, (255, 255, 255), -1)

                # 第三行：最终分类结果
                st.markdown("##### 最终分类结果")
                cols3 = st.columns(4)
                with cols3[0]:
                    st.image(colored_truss_image, caption="9. 桁架分类结果")
                with cols3[1]:
                    st.image(clustered_truss_image, caption="10. 端点聚类结果")
                with cols3[2]:
                    # 显示统计信息
                    st.subheader("分析结果统计")
                    st.metric("水平桁架数量", len(h_truss_lines))
                    st.metric("垂直桁架数量", len(v_truss_lines))
                    st.metric("斜撑数量", len(d_truss_lines))

        # 显示最终结果（大图）
        st.subheader("最终分析结果")
        
        # 创建结果可视化区域
        cols_final = st.columns(2)
        
        with cols_final[0]:
            # 使用matplotlib绘制原始桁架结果
            fig1, ax1 = plt.subplots(figsize=(10, 10))
            
            # 构建与normalized_truss相同格式的字典，便于统一绘图
            original_truss_dict = {
                "H-truss": h_clustered_lines,
                "V_truss": v_clustered_lines,
                "D_truss": d_clustered_lines
            }
            
            # 绘制水平桁架构件 (红色)
            for line in original_truss_dict.get("H-truss", []):
                ax1.plot([line[0], line[2]], [line[1], line[3]], 'r-', linewidth=2, label='H-truss')
            
            # 绘制垂直桁架构件 (绿色)
            for line in original_truss_dict.get("V_truss", []):
                ax1.plot([line[0], line[2]], [line[1], line[3]], 'g-', linewidth=2, label='V-truss')
            
            # 绘制斜向桁架构件 (蓝色)
            for line in original_truss_dict.get("D_truss", []):
                ax1.plot([line[0], line[2]], [line[1], line[3]], 'b-', linewidth=2, label='D-truss')
            
            # 收集所有节点坐标
            all_points = []
            for line in original_truss_dict.get("H-truss", []) + original_truss_dict.get("V_truss", []) + original_truss_dict.get("D_truss", []):
                all_points.append((line[0], line[1]))
                all_points.append((line[2], line[3]))
            
            # 绘制节点 - 先绘制黑色轮廓增加可见性
            for point in all_points:
                ax1.plot(point[0], point[1], 'ko', markersize=8, zorder=10)  # 黑色轮廓
            
            # 再绘制白色内部
            for i, point in enumerate(all_points):
                # 只在第一个点添加标签，避免图例重复
                if i == 0:
                    ax1.plot(point[0], point[1], 'wo', markersize=6, zorder=11, label='Nodes')  # 白色内部带标签
                else:
                    ax1.plot(point[0], point[1], 'wo', markersize=6, zorder=11)  # 白色内部不带标签
                
            # 设置图表格式
            ax1.set_title("Original Truss Analysis")
            ax1.axis('equal')
            ax1.grid(True)
            
            # 去除重复的图例条目
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys(), loc='best')
            
            # 显示图表
            st.pyplot(fig1)
        
        with cols_final[1]:
            # 使用matplotlib绘制归一化的桁架
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            
            # 绘制水平桁架构件 (红色)
            for line in normalized_truss.get("H-truss", []):
                ax2.plot([line[0], line[2]], [line[1], line[3]], 'r-', linewidth=2, label='H-truss')
            
            # 绘制垂直桁架构件 (绿色)
            for line in normalized_truss.get("V_truss", []):
                ax2.plot([line[0], line[2]], [line[1], line[3]], 'g-', linewidth=2, label='V-truss')
            
            # 绘制斜向桁架构件 (蓝色)
            for line in normalized_truss.get("D_truss", []):
                ax2.plot([line[0], line[2]], [line[1], line[3]], 'b-', linewidth=2, label='D-truss')
            
            # 设置图表格式
            ax2.set_title("Normalized Truss Structure")
            ax2.axis('equal')
            ax2.grid(True)
            
            # 去除重复的图例条目
            handles, labels = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax2.legend(by_label.values(), by_label.keys(), loc='best')
            
            # 显示图表
            st.pyplot(fig2)
            
            # 显示归一化桁架数据统计
            st.subheader("归一化桁架数据")
            st.metric("水平桁架数量", len(normalized_truss.get("H-truss", [])))
            st.metric("垂直桁架数量", len(normalized_truss.get("V_truss", [])))
            st.metric("斜撑数量", len(normalized_truss.get("D_truss", [])))

        # 清理临时文件
        os.unlink(input_image_path)
else:
    st.info("请上传一张图片开始分析。")
