import streamlit as st
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from OpenCV_core import better_cluster_endpoints, merge_lines, snap_endpoints_to_lines
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# 设置页面标题和布局
st.set_page_config(page_title="线段检测与聚类", layout="wide")
st.title("线段检测与端点聚类工具")

# 创建选项卡，用于切换绘图和上传图片模式
tab_selection = st.radio("选择输入模式:", ["绘图模式", "上传图片模式"])

# 初始化变量
canvas_result = None
uploaded_image = None
processed_image = None

if tab_selection == "绘图模式":
    # 设置画板模式
    drawing_mode = st.sidebar.selectbox(
        "绘制工具:", ("line", "freedraw", "point", "rect", "circle", "transform")
    )

    # 设置笔画宽度
    stroke_width = st.sidebar.slider("笔画宽度: ", 1, 25, 7)

    # 设置笔画颜色
    stroke_color = st.sidebar.color_picker("笔画颜色: ", "#000000")

    # 创建画板
    st.subheader("绘制区域")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # 透明填充色
        stroke_width=stroke_width,  # 笔画宽度
        stroke_color=stroke_color,  # 笔画颜色
        background_color="#eee",  # 背景颜色
        width=1000,  # 画板宽度
        height=600,  # 画板高度
        drawing_mode=drawing_mode,  # 绘制模式
        key="canvas",
        display_toolbar=True,
        update_streamlit=True,
    )

elif tab_selection == "上传图片模式":
    st.subheader("上传图片")
    uploaded_image = st.file_uploader("选择一张图片上传", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # 显示上传的图片
        image_pil = Image.open(uploaded_image)
        # 调整大小为1000x1000
        image_pil = image_pil.resize((1000, 1000), Image.LANCZOS)
        st.image(image_pil, caption="上传的图片", width=1000)
        # 保存处理后的图像以供后续使用
        processed_image = np.array(image_pil)

# 添加端点聚类阈值滑块
clustering_threshold = st.sidebar.slider("端点聚类阈值: ", 1, 50, 15)

# 添加端点吸附阈值滑块
snap_threshold = st.sidebar.slider("端点吸附阈值: ", 1, 50, 15)

# 添加线段合并参数
angle_threshold = st.sidebar.slider("线段合并角度阈值: ", 1, 30, 10)
parallel_distance_threshold = st.sidebar.slider("平行线段合并距离阈值: ", 5, 100, 30)

# 添加Hough变换参数
hough_threshold = st.sidebar.slider("Hough阈值: ", 10, 100, 40)
min_line_length = st.sidebar.slider("Hough最小线长: ", 10, 100, 40)
max_line_gap = st.sidebar.slider("Hough最大线间隔: ", 1, 50, 10)

# 添加确认按钮
analyze_button = st.button("开始分析")

# 确定要处理的图像
image_to_process = None
if analyze_button:
    if tab_selection == "绘图模式" and canvas_result is not None and canvas_result.image_data is not None:
        # 使用绘图模式的图像
        image_to_process = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2BGR)
    elif tab_selection == "上传图片模式" and processed_image is not None:
        # 使用上传的图像
        image_to_process = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

if analyze_button and image_to_process is not None:
    # 使用expander来包装处理过程的图片展示
    with st.expander("点击展开查看处理过程", expanded=True):
        st.subheader("处理过程")
        cols = st.columns(3)
        
        with cols[0]:
            st.image(cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGB), caption="1. 原始图片")
        
        # 1. 转换为灰度图
        gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
        with cols[1]:
            st.image(gray, caption="2. 灰度图")

        # 2. 二值化处理
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        with cols[2]:
            st.image(binary, caption="3. 二值化")

        # 3. 腐蚀操作，细化线条
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        
        # 4. 使用Hough变换检测线段
        lines = cv2.HoughLinesP(
            eroded, 
            rho=1, 
            theta=np.pi/180, 
            threshold=hough_threshold, 
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )

        if lines is not None:
            # 显示原始检测的线段
            original_lines_image = np.zeros_like(image_to_process)
            lines_list = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(original_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                lines_list.append([x1, y1, x2, y2])
            
            # 5. 线段合并
            merged_lines = merge_lines(lines_list, angle_threshold=angle_threshold, parallel_distance_threshold=parallel_distance_threshold)
            
            # 显示合并后的线段
            merged_lines_image = np.zeros_like(image_to_process)
            for line in merged_lines:
                x1, y1, x2, y2 = line
                cv2.line(merged_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # 6. 端点聚类
            clustered_lines = better_cluster_endpoints(merged_lines, threshold=clustering_threshold)
            
            # 显示聚类后的线段
            clustered_lines_image = np.zeros_like(image_to_process)
            for line in clustered_lines:
                x1, y1, x2, y2 = line
                cv2.line(clustered_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 在端点位置绘制小圆点
                cv2.circle(clustered_lines_image, (int(x1), int(y1)), 3, (255, 0, 0), -1)
                cv2.circle(clustered_lines_image, (int(x2), int(y2)), 3, (255, 0, 0), -1)
            
            # 7. 端点吸附
            snapped_lines = snap_endpoints_to_lines(clustered_lines, threshold=snap_threshold)
            
            # 显示吸附后的线段
            snapped_lines_image = np.zeros_like(image_to_process)
            for line in snapped_lines:
                x1, y1, x2, y2 = line
                cv2.line(snapped_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                # 在端点位置绘制小圆点
                cv2.circle(snapped_lines_image, (int(x1), int(y1)), 3, (255, 0, 0), -1)
                cv2.circle(snapped_lines_image, (int(x2), int(y2)), 3, (255, 0, 0), -1)
            
            # 第一行展示前4个步骤
            cols2 = st.columns(4)
            with cols2[0]:
                st.image(eroded, caption="4. 腐蚀")
            with cols2[1]:
                st.image(cv2.cvtColor(original_lines_image, cv2.COLOR_BGR2RGB), caption="5. 检测到的线段")
            with cols2[2]:
                st.image(cv2.cvtColor(merged_lines_image, cv2.COLOR_BGR2RGB), caption="6. 合并后的线段")
            with cols2[3]:
                st.image(cv2.cvtColor(clustered_lines_image, cv2.COLOR_BGR2RGB), caption="7. 端点聚类后的线段")

            # 第二行展示吸附结果
            st.image(cv2.cvtColor(snapped_lines_image, cv2.COLOR_BGR2RGB), caption="8. 端点吸附后的线段")

            # 显示统计信息
            st.subheader("统计信息")
            cols3 = st.columns(4)
            with cols3[0]:
                st.metric("检测到的线段数量", len(lines))
            with cols3[1]:
                st.metric("合并后的线段数量", len(merged_lines))
            with cols3[2]:
                st.metric("聚类后的线段数量", len(clustered_lines))
            with cols3[3]:
                st.metric("吸附后的线段数量", len(snapped_lines))
            
            # 使用Matplotlib绘制最终结果
            st.subheader("最终分析结果")
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 只显示吸附后的最终线段（使用蓝色线段和红色端点）
            for line in snapped_lines:
                x1, y1, x2, y2 = line
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
                ax.plot(x1, y1, 'ro', markersize=6)  # 绘制起点
                ax.plot(x2, y2, 'ro', markersize=6)  # 绘制终点
            
            ax.set_title("Final Line Detection Results")
            ax.axis('equal')
            ax.grid(True)
            ax.invert_yaxis()  # 图像坐标系y轴向下，matplotlib默认向上，需要反转
            
            # 创建图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='b', lw=2, label='Lines'),
                Line2D([0], [0], marker='o', color='r', label='Endpoints',
                       markersize=6, linestyle='None')
            ]
            ax.legend(handles=legend_elements, loc='best')
            
            st.pyplot(fig)
        else:
            st.error("未能检测到任何线段，请尝试调整图片或参数。")
