import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime
import tempfile
from OpenCV_core import merge_lines, classify_H_truss_members, classify_V_truss_members, cluster_endpoints
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# 设置页面标题和布局
st.set_page_config(page_title="桁架分析工具", layout="wide")
st.title("桁架结构分析工具")

# 设置画板模式
drawing_mode = st.sidebar.selectbox(
    "绘制工具:", ("freedraw", "point", "line", "rect", "circle", "transform")
)

# 设置笔画宽度
stroke_width = st.sidebar.slider("笔画宽度: ", 1, 25, 3)

# 设置点显示半径
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("点显示半径: ", 1, 25, 3)

# 设置笔画颜色
stroke_color = st.sidebar.color_picker("笔画颜色: ")

# 设置背景颜色
bg_color = st.sidebar.color_picker("背景颜色: ", "#eee")

# 设置背景图片
bg_image = st.sidebar.file_uploader("背景图片: ", type=["png", "jpg"])

# 设置是否实时更新
realtime_update = st.sidebar.checkbox("实时更新", True)

# 创建画板
st.subheader("绘制区域")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # 透明填充色
    stroke_width=stroke_width,  # 笔画宽度
    stroke_color=stroke_color,  # 笔画颜色
    background_color=bg_color,  # 背景颜色
    background_image=bg_image,  # 背景图片
    width=1000,  # 画板宽度
    height=1000,  # 画板高度
    drawing_mode=drawing_mode,  # 绘制模式
    key="canvas",
    display_toolbar=True,
    update_streamlit=True,
)

# 添加确认按钮
analyze_button = st.button("开始分析")

if analyze_button and canvas_result.image_data is not None:
    # 将canvas的图像数据转换为OpenCV格式
    image = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2BGR)
    
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
    st.image(clustered_truss_image, caption="最终分析结果（包含端点标记）", use_container_width=True)

else:
    st.error("未能检测到任何线段，请尝试调整图片或参数。")
