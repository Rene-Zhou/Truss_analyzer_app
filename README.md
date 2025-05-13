# 桁架分析工具 (Truss Analyzer Tool)

这是一个包含两个主要模块的桁架结构分析工具：
1. **OpenCV 图像分析模块** (`OpenCV_app.py` + `OpenCV_core.py`)：用于从上传的桁架图片中识别并分类水平桁架、垂直桁架和斜撑。
2. **有限元分析模块** (`PyFEM_app.py`)：用于对桁架结构进行有限元分析，计算位移、杆件力和支座反力。

## 结构说明

- `OpenCV_app.py` 只负责 Streamlit 前端界面和主流程，所有图像处理与桁架识别的核心算法已迁移到 `OpenCV_core.py`。
- `OpenCV_core.py` 提供线段合并、桁架分类、端点聚类等核心算法函数，供 `OpenCV_app.py` 调用。
- 这样结构更清晰，便于维护和扩展：界面与算法解耦，后续如需命令行、API等调用方式可直接复用 `OpenCV_core.py`。

## 功能

### OpenCV 图像分析模块 (`OpenCV_app.py` + `OpenCV_core.py`)
1. **图片上传**：支持上传 JPG、JPEG 或 PNG 格式的桁架图片。
2. **预处理**：包括灰度转换、高斯模糊、自适应阈值处理和边缘检测。
3. **线段检测**：使用 Hough 变换检测线段。
4. **线段合并**：合并重叠或部分重叠的线段（由 `OpenCV_core.py` 提供）。
5. **桁架分类**：
   - 水平桁架（红色）
   - 垂直桁架（绿色）
   - 斜撑（蓝色）
6. **端点聚类**：对齐桁架构件的端点，使其更加整齐（由 `OpenCV_core.py` 提供）。
7. **可视化**：实时显示处理过程中的每一步结果。

### 有限元分析模块 (`PyFEM_app.py`)
1. **参数设置**：
   - 桁架几何参数（跨度、高度、面板数量）。
   - 材料属性（弹性模量、截面面积）。
   - 荷载条件（荷载大小）。
   - 支座条件（固定、铰接、滚动支座）。
2. **分析功能**：
   - 计算节点位移。
   - 计算杆件内力。
   - 计算支座反力。
3. **可视化**：
   - 桁架结构图（支持变形放大）。
   - 详细结果展示（节点位移、杆件力）。

## 使用方法

### OpenCV 图像分析模块
1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
2. **运行应用**：
   ```bash
   streamlit run OpenCV_app.py
   ```
3. **上传图片**：
   - 打开应用后，点击 "上传" 按钮选择桁架图片。
   - 工具会自动处理图片并显示分析结果。

### 有限元分析模块
1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
2. **运行应用**：
   ```bash
   streamlit run PyFEM_app.py
   ```
3. **设置参数**：
   - 在侧边栏调整桁架参数、材料属性、荷载和支座条件。
   - 点击运行后，工具会计算并显示分析结果。

## 使用 pm2 保持 Streamlit 服务运行

为了确保 `streamlit` 应用在后台持续运行，可以使用 `pm2` 工具。以下是设置步骤：

1. **安装 pm2**（如果尚未安装）：
   ```bash
   npm install -g pm2
   ```

2. **给予脚本执行权限**：
   ```bash
   chmod +x OpenCV_app.sh
   chmod +x PyFEM_app.sh
   ```

3. **使用 pm2 启动应用**：
   运行以下命令来使用 `pm2` 启动你的应用：
   ```bash
   pm2 start ./OpenCV_app.sh --name OpenCV_app
   ```

   对于 `PyFEM_app.py`，可以使用类似的命令：
   ```bash
   pm2 start ./PyFEM_app.sh --name PyFEM_app
   ```

4. **管理 pm2 进程**：
   - 查看运行的进程：
     ```bash
     pm2 list
     ```
   - 停止应用：
     ```bash
     pm2 stop OpenCV_app
     pm2 stop PyFEM_app
     ```
   - 重启应用：
     ```bash
     pm2 restart OpenCV_app
     pm2 restart PyFEM_app
     ```
   - 查看日志：
     ```bash
     pm2 logs OpenCV_app
     pm2 logs PyFEM_app
     ```

通过以上步骤，你可以使用 `pm2` 保持 `streamlit` 应用在虚拟环境中持续运行。

## 依赖项

- Python 3.7+
- OpenCV (`opencv-python`)
- Streamlit (`streamlit`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

完整的依赖项列表见 `requirements.txt`。

## 示例

### OpenCV 图像分析模块
1. 上传图片后，工具会显示原始图片、灰度图、高斯模糊图和自适应阈值处理图。
2. 接着显示边缘检测、原始线段、合并后的线段和水平桁架识别结果。
3. 最后显示桁架分类结果和端点聚类后的最终分析结果。

### 有限元分析模块
1. 设置桁架参数后，工具会计算并显示最大位移、最大杆件力和支座反力。
2. 可视化桁架结构，支持变形放大。
3. 展开详细结果查看节点位移和杆件力。

## 注意事项

- **OpenCV 模块**：
  - 确保上传的图片清晰，桁架构件边缘明显。
  - 如果工具未能检测到线段，可以尝试调整图片或参数。
  - 如需自定义或复用核心算法，可直接调用 `OpenCV_core.py` 中的函数。
- **有限元模块**：
  - 确保输入的参数合理（如荷载大小、材料属性）。
  - 支座条件会影响计算结果，请根据实际需求选择。

### 常见问题

#### 缺少 `libGL.so.1` 共享库
如果你的 Streamlit 应用在导入 `cv2`（OpenCV）时提示缺少 `libGL.so.1` 共享库，可以按照以下步骤解决：

1. **安装缺失的库**：
   在 Linux 系统上运行以下命令安装 `libgl1` 包（包含 `libGL.so.1`）：
   ```bash
   sudo apt-get update
   sudo apt-get install libgl1
   ```

2. **确认安装**：
   安装完成后，运行以下命令确认 `libGL.so.1` 是否存在：
   ```bash
   ls /usr/lib/x86_64-linux-gnu/libGL.so.1
   ```
   如果文件存在，说明安装成功。

3. **重启 Streamlit 应用**：
   重新启动你的 Streamlit 应用：
   ```bash
   pm2 restart your-app-name  # 替换为你的应用名称
   ```
