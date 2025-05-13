import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from math import sqrt
from PyFEM_core import PrattTruss
import pandas as pd

# Language dictionaries
TEXTS = {
    'en': {
        'title': "Truss Analysis Tool",
        'parameters': "Truss Parameters",
        'span': "Span (m)",
        'height': "Height (m)",
        'panels': "Number of Panels",
        'material': "Material Properties",
        'elastic_modulus': "Elastic Modulus (GPa)",
        'cross_section': "Cross-sectional Area (cm²)",
        'loading': "Loading Conditions",
        'load_magnitude': "Load Magnitude (kN)",
        'supports': "Support Conditions",
        'left_support': "Left Support",
        'right_support': "Right Support",
        'support_types': ["Fixed", "Pinned", "Roller"],
        'visualization': "Visualization Parameters",
        'deformation_scale': "Deformation Scale",
        'results': "Analysis Results",
        'max_displacement': "Maximum Displacement",
        'max_force': "Maximum Member Force",
        'support_reactions': "Support Reactions",
        'truss_visualization': "Truss Visualization",
        'detailed_displacements': "Detailed Node Displacements",
        'detailed_forces': "Detailed Member Forces",
        'node': "Node",
        'member': "Member",
        'nodes': "Nodes",
        'mm': "mm",
        'kn': "kN",
        'language': "Language",
        'diagonal_direction': "Diagonal Member Direction",
        'diagonal_types': ["Alternating", "All Upward", "All Downward", "All Tension", "All Compression"]
    },
    'zh': {
        'title': "桁架分析工具",
        'parameters': "桁架参数",
        'span': "跨度 (m)",
        'height': "高度 (m)",
        'panels': "面板数量",
        'material': "材料属性",
        'elastic_modulus': "弹性模量 (GPa)",
        'cross_section': "截面面积 (cm²)",
        'loading': "荷载条件",
        'load_magnitude': "荷载大小 (kN)",
        'supports': "支座条件",
        'left_support': "左端支座",
        'right_support': "右端支座",
        'support_types': ["固定支座", "铰支座", "滚动支座"],
        'visualization': "可视化参数",
        'deformation_scale': "变形放大系数",
        'results': "分析结果",
        'max_displacement': "最大位移",
        'max_force': "最大杆件力",
        'support_reactions': "支座反力",
        'truss_visualization': "桁架可视化",
        'detailed_displacements': "详细节点位移",
        'detailed_forces': "详细杆件力",
        'node': "节点",
        'member': "杆件",
        'nodes': "节点",
        'mm': "毫米",
        'kn': "千牛",
        'language': "语言",
        'diagonal_direction': "斜撑方向",
        'diagonal_types': ["交替布置", "全部向上", "全部向下", "全部受拉", "全部受压"]
    }
}

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Language switcher
def switch_language():
    st.session_state.language = 'zh' if st.session_state.language == 'en' else 'en'

# Get current language texts
text = TEXTS[st.session_state.language]

st.set_page_config(page_title=text['title'], layout="wide")

# Add language switcher button in the top right
col1, col2 = st.columns([6, 1])
with col1:
    st.title(text['title'])
with col2:
    st.button(text['language'], on_click=switch_language)

# Sidebar for input parameters
st.sidebar.header(text['parameters'])

# Basic parameters
span = st.sidebar.slider(text['span'], min_value=10.0, max_value=50.0, value=24.0, step=1.0)
height = st.sidebar.slider(text['height'], min_value=2.0, max_value=10.0, value=4.0, step=0.5)
num_panels = st.sidebar.slider(text['panels'], min_value=4, max_value=12, value=6, step=1)

# Diagonal direction
diagonal_direction = st.sidebar.selectbox(
    text['diagonal_direction'],
    text['diagonal_types'],
    index=0
)

# Convert diagonal direction to internal format
diagonal_map = {
    text['diagonal_types'][0]: 'alternating',
    text['diagonal_types'][1]: 'upward',
    text['diagonal_types'][2]: 'downward',
    text['diagonal_types'][3]: 'all_tension',
    text['diagonal_types'][4]: 'all_compression',
}
diagonal_direction = diagonal_map[diagonal_direction]

# Material properties
st.sidebar.subheader(text['material'])
E = st.sidebar.number_input(text['elastic_modulus'], min_value=100.0, max_value=300.0, value=210.0, step=10.0) * 1e9
A = st.sidebar.number_input(text['cross_section'], min_value=0.1, max_value=10.0, value=1.0, step=0.1) * 1e-4

# Loading conditions
st.sidebar.subheader(text['loading'])
load_magnitude = st.sidebar.number_input(text['load_magnitude'], min_value=10.0, max_value=100.0, value=50.0, step=5.0) * 1000

# Support conditions
st.sidebar.subheader(text['supports'])
left_support = st.sidebar.selectbox(
    text['left_support'],
    text['support_types'],
    index=0
)

right_support = st.sidebar.selectbox(
    text['right_support'],
    text['support_types'],
    index=2
)

# Visualization parameters
st.sidebar.subheader(text['visualization'])
deformation_scale = st.sidebar.slider(text['deformation_scale'], min_value=10, max_value=500, value=100, step=10)

# Create truss instance
truss = PrattTruss(span, height, num_panels, diagonal_direction)
truss.E = E
truss.A = A

# Apply loads
loads = np.zeros(2 * truss.num_nodes)
for i in range(1, num_panels):
    node_idx = 2 * i
    loads[2 * node_idx + 1] = -load_magnitude

# Define support conditions
fixed_dofs = []
left_node = 0
right_node = 2 * num_panels

if left_support == text['support_types'][0] or left_support == text['support_types'][1]:  # Fixed or Pinned
    fixed_dofs.extend([2 * left_node, 2 * left_node + 1])
elif left_support == text['support_types'][2]:  # Roller
    fixed_dofs.append(2 * left_node + 1)

if right_support == text['support_types'][0] or right_support == text['support_types'][1]:  # Fixed or Pinned
    fixed_dofs.extend([2 * right_node, 2 * right_node + 1])
elif right_support == text['support_types'][2]:  # Roller
    fixed_dofs.append(2 * right_node + 1)

# Solve truss
displacements, element_forces = truss.solve(loads, fixed_dofs)

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader(text['results'])
    
    # Display maximum displacement
    max_disp = np.max(np.abs(displacements))
    st.metric(text['max_displacement'], f"{max_disp*1000:.2f} {text['mm']}")
    
    # Display maximum force
    max_force = np.max(np.abs(element_forces))
    st.metric(text['max_force'], f"{max_force/1000:.2f} {text['kn']}")
    
    # Display support reactions
    st.subheader(text['support_reactions'])
    reactions = np.dot(truss.K, displacements)
    for i in range(len(fixed_dofs)):
        dof = fixed_dofs[i]
        st.write(f"DOF {dof}: {reactions[dof]/1000:.2f} {text['kn']}")

with col2:
    st.subheader(text['truss_visualization'])
    fig, ax = truss.plot_truss(scale=deformation_scale)
    st.pyplot(fig)
    plt.close()

# Display detailed results in expandable sections
with st.expander(text['detailed_displacements']):
    # Create displacement data
    displacement_data = {
        text['node']: list(range(truss.num_nodes)),
        'X (mm)': [displacements[2*i]*1000 for i in range(truss.num_nodes)],
        'Y (mm)': [displacements[2*i+1]*1000 for i in range(truss.num_nodes)]
    }
    df_displacements = pd.DataFrame(displacement_data)
    df_displacements = df_displacements.round(2)  # Round to 2 decimal places
    st.dataframe(df_displacements, hide_index=True)

with st.expander(text['detailed_forces']):
    # Create member forces data
    force_data = {
        text['member']: list(range(truss.num_elements)),
        f"{text['nodes']} (i-j)": [f"{truss.elements[i][0]}-{truss.elements[i][1]}" for i in range(truss.num_elements)],
        f"{text['kn']}": [element_forces[i]/1000 for i in range(truss.num_elements)]
    }
    df_forces = pd.DataFrame(force_data)
    df_forces = df_forces.round(2)  # Round to 2 decimal places
    st.dataframe(df_forces, hide_index=True)
