import numpy as np
from PyFEM_core import PrattTruss  # Import from the refactored core module
import matplotlib.pyplot as plt   # For saving the plot

def main():
    # 桁架参数
    span = 24.0   # 跨度 (m)
    height = 4.0  # 高度 (m)
    num_panels = 6  # 面板数
    
    # 创建桁架, using default diagonal_direction='alternating'
    # You can specify other directions, e.g., diagonal_direction='upward'
    truss = PrattTruss(span, height, num_panels, diagonal_direction='alternating') 
    
    # 施加荷载 (example: downward force on lower chord nodes)
    loads = np.zeros(2 * truss.num_nodes)
    # Apply -50kN load to lower chord nodes (excluding end supports if they are lower chord)
    for i in range(1, num_panels): # Nodes 2, 4, 6, ... (lower chord, excluding ends)
        node_idx = 2 * i  # Index for lower chord nodes in PrattTruss configuration
        loads[2 * node_idx + 1] = -50000  # Vertical downward force (N)
    
    # 固定支座节点 (example: pinned left, roller right)
    left_node_idx = 0  # Typically the first node (0,0)
    right_node_idx = 2 * num_panels # Typically the last lower chord node

    fixed_dofs = [
        2 * left_node_idx,     # Left support, X-direction (pinned)
        2 * left_node_idx + 1, # Left support, Y-direction (pinned)
        2 * right_node_idx + 1 # Right support, Y-direction (roller)
    ]
    
    # 求解桁架
    print("Solving truss...")
    displacements, element_forces = truss.solve(loads, fixed_dofs)
    
    # 打印结果
    if displacements is not None and np.all(np.isfinite(displacements)):
        print("\n节点位移 (m):")
        for i in range(truss.num_nodes):
            print(f"节点 {i}: X={displacements[2*i]:.6f}, Y={displacements[2*i+1]:.6f}")
    else:
        print("\n节点位移计算失败或包含无效值.")

    if element_forces is not None and np.all(np.isfinite(element_forces)):
        print("\n构件轴力 (N):")
        for i in range(truss.num_elements):
            print(f"构件 {i} (节点 {truss.elements[i][0]}-{truss.elements[i][1]}): {element_forces[i]:.2f}")
    else:
        print("\n构件轴力计算失败或包含无效值.")
    
    # 绘制桁架并保存
    print("\nPlotting truss...")
    fig, ax = truss.plot_truss(scale=100) # Deformation scale factor
    output_filename = 'truss_analysis_cli_output.png'
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure explicitly to free memory
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    main() 