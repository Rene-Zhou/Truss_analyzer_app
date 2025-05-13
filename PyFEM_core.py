import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from math import sqrt

# 定义普拉特桁架类
class PrattTruss:
    def __init__(self, span, height, num_panels, diagonal_direction='alternating'):
        """
        初始化普拉特桁架
        span: 桁架跨度
        height: 桁架高度
        num_panels: 分隔面板数量
        diagonal_direction: 斜撑方向 ('alternating', 'upward', 'downward', 'all_tension', 'all_compression')
        """
        self.span = span
        self.height = height
        self.num_panels = num_panels
        self.panel_length = span / num_panels
        self.diagonal_direction = diagonal_direction
        
        # 创建节点和构件
        self.create_nodes_and_elements()
        
        # 材料和截面属性
        self.E = 210e9  # 弹性模量 (Pa)
        self.A = 0.001  # 截面面积 (m^2)
        
        # 刚度矩阵
        self.K = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))
        
        # 位移和力
        self.displacements = None
        self.element_forces = None
        
    def create_nodes_and_elements(self):
        """创建桁架的节点和构件"""
        # 创建节点
        self.nodes = []
        for i in range(self.num_panels + 1):
            # 下弦节点
            self.nodes.append([i * self.panel_length, 0])
            # 上弦节点
            self.nodes.append([i * self.panel_length, self.height])
        
        self.num_nodes = len(self.nodes)
        self.nodes = np.array(self.nodes)
        
        # 创建杆件（定义杆件连接的节点）
        self.elements = []
        
        # 下弦杆件
        for i in range(self.num_panels):
            self.elements.append([2*i, 2*(i+1)])
        
        # 上弦杆件
        for i in range(self.num_panels):
            self.elements.append([2*i+1, 2*(i+1)+1])
        
        # 垂直杆件
        for i in range(self.num_panels + 1):
            self.elements.append([2*i, 2*i+1])
        
        # 斜杆件（根据方向设置）
        for i in range(self.num_panels):
            if self.diagonal_direction == 'alternating':
                if i % 2 == 0:  # 偶数面板
                    self.elements.append([2*i+1, 2*(i+1)])
                else:  # 奇数面板
                    self.elements.append([2*i, 2*(i+1)+1])
            elif self.diagonal_direction == 'upward': # All diagonals point upward from left to right
                self.elements.append([2*i, 2*(i+1)+1])
            elif self.diagonal_direction == 'downward': # All diagonals point downward from left to right
                self.elements.append([2*i+1, 2*(i+1)])
            elif self.diagonal_direction == 'all_tension': # Pratt-like: / / / \ \ \
                mid_point = self.num_panels / 2.0
                if i < mid_point:
                    self.elements.append([2*i, 2*(i+1)+1])  # Left half: bottom-left to top-right
                else:
                    self.elements.append([2*i+1, 2*(i+1)])  # Right half: top-left to bottom-right
            elif self.diagonal_direction == 'all_compression': # Howe-like: \ \ \ / / /
                mid_point = self.num_panels / 2.0
                if i < mid_point:
                    self.elements.append([2*i+1, 2*(i+1)])  # Left half: top-left to bottom-right
                else:
                    self.elements.append([2*i, 2*(i+1)+1])  # Right half: bottom-left to top-right
        
        self.num_elements = len(self.elements)
        self.elements = np.array(self.elements)
        
    def assemble_stiffness_matrix(self):
        """装配整体刚度矩阵"""
        self.K = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))
        
        for el in range(self.num_elements):
            # 获取单元对应的节点
            node_i, node_j = self.elements[el]
            
            # 节点坐标
            xi, yi = self.nodes[node_i]
            xj, yj = self.nodes[node_j]
            
            # 计算杆长和方向余弦
            L = sqrt((xj - xi)**2 + (yj - yi)**2)
            if L == 0: continue # Avoid division by zero for zero-length elements
            c = (xj - xi) / L
            s = (yj - yi) / L
            
            # 计算单元刚度矩阵
            k_factor = (self.E * self.A / L)
            k_matrix_terms = np.array([
                [c*c, c*s, -c*c, -c*s],
                [c*s, s*s, -c*s, -s*s],
                [-c*c, -c*s, c*c, c*s],
                [-c*s, -s*s, c*s, s*s]
            ])
            k = k_factor * k_matrix_terms
            
            # 组装全局刚度矩阵的索引
            dofs = [2*node_i, 2*node_i+1, 2*node_j, 2*node_j+1]
            
            # 将单元刚度矩阵添加到全局刚度矩阵
            for i_idx in range(4):
                for j_idx in range(4):
                    self.K[dofs[i_idx], dofs[j_idx]] += k[i_idx, j_idx]
    
    def solve(self, loads, fixed_dofs):
        """
        求解桁架
        loads: 外荷载向量 (2*num_nodes)
        fixed_dofs: 固定自由度列表
        """
        # 装配刚度矩阵
        self.assemble_stiffness_matrix()
        
        # 分离自由度
        all_dofs = np.arange(2*self.num_nodes)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        # 求解位移
        # Ensure K_ff is not singular or empty
        if free_dofs.size == 0:
             self.displacements = np.zeros(2*self.num_nodes) # All DOFs are fixed
        else:
            K_ff = self.K[np.ix_(free_dofs, free_dofs)]
            loads_f = loads[free_dofs]
            
            try:
                u_free = np.linalg.solve(K_ff, loads_f)
                # 完整位移向量
                self.displacements = np.zeros(2*self.num_nodes)
                self.displacements[free_dofs] = u_free
            except np.linalg.LinAlgError:
                # Handle cases where the matrix is singular (e.g., unstable structure)
                print("Warning: Singular stiffness matrix. Check supports and structure stability.")
                self.displacements = np.zeros(2*self.num_nodes) # Or handle error as appropriate
                self.displacements[free_dofs] = np.nan # Indicate failure

        # 计算杆件力
        self.calculate_element_forces()
        
        return self.displacements, self.element_forces
    
    def calculate_element_forces(self):
        """计算杆件轴力"""
        if self.displacements is None:
            print("Warning: Displacements not calculated. Run solve() first.")
            self.element_forces = np.zeros(self.num_elements)
            return

        self.element_forces = np.zeros(self.num_elements)
        
        for el in range(self.num_elements):
            # 获取单元对应的节点
            node_i, node_j = self.elements[el]
            
            # 节点坐标
            xi, yi = self.nodes[node_i]
            xj, yj = self.nodes[node_j]
            
            # 计算杆长和方向余弦
            L = sqrt((xj - xi)**2 + (yj - yi)**2)
            if L == 0: 
                self.element_forces[el] = 0 # No force in zero-length element
                continue
            c = (xj - xi) / L
            s = (yj - yi) / L
            
            # 获取节点位移
            u_i = self.displacements[2*node_i]
            v_i = self.displacements[2*node_i+1]
            u_j = self.displacements[2*node_j]
            v_j = self.displacements[2*node_j+1]

            if any(np.isnan([u_i, v_i, u_j, v_j])):
                self.element_forces[el] = np.nan # Propagate NaN if displacements are NaN
                continue
            
            # 计算单元力
            self.element_forces[el] = (self.E * self.A / L) * (c*(u_j - u_i) + s*(v_j - v_i))
    
    def plot_truss(self, scale=10, figsize=(12, 8)):
        """
        Plot truss original and deformed shape
        scale: deformation magnification factor
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine max_abs_force for color mapping and line width
        if self.element_forces is not None and self.element_forces.size > 0 and np.all(np.isfinite(self.element_forces)):
            max_abs_force = np.max(np.abs(self.element_forces))
            if max_abs_force == 0:  # All forces are zero
                max_abs_force = 1.0 # Avoid division by zero for line width
        else:
            max_abs_force = 1.0 # Default if no forces or NaN forces

        norm = Normalize(vmin=-max_abs_force, vmax=max_abs_force)
        
        blue_cmap = cm.get_cmap('Blues')
        red_cmap = cm.get_cmap('Reds')
        
        # Calculate deformed node coordinates
        deformed_nodes = np.zeros_like(self.nodes, dtype=float)
        if self.displacements is not None and np.all(np.isfinite(self.displacements)):
            for i in range(self.num_nodes):
                deformed_nodes[i, 0] = self.nodes[i, 0] + scale * self.displacements[2*i]
                deformed_nodes[i, 1] = self.nodes[i, 1] + scale * self.displacements[2*i+1]
        else: # If displacements are None or NaN, plot original shape as deformed
            deformed_nodes = self.nodes.astype(float)

        # Plot original truss
        for el in range(self.num_elements):
            node_i, node_j = self.elements[el]
            x_orig = [self.nodes[node_i, 0], self.nodes[node_j, 0]]
            y_orig = [self.nodes[node_i, 1], self.nodes[node_j, 1]]
            ax.plot(x_orig, y_orig, 'k--', alpha=0.3)  # Original state shown in dashed lines
        
        # Plot deformed truss
        if self.element_forces is not None and np.all(np.isfinite(self.element_forces)):
            for el in range(self.num_elements):
                node_i, node_j = self.elements[el]
                
                xi_new, yi_new = deformed_nodes[node_i]
                xj_new, yj_new = deformed_nodes[node_j]
                
                force = self.element_forces[el]
                if force < 0:  # Compression
                    color_val = norm(-force) if max_abs_force != 0 else 0.5
                    color = blue_cmap(color_val)
                    lw = 1 + 3 * (abs(force) / max_abs_force) if max_abs_force != 0 else 1
                else:  # Tension
                    color_val = norm(force) if max_abs_force != 0 else 0.5
                    color = red_cmap(color_val)
                    lw = 1 + 3 * (abs(force) / max_abs_force) if max_abs_force != 0 else 1
                
                ax.plot([xi_new, xj_new], [yi_new, yj_new], '-', color=color, linewidth=max(0.5, lw))
        else: # If forces are None or NaN, plot deformed shape in grey
            for el in range(self.num_elements):
                node_i, node_j = self.elements[el]
                xi_new, yi_new = deformed_nodes[node_i]
                xj_new, yj_new = deformed_nodes[node_j]
                ax.plot([xi_new, xj_new], [yi_new, yj_new], '-', color='grey', linewidth=1)


        # Add colorbars if forces are valid
        if self.element_forces is not None and np.all(np.isfinite(self.element_forces)):
            sm_tension = cm.ScalarMappable(cmap=red_cmap, norm=norm)
            sm_tension.set_array([])
            cbar_tension = fig.colorbar(sm_tension, ax=ax, location='right', pad=0.1, aspect=30)
            cbar_tension.set_label('Tension Force (N)')
            
            sm_compression = cm.ScalarMappable(cmap=blue_cmap, norm=norm)
            sm_compression.set_array([])
            cbar_compression = fig.colorbar(sm_compression, ax=ax, location='right', pad=0.05, aspect=30) # pad adjusted
            cbar_compression.set_label('Compression Force (N)')
        
        # Plot nodes
        ax.plot(deformed_nodes[:, 0], deformed_nodes[:, 1], 'o', markersize=6, color='black')
        
        # Set appropriate axis limits
        all_x = np.concatenate([self.nodes[:, 0], deformed_nodes[:, 0]])
        all_y = np.concatenate([self.nodes[:, 1], deformed_nodes[:, 1]])
        
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)

        x_margin = 0.1 * (x_max - x_min) if (x_max - x_min) > 0 else 1.0
        y_margin = 0.1 * (y_max - y_min) if (y_max - y_min) > 0 else 1.0
        
        ax.set_xlim([x_min - x_margin, x_max + x_margin])
        ax.set_ylim([y_min - y_margin, y_max + y_margin])
        
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Truss Analysis (Deformation Magnified {}x)'.format(scale))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        return fig, ax 