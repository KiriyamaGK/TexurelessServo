import numpy as np
import trimesh
from typing import Tuple, List
from utils.mesh import load_mesh, sample_face_points
import time

# ======================== SAT算法实现 ========================
def sat_collision(hull1: trimesh.Trimesh, hull2: trimesh.Trimesh) -> bool:
    """分离轴定理(SAT)实现"""
    # 获取所有待测试的轴（面法向 + 边叉积）
    axes = []
    
    # 添加所有面法向
    axes.extend(hull1.facets_normal)
    axes.extend(hull2.facets_normal)
    
    # 添加边叉积组合
    for e1 in hull1.edges_unique:
        for e2 in hull2.edges_unique:
            edge1 = hull1.vertices[e1[1]] - hull1.vertices[e1[0]]
            edge2 = hull2.vertices[e2[1]] - hull2.vertices[e2[0]]
            cross = np.cross(edge1, edge2)
            if np.linalg.norm(cross) > 1e-6:
                axes.append(cross / np.linalg.norm(cross))
    
    # 归一化所有轴
    axes = [a / np.linalg.norm(a) for a in axes if np.linalg.norm(a) > 1e-6]
    
    # 对每个轴进行投影测试
    for axis in axes:
        # 投影第一个凸包
        dots1 = np.dot(hull1.vertices, axis)
        min1, max1 = np.min(dots1), np.max(dots1)
        
        # 投影第二个凸包
        dots2 = np.dot(hull2.vertices, axis)
        min2, max2 = np.min(dots2), np.max(dots2)
        
        # 检查投影是否重叠
        if max1 < min2 or max2 < min1:
            return False  # 存在分离轴
    
    return True  # 所有轴都重叠，发生碰撞



# ======================== 统一接口 ========================
def check_collision(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, method='auto') -> bool:
    """
    碰撞检测统一接口
    :param method: 'aabb', 'sat', 'gjk', 'auto'(自动选择)
    """
    # 先计算凸包
    hull1 = mesh1.convex_hull if not mesh1.is_convex else mesh1
    hull2 = mesh2.convex_hull if not mesh2.is_convex else mesh2
    
    # 执行指定算法
    if method == 'sat':
        return sat_collision(hull1, hull2)
    else:
        raise ValueError(f"Unknown method: {method}")

# ======================== 测试用例 ========================
if __name__ == "__main__":
    # 创建线框版本的网格
    def create_wireframe(mesh, color=[255, 255, 255, 255]):
        """创建网格的线框版本"""
        # 提取边
        edges = mesh.edges_unique
        vertices = mesh.vertices
        
        # 创建线框几何体
        from trimesh.creation import line
        lines = []
        for edge in edges:
            start = vertices[edge[0]]
            end = vertices[edge[1]]
            line_geom = line([start, end])
            line_geom.visual.face_colors = color
            lines.append(line_geom)
        
        return lines
        
    # 创建测试物体
    gripper_path = "/home/kiriyamagk/桌面/collision_detection/meshes/gripper/meshes/l_gripper_tip_scaled.stl"   
    object_path = "/home/kiriyamagk/桌面/collision_detection/meshes/objs/part15.STL"

    gripper_mesh = load_mesh(gripper_path,scale = 1000)
    object_mesh = load_mesh(object_path)
    object_mesh.apply_translation([30.0, 0.0, 0.0])  # 向右移动10个单位
    
    print("=== 碰撞检测测试 ===")
    t_start = time.time()
    print(f"SAT 检测结果: {check_collision(gripper_mesh, object_mesh, 'sat')}")
    t_end = time.time()
    print(f"SAT 检测时间: {t_end - t_start} 秒")
    
    # 计算凸包
    hull1 = gripper_mesh.convex_hull if not gripper_mesh.is_convex else gripper_mesh
    hull2 = object_mesh.convex_hull if not object_mesh.is_convex else object_mesh
    
    # 创建场景并显示
    scene = trimesh.Scene([gripper_mesh, object_mesh])
    scene.show(viewer="gl")