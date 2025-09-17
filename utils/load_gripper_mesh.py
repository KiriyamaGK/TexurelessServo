import time
import xml.etree.ElementTree as ET
import trimesh
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import random
from utils.mesh import load_mesh

def parse_origin(origin_elem):
    """解析URDF中的origin元素，返回4x4变换矩阵"""
    transform = np.eye(4)

    if origin_elem is not None:
        # 获取xyz属性
        xyz = [0, 0, 0]
        if 'xyz' in origin_elem.attrib:
            xyz = list(map(float, origin_elem.attrib['xyz'].split()))

        # 获取rpy属性
        rpy = [0, 0, 0]
        if 'rpy' in origin_elem.attrib:
            rpy = list(map(float, origin_elem.attrib['rpy'].split()))

        # 创建旋转矩阵
        rotation = R.from_euler('xyz', rpy).as_matrix()

        # 创建变换矩阵
        transform[:3, :3] = rotation
        transform[:3, 3] = xyz

    return transform


def build_transform_tree(urdf_path, joint_angles=None):
    """构建URDF的完整变换树，支持设置关节角度"""
    if joint_angles is None:
        joint_angles = {}
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 存储链接的变换矩阵（相对于世界坐标系）
    link_transforms = {}
    # 存储父子关系
    parent_map = {}
    joint_info = {}  # 存储关节信息（类型、轴、限制等）
    joint_transforms = {}

    # 首先找到所有关节和它们的变换
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        joint_type = joint.get('type')
        parent_link = joint.find('parent').get('link')
        child_link = joint.find('child').get('link')

        parent_map[child_link] = parent_link

        # 获取关节变换
        origin_elem = joint.find('origin')
        base_transform = parse_origin(origin_elem)
        
        # 获取关节轴
        axis_elem = joint.find('axis')
        axis = [0, 0, 1]  # 默认Z轴
        if axis_elem is not None and 'xyz' in axis_elem.attrib:
            axis = list(map(float, axis_elem.attrib['xyz'].split()))
        
        # 获取关节限制
        limit_elem = joint.find('limit')
        lower_limit = 0
        upper_limit = 0
        if limit_elem is not None:
            lower_limit = float(limit_elem.get('lower', 0))
            upper_limit = float(limit_elem.get('upper', 0))
        
        # 获取mimic信息
        mimic_elem = joint.find('mimic')
        mimic_joint = None
        multiplier = 1.0
        offset = 0.0
        if mimic_elem is not None:
            mimic_joint = mimic_elem.get('joint')
            multiplier = float(mimic_elem.get('multiplier', 1.0))
            offset = float(mimic_elem.get('offset', 0.0))
        
        # 存储关节信息
        joint_info[joint_name] = {
            'type': joint_type,
            'axis': axis,
            'lower_limit': lower_limit,
            'upper_limit': upper_limit,
            'mimic_joint': mimic_joint,
            'multiplier': multiplier,
            'offset': offset,
            'base_transform': base_transform
        }
        
        joint_transforms[child_link] = base_transform

    # 计算所有关节的角度（考虑mimic关系）
    def get_joint_angle(joint_name):
        if joint_name in joint_angles:
            return joint_angles[joint_name]
        
        info = joint_info[joint_name]
        if info['mimic_joint']:
            mimic_angle = get_joint_angle(info['mimic_joint'])
            return mimic_angle * info['multiplier'] + info['offset']
        
        # 默认返回0
        return 0.0

    # 计算关节变换矩阵（考虑关节角度）
    def get_joint_transform(joint_name, angle):
        info = joint_info[joint_name]
        base_transform = info['base_transform']
        
        if info['type'] == 'revolute':
            # 创建旋转矩阵
            axis = info['axis']
            rotation = R.from_rotvec(np.array(axis) * angle).as_matrix()
            
            # 将旋转矩阵转换为齐次变换矩阵
            rotation_transform = np.eye(4)
            rotation_transform[:3, :3] = rotation
            
            # 组合变换：先基础变换，再旋转
            return np.dot(base_transform, rotation_transform)
        else:
            # 固定关节或其它类型，直接返回基础变换
            return base_transform

    # 找到基础链接（没有父关节的链接）
    all_links = {link.get('name') for link in root.findall('link')}
    all_children = set(parent_map.keys())
    base_links = all_links - all_children

    if not base_links:
        print("警告: 没有找到基础链接，使用第一个链接")
        first_link = root.find('link')
        if first_link:
            base_links = [first_link.get('name')]

    # 从基础链接开始，广度优先遍历计算所有链接的变换
    for base_link in base_links:
        link_transforms[base_link] = np.eye(4)  # 基础链接的变换是单位矩阵

        queue = deque([base_link])
        while queue:
            current_link = queue.popleft()

            # 找到所有以当前链接为父链接的子链接
            for joint in root.findall('joint'):
                parent_link_elem = joint.find('parent')
                if parent_link_elem is not None and parent_link_elem.get('link') == current_link:
                    joint_name = joint.get('name')
                    child_link = joint.find('child').get('link')
                    
                    # 获取关节角度
                    angle = get_joint_angle(joint_name)
                    
                    # 获取关节变换矩阵（考虑角度）
                    joint_transform = get_joint_transform(joint_name, angle)
                    
                    # 计算子链接的变换 = 父链接变换 × 关节变换
                    link_transforms[child_link] = np.dot(
                        link_transforms[current_link],
                        joint_transform
                    )
                    queue.append(child_link)

    return link_transforms, joint_info


def load_urdf_with_joint_angles(urdf_path, joint_angles=None, mesh_scale=1.0):
    """加载URDF模型并设置关节角度"""
    if joint_angles is None:
        joint_angles = {}
    
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    meshes_dir = os.path.join(urdf_dir, "meshes")
    
    # 构建完整的变换树（考虑关节角度）
    link_transforms, joint_info = build_transform_tree(urdf_path, joint_angles)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    meshes = []
    mesh_info = []

    for link in root.findall('link'):
        link_name = link.get('name')

        # 获取该链接的世界变换
        world_transform = link_transforms.get(link_name, np.eye(4))

        # 处理视觉元素
        visual = link.find('visual')
        if visual is not None:
            geometry = visual.find('geometry')
            if geometry is not None:
                mesh_elem = geometry.find('mesh')
                if mesh_elem is not None:
                    mesh_filename = mesh_elem.get('filename')

                    # 处理路径
                    if mesh_filename.startswith('package://'):
                        # 移除package://前缀
                        relative_path = mesh_filename.replace('package://', '')
                        # 移除可能的多余路径部分
                        if '/' in relative_path:
                            relative_path = relative_path.split('/', 1)[1]
                        mesh_filename = os.path.join(meshes_dir, os.path.basename(relative_path))
                    else:
                        # 直接使用相对路径
                        mesh_filename = os.path.join(meshes_dir, os.path.basename(mesh_filename))

                    if os.path.exists(mesh_filename):
                        try:
                            mesh = load_mesh(mesh_filename, mesh_scale)
                            if mesh is None:
                                continue

                            # 应用视觉元素的局部变换
                            visual_origin = visual.find('origin')
                            if visual_origin is not None:
                                visual_transform = parse_origin(visual_origin)
                                mesh.apply_transform(visual_transform)

                            # 应用世界变换（关节变换）
                            mesh.apply_transform(world_transform)

                            meshes.append(mesh)
                            mesh_info.append({
                                'link_name': link_name,
                                'filename': os.path.basename(mesh_filename),
                                'transform': world_transform
                            })

                        except Exception as e:
                            print(f"加载失败 {mesh_filename}: {e}")
                    else:
                        print(f"网格文件不存在: {mesh_filename}")

    return meshes, mesh_info, joint_info


def show_gripper_at_angles(joint_angles=None):
    """显示指定关节角度的夹爪"""
    if joint_angles is None:
        joint_angles = {}
    
    urdf_path = "/home/kiriyamagk/桌面/AlignAnything/meshes/zhixing/crt_ctag2f120.urdf"
    
    # 加载URDF模型
    meshes, mesh_info, joint_info = load_urdf_with_joint_angles(urdf_path, joint_angles)
    
    if meshes:
        print(f"成功加载 {len(meshes)} 个网格")
        
        # 显示场景
        scene = trimesh.Scene()
        for mesh in meshes:
            scene.add_geometry(mesh)
        
        # 设置相机位置以便更好地查看
        scene.set_camera(angles=(np.pi/4, np.pi/4, 0), distance=0.5)
        scene.show()
    else:
        print("没有加载到任何网格")


# 示例用法
if __name__ == "__main__":
    # 示例1：0开度
    print("=== 0开度 ===")
    show_gripper_at_angles({})
    
    # 示例2：R2关节打开到0.3弧度
    print("=== R2关节打开到0.3弧度 ===")
    show_gripper_at_angles({'R2': 0.3})
    
    # 示例3：R2关节完全打开到0.7弧度
    print("=== R2关节完全打开到0.7弧度 ===")
    show_gripper_at_angles({'R2': 0.7})
    
    # # 示例4：自定义多个关节角度
    # print("=== 自定义多个关节角度 ===")
    # show_gripper_at_angles({'R2': 0.5, 'L2': -0.3})