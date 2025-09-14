import time
import numpy as np
from build.sdf_module import SDFCalculator
from utils.mesh import load_mesh, sample_face_points
import trimesh
from utils.paths import PROJECT_ROOT_DIR
import os

def create_sdf_calculator(mesh):
    sdf_calculator = SDFCalculator()
    vertices = mesh.vertices
    faces = mesh.faces
    verts_cpp = [tuple(v) for v in vertices] # 转换为C++需要的格式
    tris_cpp = [tuple(t) for t in faces]
    sdf_calculator.load_mesh(verts_cpp, tris_cpp)
    return sdf_calculator

def cross_check_collision(obj1_sdf, obj2_sdf, obj1_mesh, obj2_mesh, num_sample_points=500):
    # 互相检测表面点
    t_start = time.time()
    points1 = obj1_mesh.sample(num_sample_points)
    points2 = obj2_mesh.sample(num_sample_points)
    t_end = time.time()
    print("use sample time:",t_end - t_start)
    
    # 检测obj1的点是否在obj2内部
    collision1 = any(obj2_sdf.signed_distance(p) < 0 for p in points1)
    # 检测obj2的点是否在obj1内部
    collision2 = any(obj1_sdf.signed_distance(p) < 0 for p in points2)
    
    return collision1 or collision2

def check_collision_hybrid(obj1_sdf, obj2_sdf, obj1_mesh, obj2_mesh, num_sample_points=500,a_to_b=True):
    # 第一步：快速AABB（axis-aligned bounding box）检测
    obj1_bounds = obj1_mesh.bounds
    obj2_bounds = obj2_mesh.bounds
    
    if not (obj1_bounds[0][0] <= obj2_bounds[1][0] and 
            obj1_bounds[1][0] >= obj2_bounds[0][0] and
            obj1_bounds[0][1] <= obj2_bounds[1][1] and 
            obj1_bounds[1][1] >= obj2_bounds[0][1] and
            obj1_bounds[0][2] <= obj2_bounds[1][2] and 
            obj1_bounds[1][2] >= obj2_bounds[0][2]):
        return False  # 包围盒不相交
    
    # 第二步：精确SDF检测
    if a_to_b:
        t_start = time.time()
        points = obj1_mesh.sample(num_sample_points)  # 增加采样点
        t_end = time.time()
        print("use sample time:",t_end - t_start)
        return any(obj2_sdf.signed_distance(p) < 0 for p in points)
    else:
        points = obj2_mesh.sample(num_sample_points)  # 增加采样点
        return any(obj1_sdf.signed_distance(p) < 0 for p in points)

if __name__ == "__main__":
    gripper_path = os.path.join(PROJECT_ROOT_DIR,"meshes/gripper/meshes/l_gripper_tip_scaled.stl")   
    object_path = os.path.join(PROJECT_ROOT_DIR,"meshes/objs/part15.STL")

    gripper_mesh = load_mesh(gripper_path,scale = 1000) # unit is mm
    object_mesh = load_mesh(object_path) # unit is already mm
    gripper_mesh = gripper_mesh.convex_hull  #if not gripper_mesh.is_convex else gripper_mesh
    object_mesh = object_mesh.convex_hull  #if not object_mesh.is_convex else object_mesh

    object_mesh.apply_translation([0.0, 0.0, 50])  # 向右移动10个单位

    gripper_sdf = create_sdf_calculator(gripper_mesh)
    object_sdf = create_sdf_calculator(object_mesh)


    t_start = time.time()
    res_1 = check_collision_hybrid(gripper_sdf, object_sdf, gripper_mesh, object_mesh, num_sample_points=500,a_to_b=True)
    t_end = time.time()
    print("use time:",t_end - t_start)
    print("res_1:",res_1) # True

    t_start = time.time()
    res_2 = cross_check_collision(gripper_sdf, object_sdf, gripper_mesh, object_mesh, num_sample_points=500)
    t_end = time.time()
    print("use time:",t_end - t_start)
    print("res_2:",res_2) # True

    scene = trimesh.Scene([gripper_mesh, object_mesh])
    scene.show(viewer="gl")