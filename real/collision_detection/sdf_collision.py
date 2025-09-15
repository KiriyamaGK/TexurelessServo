import time
import numpy as np
from build.sdf_module import SDFCalculator
from utils.mesh import load_mesh, sample_face_points
import trimesh
from utils.paths import PROJECT_ROOT_DIR
import os

class CollisionDetector():
    def __init__(self,obj1_mesh_path, obj2_mesh_path,scalar_1=1000.0,scalar_2=1.0,use_convex_hull_1 = True,use_convex_hull_2 = True,T_1 = None,T_2 = None):
        self.obj1_mesh = load_mesh(obj1_mesh_path,scalar_1) #gripper is 1, object is 2
        self.obj2_mesh = load_mesh(obj2_mesh_path,scalar_2)
        if use_convex_hull_1:
            self.obj1_mesh = self.obj1_mesh.convex_hull
        if use_convex_hull_2:
            self.obj2_mesh = self.obj2_mesh.convex_hull
        self.obj1_sdf = self.create_sdf_calculator(self.obj1_mesh)
        self.obj2_sdf = self.create_sdf_calculator(self.obj2_mesh)
        if T_1 is not None:
            self.obj1_mesh.apply_transform(T_1)
        if T_2 is not None:
            self.obj2_mesh.apply_transform(T_2)
        
    def create_sdf_calculator(self,mesh):
        sdf_calculator = SDFCalculator()
        vertices = mesh.vertices
        faces = mesh.faces
        verts_cpp = [tuple(v) for v in vertices] # 转换为C++需要的格式
        tris_cpp = [tuple(t) for t in faces]
        sdf_calculator.load_mesh(verts_cpp, tris_cpp)
        return sdf_calculator

    def cross_check_collision(self, num_sample_points=500):
        # 互相检测表面点
        t_start = time.time()
        points1 = self.obj1_mesh.sample(num_sample_points)
        points2 = self.obj2_mesh.sample(num_sample_points)
        t_end = time.time()
        print("use sample time:",t_end - t_start)
        
        # 检测obj1的点是否在obj2内部
        collision1 = any(self.obj2_sdf.signed_distance(p) < 0 for p in points1)
        # 检测obj2的点是否在obj1内部
        collision2 = any(self.obj1_sdf.signed_distance(p) < 0 for p in points2)
        
        return collision1 or collision2

    def check_collision_hybrid(self, num_sample_points=500,a_to_b=True,threshold=0.0):
        # 第一步：快速AABB（axis-aligned bounding box）检测
        obj1_bounds = self.obj1_mesh.bounds
        obj2_bounds = self.obj2_mesh.bounds
        
        if not (obj1_bounds[0][0] <= obj2_bounds[1][0] and 
                obj1_bounds[1][0] >= obj2_bounds[0][0] and
                obj1_bounds[0][1] <= obj2_bounds[1][1] and 
                obj1_bounds[1][1] >= obj2_bounds[0][1] and
                obj1_bounds[0][2] <= obj2_bounds[1][2] and 
                obj1_bounds[1][2] >= obj2_bounds[0][2]):
            return False,-10000  # 包围盒不相交
        
        # 第二步：精确SDF检测
        if a_to_b:
            points = self.obj1_mesh.sample(num_sample_points)  # 增加采样点
            min_distance = 10000000
            for p in points:
                dis = self.obj2_sdf.signed_distance(p)
                if dis < threshold:
                    min_distance = dis
            return min_distance < threshold,min_distance

        else:
            points = self.obj2_mesh.sample(num_sample_points)  # 增加采样点
            min_distance = 10000000
            for p in points:
                dis = self.obj1_sdf.signed_distance(p) 
                if dis < threshold:
                    min_distance = dis
            return min_distance < threshold,min_distance
    
    def apply_transform(self,dT,obj_id):
        if obj_id == 1:
            self.obj1_mesh.apply_transform(dT)
        elif obj_id == 2:
            self.obj2_mesh.apply_transform(dT)
        else:
            raise ValueError("obj_id must be 1 or 2")
    
    def apply_translation(self,delta_t,obj_id):
        if obj_id == 1:
            self.obj1_mesh.apply_translation(delta_t)
        elif obj_id == 2:
            self.obj2_mesh.apply_translation(delta_t)
        else:
            raise ValueError("obj_id must be 1 or 2")
    

if __name__ == "__main__":
    gripper_path = os.path.join(PROJECT_ROOT_DIR,"meshes/gripper/meshes/l_gripper_tip_scaled.stl")   
    object_path = os.path.join(PROJECT_ROOT_DIR,"meshes/objs/part15.STL")

    detector = CollisionDetector(gripper_path,object_path,scalar_1=1000.0,scalar_2=1.0,use_convex_hull_1=True,use_convex_hull_2=True,T_1=None,T_2=None)

    for _ in range(10):
        detector.apply_translation([0.0, 0.0, 50],obj_id=2)  # 向右移动10个单位,delta_type

        t_start = time.time()
        res_1,distance = detector.check_collision_hybrid(num_sample_points=500,a_to_b=True)
        t_end = time.time()
        print("use time:",t_end - t_start)
        print("res_1:",res_1) # True

        t_start = time.time()
        res_2 = detector.cross_check_collision(num_sample_points=500)
        t_end = time.time()
        print("use time:",t_end - t_start)
        print("res_2:",res_2) # True

        scene = trimesh.Scene([detector.obj1_mesh, detector.obj2_mesh])
        scene.show(viewer="gl")