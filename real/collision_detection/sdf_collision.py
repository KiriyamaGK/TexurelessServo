import time
import numpy as np
from real.collision_detection.build.sdf_module import SDFCalculator
from utils.load_gripper_mesh import load_urdf_with_joint_angles
from utils.mesh import load_mesh, sample_face_points
import trimesh
from utils.paths import PROJECT_ROOT_DIR
import os
import threading
import random
import open3d as o3d
import numpy as np
import time
from utils.paths import PROJECT_ROOT_DIR
import os

class CollisionDetector():
    def __init__(self,obj1_mesh_path, obj2_mesh_path,scalar_1=1.0,scalar_2=1.0,use_convex_hull_1 = True,use_convex_hull_2 = True, cali_T = None):
        # The world frame is at the bottom of the object,where is also the initial local frame of the object and the gripper.
        # axis x,z is defined parallel to the object surface,there x points from ear to curve,z from curve right to left
        # ,and x,z axis counter clockwise 22.5 deg. y is along the normal of the object surface from bottom to top.
        # cali_T: transform matrix from obj's desired local frame to gripper's desired local frame.

        self.obj1_mesh_list,self.obj1_mesh_info,_ = load_urdf_with_joint_angles(obj1_mesh_path,joint_angles={'R2': 1.035},mesh_scale=scalar_1) # gripper is 1, object is 2 ,1.035 correspond to the maximum width
        self.obj2_mesh = load_mesh(obj2_mesh_path,scalar_2)

        assert len(self.obj1_mesh_info) == len(self.obj1_mesh_list)
        
        assert not use_convex_hull_1

        if use_convex_hull_2:
            self.obj2_mesh = self.obj2_mesh.convex_hull
        
        self.obj2_sdf = self.create_sdf_calculator(self.obj2_mesh)

        #transform matrix from world frame to local frame(determined by trimesh and stl file)
        self.wpos1_T = np.eye(4)
        self.wpos2_T = np.eye(4)

        #transform matrix from local frame to desired local frame
        self.lcl_deslcl_1 = np.array([[0.,1.,0.,0.],
                                      [-1.,0.,0.,0.],
                                      [0.,0.,1.,0.25326], #this value is the length of gripper,which is also relative with the gripper width
                                      [0.,0.,0.,1.]])

        self.lcl_deslcl_2 = np.array([[np.cos(np.pi/8),  np.sin(np.pi/8),  0.,                0.],
                                      [0.,               0.,               1.,                0.],
                                      [np.sin(np.pi/8),  -np.cos(np.pi/8), 0.,               0],
                                      [0.,               0.,               0.,                1.]])

        assert cali_T is not None
        self.initial_calibration(cali_T)

    def initial_calibration(self,cali_T):
        """
        calibrate initial wpos1_T according to given cali_T,which is bias between obj and gripper.
        :param cali_T:
        :return:
        """
        # self.wpos2_T @ self.lcl_deslcl_2 @ T_cali @ np.linalg.inv(self.lcl_deslcl_1) = T_apply @ self.wpos1_T
        T_apply = self.wpos2_T @ self.lcl_deslcl_2 @ cali_T @ np.linalg.inv(self.lcl_deslcl_1) @ np.linalg.inv(self.wpos1_T)
        self.apply_transform(T_apply,1)

    def create_sdf_calculator(self,mesh):
        sdf_calculator = SDFCalculator()
        if not isinstance(mesh,list):
            vertices = mesh.vertices
            faces = mesh.faces
            verts_cpp = [tuple(v) for v in vertices]
            tris_cpp = [tuple(t) for t in faces]
            sdf_calculator.load_mesh(verts_cpp, tris_cpp)
        else:
            verts_cpp = [tuple(v) for me in mesh for v in me.vertices]
            tris_cpp = [tuple(t) for me in mesh for t in me.faces]
            sdf_calculator.load_mesh(verts_cpp, tris_cpp)
        return sdf_calculator

    def check_collision(self, num_sample_points=500,threshold=0.0,key_list = []):
        '''
        key_list: list of link_name, if empty, check all links in obj1_mesh_list
        threshold: mm
        '''
        # 互相检测表面点
        t_start = time.time()
        if not len(key_list):
            points1 = [mesh.sample(num_sample_points) for mesh in self.obj1_mesh_list]
            points1 = np.concatenate(points1, axis=0)
        else:
            points1 = [mesh.sample(num_sample_points) for idx,mesh in enumerate(self.obj1_mesh_list) if self.obj1_mesh_info[idx]['link_name'] in key_list]
            points1 = np.concatenate(points1, axis=0)

        t_end = time.time()
        # print("use sample time:",t_end - t_start)
        
        min_distance = 10000000
        for p in points1:
            dis = self.obj2_sdf.signed_distance(p)
            if dis < min_distance:
                min_distance = dis
        return min_distance*1000 < threshold,min_distance*1000 #m2mm

    
    def apply_transform(self,dT,obj_id):
        """ apply transform to gripper or object and update state """

        if obj_id == 1:
            for mesh in self.obj1_mesh_list:
                mesh.apply_transform(dT)
            self.wpos1_T = dT @ self.wpos1_T

        elif obj_id == 2:
            self.obj2_mesh.apply_transform(dT)
            self.wpos2_T = dT @ self.wpos2_T
        else:
            raise ValueError("obj_id must be 1 or 2")
    
    def apply_translation(self,delta_t,obj_id):
        """ apply transform to gripper or object and update state """
        if isinstance(delta_t, list):
            delta_t = np.array(delta_t)

        if obj_id == 1:
            for mesh in self.obj1_mesh_list:
                mesh.apply_translation(delta_t)
            self.wpos1_T[0:3,3]+=delta_t

        elif obj_id == 2:
            self.obj2_mesh.apply_translation(delta_t)
            self.wpos2_T[0:3,3]+=delta_t
        else:
            raise ValueError("obj_id must be 1 or 2")

    def update_pos(self,delta_T):
        """

        :param delta_T: transform in local frame
        :return:
        """
        # T_apply * self.wpos1_T = self.wpos1_T @ self.lcl_des_lcl @ delta_t @ np.linalg.inv(self.lcl_des_lcl)

        T_apply = self.wpos1_T @ self.lcl_deslcl_1 @ delta_T @ np.linalg.inv(self.lcl_deslcl_1) @ np.linalg.inv(self.wpos1_T)
        self.apply_transform(T_apply,1)



def trimesh_to_open3d(trimesh_mesh, color=None):
    """将trimesh转换为open3d mesh"""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    if color is not None:
        mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


class Open3DVisualizer:
    def __init__(self, detector):
        self.detector = detector
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Collision Detection', width=1000, height=800)
        self.meshes = {}

        # 设置渲染选项
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.95, 0.95, 0.95])
        render_option.mesh_show_back_face = True
        render_option.light_on = True

        # 添加几何体
        self.add_geometries()

        # 设置视角
        self.vis.reset_view_point(True)

    def add_geometries(self):
        """添加所有几何体到可视化器"""
        # 添加夹爪各部分
        for mesh, mesh_info in zip(self.detector.obj1_mesh_list, self.detector.obj1_mesh_info):
            link_name = mesh_info['link_name']
            o3d_mesh = trimesh_to_open3d(mesh, color=[0.6, 0.6, 0.8])  # 蓝色
            self.vis.add_geometry(o3d_mesh, reset_bounding_box=False)
            self.meshes[link_name] = o3d_mesh

        # 添加物体
        o3d_obj = trimesh_to_open3d(self.detector.obj2_mesh, color=[0.8, 0.2, 0.2])  # 红色
        self.vis.add_geometry(o3d_obj, reset_bounding_box=False)
        self.meshes["object"] = o3d_obj

    def update_geometries(self):
        """更新几何体位置"""
        updated = False

        # 更新夹爪各部分
        for mesh, mesh_info in zip(self.detector.obj1_mesh_list, self.detector.obj1_mesh_info):
            link_name = mesh_info['link_name']
            if link_name in self.meshes:
                # 创建新的顶点和面数组
                new_vertices = np.asarray(mesh.vertices)
                new_faces = np.asarray(mesh.faces)

                # 更新几何体
                self.meshes[link_name].vertices = o3d.utility.Vector3dVector(new_vertices)
                self.meshes[link_name].triangles = o3d.utility.Vector3iVector(new_faces)
                self.meshes[link_name].compute_vertex_normals()
                self.vis.update_geometry(self.meshes[link_name])
                updated = True

        # 更新物体
        if "object" in self.meshes:
            new_vertices = np.asarray(self.detector.obj2_mesh.vertices)
            new_faces = np.asarray(self.detector.obj2_mesh.faces)
            self.meshes["object"].vertices = o3d.utility.Vector3dVector(new_vertices)
            self.meshes["object"].triangles = o3d.utility.Vector3iVector(new_faces)
            self.meshes["object"].compute_vertex_normals()
            self.vis.update_geometry(self.meshes["object"])
            updated = True

        return updated

    def run_iteration(self):
        """运行一次迭代"""
        if self.update_geometries():
            # 处理事件并更新渲染
            self.vis.poll_events()
            self.vis.update_renderer()
            return True
        return False


# if __name__ == "__main__":
#     gripper_path = os.path.join(PROJECT_ROOT_DIR, "meshes/zhixing/crt_ctag2f120.urdf")
#     object_path = os.path.join(PROJECT_ROOT_DIR, "meshes/classical_part.STL")
#     cali_T = np.eye(4)
#     cali_T[0, 0] *= -1
#     cali_T[2, 2] *= -1
#     cali_T[2, 3] = 0
#
#     detector = CollisionDetector(gripper_path, object_path, scalar_1=1.0, scalar_2=0.001,
#                                  use_convex_hull_1=False, use_convex_hull_2=False, cali_T=cali_T)
#
#     # 创建可视化器
#     visualizer = Open3DVisualizer(detector)
#
#     # 初始渲染
#     visualizer.vis.poll_events()
#     visualizer.vis.update_renderer()
#     for i in range(15):
#         dT = np.eye(4)
#         dT[1, 3] += 0.03
#         detector.update_pos(dT)
#
#         # 更新可视化
#         visualizer.run_iteration()
#
#         t_start = time.time()
#         res_1, distance = detector.check_collision(num_sample_points=500, threshold=0.0)
#         t_end = time.time()
#
#         print(f"Iteration {i}:")
#         print("res_1:", res_1)
#         print("distance:", distance)
#         print("use time:", t_end - t_start)
#         print("---")
#
#         # 短暂暂停
#         time.sleep(0.1)
#
#     # # 保持窗口打开直到用户关闭
#     # print("可视化完成，按q关闭窗口或直接关闭窗口")
#     # while visualizer.vis.poll_events():
#     #     visualizer.vis.update_renderer()
#     #     time.sleep(0.01)
#
#     visualizer.vis.destroy_window()

if __name__ == "__main__":
    gripper_path = os.path.join(PROJECT_ROOT_DIR,"meshes/zhixing/crt_ctag2f120.urdf")
    object_path = os.path.join(PROJECT_ROOT_DIR,"meshes/classical_part.STL")
    cali_T = np.eye(4)
    cali_T[1,1] *= -1
    cali_T[2,2] *= -1
    cali_T[2,3] = 0.09

    detector = CollisionDetector(gripper_path,object_path,scalar_1=1.0,scalar_2=0.001,use_convex_hull_1=False,use_convex_hull_2=False,cali_T = cali_T)

    scene = trimesh.Scene()
    for mesh in detector.obj1_mesh_list:
        scene.add_geometry(mesh)
    scene.add_geometry(detector.obj2_mesh)

    from scipy.spatial.transform import Rotation as R

    scene.show(viewer="gl")

    for i in range(15):
        dT = np.eye(4)
        dT[0:3,0:3] = R.from_euler("xyz",[10,0,0],degrees=True).as_matrix()
        dT[2,3] = 0.01
        # dT[1,3]+=0.03
        detector.update_pos(dT)
        t_start = time.time()
        res_1,distance = detector.check_collision(num_sample_points=500,threshold=0.0)
        t_end = time.time()
        print("res_1:",res_1)
        print("distance:",distance)
        print("use time:",t_end - t_start)
        scene.show(viewer="gl")