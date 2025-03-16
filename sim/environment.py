import os
import json
import time
import numpy as np
import pybullet as p
import pybullet_data
import random
from scipy.spatial.transform import Rotation as R
from typing import Union, Optional
from sim.perception import Camera, CameraIntrinsic
from utils.transform import rmat2quat
from math import cos,sin

class Environment(object):
    def __init__(
        self,
        camera_config: Optional[Union[str, CameraIntrinsic]],
        objs_descriptor=20,
        use_max_rot=False
    ):
        cwT = np.array([[-1, 0, 0, 0],  # 外参，左上角c右下角w，前三列是c系在w系中的表示,按列排列，最后一列是从c的原点指向w的原点并在c系中表示
                        [0, 1, 0, -0.02],
                        [0, 0, -1, 2.1],
                        [0, 0, 0, 1]])
        wcT = np.linalg.inv(cwT)
        dTx = np.eye(4)
        dTx[0:3, 0:3] = R.from_rotvec(np.array([1, 0, 0]) * 4 / 180 * np.pi).as_matrix()

        self.angle_eps = 0.4  # degree
        self.dist_eps = 0.0007  # m
        self.init_trans=0.05    # m
        self.init_rot=60       # degree
        self.use_max_rot=use_max_rot

        self.obj_idx = 0
        self.obj_idx_pointer = 0
        self.obj_total_num=23
        self.objs_descriptor=objs_descriptor
        self.obj_scale_factor = 0.001                     #mm2m
        self.table_scale_factor = 3
        self.objStartPos = [0, 0, 1.88]
        self.objStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.world_ori_axis=np.array([0, 0.2, 1.9])

        self.gripStartPos = [0, 0, 2.25]
        self.gripStartOrientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        self.wcT_tar=wcT @ dTx
        self.cwT_tar=np.linalg.inv(self.wcT_tar)
        self.gwT_tar=np.eye(4)
        self.gwT_tar[0:3,0:3]=R.from_rotvec(np.array([np.pi, 0, 0])).as_matrix()
        self.gwT_tar[0:3,3]=np.array(self.gripStartPos)#绕世界系先旋转再平移的结果
        self.cgT=self.cwT_tar @ np.linalg.inv(self.gwT_tar)
        self.wgT_tar = self.wcT_tar @ self.cgT
        self.init_flag=False

        self.close_enough_flag=False
        self.vel_in_threshold_flag=False


        if isinstance(camera_config, str):
            with open(camera_config, "r") as j:
                config = json.load(j)
            camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
        elif isinstance(camera_config, CameraIntrinsic):
            camera_intrinsic = camera_config
        self.camera = Camera(camera_intrinsic)
        if not hasattr(self, "client"):
            # self.client = p.connect(p.SHARED_MEMORY)
            self.client=p.connect(p.GUI)
            print("[INFO] Client (id = {}) initialized".format(self.client))
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=10, cameraPitch=-45, cameraTargetPosition=self.objStartPos)
        p.setRealTimeSimulation(0, physicsClientId=self.client) #用于设置仿真是否实时进行。参数 0 表示关闭实时仿真
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.axes_cam = DebugAxesCam(self.client)
        self.axes_gripper = DebugAxesGripper(self.client)

        self.determine_obj_idxes()

    def determine_obj_idxes(self):
        if isinstance(self.objs_descriptor,int):
            if self.objs_descriptor>0:
                self.obj_idxs=list(range(1, self.objs_descriptor+ 1))
            elif self.objs_descriptor<0:
                self.obj_idxs = list(range(self.obj_total_num+self.objs_descriptor+1, self.obj_total_num + 1))
            else:
                raise RuntimeError('objs_descriptor must not be 0 as an integer')
        elif isinstance(self.objs_descriptor, list):
            self.obj_idxs = sorted(self.objs_descriptor)
            if len(self.obj_idxs)!=len(set(self.obj_idxs)):
                raise RuntimeError('objs_descriptor must not contain duplicates')
            if self.obj_idxs[0]<1 or self.obj_idxs[-1]>self.obj_total_num:
                raise RuntimeError('objs_descriptor has invalid index')
        else:
            raise RuntimeError('objs_descriptor must be an integer or a list')
    def gen_scene(self):
        self.clear_axes()

        p.resetSimulation(physicsClientId=self.client)  # 重置仿真环境
        p.setRealTimeSimulation(0, physicsClientId=self.client)

        ObjFileName = os.path.join("../meshes/objs", str(self.obj_idx) + ".urdf")
        TableFileName = "../meshes/table/table.urdf"
        GripFileName = "../meshes/gripper/custom_wsg50_with_r2d2_gripper.sdf"

        self.planeId = p.loadURDF("plane.urdf")
        self.objId = p.loadURDF(ObjFileName, globalScaling=self.obj_scale_factor, useFixedBase=True)
        self.tableId = p.loadURDF(TableFileName, globalScaling=self.table_scale_factor, useFixedBase=True)
        self.gripId = p.loadSDF(GripFileName, globalScaling=1)

        p.resetBasePositionAndOrientation(self.objId, self.objStartPos, self.objStartOrientation)
        p.resetBasePositionAndOrientation(self.gripId[0], self.gripStartPos, self.gripStartOrientation)
        p.addUserDebugLine(self.world_ori_axis, self.world_ori_axis+np.array([0.05, 0, 0]), lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
        p.addUserDebugLine(self.world_ori_axis, self.world_ori_axis+np.array([0, 0.05, 0]), lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
        p.addUserDebugLine(self.world_ori_axis, self.world_ori_axis+np.array([0, 0, 0.05]), lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0)

        self.axes_gripper.update(self.wgT)
        self.axes_cam.update(self.wcT)

        p.setGravity(0, 0, -10, physicsClientId=self.client)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)

    def sample_init_pos(self):
        if not self.use_max_rot:
            ang = random.uniform(0, self.init_rot) * (random.randint(0, 1) - 0.5) * 2
        else:
            ang=self.init_rot*(random.randint(0, 1) - 0.5) * 2
        ori = random.uniform(0, np.pi*2)

        dT=np.eye(4)
        dT[0:3,0:3]=R.from_rotvec(np.array([0, 0, ang * np.pi / 180])).as_matrix()
        dT[0:2,3]=np.array([cos(ori)*self.init_trans, sin(ori)*self.init_trans])  #绕世界系先旋转再平移的结果
        # dT[0:2, 3] = np.array([0,0])  # 绕世界系先旋转再平移的结果
        self.wgT=dT@self.wgT_tar
        self.gwT=np.linalg.inv(self.wgT)
        self.cwT=self.cgT@self.gwT
        self.wcT=dT@self.wcT_tar

    def init(self):
        if self.obj_idx_pointer>=len(self.obj_idxs):
            self.obj_idx_pointer=0
        self.obj_idx=self.obj_idxs[self.obj_idx_pointer]
        self.sample_init_pos()
        self.gen_scene()
        self.init_flag = True
        self.obj_idx_pointer+=1
        self.task_timer=time.time()
        self.vel_timer=time.time()

    def observation(self,random_light_dir=False,use_prob=False):
        if random_light_dir:# 50% percent random_light_dir
            if use_prob:
                rand=random.uniform(0,1)
                if rand<0.5:
                    p.configureDebugVisualizer(lightPosition=random_light_direction())
            else:
                p.configureDebugVisualizer(lightPosition=random_light_direction())
        else:
            p.configureDebugVisualizer(lightPosition=[0,0,1])
        frame = self.camera.render(self.cwT, self.client)
        rgb = frame.color_image()
        return rgb

    def act_to_goal(self):
        self.wgT = self.wgT_tar
        self.gwT = self.gwT_tar
        self.cwT = self.cwT_tar
        self.wcT = self.wcT_tar

        p.resetBasePositionAndOrientation(self.gripId[0], self.wgT_tar[0:3, 3], rmat2quat(self.gwT_tar[0:3, 0:3]))
        self.axes_gripper.update(self.wgT_tar)
        self.axes_cam.update(self.wcT_tar)

    def act_with_abs_dict(self,pos:dict):
        # 更新矩阵,dT是对于世界坐标系下的变化量
        self.wgT = pos["wgT"]
        self.gwT = pos["gwT"]
        self.cwT = pos["cwT"]
        self.wcT = pos["wcT"]
        # 更新夹爪位置
        p.resetBasePositionAndOrientation(self.gripId[0], self.wgT[0:3, 3], rmat2quat(self.gwT[0:3, 0:3]))
        # 更新夹爪坐标轴
        self.axes_gripper.update(self.wgT)
        # 更新相机坐标轴
        self.axes_cam.update(self.wcT)

    def action(self, dT):
        #更新矩阵,dT是对于世界坐标系下的变化量
        self.wgT = dT @ self.wgT
        self.gwT = np.linalg.inv(self.wgT)
        self.cwT = self.cgT @ self.gwT
        self.wcT = dT @ self.wcT

        #更新夹爪位置
        p.resetBasePositionAndOrientation(self.gripId[0], self.wgT[0:3,3], rmat2quat(self.gwT[0:3,0:3]))
        #更新夹爪坐标轴
        self.axes_gripper.update(self.wgT)
        #更新相机坐标轴
        self.axes_cam.update(self.wcT)

    def reinit(self):
        if  self.need_reinit():
            self.init()
        else:
            self.init_flag=False
        return self.init_flag

    def reinit_eval(self):
        rtn_dict=self.need_reinit_eval()
        need_reinit=rtn_dict["need_reinit"]
        if need_reinit:
            self.init()
        else:
            self.init_flag=False
        return rtn_dict

    def clear_axes(self):
        self.axes_cam.clear()
        self.axes_gripper.clear()

    def need_reinit(self):
        err_dict = self.compute_error(self.wcT_tar, self.wcT)
        if err_dict["close_enough"]:
            return True
        else:
            return False

    def need_reinit_eval(self):
        err_dict=self.compute_error(self.wcT_tar, self.wcT)

        if time.time()-self.task_timer>=self.time_up_bound:
            return {"need_reinit": True,
                    "dist": err_dict["dist"],
                    "angle": err_dict["angle"]
                    }

        else:
            if time.time()-self.vel_timer>=self.in_threshold_range_time and self.vel_in_threshold_flag:
                return {"need_reinit": True,
                        "dist": err_dict["dist"],
                        "angle": err_dict["angle"]
                        }
            else:
                return {"need_reinit": False,
                        "dist": err_dict["dist"],
                        "angle": err_dict["angle"]
                        }

    def compute_error(self, T0, T1):
        dT = np.linalg.inv(T0) @ T1
        angle = np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec()) / np.pi * 180
        dist = np.linalg.norm(dT[:3, 3])
        self.close_enough_flag=(angle < self.angle_eps) and (dist < self.dist_eps)
        return {
            "close_enough":self.close_enough_flag,
            "dist":dist,
            "angle":angle,
        }
    def determine_vel_in_threshold(self,vel_tr,vel_rot): #self.vel_in_threshold_flag = True当且仅当速度在误差内
        if not self.vel_in_threshold_flag:
            if vel_tr<=self.trans_vel_threshold and vel_rot<=self.rot_vel_threshold:
                self.vel_in_threshold_flag=True
                self.vel_timer=time.time()

        else:
            if vel_tr>self.trans_vel_threshold or vel_rot>self.rot_vel_threshold:
                self.vel_in_threshold_flag = False

    def setup_stop_policy(self,metrics:dict):
        self.rot_vel_threshold=metrics["rot_vel_threshold"]  #deg
        self.trans_vel_threshold=metrics["trans_vel_threshold"] #m
        self.time_up_bound = metrics["use_time_upperbound"] #maximum used time during a rollout
        self.in_threshold_range_time = metrics["in_threshold_range_time"] #maximum time stay in error threshold before entering the next rollout

    def return_cur_pos_info(self):
        return {
        'wgT':self.wgT,
        'gwT':self.gwT,
        'wcT':self.wcT,
        'cwT':self.cwT,
        }
    def return_hand_eye_info(self):
        return {
        'cgT':self.cgT,
        }
    def return_tar_pos_info(self):
        return {
        'wgT_tar':self.wgT_tar,
        'gwT_tar':self.gwT_tar,
        'wcT_tar':self.wcT_tar,
        'cwT_tar':self.cwT_tar,
        }

class DebugAxesCam(object):
    """Visualize axes, red for x axis, green for y axis, blue for z axis"""

    def __init__(self, client=0):
        self.uids = [-1, -1, -1]
        self.client = client

    def update(self, pose):  # 根据位姿显示相机坐标轴
        pos = pose[:3, 3]
        rot3x3 = pose[:3, :3]
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0],  # p.addUserDebugLine:添加调试线
                                          replaceItemUniqueId=self.uids[0], physicsClientId=self.client)
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0],
                                          replaceItemUniqueId=self.uids[1], physicsClientId=self.client)
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1],
                                          replaceItemUniqueId=self.uids[2], physicsClientId=self.client)

    def clear(self):
        p.removeUserDebugItem(self.uids[0], physicsClientId=self.client)
        p.removeUserDebugItem(self.uids[1], physicsClientId=self.client)
        p.removeUserDebugItem(self.uids[2], physicsClientId=self.client)
        self.uids = [-1, -1, -1]

class DebugAxesGripper(object):
    """Visualize axes, red for x axis, green for y axis, blue for z axis"""

    def __init__(self, client=0):
        self.uids = [-2, -2, -2]
        self.client = client

    def update(self, pose):  # 根据位姿显示相机坐标轴
        pos = pose[:3, 3]
        rot3x3 = pose[:3, :3]
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = p.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0],  # p.addUserDebugLine:添加调试线
                                          replaceItemUniqueId=self.uids[0], physicsClientId=self.client)
        self.uids[1] = p.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0],
                                          replaceItemUniqueId=self.uids[1], physicsClientId=self.client)
        self.uids[2] = p.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1],
                                          replaceItemUniqueId=self.uids[2], physicsClientId=self.client)

    def clear(self):
        p.removeUserDebugItem(self.uids[0], physicsClientId=self.client)
        p.removeUserDebugItem(self.uids[1], physicsClientId=self.client)
        p.removeUserDebugItem(self.uids[2], physicsClientId=self.client)
        self.uids = [-2, -2, -2]

def random_light_direction():
    # 随机生成一个单位向量作为光照方向
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    sintheta = np.sqrt(1 - costheta**2)*random.choice([1, -1])
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = costheta
    return [x, y, z]