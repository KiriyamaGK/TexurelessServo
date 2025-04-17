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
        init_horizon_trans=0.05,
        init_vertical_trans=0.05,
        init_transform_frame="grip",
        using_minus_vertical=False,
        init_rot=60,
        use_max_rot=False,
        use_max_trans=False,
        dof=3,
        angle_eps = 0.4,
        dist_eps=0.001,
        depth_info=None,
        pose_and_orientations=None,
    ):
        '''
        self.obj_idx:当前物体id
        self.obj_idxs:所有物体总id，list
        self.obj_idx_pointer:指针,self.obj_idxs=self.obj_idxs[self.obj_idx_pointer]
        '''
        if pose_and_orientations is None:
            pose_and_orientations={
                "cam1": {
                    "cwT_trans": [0, -0.09, 2.1],
                    "rotation": 30
                },
                "cam2": {
                    "c2wT_trans": [0, -0.08, 2.1],
                    "rotation": 30
                },
                "grip": {
                    "start_pos_z": 2.25,
                    "offset_z": 0
                },
            }

        #cam1
        cwT = np.array([[-1., 0, 0, 0],  # 左上角c右下角w，前三列是c系在w系中的表示,按列排列，最后一列是从c的原点指向w的原点并在c系中表示
                        [0, 1., 0, 0],
                        [0, 0, -1., 0],
                        [0, 0, 0, 1.]])
        cwT[0:3,3]=np.array(pose_and_orientations["cam1"]["cwT_trans"])
        wcT = np.linalg.inv(cwT)
        dTx = np.eye(4)
        dTx[0:3, 0:3] = R.from_rotvec(np.array([1, 0, 0]) * pose_and_orientations["cam1"]["rotation"] / 180 * np.pi).as_matrix()

        #cam2
        c2wT = np.array([[1., 0, 0, 0],  # 左上角c右下角w，前三列是c系在w系中的表示,按列排列，最后一列是从c的原点指向w的原点并在c系中表示
                        [0, -1., 0, 0],
                        [0, 0, -1., 0],
                        [0, 0, 0, 1.]])
        c2wT[0:3,3]=np.array(pose_and_orientations["cam2"]["c2wT_trans"])
        wc2T = np.linalg.inv(c2wT)
        dT2x = np.eye(4)
        dT2x[0:3, 0:3] = R.from_rotvec(np.array([1, 0, 0]) * pose_and_orientations["cam2"]["rotation"] / 180 * np.pi).as_matrix()

        #determine grip startpos
        self.gripStartPos = [0, 0, 0]   #it is for determining hand-eye, not actual goal position,which should be obtained by adding to offset
        self.gripStartPos[2]=pose_and_orientations["grip"]["start_pos_z"]

        #others
        self.dof = dof
        assert self.dof in [3,6]

        self.angle_eps = angle_eps  # degree
        self.dist_eps = dist_eps  # m

        self.init_horizon_trans = init_horizon_trans # m
        self.init_vertical_trans=init_vertical_trans if self.dof==6 else 0   # m
        self.init_transform_frame = init_transform_frame
        self.using_minus_vertical = using_minus_vertical

        if self.dof == 3:
            assert isinstance(init_rot,(int, float))
            self.init_rot = np.array([0,0,init_rot]) # degree
        else:
            assert isinstance(init_rot, list) and len(init_rot) == 3
            self.init_rot = np.array(init_rot)  # degree


        self.use_max_rot=use_max_rot
        self.use_max_trans=use_max_trans

        self.obj_idx = 0
        self.obj_idx_pointer = 0
        self.obj_total_num=23
        self.objs_descriptor=objs_descriptor
        self.obj_scale_factor = 0.0007                     #mm2m
        self.table_scale_factor = 3
        self.objStartPos = [0, 0, 1.88]
        self.objStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.world_ori_axis=np.array([0, 0.2, 1.9]) #世界坐标系设置在原点，但显示的时候按照[0, 0.2, 1.9]平移

        self.gripStartOrientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        self.gwT_tar=np.eye(4)
        self.gwT_tar[0:3,0:3]=R.from_rotvec(np.array([np.pi, 0, 0])).as_matrix()
        self.gwT_tar[0:3,3]=np.array(self.gripStartPos)#绕世界系先旋转再平移的结果

        self.wcT_tar = wcT.copy() @ dTx.copy()
        # print(self.wcT_tar)
        self.cwT_tar = np.linalg.inv(self.wcT_tar.copy())
        self.cgT=self.cwT_tar.copy() @ np.linalg.inv(self.gwT_tar.copy())
        self.wgT_tar = self.wcT_tar.copy() @ self.cgT.copy()

        if self.dof == 6:
            self.wc2T_tar = wc2T.copy() @ dT2x.copy()
            # print(self.wc2T_tar)
            self.c2wT_tar = np.linalg.inv(self.wc2T_tar.copy())
            self.c2gT = self.c2wT_tar.copy() @ np.linalg.inv(self.gwT_tar.copy())

        self.init_flag=False
        self.close_enough_flag=False
        self.vel_in_threshold_flag=False
        self.depth_info=depth_info if depth_info is not None else {"utilized":False,"normalize_scaler": 10}

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
        if self.dof==6:
            self.axes_cam2 = DebugAxesCam2(self.client)
        self.axes_gripper = DebugAxesGripper(self.client)

        self.determine_obj_idxes()
        self.determine_offsetz_idxes(pose_and_orientations["grip"]["offset_z"])

    def determine_obj_idxes(self):
        '''
        self.obj_idxs:list
        self.objs_descriptor:int or list
        '''
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

    def determine_offsetz_idxes(self,gripOffset_z):
        if isinstance(gripOffset_z,list):
            assert len(gripOffset_z)==self.obj_total_num
        else:
            assert isinstance(gripOffset_z,(int,float))

        gripOffset_z_dict={}
        for id in range(self.obj_total_num):
            gripOffset_z_dict[id+1]=gripOffset_z[id] if isinstance(gripOffset_z,list) else gripOffset_z
        self.gripOffset_z_dict=gripOffset_z_dict


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
        p.resetBasePositionAndOrientation(self.gripId[0], self.wgT_tar[0:3, 3], rmat2quat(self.gwT_tar[0:3, 0:3]))
        p.addUserDebugLine(self.world_ori_axis, self.world_ori_axis+np.array([0.05, 0, 0]), lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
        p.addUserDebugLine(self.world_ori_axis, self.world_ori_axis+np.array([0, 0.05, 0]), lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
        p.addUserDebugLine(self.world_ori_axis, self.world_ori_axis+np.array([0, 0, 0.05]), lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0)

        self.axes_gripper.update(self.wgT)
        self.axes_cam.update(self.wcT)
        if self.dof==6:
            self.axes_cam2.update(self.wc2T)

        p.setGravity(0, 0, -10, physicsClientId=self.client)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.client)

    def update_goal_position(self):
        self.gwT_tar[2,3]=self.gripStartPos[2]+self.gripOffset_z_dict[self.obj_idx]
        self.wgT_tar=np.linalg.inv(self.gwT_tar.copy())
        self.cwT_tar=self.cgT.copy()@self.gwT_tar.copy()
        self.wcT_tar=np.linalg.inv(self.cwT_tar.copy())
        if self.dof==6:
            self.c2wT_tar = self.c2gT.copy() @ self.gwT_tar.copy()
            self.wc2T_tar = np.linalg.inv(self.c2wT_tar.copy())

    def sample_init_pos(self):
        assert self.init_transform_frame in ["grip","cam"]
        if not self.use_max_rot:
            if random.uniform(0, 1) < 0.95:
                ang = np.array([random.uniform(0, self.init_rot[i])* (random.randint(0, 1) - 0.5) * 2 for i in range(self.init_rot.shape[0])])
            else:
                ang = np.array([self.init_rot[i] * (random.randint(0, 1) - 0.5) * 2 for i in range(self.init_rot.shape[0])])
        else:
            ang =np.array([self.init_rot[i]* (random.randint(0, 1) - 0.5) * 2 for i in range(self.init_rot.shape[0])])
        print("angle:",ang)
        ori = random.uniform(0, np.pi*2)
        trans_dev=self.init_horizon_trans if self.use_max_trans else self.init_horizon_trans*random.uniform(0, 1)

        dT=np.eye(4)
        dT[0:3,0:3]=R.from_rotvec(ang * np.pi / 180).as_matrix()
        dT[0:3,3]=np.array([cos(ori)*trans_dev, sin(ori)*trans_dev,-self.init_vertical_trans*random.uniform(0, 1)])
        if self.using_minus_vertical:
            dT[2,3]=dT[2,3]*(random.randint(0,1)-0.5)*2

        if self.init_transform_frame=="grip":
            self.wgT=self.wgT_tar@dT #绕夹爪系
            self.gwT=np.linalg.inv(self.wgT)
            self.cwT=self.cgT@self.gwT
            self.wcT=np.linalg.inv(self.cwT)

            if self.dof == 6:
                self.c2wT = self.c2gT @ self.gwT
                self.wc2T = np.linalg.inv(self.c2wT)

        else:
            if self.dof==3:
                self.wcT=self.wcT_tar@dT
                self.cwT=np.linalg.inv(self.wcT)
                self.wgT=self.wcT@self.cgT
                self.gwT=np.linalg.inv(self.wgT)
            else:
                #设置两个相机之间的中间坐标系
                mid_frame=self.wcT_tar.copy()
                mid_frame[0:3,3]=0.5*(self.wcT_tar[0:3,3].copy()+self.wc2T_tar[0:3,3].copy())
                cmT=self.cwT_tar@mid_frame.copy()
                c2mT = self.c2wT_tar@mid_frame.copy()

                mid_frame= mid_frame @ dT
                self.cwT=cmT@np.linalg.inv(mid_frame)
                self.wcT=np.linalg.inv(self.cwT)
                self.c2wT = c2mT @ np.linalg.inv(mid_frame)
                self.wc2T = np.linalg.inv(self.c2wT)


    def init(self):
        if self.obj_idx_pointer>=len(self.obj_idxs):
            self.obj_idx_pointer=0
        self.obj_idx=self.obj_idxs[self.obj_idx_pointer]
        self.update_goal_position()
        self.sample_init_pos()
        # print("init_wgT_tar:", self.wgT_tar)
        # print("init_wgT:", self.wgT)
        # print("init_error(mm):", np.linalg.norm(self.wgT_tar[0:3, 3] - self.wgT[0:3, 3]) * 1000)
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

        if self.dof==6:
            frame2 = self.camera.render(self.c2wT, self.client)
            return {"img_1":frame.color_image(), "img_2":frame2.color_image()} if not self.depth_info["utilized"] else {"img_1":frame.color_image(), "img_2":frame2.color_image(),"img_1_depth":frame.depth_image(), "img_2_depth":frame2.depth_image()}
        else:
            return {"img_1":frame.color_image()} if not self.depth_info["utilized"] else {"img_1":frame.color_image(), "img_1_depth":frame.depth_image()}#TODO:万一有问题可以/self.depth_info["normalize_scaler"]

    def act_to_goal(self):
        self.wgT = self.wgT_tar
        self.gwT = self.gwT_tar
        self.cwT = self.cwT_tar
        self.wcT = self.wcT_tar

        p.resetBasePositionAndOrientation(self.gripId[0], self.wgT_tar[0:3, 3], rmat2quat(self.gwT_tar[0:3, 0:3]))
        self.axes_gripper.update(self.wgT_tar)
        self.axes_cam.update(self.wcT_tar)

        if self.dof==6:
            self.c2wT = self.c2wT_tar
            self.wc2T = self.wc2T_tar
            self.axes_cam2.update(self.wc2T_tar)

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
        if self.dof==6:
            self.c2wT = pos["c2wT"]
            self.wc2T = pos["wc2T"]
            self.axes_cam2.update(self.wc2T)

    def action(self, dT):
        if self.dof==3:
            self.wgT = dT @ self.wgT
        else:
            self.wgT =  self.wgT @ dT

        self.gwT = np.linalg.inv(self.wgT)

        self.cwT = self.cgT @ self.gwT
        self.wcT = np.linalg.inv(self.cwT)

        # 更新夹爪坐标轴
        self.axes_gripper.update(self.wgT)
        # 更新相机坐标轴
        self.axes_cam.update(self.wcT)
        if self.dof == 6:
            self.c2wT = self.c2gT @ self.gwT
            self.wc2T = np.linalg.inv(self.c2wT)
            self.axes_cam2.update(self.wc2T)

        #更新夹爪位置
        p.resetBasePositionAndOrientation(self.gripId[0], self.wgT[0:3,3], rmat2quat(self.gwT[0:3,0:3].T))

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
        if self.dof==6:
            self.axes_cam2.clear()
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
            need_reinit=True
        else:
            if time.time()-self.vel_timer>=self.in_threshold_range_time and self.vel_in_threshold_flag:
                need_reinit=True
            else:
                need_reinit=False

        err_dict["need_reinit"]=need_reinit
        return err_dict

    def compute_error(self, T0, T1):
        dT = np.linalg.inv(T0) @ T1
        angle = np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec()) / np.pi * 180
        dist = np.linalg.norm(dT[:3, 3])
        z_error=abs(dT[2, 3])
        self.close_enough_flag=(angle < self.angle_eps) and (dist < self.dist_eps)
        return {
            "close_enough":self.close_enough_flag,
            "dist":dist,
            "angle":angle,
            "z_error":z_error,
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
        rtn_dict={
            'wgT':self.wgT,
            'gwT':self.gwT,
            'wcT':self.wcT,
            'cwT':self.cwT,
            }
        if self.dof==6:
            rtn_dict['c2wT']=self.c2wT
            rtn_dict['wc2T'] = self.wc2T
        return rtn_dict

    def return_hand_eye_info(self):
        rtn_dict={'cgT':self.cgT}
        if self.dof==6:
            rtn_dict['c2gT']=self.c2gT
        return rtn_dict

    def return_tar_pos_info(self):
        rtn_dict ={
        'wgT_tar':self.wgT_tar,
        'gwT_tar':self.gwT_tar,
        'wcT_tar':self.wcT_tar,
        'cwT_tar':self.cwT_tar,
        }
        if self.dof==6:
            rtn_dict['c2wT_tar']=self.c2wT_tar
            rtn_dict['wc2T_tar'] = self.wc2T_tar
        return rtn_dict

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

class DebugAxesCam2(object):
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