from real.perception import Camera
from real.fr_robot import FR_Robot
from real.gripper import Gripper
import numpy as np
import time
import random
from math import pi,sin,cos
from utils.transform import make_an_angle_in_180, euler2Matrix, rmat2euler_degree
from scipy.spatial.transform import Rotation as R
from itertools import product

class Environment:
    def __init__(self,robot_address,w,h,fps,cam_devices,use_devices_type,dof,down_to_grasp_distance,init,stop_policy,velocity):
        self.robot_ins=FR_Robot(robot_address)
        self.camera=Camera(devices=cam_devices,use_devices_type=use_devices_type,width=w, height=h, fps=fps)
        # self.gripper=Gripper()

        self.corner_1 = [-405.3097229003906, 199.8372497558593]  # 工作区域左上角点
        self.corner_2 = [-707.1678466796875, -249.3986206054687]  # 工作区域右下角点

        self.x_max = max(self.corner_1[0], self.corner_2[0])
        self.x_min = min(self.corner_1[0], self.corner_2[0])
        self.y_max = max(self.corner_1[1], self.corner_2[1])
        self.y_min = min(self.corner_1[1], self.corner_2[1])

        self.down_dis = down_to_grasp_distance  # 夹爪下移距离，mm

        self.dof = dof
        assert self.dof in [3, 6]

        self.angle_eps = stop_policy["angle_eps"]  # degree
        self.dist_eps = stop_policy["dist_eps"]  # mm
        self.trans_vel = velocity["trans_vel"]["value"]
        self.rot_vel = velocity["rot_vel"]["value"]

        self.init_horizon_trans = init["init_horizon_trans"]["value"]                             #最大平移距离，mm
        self.init_vertical_trans = init["init_vertical_trans"]["value"] if self.dof==6 else 0     #mm
        init_rot = init["init_rot"]["value"]                                                      #最大旋转偏差（axis-angle，deg）
        self.uniform_eval_settings = init["uniform_evaluation"]

        if self.dof == 3:
            assert isinstance(init_rot,(int, float))
            self.init_rot = np.array([0,0,init_rot]) # degree
        else:
            assert isinstance(init_rot, list) and len(init_rot) == 3
            self.init_rot = np.array(init_rot)  # degree

        self.use_max_rot = init["init_rot"]["use_max_rot"]
        self.use_max_trans = init["init_horizon_trans"]["use_max_h_trans"]
        self.using_max_v_trans = init["init_vertical_trans"]["using_max_v_trans"]
        self.using_minus_vertical = init["init_vertical_trans"]["using_minus"]
        self.conditioned_sampling = init["conditioned_sampling"] if "conditioned_sampling" in init else False

        if self.conditioned_sampling:
            assert (not self.use_max_rot) and (not self.use_max_trans) and (not self.using_max_v_trans)
            assert self.dof == 6
            self.max_transxy_points = int(self.init_horizon_trans / self.trans_vel[0])
            self.max_transz_points = int(self.init_vertical_trans / self.trans_vel[1])
            self.max_rot_points = int(np.linalg.norm(self.init_rot) / self.rot_vel)
            print("==================================================================")
            print("max_xy_points:",self.max_transxy_points)
            print("max_z_points:",self.max_transz_points)
            print("max_rot_points:",self.max_rot_points)
            self.demo_cond_p=0

        self.init_flag = False
        self.vel_in_threshold_flag = False

        self.task_timer = time.time()
        self.vel_timer = time.time()

        if self.uniform_eval_settings["utilized"]:
            self.return_evenly_distributed_poses()
            self.evenly_posid=0

        self.wgT_tar=None
        self.gwT_tar=None
        self.wgT=None
        self.gwT=None

    def place(self,p_0):
        '''
        在工作空间内随机选取一点p_1,机器人从初始位置p_0抓取零件放置到p_1
        :param p_0: 初始位置
        :param down_dis: 预抓取时夹爪下移距离
        :return: p_1:放置零件的新位置，p_1与p_0仅在x,y,rz上有差异
        '''

        #开始下去，抓零件
        pose = p_0.copy()
        pose[2] -= self.down_dis
        self.gripper.move_gripper(0,60,60)
        time.sleep(2)
        self.robot_ins.move_l(pose, tool=1, user=0, vel=10)
        self.gripper.move_gripper(3000, 60, 60)
        time.sleep(2)

        #开始上来
        self.robot_ins.move_l(p_0, tool=1, user=0, vel=30)

        # 开始平移到新位置
        p_1=[0,0,0,0,0,0]
        p_1[0]=random.uniform(self.x_min,self.x_max)
        p_1[1]=random.uniform(self.y_min,self.y_max)
        p_1[2]=p_0[2]
        p_1[3]=p_0[3]
        p_1[4]=p_0[4]
        p_1[5]=random.randint(130, 175)*(random.randint(0, 1) - 0.5) * 2
        self.robot_ins.move_l(p_1, tool=1, user=0, vel=30)

        #开始下去
        pose = p_1.copy()
        pose[2] -= (self.down_dis*0.9)
        self.robot_ins.move_l(pose, tool=1, user=0, vel=10)
        self.gripper.move_gripper(0,60,60)
        time.sleep(1)

        self.robot_ins.move_l(p_1, tool=1, user=0, vel=10)
        return p_1

    def cond_sample_init_pos_algo(self):
        if self.demo_cond_p == 0:
            trans_xy_points = random.randint(int(0.8 * self.max_transxy_points), self.max_transxy_points)
            trans_z_points = random.randint(int(0.8 * min(trans_xy_points, self.max_transz_points)),
                                            min(trans_xy_points, self.max_transz_points))
            rot_points = random.randint(int(0.8 * min(trans_xy_points, self.max_rot_points)),
                                        min(trans_xy_points, self.max_rot_points))
            self.demo_cond_p += 1
        elif self.demo_cond_p == 1:
            trans_z_points = random.randint(int(0.8 * self.max_transz_points), self.max_transz_points)
            trans_xy_points = random.randint(int(0.8 * min(trans_z_points, self.max_transxy_points)),
                                             min(trans_z_points, self.max_transxy_points))
            rot_points = random.randint(int(0.8 * min(trans_z_points, self.max_rot_points)),
                                        min(trans_z_points, self.max_rot_points))
            self.demo_cond_p += 1
        else:
            rot_points = random.randint(int(0.8 * self.max_rot_points), self.max_rot_points)
            trans_xy_points = random.randint(int(0.8 * min(rot_points, self.max_transxy_points)),
                                             min(rot_points, self.max_transxy_points))
            trans_z_points = random.randint(int(0.8 * min(rot_points, self.max_transz_points)),
                                            min(rot_points, self.max_transz_points))
            self.demo_cond_p = 0
        print("===============================================")
        print("trans_xy_points:", trans_xy_points)
        print("trans_z_points:", trans_z_points)
        print("rot_points:", rot_points)
        return trans_xy_points, trans_z_points, rot_points

    def return_evenly_distributed_poses(self):
        # 生成示例数据
        x_inteval=self.uniform_eval_settings["trans"]["x"]["inteval"]
        x_min=self.uniform_eval_settings["trans"]["x"]["range"][0]
        x_max=self.uniform_eval_settings["trans"]["x"]["range"][1]

        y_inteval = self.uniform_eval_settings["trans"]["y"]["inteval"]
        y_min = self.uniform_eval_settings["trans"]["y"]["range"][0]
        y_max = self.uniform_eval_settings["trans"]["y"]["range"][1]

        z_inteval = self.uniform_eval_settings["trans"]["z"]["inteval"]
        z_min = self.uniform_eval_settings["trans"]["z"]["range"][0]
        z_max = self.uniform_eval_settings["trans"]["z"]["range"][1]

        rx_inteval = self.uniform_eval_settings["rot"]["rx"]["inteval"]
        rx_min = self.uniform_eval_settings["rot"]["rx"]["range"][0]
        rx_max = self.uniform_eval_settings["rot"]["rx"]["range"][1]

        ry_inteval = self.uniform_eval_settings["rot"]["ry"]["inteval"]
        ry_min = self.uniform_eval_settings["rot"]["ry"]["range"][0]
        ry_max = self.uniform_eval_settings["rot"]["ry"]["range"][1]

        rz_inteval = self.uniform_eval_settings["rot"]["rz"]["inteval"]
        rz_min = self.uniform_eval_settings["rot"]["rz"]["range"][0]
        rz_max = self.uniform_eval_settings["rot"]["rz"]["range"][1]

        translations = list(product(
            np.arange(x_min, x_max+0.1*x_inteval, x_inteval),  # x
            np.arange(y_min, y_max+0.1*y_inteval, y_inteval),  # y
            np.arange(z_min, z_max+0.1*z_inteval, z_inteval)  # z
        ))
        rotations = list(product(
            np.arange(rx_min, rx_max+0.1*rx_inteval, rx_inteval),  # rx
            np.arange(ry_min, ry_max+0.1*ry_inteval, ry_inteval),  # ry
            np.arange(rz_min, rz_max+0.1*rz_inteval, rz_inteval)  # rz
        ))

        self.all_even_poses = []
        for tx, ty, tz in translations:
            for rx, ry, rz in rotations:
                dT=np.eye(4)
                dRx=R.from_rotvec(np.array([rx/180*np.pi, 0, 0])).as_matrix()
                dRy=R.from_rotvec(np.array([0, ry/180*np.pi, 0])).as_matrix()
                dRz=R.from_rotvec(np.array([0, 0, rz/180*np.pi])).as_matrix()
                dT[0:3,3] = tx, ty, tz
                dT[0:3,0:3]=dRz @ dRy @ dRx
                self.all_even_poses.append({
                    'x': tx, 'y': ty, 'z': tz,
                    'rx': rx, 'ry': ry, 'rz': rz,
                    'dT': dT
                })

    def sample_init_pos(self):
        """
        get delta_T : g_tar_g_T
        """
        # print("===============================")
        # print(self.evenly_posid)
        if self.uniform_eval_settings["utilized"]:
            dT=self.all_even_poses[self.evenly_posid]["dT"]
            x = self.all_even_poses[self.evenly_posid]["x"]
            y = self.all_even_poses[self.evenly_posid]["y"]
            z = self.all_even_poses[self.evenly_posid]["z"]
            rx=self.all_even_poses[self.evenly_posid]["rx"]
            ry=self.all_even_poses[self.evenly_posid]["ry"]
            rz=self.all_even_poses[self.evenly_posid]["rz"]
            print("------------------------------------")
            print(x, y, z, rx, ry, rz)
            print("------------------------------------")

        else:
            dT = np.eye(4)
            ori = random.uniform(0, np.pi * 2)

            if not self.conditioned_sampling:
                #rot
                if (self.init_rot==np.array([0,0,0])).all():
                    dT[0:3,0:3]=np.eye(3)
                else:
                    if not self.use_max_rot:
                        if random.uniform(0, 1) < 0.95:
                            ang = np.array([random.uniform(0, self.init_rot[i])* (random.randint(0, 1) - 0.5) * 2 for i in range(self.init_rot.shape[0])])
                        else:
                            ang = np.array([self.init_rot[i] * (random.randint(0, 1) - 0.5) * 2 for i in range(self.init_rot.shape[0])])
                    else:
                        ang =np.array([self.init_rot[i]* (random.randint(0, 1) - 0.5) * 2 for i in range(self.init_rot.shape[0])])
                    print("angle:",ang)
                    dT[0:3, 0:3] = R.from_rotvec(ang * np.pi / 180).as_matrix()
                #trans
                trans_dev=self.init_horizon_trans if self.use_max_trans else np.sqrt(random.uniform(0, self.init_horizon_trans**2))
                dT[0:3,3]=np.array([cos(ori)*trans_dev, sin(ori)*trans_dev,-self.init_vertical_trans])

                if not self.using_max_v_trans:
                    dT[2, 3]*=random.uniform(0, 1)
                if self.using_minus_vertical:    #TODO:对z轴从下往上做了限制
                    if random.randint(0,1)>0.65:
                        dT[2,3]=max(dT[2,3],-10)*(-1)
            else:
                # print("1")
                trans_xy_points, trans_z_points, rot_points = self.cond_sample_init_pos_algo()
                trans_dev = trans_xy_points*self.trans_vel[0]
                vertical_dev=trans_z_points*self.trans_vel[1]
                if self.using_minus_vertical:
                    if random.uniform(0,1)>0.65:
                        vertical_dev=min(10,vertical_dev)*(-1)    #TODO:对z轴从下往上做了限制
                # dr = np.array([random.uniform(0, 5) for _ in range(3)])#TODO:记得注释这四行
                # rot_dev_mat = R.from_rotvec(self.init_rot * np.pi / 180).as_matrix() @ R.from_rotvec(
                #     dr * np.pi / 180).as_matrix()
                # rot_dev_vec = R.from_matrix(rot_dev_mat).as_rotvec()
                # rot_dev_vec /= np.linalg.norm(rot_dev_vec) / (rot_points * self.rot_vel)

                rot_dev_vec = np.array([random.uniform(0.5*self.init_rot[i], self.init_rot[i])* (random.randint(0, 1) - 0.5) * 2 for i in range(self.init_rot.shape[0])])
                rot_dev_vec/=np.linalg.norm(rot_dev_vec)/(rot_points*self.rot_vel) #TODO:记得解注释这两行
                print("trans_dev:",trans_dev)
                print("vertical_dev:",vertical_dev)
                print("rot_dev_vec:",rot_dev_vec)

                dT[0:3, 0:3] = R.from_rotvec(rot_dev_vec * np.pi / 180).as_matrix()
                dT[0:3, 3] = np.array([cos(ori) * trans_dev, sin(ori) * trans_dev, -vertical_dev])

        self.g_tar_g_init_T = dT

    def set_target_coordinate(self,pos=None,use_cur=True):  ##TODO: new
        """
        :param pos: mm ,deg
        :param use_cur:
        :return: pos:mm ,deg
        """
        if pos is None and not use_cur:
            raise ValueError("set target coordinate error")

        if use_cur:
            self.update_state_matrix()
            self.wgT_tar = self.wgT
            self.gwT_tar = self.gwT
        else:
            self.wgT_tar =  euler2Matrix(pos)
            self.gwT_tar = np.linalg.inv(self.wgT_tar)

    def update_state_matrix(self):
        #update wgT and gwT using current TCP POSE
        cur_pos = np.array(self.robot_ins.get_gripper_TCP_pose())
        self.wgT = euler2Matrix(cur_pos)
        self.gwT = np.linalg.inv(self.wgT)

    def tcp_frame_dT_to_command(self,dT):                  ##TODO: new
        """

        :param dT: gstart_gendT
        :return: cmd=[dx,dy,dz,drx,dry,drz] in start tcp frame,rotation euler sequence is z,y,x(tcp frame)/x ,y ,z(base frame)
        """
        cmd = np.zeros(6)
        cmd[0:3] = dT[0:3,3]                      ##mm
        cmd[3:6] = rmat2euler_degree(dT[0:3,0:3]) ##deg
        return cmd

    def absolute_T_to_pose(self,T):
        pose = np.zeros(6)
        pose[0:3] = T[0:3, 3]                       ##mm
        pose[3:6] = rmat2euler_degree(T[0:3, 0:3])  ##deg
        return pose

    def init(self):
        self.sample_init_pos()
        self.init_flag = True
        self.task_timer = time.time()
        self.vel_timer = time.time()
        print("environment initialized")

    def observation(self):
        return self.camera.get_frame()

    def act_to_goal(self):
        self.action_abs_T(self.wgT_tar)

    def action_abs_T(self,T):
        """
        基类action方法1：根据绝对齐次变换矩阵T进行运动，并且运动完后更新wgT,gwT
        """
        # print("actioned")
        tar_pose = self.absolute_T_to_pose(T)
        self.robot_ins.move_cart(pose=tar_pose, tool=1, user=0, vel=40)  ##servo cart is a blocked-type cmd
        self.update_state_matrix()

    def action_dT(self, dT,update_state=True):
        """
        基类action方法2：根据相对齐次变换矩阵dT(工具坐标系下的相对运动描述)进行运动
        """
        dT[0:3,0:3]=dT[0:3,0:3]
        cmd = self.tcp_frame_dT_to_command(dT)
        self.robot_ins.servo_cart(desc_pos=cmd, mode=2, vel=10.0)
        if update_state:
            self.update_state_matrix()

    def reinit(self):
        res_dict = self.need_reinit()
        if  res_dict["close_enough"]:
            self.init()
        else:
            self.init_flag=False
        return res_dict

    def reinit_eval(self,all_epochs_num=None,cur_epoch=None,freq_per_pos=None):
        rtn_dict=self.need_reinit_eval()
        need_reinit=rtn_dict["need_reinit"]
        if need_reinit:
            if self.uniform_eval_settings["utilized"] and (1+cur_epoch) % freq_per_pos == 0 and cur_epoch<all_epochs_num-1:
                self.evenly_posid += 1
            self.init()
        else:
            self.init_flag=False
        return rtn_dict

    def need_reinit(self):
        return self.compute_error(self.wgT_tar, self.wgT)

    def need_reinit_eval(self):
        err_dict = self.compute_error(self.wgT_tar, self.wgT) #eval stage,to get accurate z_zrror,compute dT = g_tar,gT is more acceptable

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
        # print("======================================")
        dT = np.linalg.inv(T0) @ T1
        angle = np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec()) / np.pi * 180
        dist = np.linalg.norm(dT[:3, 3])
        # print(f"compute_error_res--------trans:{dist},rot:{angle}")
        z_error=abs(dT[2, 3])
        self.close_enough_flag=(angle < self.angle_eps) and (dist < self.dist_eps)
        # print("angle:",angle)
        # print("dist:",dist)
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
        self.trans_vel_threshold=metrics["trans_vel_threshold"] #mm
        self.time_up_bound = metrics["use_time_upperbound"] #maximum used time during a rollout
        self.in_threshold_range_time = metrics["in_threshold_range_time"] #maximum time stay in error threshold before entering the next rollout

    def return_cur_pos_info(self):
        rtn_dict={
            'wgT':self.wgT,
            'gwT':self.gwT,
            }
        return rtn_dict

    def return_tar_pos_info(self):
        rtn_dict ={
        'wgT_tar':self.wgT_tar,
        'gwT_tar':self.gwT_tar,
        }
        return rtn_dict

    def setup_desire_pt(self,desire_pt):                       ##TODO:outdated
        self.desire_pt=np.array(desire_pt)
