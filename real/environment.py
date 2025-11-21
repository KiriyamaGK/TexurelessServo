from real.perception import Camera
from real.fr_robot import FR_Robot
from real.gripper import Gripper
import numpy as np
import time
import random
from math import pi,sin,cos
from utils.transform import make_an_angle_in_180, euler2Matrix, rmat2euler_degree,_6d_pose_to_mat,mat_to_6d_pose
from scipy.spatial.transform import Rotation as R
from itertools import product
from typing import Union,List

class Environment:
    def __init__(self,robot_address,w,h,fps,cam_devices,use_devices_type,dof,down_to_grasp_distance,init,stop_policy,velocity,pick_and_place_from_slot = None):
        self.robot_ins=FR_Robot(robot_address)
        self.camera=Camera(devices=cam_devices,use_devices_type=use_devices_type,width=w, height=h, fps=fps)
        self.gripper=Gripper()

        self.down_dis = down_to_grasp_distance  # 夹爪下移距离，mm

        self.dof = dof
        assert self.dof in [3, 6]

        self.angle_eps = stop_policy["angle_eps"]  # degree
        self.dist_eps = stop_policy["dist_eps"]  # mm
        self.trans_vel = velocity["trans_vel"]["value"]
        self.rot_vel = velocity["rot_vel"]["value"]

        # 保存参数列表用于动态选择
        self.init_horizon_trans_list = init["init_horizon_trans"]["value"]
        self.init_vertical_trans_list = init["init_vertical_trans"]["value"] if self.dof==6 else [[0, 0, 0]]
        self.init_rot_list = init["init_rot"]["value"]
        
        # 初始化参数（将在init()中动态更新）
        self.init_horizon_trans = self.init_horizon_trans_list[0][0] if isinstance(self.init_horizon_trans_list[0], list) else self.init_horizon_trans_list[0]
        self.init_vertical_trans = self.init_vertical_trans_list[0][0] if isinstance(self.init_vertical_trans_list[0], list) else self.init_vertical_trans_list[0]
        init_rot = self.init_rot_list[0][:3] if isinstance(self.init_rot_list[0], list) else self.init_rot_list[0]
        self.uniform_eval_settings = init["uniform_evaluation"]

        if self.dof == 3:
            if isinstance(init_rot, list):
                self.init_rot = np.array([0,0,init_rot[0]]) # degree
            else:
                self.init_rot = np.array([0,0,init_rot]) # degree
        else:
            if isinstance(init_rot, list) and len(init_rot) == 3:
                self.init_rot = np.array(init_rot)  # degree
            else:
                self.init_rot = np.array([0,0,0])  # degree

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

        self.vel_in_threshold_flag = False
        self.episode = -1

        self.task_timer = time.time()
        self.vel_timer = time.time()

        if self.uniform_eval_settings["utilized"]:
            self.return_evenly_distributed_poses()
            self.evenly_posid=0

        self.wgT_tar=None
        self.gwT_tar=None
        self.wgT=None
        self.gwT=None

        #=======================pick and place========================
        if pick_and_place_from_slot is not None and pick_and_place_from_slot["utilized"] == True:
            self.safe_vel = pick_and_place_from_slot["velocity"]["safe"]
            self.unsafe_vel = pick_and_place_from_slot["velocity"]["unsafe"]

            self.wpts = pick_and_place_from_slot["slot"]["waypoints"]
            self.num_wpts = len(self.wpts)

            self.slot_wpts = []
            slot_xy_rz = pick_and_place_from_slot["slot"]["slot_xy_rz"]
            slot_down_hs = pick_and_place_from_slot["slot"]["slot_down_hs"]
            for idx in range(len(slot_down_hs)):
                if idx < len(slot_down_hs) -1 :
                    self.slot_wpts.append([slot_xy_rz[0],slot_xy_rz[1],slot_down_hs[idx],-180.,0.,slot_xy_rz[2]])
                else:
                    self.slot_grasp_pose = [slot_xy_rz[0],slot_xy_rz[1],slot_down_hs[idx],-180.,0.,slot_xy_rz[2]]
            self.num_slot_wpts = len(self.slot_wpts)

            self.place_x_range = pick_and_place_from_slot["table"]["place_x_range"]
            self.place_y_range = pick_and_place_from_slot["table"]["place_y_range"]
            self.place_rz_range = pick_and_place_from_slot["table"]["place_rz_range"]
            self.place_rxry_z = pick_and_place_from_slot["table"]["place_rxry_z"]

            self.safeup_dev = pick_and_place_from_slot["deviation"]["safeup_z"]
            self.tar_dev = pick_and_place_from_slot["deviation"]["tar_dev"]

            self.g_place_T_safeup = np.eye(4)
            self.g_place_T_safeup[2, 3] = self.safeup_dev

            self.g_place_T_g_tar = _6d_pose_to_mat(self.tar_dev)
        # =======================pick and place========================

    def generate_table_pose(self):
        _place_pose = np.zeros(6)
        _place_pose[0] = random.uniform(self.place_x_range[0], self.place_x_range[1])
        _place_pose[1] = random.uniform(self.place_y_range[0], self.place_y_range[1])
        _place_pose[2] = self.place_rxry_z[2]
        _place_pose[3] = self.place_rxry_z[0]
        _place_pose[4] = self.place_rxry_z[1]
        _place_pose[5] = random.uniform(self.place_rz_range[0], self.place_rz_range[1])

        _w_T_g_place = _6d_pose_to_mat(_place_pose)
        return _w_T_g_place,_place_pose


    def pick_slot_and_place_table_once(self,_w_T_g_place = None, _place_pose = None):
        """
        The function should be utilized only when the gripper is nearby the slot pose., and is finalized with the gripper at the goal pose.
        :return:
        """
        # execute pre-grasp points
        for idx in range(self.num_slot_wpts):
            self.robot_ins.move_cart(pose=self.slot_wpts[idx], tool=2, user=0, vel=self.safe_vel)

        # open gripper and go down
        self.gripper.move_gripper(400, 60, 60)  # 考虑料盘的夹爪槽宽度，gripper 张开对应cmd400，闭合对应cmd600
        time.sleep(3)
        self.robot_ins.move_cart(pose=self.slot_grasp_pose, tool=2, user=0, vel=self.unsafe_vel)

        # close gripper, go up and place on table
        self.gripper.move_gripper(600, 60, 60)  # close
        time.sleep(3)
        print("Grasped part!")

        for idx in range(self.num_slot_wpts):
            self.robot_ins.move_cart(pose=self.slot_wpts[self.num_slot_wpts-1-idx], tool=2, user=0, vel=self.unsafe_vel if idx == 0 else self.safe_vel)

        for idx in range(self.num_wpts):
            self.robot_ins.move_cart(pose=self.wpts[self.num_wpts-1-idx], tool=2, user=0, vel=self.safe_vel)

        if _w_T_g_place is None and _place_pose is None:
            _w_T_g_place, _place_pose = self.generate_table_pose()

        self.robot_ins.move_cart(pose=_place_pose, tool=2, user=0, vel=self.safe_vel)
        self.gripper.move_gripper(400, 60, 60)  # open
        time.sleep(2)
        print("Placed part!")

        # go to safe pose and goal pose
        w_T_safeup = _w_T_g_place @ self.g_place_T_safeup
        safeup_pose = mat_to_6d_pose(w_T_safeup)
        self.robot_ins.move_cart(pose=safeup_pose, tool=2, user=0, vel=self.safe_vel)
        print("Went to safeup pose!")

        _w_T_g_tar = _w_T_g_place @ self.g_place_T_g_tar
        goal_pose = mat_to_6d_pose(_w_T_g_tar)
        self.robot_ins.move_cart(pose=goal_pose, tool=2, user=0, vel=self.safe_vel)
        print("Went to goal pose!")
        self.set_target_coordinate() #goal pose set here
        return _w_T_g_place, _place_pose

    def pick_table_and_place_slot_once(self,_place_pose: Union[np.ndarray, List], _w_T_g_place: Union[np.ndarray, None]):
        """
        The function should be utilized only when the gripper is nearby the table goal pose , and is finalized with the gripper at the top of the slot pose.
        :return:
        """
        # go to safe pose
        if _w_T_g_place is None:
            _w_T_g_place = _6d_pose_to_mat(_place_pose)
        w_T_safeup = _w_T_g_place @ self.g_place_T_safeup
        safeup_pose = mat_to_6d_pose(w_T_safeup)
        self.robot_ins.move_cart(pose=safeup_pose, tool=2, user=0, vel=self.safe_vel)
        print("Went to safeup pose!")

        # open gripper and grasp
        self.gripper.move_gripper(400, 60, 60)  # open
        time.sleep(3)
        self.robot_ins.move_cart(pose=_place_pose, tool=2, user=0, vel=self.safe_vel)
        print("Went to table pose!")
        self.gripper.move_gripper(600, 60, 60)  # close
        time.sleep(3)
        print("Grasped part!")

        # execute pre-grasp points
        for idx in range(self.num_wpts):
            self.robot_ins.move_cart(pose=self.wpts[idx], tool=2, user=0, vel=self.safe_vel)
        for idx in range(self.num_slot_wpts): # len(“slot_down_hs") - 1
            self.robot_ins.move_cart(pose=self.slot_wpts[idx], tool=2, user=0, vel=self.unsafe_vel if idx == self.num_slot_wpts-1 else self.safe_vel)

        # plug in, open gripper and step back
        self.robot_ins.move_cart(pose=self.slot_grasp_pose, tool=2, user=0, vel=self.unsafe_vel)
        self.gripper.move_gripper(400, 60, 60)  # open
        time.sleep(3)
        print("Placed part in slot!")
        for idx in range(self.num_slot_wpts):
            self.robot_ins.move_cart(pose=self.slot_wpts[self.num_slot_wpts - 1 - idx], tool=2, user=0,vel=self.safe_vel)
        print("Stepped back!")

    def pick_table_and_place_slot_test(self,_place_pose: Union[np.ndarray, List], _w_T_g_place: Union[np.ndarray, None]):
        """
        The function should be utilized only when the gripper is nearby the table goal pose , and is finalized with the gripper at the top of the slot pose.
        :return:
        """
        # go to safe pose
        if _w_T_g_place is None:
            _w_T_g_place = _6d_pose_to_mat(_place_pose)
        w_T_safeup = _w_T_g_place @ self.g_place_T_safeup
        safeup_pose = mat_to_6d_pose(w_T_safeup)
        self.robot_ins.move_cart(pose=safeup_pose, tool=2, user=0, vel=self.safe_vel)
        print("Went to safeup pose!")

        # open gripper and grasp
        self.gripper.move_gripper(400, 60, 60)  # open
        time.sleep(3)
        self.robot_ins.move_cart(pose=_place_pose, tool=2, user=0, vel=self.safe_vel)
        print("Went to table pose!")
        self.gripper.move_gripper(600, 60, 60)  # close
        time.sleep(3)
        print("Grasped part!")

        # execute pre-grasp points
        for idx in range(self.num_wpts):
            self.robot_ins.move_cart(pose=self.wpts[idx], tool=2, user=0, vel=self.safe_vel)
        self.robot_ins.move_cart(pose=self.slot_wpts[0], tool=2, user=0, vel=self.safe_vel)
        self.robot_ins.move_cart(pose=self.slot_wpts[1], tool=2, user=0, vel=1)
        # open gripper
        self.gripper.move_gripper(400, 60, 60)  # open
        time.sleep(3)

    def get_dynamic_params(self, param_list, episode):
        """
        根据episode动态选择参数
        :param param_list: 参数列表，格式为[[value, epi_start, epi_end], ...] 或 [[theta1, theta2, theta3, epi_start, epi_end], ...]
        :param episode: 当前episode
        :return: 对应的参数值
        """
        for param_config in param_list:
            if len(param_config) == 3:  # [value, epi_start, epi_end]
                value, epi_start, epi_end = param_config
                if epi_start <= episode <= epi_end:
                    return value
            elif len(param_config) == 5:  # [theta1, theta2, theta3, epi_start, epi_end]
                theta1, theta2, theta3, epi_start, epi_end = param_config
                if epi_start <= episode <= epi_end:
                    return [theta1, theta2, theta3]
        
        # 如果没有找到匹配的episode范围，返回最后一个配置的值
        if len(param_list) > 0:
            last_config = param_list[-1]
            if len(last_config) == 3:
                return last_config[0]
            elif len(last_config) == 5:
                return last_config[:3]
        
        # 默认值
        return param_list[0][0] if len(param_list) > 0 else 0

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
        self.episode += 1
        
        # 动态选择参数
        if hasattr(self, 'init_horizon_trans_list'):
            self.init_horizon_trans = self.get_dynamic_params(self.init_horizon_trans_list, self.episode)
        if hasattr(self, 'init_vertical_trans_list') and self.dof == 6:
            self.init_vertical_trans = self.get_dynamic_params(self.init_vertical_trans_list, self.episode)
        if hasattr(self, 'init_rot_list'):
            rot_params = self.get_dynamic_params(self.init_rot_list, self.episode)
            if self.dof == 3:
                self.init_rot = np.array([0, 0, rot_params[0] if isinstance(rot_params, list) else rot_params])
            else:
                self.init_rot = np.array(rot_params)
        
        self.sample_init_pos()
        self.task_timer = time.time()
        self.vel_timer = time.time()
        print(f"environment initialized for episode {self.episode}")

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
        self.robot_ins.move_cart(pose=tar_pose, tool=2, user=0, vel=40)  ##servo cart is a blocked-type cmd
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
        return res_dict

    def reinit_eval(self,all_epochs_num=None,cur_epoch=None,freq_per_pos=None):
        rtn_dict=self.need_reinit_eval()
        need_reinit=rtn_dict["need_reinit"]
        if need_reinit:
            if self.uniform_eval_settings["utilized"] and (1+cur_epoch) % freq_per_pos == 0 and cur_epoch<all_epochs_num-1:
                self.evenly_posid += 1
            # self.init()
        return rtn_dict

    def need_reinit(self):
        return self.compute_error(self.wgT_tar, self.wgT)

    def need_reinit_eval(self):
        err_dict = self.compute_error(self.wgT_tar, self.wgT) #eval stage,to get accurate z_zrror,compute dT = g_tar,gT is more acceptable

        if time.time()-self.task_timer>=self.time_up_bound:
            print("time limit reached,reinit.....")
            need_reinit=True
        else:
            if time.time()-self.vel_timer>=self.in_threshold_range_time and self.vel_in_threshold_flag:
                need_reinit=True
                print(f"velocity too small,reinit.....,vel_timer:{self.vel_timer}")
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
