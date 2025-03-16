from real.perception import Camera
from real.fr_robot import FR_Robot
from real.gripper import Gripper
import numpy as np
import time
import random
from math import pi,sin,cos
from utils.transform import make_an_angle_in_180
from scipy.spatial.transform import Rotation as R

class Environment:
    def __init__(self,robot_address,w,h,fps,cam_devices,use_devices_type,trans_thres,rot_thres,down_dis):

        self.robot_ins=FR_Robot(robot_address)
        self.camera=Camera(devices=cam_devices,use_devices_type=use_devices_type,width=w, height=h, fps=fps)
        self.gripper=Gripper()

        self.corner_1 = [-405.3097229003906, 199.8372497558593]  # 工作区域左上角点
        self.corner_2 = [-707.1678466796875, -249.3986206054687]  # 工作区域右下角点

        self.x_max = max(self.corner_1[0], self.corner_2[0])
        self.x_min = min(self.corner_1[0], self.corner_2[0])
        self.y_max = max(self.corner_1[1], self.corner_2[1])
        self.y_min = min(self.corner_1[1], self.corner_2[1])

        self.trans_thres = trans_thres #最大平移距离，mm
        self.rot_thres = rot_thres  #最大rz，°
        self.down_dis = down_dis    #夹爪下移距离，mm

        self.task_timer = time.time()
        self.vel_timer = time.time()
        self.vel_in_threshold_flag = False
        print("environment initialized")

    def init(self):
        # if self.obj_idx_pointer>=len(self.obj_idxs):
        #     self.obj_idx_pointer=0
        # self.obj_idx=self.obj_idxs[self.obj_idx_pointer]
        # self.sample_init_pos()
        # self.gen_scene()
        # self.obj_idx_pointer+=1
        self.task_timer=time.time()
        self.vel_timer=time.time()



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

    def generate_motion_paras(self,desire_pt):
        '''
        给定目标位置生成运动参数，包括theta（xy方位），alpha（rz）,生成的起始点start_pt.要确保目标位姿的rz的绝对值在120°-180°，否则可能位姿无法到达
        :param desire_pt: 目标位姿
        :param rot_thres: 最大rz旋转角（°）
        :param trans_thres: 最大平移量(mm)
        :return: start_pt,alpha（°）,theta(rad)
        '''
        for i in range(100):
            theta = random.uniform(-2 * pi, 2 * pi)
            alpha = random.uniform(10, self.rot_thres) * (random.randint(0, 1) - 0.5) * 2
            assert self.rot_thres < 85
            if alpha > 0:  # 逆时针转
                if desire_pt[5] > 0:  # 腕部相机在desire_pt向left_cam偏
                    assert desire_pt[5] > 120 and desire_pt[5] <= 180
                if desire_pt[5] < 0:  # 腕部相机在desire_pt向right_cam偏
                    assert desire_pt[5] < -120 and desire_pt[5] >= -180
                    alpha = min(alpha, -120 - desire_pt[5])
            elif alpha < 0:  # 顺时针转
                if desire_pt[5] > 0:  # 腕部相机在desire_pt向left_cam偏
                    assert desire_pt[5] > 120 and desire_pt[5] <= 180
                    alpha = -1 * min(-alpha, desire_pt[5] - 120)
                if desire_pt[5] < 0:  # 腕部相机在desire_pt向right_cam偏
                    assert desire_pt[5] < -120 and desire_pt[5] >= -180

            delta = [self.trans_thres * cos(theta), self.trans_thres * sin(theta)]
            start_pt = np.array(desire_pt.copy()) + np.array([delta[0], delta[1], 0, 0, 0, alpha])
            start_pt[5]=make_an_angle_in_180(start_pt[5])
            print("start_pt", start_pt[5])
            print('desire_pt', desire_pt[5])
            print("alpha", alpha)
            assert abs(start_pt[5]) >= 120 and abs(start_pt[5]) <= 180
            ret = self.robot_ins.robot.GetInverseKin(0, start_pt, config=-1)
            if isinstance(ret, tuple) and ret[0] == 0: #如果能求出逆解
                return theta, alpha, start_pt
            else:
                continue

    def setup_stop_policy(self,metrics:dict):
        self.rot_vel_threshold=metrics["rot_vel_threshold"]  #deg
        self.trans_vel_threshold=metrics["trans_vel_threshold"] #m
        self.time_up_bound = metrics["use_time_upperbound"] #maximum used time during a rollout
        self.in_threshold_range_time = metrics["in_threshold_range_time"] #maximum time stay in error threshold before entering the next rollout

    def setup_desire_pt(self,desire_pt):
        self.desire_pt=np.array(desire_pt)

    def need_reinit_eval(self):
        err_dict=self.compute_error()
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

    def compute_error(self):
        current_pos=np.array(self.robot_ins.get_gripper_TCP_pose())
        angle = make_an_angle_in_180(current_pos.copy()[-1] - self.desire_pt[-1])
        angle=abs(angle)
        dist = np.linalg.norm(current_pos.copy()[0:3] - self.desire_pt[0:3])
        return {
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

    def reinit_eval(self):
        rtn_dict=self.need_reinit_eval()
        need_reinit=rtn_dict["need_reinit"]
        if need_reinit:
            self.init()
        return rtn_dict