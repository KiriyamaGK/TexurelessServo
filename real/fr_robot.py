import numpy as np
import Robot
import random
import time


class FR_Robot:
    def __init__(self,address='192.168.58.2'):
        self.robot = Robot.RPC(address)

    @staticmethod
    def check_pose_type(pose):
        if isinstance(pose,np.ndarray):
            assert pose.shape == (6,)
        elif isinstance(pose,list):
            assert len(pose) == 6
        else:
            raise TypeError('Pose must be a numpy array or list.')

    def move_cart(self,pose,tool,user,vel):
        self.check_pose_type(pose)
        ret = self.robot.MoveCart(pose, tool=tool, user=user, vel=vel)
        if ret != 0:
            raise RuntimeError('point cannot arrive,error code:',ret)

    def move_l(self,pose,tool,user,vel):
        self.check_pose_type(pose)
        ret = self.robot.MoveL(pose, tool=tool, user=user, vel=vel)
        if ret != 0:
            raise RuntimeError('point cannot arrive,error code:',ret)

    def servo_cart(self,desc_pos,mode,vel):
        self.check_pose_type(desc_pos)
        ret = self.robot.ServoCart(desc_pos=desc_pos, mode=mode, vel=vel)
        if ret != 0:
            raise RuntimeError('point cannot arrive,error code:',ret)

    def get_gripper_TCP_pose(self):  ##degree
        ret = self.robot.GetActualTCPPose()
        if ret[0]==0:
            return ret[1]
        else:
            raise RuntimeError("Meet error when querying TCP pose,error code:",ret[0])

    def get_actual_joints_degree(self):
        ret = self.robot.GetActualJointPosDegree()
        if ret[0] == 0:
            return ret[1]
        else:
            raise RuntimeError("Meet error when querying TCP pose,error code:", ret[0])

    def check_trans_ability(self,point,absolute_cmd):
        desc_pos = np.array(point)
        ret = self.robot.GetInverseKin(0, desc_pos, config=-1)
        if isinstance(ret, tuple) and ret[0] == 0:
            if absolute_cmd:
                return desc_pos
            else:
                return [0,0,0,0,0,0]
        else:
            raise RuntimeError('point cannot arrive finally when translating')

    def check_rot_ability(self,point,absolute_cmd):
        desc_pos = np.array(point)
        ret = self.robot.GetInverseKin(0, desc_pos, config=-1)
        if isinstance(ret, tuple) and ret[0] == 0:
            if absolute_cmd:
                return desc_pos
            else:
                return [0,0,0,0,0,0]
        else:
            raise RuntimeError('point cannot arrive finally when rotating')

    def rectify_angle(self,pos):
        start_id = 3 if pos.shape[0]==6 else 0
        for i,p in enumerate(pos):
            if i>=start_id:
                if p>180:
                    pos[i]-=180
                elif p<-180:
                    pos[i]+=180
        return pos

    def go_vertical(self):
        pos = fr_robot.get_gripper_TCP_pose()
        pos[3] = -180.
        pos[4] = 0.
        self.move_cart(np.array(pos), tool=2, user=0, vel=40)

if __name__ == '__main__':
    # task = "go_vertical" #go_vertical/go_pose/print_pose
    task = "go_pose"
    # task = "print_pose"
    # 1：320., 226.5
    # 3:323.9  26.54
    _dir = np.array([323.9 , 26.54]) - np.array([320., 226.5])
    old = np.array([320., 226.5])
    new = old + _dir/np.linalg.norm(_dir)*300.
    print(new)
    # init_pos = np.array(
    #     [new[0], new[1], 160, 180.0, 0., 86.5])
    init_pos  = [325.85005765, -68.5, 170., -180.0, 0., 86.5]
    print(init_pos)

    fr_robot = FR_Robot()
    if task == "go_pose":
        print("Go pose!")
        fr_robot.move_cart(np.array(init_pos), tool=2, user=0, vel=5)
    elif task == "print_pose":
        print("Print pose!")
        pos = fr_robot.get_gripper_TCP_pose()
        print("current pos:",pos)
    elif task == "go_vertical":
        print("Go vertical!")
        fr_robot.go_vertical()
    else:
        raise ValueError("Required task is not defined.")
    # while True:
    #     cur_pose = fr_robot.get_gripper_TCP_pose()
    #     fr_robot.servo_cart(cur_pose,mode=0,vel=10)
    #     time.sleep(1)
    #     print(cur_pose)
    #     time.sleep(0.1)