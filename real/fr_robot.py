import numpy as np
import Robot
import random
import time


class FR_Robot:
    def __init__(self,address='192.168.58.2'):
        self.robot = Robot.RPC(address)

    def move_cart(self,pose,tool,user,vel):
        ret = self.robot.MoveCart(pose, tool=tool, user=user, vel=vel)
        if ret != 0:
            raise RuntimeError('point cannot arrive,error code:',ret)

    def move_l(self,pose,tool,user,vel):
        ret = self.robot.MoveL(pose, tool=tool, user=user, vel=vel)
        if ret != 0:
            raise RuntimeError('point cannot arrive,error code:',ret)

    def servo_cart(self,desc_pos,mode,vel):
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

if __name__ == '__main__':
    fr_robot = FR_Robot()
    init_pos = np.array(
        [-509.0465087890625, -101.97378540039062, 447.5419921875, -179.19952392578125, -0.5334994196891785, -162.46022033691406])
    fr_robot.move_cart(np.array(init_pos),tool=0,user=0,vel=40)
    init_pos[2]-=300
    fr_robot.move_cart(np.array(init_pos), tool=2, user=0, vel=40)
    pos = fr_robot.get_gripper_TCP_pose()
    print(pos)
    time.sleep(1)
    print("move finished")
    # while True:
    #     cur_pose = fr_robot.get_gripper_TCP_pose()
    #     fr_robot.servo_cart(cur_pose,mode=0,vel=10)
    #     time.sleep(1)
    #     print(cur_pose)
    #     time.sleep(0.1)