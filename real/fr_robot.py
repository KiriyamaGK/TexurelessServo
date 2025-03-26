import numpy as np
import Robot

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

    def get_gripper_TCP_pose(self):
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
if __name__ == '__main__':
    fr_robot = FR_Robot()
    fr_robot.move_cart([-580.7462158203125, -91.12007141113281, 106.64298248291016, 179.99981689453125, -0.00024279687204398215, 171.9166717529297],tool=1,user=0,vel=40)
    pos=fr_robot.get_gripper_TCP_pose()
    print(pos)