import pygame
import Robot
import time
import numpy as np
import os
import math
from math import *
import minimalmodbus
import threading
from utils.transform import from_mov_vel2rots,get_rot_matrix_from_delta,rmat2euler_degree
from real.fr_robot import FR_Robot
from real.gripper import Gripper
from real.perception import Camera
import cv2

def filter_axis_0_3(x):
    if x>=-0.2 and x<=0.2:
        return 0
    else:
        return x
def filter_axis_4_5(x):
    if x>=-0.95:
        return x
    else:
        return -1

def unit_transform(pose):
    pose[0]=pose[0]/1000
    pose[1]=pose[1]/1000
    pose[2]=pose[2]/1000
    pose[3]=pose[3]/180*np.pi
    pose[4]=pose[4]/180*np.pi
    pose[5]=pose[5]/180*np.pi
    return pose

class Teleop:
    def __init__(self,robot_address,trans_coeff,rot_coeff,use_rxry,use_z,use_camera,stay_vertical,allow_teleop=True):

        self.allow_teleop=allow_teleop
        if self.allow_teleop:
            self.robo_ins=FR_Robot(robot_address)
        self.gripper_ins=Gripper()
        self.trans_coeff=trans_coeff
        self.rot_coeff=rot_coeff
        self.use_rxry=use_rxry
        self.use_camera=use_camera
        self.use_z=use_z
        self.stay_vertical=stay_vertical
        if self.use_camera:
            self.camera = Camera(devices={"wrist":'215222073421'},use_devices_type=["wrist"],width=640, height=480, fps=30)
        self.start_flag=0
        self.skip_flag=0
        self.stop_flag=0


    def operation(self):
        times=1000

        # 用于控制屏幕的刷新速度
        clock = pygame.time.Clock()

        # 这个字典可以保持原样，
        # 因为pygame将为程序开始时连接的每个操纵杆生成一个
        # pygame.JOYDEVICEADDED事件。
        joysticks = {}

        done = False
        i=0
        k=1   #num of pose collection
        time_list=[]
        pose_list=[]
        t_op=time.time()
        t_cls=time.time()
        t_pos_col=time.time()
        t0=time.time()
        if self.allow_teleop:
            in_pos=self.robo_ins.get_gripper_TCP_pose()
            if self.stay_vertical:
                in_pos[3]=-180
                in_pos[4]=0
            self.robo_ins.move_cart(in_pos, tool=1, user=0, vel=40)

        try:
            while not done:
                if self.use_camera:
                    if time.time() - t0 > 1 / 30:  # 此程序运行约0.01s（10hz）,因此循环频率需要低于10hz
                        print("camera circulation_time:", time.time() - t0)
                        t0 = time.time()
                        # 读取图像帧，包括RGB图
                        frame_dict = self.camera.get_frame()
                        assert frame_dict is not None

                        for type, img in frame_dict.items():
                            cv2.imshow(type, img)
                            cv2.waitKey(1)

                # 处理事件
                # 可能的游戏手柄事件：JOYAXISMOTION, JOYBALLMOTION, JOYBUTTONDOWN,
                # JOYBUTTONUP, JOYHATMOTION, JOYDEVICEADDED, JOYDEVICEREMOVED
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True  # 退出循环
                    if event.type == pygame.JOYDEVICEADDED:
                        # 该事件将在程序启动每个操纵杆时生成，无需手动创建即可填充列表。
                        joy = pygame.joystick.Joystick(event.device_index)
                        joysticks[joy.get_instance_id()] = joy
                        print(f"Joystick {joy.get_instance_id()} connencted")

                    if event.type == pygame.JOYDEVICEREMOVED:
                        del joysticks[event.instance_id]
                        print(f"Joystick {event.instance_id} disconnected")

                # 遍历所有的游戏手柄
                for joystick in joysticks.values():
                    self.start_flag = joystick.get_button(10) #左
                    self.skip_flag = joystick.get_button(15)  #中
                    self.stop_flag = joystick.get_button(11)  #右

                    if i==0:
                        t0=time.time()
                    if i==times:
                        print("===============================================================================================================================")
                        print("time used in iter {} to {}:".format(i - times, i), time.time()-t0)
                        time_list.append(time.time())
                    if (i%times==0 and i>=2*times):
                        print("===============================================================================================================================")
                        print("time used in iter {} to {}:".format(i-times,i),time.time()-time_list[-1])
                        time_list.append(time.time())

                    #teleop
                    if self.allow_teleop:
                        #x,y,z are all in [0,1]
                        x = joystick.get_axis(1)
                        x=filter_axis_0_3(x)
                        mov_x = -self.trans_coeff * x

                        y = joystick.get_axis(0)
                        y=filter_axis_0_3(y)
                        mov_y = -self.trans_coeff * y

                        if not self.use_z:
                            mov_z=0
                        else:
                            z=joystick.get_axis(5)
                            z = (filter_axis_4_5(z)+1)/2
                            z_d=joystick.get_button(6)
                            if z_d == 0:
                                mov_z = -self.trans_coeff * z
                            if z_d == 1:
                                mov_z = self.trans_coeff * z

                        if not self.use_rxry:
                            mov_rx=0
                            mov_ry=0
                        else:
                            rx=joystick.get_axis(2)
                            rx=filter_axis_0_3(rx)
                            mov_rx = self.rot_coeff * rx

                            ry=joystick.get_axis(3)
                            ry=filter_axis_0_3(ry)
                            mov_ry = -self.rot_coeff * ry

                        rz=joystick.get_axis(4)
                        rz=(filter_axis_4_5(rz)+1)/2
                        rz_d=joystick.get_button(7)
                        if rz_d==0:
                            mov_rz=-self.rot_coeff * rz
                        if rz_d==1:
                            mov_rz=self.rot_coeff * rz

                        a = self.robo_ins.get_gripper_TCP_pose()
                        x_0, y_0, z_0, rx_0, ry_0, rz_0 = a[0], a[1], a[2], a[3], a[4], a[5]
                        if x_0==0 and y_0==0 and z_0==0 and rx_0==0 and ry_0==0 and rz_0==0:
                            continue

                        mov_vel=[mov_x,mov_y,mov_z,mov_rx,mov_ry,mov_rz]
                        Rx, Ry, Rz = from_mov_vel2rots([mov_vel[3], mov_vel[4], mov_vel[5]])
                        mov_rot_in_world = get_rot_matrix_from_delta(a, Rx, Ry, Rz)
                        rx, ry, rz = rmat2euler_degree(mov_rot_in_world)
                        if self.stay_vertical:
                            rx=-180
                            ry=0
                        desc_pos = [mov_vel[0] + x_0, mov_vel[1] + y_0, mov_vel[2] + z_0, rx, ry, rz]
                        # print('desc_pos:', desc_pos)
                        self.robo_ins.servo_cart(desc_pos=desc_pos, mode=0, vel=10.0)  # mode=0:绝对运动基坐标系

                    #grip
                    grip_op = joystick.get_button(0)
                    grip_cls = joystick.get_button(1)
                    if grip_op == 1 and time.time()-t_op>1:
                        try:
                            t_op=time.time()
                            self.gripper_ins.move_gripper(0,60,60)
                        except Exception:
                            continue

                    if grip_cls == 1 and time.time()-t_cls>1:
                        try:
                            t_cls=time.time()
                            self.gripper_ins.move_gripper(1000,60,60)
                        except Exception:
                            continue

                    #print pose
                    print_pos = joystick.get_button(4)
                    if print_pos == 1 and time.time()-t_pos_col>1:
                        t_pos_col=time.time()
                        cur_pose=self.robo_ins.get_gripper_TCP_pose()
                        print('==================================================================================')
                        print("pose_{}:".format(k),cur_pose)
                        pose_list.append(cur_pose)
                        k+=1
                    i+=1
                    time.sleep(0.008)
                # 控制30帧每秒
                clock.tick(times)
        except KeyboardInterrupt or SystemExit:
            print('==============================pose_list=======================================')
            print(pose_list)

if __name__ == "__main__":
    robot_address="192.168.58.2"
    trans_coeff=2
    rot_coeff=1
    use_rxry=True
    stay_vertical=False
    use_z=True
    use_camera=True

    pygame.init()
    # main()
    teleop=Teleop(robot_address=robot_address,trans_coeff=trans_coeff,rot_coeff=rot_coeff,use_rxry=use_rxry,use_z=use_z,use_camera=use_camera,stay_vertical=stay_vertical)
    teleop.operation()
    # pygame.quit()