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
from pynput import keyboard

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
    def __init__(self,robot_ins,trans_coeff,rot_coeff,use_rxry,use_z,use_camera,ctrl_freq=30,listen_finish=False):

        self.robo_ins=robot_ins
        # self.gripper_ins=Gripper()
        self.trans_coeff=trans_coeff
        self.rot_coeff=rot_coeff
        self.use_rxry=use_rxry
        self.use_camera=use_camera
        self.use_z=use_z
        if self.use_camera:
            self.camera = Camera(devices={"wrist":'215222073421'},use_devices_type=["wrist"],width=640, height=480, fps=30)
        self.start_flag=0
        self.skip_flag=0
        self.stop_flag=0
        self.ctrl_freq=ctrl_freq
        self.listen_finish=listen_finish
        self.stop_teleop=False
        pygame.init()

    def operation(self):
        self.stop_teleop=False
        self.joysticks = {}
        done = False

        if self.listen_finish:
            self.listen_process()

        t_op = time.time()
        t_cls = time.time()

        t0 = time.time()

        try:
            while not done:
                # print("1")
                t_teleop = time.time()
                if self.use_camera:
                    if time.time() - t0 > 1 / 30:  # 此程序运行约0.01s（10hz）,因此循环频率需要低于10hz
                        # print("camera circulation_time:", time.time() - t0)
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
                        self.joysticks[joy.get_instance_id()] = joy
                        print(f"Joystick {joy.get_instance_id()} connencted")

                    if event.type == pygame.JOYDEVICEREMOVED:
                        del self.joysticks[event.instance_id]
                        print(f"Joystick {event.instance_id} disconnected")

                # 遍历所有的游戏手柄
                for joystick in self.joysticks.values():
                    # print("2")
                    self.start_flag = joystick.get_button(10) #左
                    self.skip_flag = joystick.get_button(15)  #中
                    self.stop_flag = joystick.get_button(11)  #右

                    #teleop
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
                        else:
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
                    else:
                        mov_rz=self.rot_coeff * rz

                    cur_pose = self.robo_ins.get_gripper_TCP_pose()
                    mov_vel=np.array([mov_x,mov_y,mov_z,mov_rx,mov_ry,mov_rz])

                    # print("cur_pose:",cur_pose)
                    # print("mov_vel:",mov_vel)

                    self.robo_ins.servo_cart(desc_pos=mov_vel, mode=1, vel=10.0)  # mode=1:增量运动基坐标系

                    #grip
                    grip_op = joystick.get_button(0)
                    grip_cls = joystick.get_button(1)
                    if grip_op == 1 and time.time()-t_op>1:
                        try:
                            t_op=time.time()
                            # self.gripper_ins.move_gripper(0,60,60)
                        except Exception:
                            continue

                    if grip_cls == 1 and time.time()-t_cls>1:
                        try:
                            t_cls=time.time()
                            # self.gripper_ins.move_gripper(1000,60,60)
                        except Exception:
                            continue

                if self.listen_finish:
                    if self.stop_teleop:
                        pygame.quit()
                        if self.keyboard_listener is not None:
                            self.keyboard_listener.stop()
                        break

                dt=time.time()-t_teleop
                time.sleep(max(0,1/self.ctrl_freq-dt))

        except KeyboardInterrupt or SystemExit:
            print('exit')
            pygame.quit()  # 清理 pygame
            self.stop_teleop = True  # 确保标志位正确
            if self.keyboard_listener is not None:
                self.keyboard_listener.stop()

    def listen_process(self):
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()

    def _on_key_press(self, key):
        try:
            if key.char == 'f':
                self.stop_teleop = True
                time.sleep(0.1)
        except AttributeError:
            pass


if __name__ == "__main__":
    robot_address="192.168.58.2"
    robot_ins=FR_Robot(robot_address)

    trans_coeff=2
    rot_coeff=1
    use_rxry=True
    use_z=False
    use_camera=False
    ctrl_freq = 100

    # main()
    teleop=Teleop(robot_ins=robot_ins,trans_coeff=trans_coeff,rot_coeff=rot_coeff,use_rxry=use_rxry,use_z=use_z,use_camera=use_camera,ctrl_freq=ctrl_freq)
    teleop.operation()
    # pygame.quit()