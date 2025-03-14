import time

import numpy as np
import pygame


from real.perception import Camera
from real.fr_robot import FR_Robot
from utils.paths import return_disc_route
from utils.file import ensure_dir
from utils.input_process import clip_image
import os
import cv2
import random
from real.teleop_with_joystick import Teleop
import threading
import json

def make_an_angle_in_180(ang,max_attempts_num=10):
    attempts_num=0
    while True:
        if attempts_num >= max_attempts_num:
            raise RuntimeError("Too many attempts")
        if abs(ang) >180:
            if ang>0:
                ang -= 360
            else:
                ang += 360
        if abs(ang) <=180:
            return ang
        attempts_num+=1

def determine_trans_vel(max_rad,cur_rad,max_vel):
    '''

    :param max_rad:
    :param cur_rad:
    :param max_vel:
    :return:
    '''
    assert cur_rad<=max_rad+max_vel
    return min(max_rad-cur_rad, max_vel)

def generate_random_dxdy(vel_norm):
    alpha=random.uniform(0,2*np.pi)
    return vel_norm *np.array([np.cos(alpha), np.sin(alpha)])

if __name__ == '__main__':
    config_dir = "../../configs/create_domain_B.json"
    with open(config_dir, "r") as j:
        config = json.load(j)

    robot_address = "192.168.58.2"
    part_name=config["overall_setting"]["part_name"]
    date_name=config["overall_setting"]["date_name"]
    img_size=config["overall_setting"]["img_size"]
    use_joystick=config["overall_setting"]["use_joystick"]
    motion_type=config["motion"]["motion_type"]

    assert motion_type in ["auto", "joystick","none"]

    base_dir = return_disc_route("One Touch")
    img_base_dir = os.path.join(base_dir, 'AlignAnything_real', date_name, 'cycle_gan')
    part_base_dir=os.path.join(img_base_dir,part_name)
    trainB_dir=os.path.join(part_base_dir, "trainB")
    testB_dir=os.path.join(part_base_dir, "testB")
    ensure_dir(trainB_dir)
    # ensure_dir(testB_dir)

    cam=Camera(devices={"wrist":'215222073421'},use_devices_type=["wrist"],width=640, height=480, fps=30)

    if motion_type == "auto":
        robo_ins=FR_Robot(robot_address)
        motion_radius_thres=config["motion"]["motion_radius_thres"]
        max_trans_vel=config["motion"]["max_tr_vel"]
        rot_vel=config["motion"]["rot_vel"]
        rz_range=config["motion"]["rz_range"]

    t_0=time.time()
    t_cam=time.time()

    if use_joystick:
        allow_teleop=True if motion_type == "joystick" else False

        trans_coeff = 2
        rot_coeff = 1
        use_rxry = False
        stay_vertical = True
        use_z = True

        pygame.init()
        teleop = Teleop(robot_address=robot_address, trans_coeff=trans_coeff, rot_coeff=rot_coeff, use_rxry=use_rxry,
                        use_z=use_z, use_camera=False, stay_vertical=stay_vertical,allow_teleop=allow_teleop)

        teleop_thread = threading.Thread(target=teleop.operation)
        teleop_thread.daemon = True  # 设置为守护线程，这样主线程结束时会自动结束子线程
        teleop_thread.start()

    photo_taken = False  # 标记是否已经拍照
    idx=0

    if motion_type == "auto":
        init_pos=robo_ins.get_gripper_TCP_pose()
        init_pos[3:5]=-180,0
        robo_ins.move_cart(pose=init_pos,tool=1, user=0, vel=40)
        desire_rz=rz_range[0]

    while True:
        img = cam.get_frame()["wrist"]
        cv2.imshow('img', img)
        key = cv2.waitKey(1) #如果在这 1 毫秒内有按键按下，函数会返回按键的 ASCII 码值（32位），如果进一步&0xFF是变成8位

        #determine events
        if not use_joystick:
            start_flag=(key & 0xFF == ord('s'))
            stop_flag=(key & 0xFF == ord('q'))
            skip_flag=(key & 0xFF == ord('j'))
        else:
            start_flag=teleop.start_flag
            stop_flag=teleop.stop_flag
            skip_flag=teleop.skip_flag

        #check events triggered
        if start_flag and not photo_taken:
            photo_taken = True
            print("start taking photo.....")
        if photo_taken:
            if time.time() - t_cam > 0.5:
            # save_base = trainB_dir if random.uniform(0, 1) < 0.8 else testB_dir
                save_name = '{}.png'.format(idx)
                img = clip_image(img, img_size)
                cv2.imwrite(os.path.join(trainB_dir, save_name), img)  #bgr2rgb
                t_cam = time.time()
            # cv2.imwrite(os.path.join(testB_dir, save_name), img)  # bgr2rgb
        if stop_flag:
            print("stop taking photo.....")
            break
        if skip_flag and time.time()-t_0 > 1:
            t_0 = time.time()
            part_name=part_name[0:5]+str(int(part_name[5:])+1)
            part_base_dir = os.path.join(img_base_dir, part_name)
            trainB_dir = os.path.join(part_base_dir, "trainB")
            testB_dir = os.path.join(part_base_dir, "testB")
            print("skipping to {}...".format(part_name))

            ensure_dir(trainB_dir)
            # ensure_dir(testB_dir)
            if motion_type == "auto":
                robo_ins.move_cart(pose=init_pos,tool=1, user=0, vel=40)
        #move robot
        if motion_type == "auto":
            tcp=robo_ins.get_gripper_TCP_pose()
            current_dist=np.linalg.norm(np.array(tcp)[0:2].copy()-np.array(init_pos)[0:2].copy())
            tr_vel=determine_trans_vel(max_rad=motion_radius_thres,cur_rad=current_dist,max_vel=max_trans_vel)
            delta_tr=generate_random_dxdy(tr_vel)

            delta_rot = desire_rz - tcp[5]
            delta_rot = make_an_angle_in_180(delta_rot)

            if abs(delta_rot)<2:
                desire_rz*=-1
                continue

            vrz = rot_vel* (delta_rot) / abs(delta_rot)
            vel = [delta_tr[0], delta_tr[1], 0, 0, 0, vrz]

            robo_ins.servo_cart(desc_pos=vel, mode=1, vel=10.0)
        time.sleep(0.1)
        idx+=1

    cam.release()
    cv2.destroyAllWindows()
