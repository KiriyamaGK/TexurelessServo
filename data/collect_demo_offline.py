import os
import numpy as np
import json
from sim.environment import Environment
from utils.transform import rotation_matrix_z,rmat2euler_rz_degree
from utils.perception import CameraIntrinsic
from datetime import datetime
import pybullet as p
import time
import cv2
from math import asin
from utils.input_process import clip_image


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    img_w=200
    img_h=200
    cut_to_square=True
    config_dir= "../configs/demo_collection.json"
    current_date = "25.01.19"

    with open(config_dir, "r") as j:
        config = json.load(j)
    num_objs=config['num_objs']
    motion_type=config['trans_and_rot_type']
    demo_total_num=config['demo_total_num']
    trans_vel_norm=config['trans_vel'] #m
    rot_vel_norm=config['rot_vel']    #deg

    # a=p.connect(p.GUI)
    # print(a)
    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    env=Environment(camera_intrinsic,num_objs=num_objs)
    env.init()
    base_dir = "/media/kiriyamagk/One Touch"
    for idx in range(demo_total_num):
        print("[INFO] start collecting demo_{} ...".format(idx))
        npy_dir = os.path.join(base_dir, 'AlignAnything', current_date, f'npys/{int(time.time())}')
        ensure_dir(npy_dir)
        # image_rgb3_path = os.path.join(base_dir, 'AlignAnything', current_date, f'image/wrist/{int(time.time())}')
        # ensure_dir(image_rgb3_path)
        action_list=[]
        img_lst=[]
        rz_list=[]
        while True:
            dT=np.eye(4)
            img=env.observation()
            wgT_tar=env.wgT_tar
            wgT=env.wgT
            rz=rmat2euler_rz_degree(wgT)
            del_tr=wgT_tar[0:2,3]-wgT[0:2,3]
            abs_del_tr=np.linalg.norm(del_tr)
            if motion_type!="simultaneously":
                if abs_del_tr > env.dist_eps:
                    vel_tr = del_tr / abs_del_tr * trans_vel_norm
                    dT[0:2, 3] = vel_tr
                    vel_rot = 0
                else:
                    vel_tr = np.array([0, 0])
                    del_rot_mat = np.linalg.inv(wgT[0:3, 0:3]) @ wgT_tar[0:3, 0:3]  # 绕夹爪自己的轴
                    # abs_del_rot=abs(asin(del_rot_mat[0,1]))
                    if del_rot_mat[0, 1] > 0:
                        vel_rot = rot_vel_norm
                    elif del_rot_mat[0, 1] < 0:
                        vel_rot = -rot_vel_norm
                    else:
                        vel_rot = 0
                    dT[0:3, 0:3] = rotation_matrix_z(vel_rot / 180 * np.pi)
            else:
                del_rot_mat = np.linalg.inv(wgT[0:3, 0:3]) @ wgT_tar[0:3, 0:3]  # 绕夹爪自己的轴
                if abs_del_tr>env.dist_eps:
                    vel_tr=del_tr/abs_del_tr*trans_vel_norm
                else:
                    vel_tr=np.array([0,0])
                if del_rot_mat[0,1]>0:
                    vel_rot=rot_vel_norm
                elif del_rot_mat[0,1]<0:
                    vel_rot=-rot_vel_norm
                else:
                    vel_rot=0
                dT[0:2, 3] = vel_tr
                dT[0:3,0:3]=rotation_matrix_z(vel_rot/180*np.pi)
            action_list.append(np.concatenate((vel_tr,np.array([vel_rot]))))
            rz_list.append(rz)
            if cut_to_square:
                img=clip_image(img,img_h)
            else:
                img=cv2.resize(img,(img_w,img_h))
            img_lst.append(img)
            env.action(dT)
            if env.reinit():
                assert len(img_lst)==len(action_list)
                episode={
                    'img': np.asarray(img_lst,dtype=np.uint8),
                    'action_list': np.asarray(action_list,dtype=np.float32),
                    'abs_rot': np.asarray(rz_list,dtype=np.float32),
                }
                npy_name = 'demo.npy'
                npy_path = os.path.join(npy_dir, npy_name)
                np.save(npy_path, episode)
                print("action_lst-1:",action_list[-1])
                print("[INFO] demo_{} collected successfully.".format(idx))
                break










