import os
import numpy as np
import json
from sim.environment import Environment
from utils.paths import return_disc_route
from utils.transform import rotation_matrix_z,rmat2euler_rz_degree
from utils.perception import CameraIntrinsic
from datetime import datetime
import pybullet as p
import time
import cv2
from math import asin
from utils.input_process import clip_image
from utils.get_stl_geometry import get_stl_geometry
from utils.transform import project_XYZw_to_uv
import open3d as o3d
from utils.policy import get_expert_policy



def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def pixel_cord_from_frame1_to_frame3(h,w,h_hat,u1,v1):
     """
     u1,v1:h,w
     u2,v2:h,h
     u3,v3:h_hat,h_hat
     """
     return np.array([(u1-w/2+h/2)*h_hat/h,v1*h_hat/h])

if __name__ == '__main__':
    img_w=220
    img_h=220
    gau_h=64
    cut_to_square=True
    config_dir= "../configs/demo_collection.json"
    current_date = "25.01.24"

    with open(config_dir, "r") as j:
        config = json.load(j)
    objs_descriptor=config['objs_descriptor']
    motion_type=config['trans_and_rot_type']
    demo_total_num=config['demo_total_num']
    trans_vel_norm=config['trans_vel'] #m
    rot_vel_norm=config['rot_vel']    #deg
    use_max_rot=config['use_max_rot']
    random_light_dir = config['random_light_dir']
    use_light_key=["use_random_light_img_key"]

    # a=p.connect(p.GUI)
    # print(a)
    if not random_light_dir:
        use_light_key=False
    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    env=Environment(camera_intrinsic,objs_descriptor=objs_descriptor,use_max_rot=use_max_rot)
    env.init()
    base_dir = return_disc_route("One Touch")
    for idx in range(demo_total_num):
        print("[INFO] start collecting demo_{} ...".format(idx))
        npy_dir = os.path.join(base_dir, 'AlignAnything', current_date, f'npys/{int(time.time())}')
        ensure_dir(npy_dir)
        # image_rgb3_path = os.path.join(base_dir, 'AlignAnything', current_date, f'image/wrist/{int(time.time())}')
        # ensure_dir(image_rgb3_path)
        action_list=[]
        img_lst=[]
        img_light_list=[]
        rz_list=[]
        gaussian_img_ct_lst=[]
        gaussian_img_kpt_lst = []
        while True:
            if not use_light_key:
                img=env.observation(random_light_dir=random_light_dir,use_prob=True)
            else:
                img=env.observation(random_light_dir=False)
                img_light=env.observation(random_light_dir=True,use_prob=False)

            wgT_tar=env.wgT_tar
            wgT=env.wgT
            rz=rmat2euler_rz_degree(wgT)
            act_dict=get_expert_policy(wgT_tar=wgT_tar,wgT=wgT,trans_vel_norm=trans_vel_norm,rot_vel_norm=rot_vel_norm,dist_eps=env.dist_eps,angle_eps=env.angle_eps,motion_type=motion_type)
            vel_tr=act_dict['vel_tr']
            vel_rot=act_dict['vel_rot']
            dT=act_dict["dT"]
            action_list.append(np.concatenate((vel_tr,np.array([vel_rot]))))
            rz_list.append(rz)
            # cv2.imshow("img",img)
            # cv2.imshow("img_light",img_light)
            cv2.waitKey(1)
            if cut_to_square:
                img=clip_image(img,img_h)
                if use_light_key:
                    img_light=clip_image(img_light,img_h)
            else:
                img=cv2.resize(img,(img_w,img_h))
                if use_light_key:
                    img_light=cv2.resize(img_light,(img_w,img_h))
            img_lst.append(img)
            if use_light_key:
                img_light_list.append(img_light)

            env.action(dT)
            if env.reinit():
                assert len(img_lst)==len(action_list)
                if not use_light_key:
                    episode={
                        'img': np.asarray(img_lst,dtype=np.uint8),
                        'action_list': np.asarray(action_list,dtype=np.float32),
                        'abs_rot': np.asarray(rz_list,dtype=np.float32),
                    }
                else:
                    episode = {
                        'img': np.asarray(img_lst, dtype=np.uint8),
                        'img_light':np.asarray(img_light_list, dtype=np.uint8),
                        'action_list': np.asarray(action_list, dtype=np.float32),
                        'abs_rot': np.asarray(rz_list, dtype=np.float32),
                    }
                npy_name = 'demo.npy'
                npy_path = os.path.join(npy_dir, npy_name)
                np.save(npy_path, episode)
                print("action_lst-1:",action_list[-1])
                print("[INFO] demo_{} collected successfully.".format(idx))
                break










