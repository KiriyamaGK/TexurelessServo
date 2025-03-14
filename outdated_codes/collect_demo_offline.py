import os
import numpy as np
import json
from sim.environment import Environment
from utils.transform import rmat2euler_rz_degree
from sim.perception import CameraIntrinsic
import time
import cv2
from utils.input_process import clip_image
from utils.get_stl_geometry import get_stl_geometry
from utils.transform import project_XYZw_to_uv
import open3d as o3d
from utils.transform import rotation_matrix_z
from outdated_codes.gaussion_kernal import gaussian_img


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
    random_light_dir=True
    config_dir= "../configs/demo_collection.json"
    current_date = "25.01.23"

    with open(config_dir, "r") as j:
        config = json.load(j)
    objs_descriptor=config['objs_descriptor']
    motion_type=config['trans_and_rot_type']
    demo_total_num=config['demo_total_num']
    trans_vel_norm=config['trans_vel'] #m
    rot_vel_norm=config['rot_vel']    #deg
    use_max_rot=config['use_max_rot']
    project_obj_kps=config['project_obj_kps']

    # a=p.connect(p.GUI)
    # print(a)
    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    env=Environment(camera_intrinsic,objs_descriptor=objs_descriptor,use_max_rot=use_max_rot)
    env.init()
    base_dir = "/media/kiriyamagk/One Touch"
    if project_obj_kps:
        scaler=env.obj_scale_factor
        obj_pos=env.objStartPos
        intr=o3d.camera.PinholeCameraIntrinsic(
                width=camera_intrinsic.width,
                height=camera_intrinsic.height,
                fx=camera_intrinsic.fx,
                fy=camera_intrinsic.fy,
                cx=camera_intrinsic.cx,
                cy=camera_intrinsic.cy,
            ).intrinsic_matrix
        intr=intr@rotation_matrix_z(np.pi)
    for idx in range(demo_total_num):
        if project_obj_kps:
            obj_id=env.obj_idx
            obj_keypt_y,obj_height=get_stl_geometry('../meshes/objs',obj_id)
            XYZw_ct=np.array(obj_pos)+np.array([0,0,scaler*obj_height])
            XYZw_keypt = np.array(obj_pos) + np.array([0, 0, scaler * obj_height])+np.array([0, scaler*obj_keypt_y, 0])
        print("[INFO] start collecting demo_{} ...".format(idx))
        npy_dir = os.path.join(base_dir, 'AlignAnything', current_date, f'npys/{int(time.time())}')
        ensure_dir(npy_dir)
        # image_rgb3_path = os.path.join(base_dir, 'AlignAnything', current_date, f'image/wrist/{int(time.time())}')
        # ensure_dir(image_rgb3_path)
        action_list=[]
        img_lst=[]
        rz_list=[]
        gaussian_img_ct_lst=[]
        gaussian_img_kpt_lst = []
        while True:
            dT=np.eye(4)
            img=env.observation(random_light_dir)
            wgT_tar=env.wgT_tar
            wgT=env.wgT
            rz=rmat2euler_rz_degree(wgT)
            del_tr=wgT_tar[0:2,3]-wgT[0:2,3]
            abs_del_tr=np.linalg.norm(del_tr)
            if project_obj_kps:
                uv_ct=project_XYZw_to_uv(intr,env.cwT,XYZw_ct)
                uv_kpt = project_XYZw_to_uv(intr, env.cwT, XYZw_keypt)
                # cv2.circle(img, (int(uv_kpt[0]),int(uv_kpt[1])), radius=5, color=255, thickness=-1)
                # cv2.circle(img, (int(uv_ct[0]),int(uv_ct[1])), radius=5, color=255, thickness=-1)
                # cv2.imshow('img', img)
                # cv2.waitKey(1)
                new_uv_ct=pixel_cord_from_frame1_to_frame3(h=480,w=640,h_hat=gau_h,u1=uv_ct[0],v1=uv_ct[1])
                new_uv_kpt = pixel_cord_from_frame1_to_frame3(h=480, w=640, h_hat=gau_h, u1=uv_kpt[0], v1=uv_kpt[1])
                gauss_img_ct=gaussian_img([gau_h,gau_h], new_uv_ct[np.newaxis,:], kernel_size=20, sigma=2.0)
                gauss_img_kpt = gaussian_img([gau_h,gau_h], new_uv_kpt[np.newaxis, :], kernel_size=20, sigma=2.0)
                # cv2.imshow('img',gauss_img_ct+gauss_img_kpt)
                # cv2.waitKey(1)
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
            if project_obj_kps:
                # cv2.imshow('img',gauss_img_ct)
                # cv2.imshow('gaussian_img',gauss_img_kpt)
                # cv2.waitKey(1)
                # img_vis=img.copy()
                # img_vis=cv2.resize(img_vis,(64,64))
                # cv2.circle(img_vis, (int(new_uv_kpt[0]), int(new_uv_kpt[1])), radius=5, color=255, thickness=-1)
                # cv2.circle(img_vis, (int(new_uv_ct[0]),int(new_uv_ct[1])), radius=5, color=255, thickness=-1)
                # cv2.imshow('img', img_vis)
                # cv2.waitKey(1)
                gaussian_img_ct_lst.append(gauss_img_ct)
                gaussian_img_kpt_lst.append(gauss_img_kpt)
            env.action(dT)
            if env.reinit():
                assert len(img_lst)==len(action_list)
                episode={
                    'img': np.asarray(img_lst,dtype=np.uint8),
                    'action_list': np.asarray(action_list,dtype=np.float32),
                    'abs_rot': np.asarray(rz_list,dtype=np.float32),
                }
                if project_obj_kps:
                    episode['gauss_img_ct']=np.asarray(gaussian_img_ct_lst,dtype=np.float32),
                    episode['gauss_img_kpt'] = np.asarray(gaussian_img_kpt_lst, dtype=np.float32),

                npy_name = 'demo.npy'
                npy_path = os.path.join(npy_dir, npy_name)
                np.save(npy_path, episode)
                print("action_lst-1:",action_list[-1])
                print("[INFO] demo_{} collected successfully.".format(idx))
                break










