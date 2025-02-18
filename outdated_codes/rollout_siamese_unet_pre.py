import os
import numpy as np
import json
from sim.environment import Environment
from utils.transform import rotation_matrix_z,rmat2euler_rz_degree
from utils.perception import CameraIntrinsic
from utils.input_process import input_dict_preprocess
import time
import cv2
import torch
from networks.helpers import get_network_cls
from utils.input_process import clip_image
from outdated_codes.gaussion_kernal import gaussian_img
from utils.transform import project_XYZw_to_uv
from utils.get_stl_geometry import get_stl_geometry
import open3d as o3d


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _setup_model(model_config: dict):
    """
    Set up the model.
    """
    model,need_init_params = get_network_cls(model_config["algorithm"]["policy"]["name"])
    return model()

if __name__ == '__main__':
    trans_vel_norm = 0.001
    rot_vel_norm = 0.5
    config_dir= "../configs/rollout_siamese_unet.json"

    with open(config_dir, "r") as j:
        config = json.load(j)
    model_config_dir = config["logs_dir"]
    with open(model_config_dir, "r") as j:
        model_config = json.load(j)

    img_w=model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"]
    img_h=img_w
    cut_to_square=config["cut_to_square"]
    ckpts_dir=config["ckpts_dir"]
    objs_descriptor=config['objs_descriptor']
    npy_size=config['npy_img_size']
    eval_epoch_num=config['eval_epoch_num']
    time_threshold=config['time_threshold']
    rgb_key = [n for n in model_config["dataset"]['specific_obs_keys'] if ('image' in n or "img" in n)]
    low_dim_key = [n for n in model_config["dataset"]['specific_obs_keys'] if n not in rgb_key]

    assert low_dim_key == ['abs_rot']
    # assert rgb_key == ["robot0_eye_in_hand_image", "robot0_eye_in_hand_image_goal"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _setup_model(model_config)
    state_dict = torch.load(ckpts_dir, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
    env=Environment(camera_intrinsic,objs_descriptor=objs_descriptor)
    env.init()

    scaler = env.obj_scale_factor
    obj_pos = env.objStartPos
    intr = o3d.camera.PinholeCameraIntrinsic(
        width=camera_intrinsic.width,
        height=camera_intrinsic.height,
        fx=camera_intrinsic.fx,
        fy=camera_intrinsic.fy,
        cx=camera_intrinsic.cx,
        cy=camera_intrinsic.cy,
    ).intrinsic_matrix
    intr = intr @ rotation_matrix_z(np.pi)
    for idx in range(eval_epoch_num):
        init_transform_dict = env.return_cur_pos_info()
        env.act_to_goal()
        img_goal = env.observation()

        if cut_to_square:
            img_goal = clip_image(img_goal, npy_size)
        else:
            img_goal = cv2.resize(img_goal, (npy_size, npy_size))
        if npy_size!= img_w or npy_size!= img_h:
            img_goal = cv2.resize(img_goal, (img_w, img_h))
        # cv2.imwrite('/media/kiriyamagk/One Touch/AlignAnything/imgs/{}.png'.format(idx+1),img_goal_vis)
        env.act_with_abs_dict(init_transform_dict)
        print("[INFO] start rollout_ ...".format(idx))

        obj_id = env.obj_idx
        obj_keypt_y, obj_height = get_stl_geometry('../meshes/objs', obj_id)
        XYZw_ct = np.array(obj_pos) + np.array([0, 0, scaler * obj_height])
        XYZw_keypt = np.array(obj_pos) + np.array([0, 0, scaler * obj_height]) + np.array([0, scaler * obj_keypt_y, 0])

        uv_ct_tar = project_XYZw_to_uv(intr, env.cwT_tar, XYZw_ct)
        uv_kpt_tar = project_XYZw_to_uv(intr, env.cwT_tar, XYZw_keypt)
        # cv2.circle(img, (int(uv_kpt[0]),int(uv_kpt[1])), radius=5, color=255, thickness=-1)
        # cv2.circle(img, (int(uv_ct[0]),int(uv_ct[1])), radius=5, color=255, thickness=-1)
        # cv2.imshow('img', img)
        # cv2.waitKey(1)
        gauss_img_goal = gaussian_img([480,640],
                                 np.concatenate((uv_kpt_tar[np.newaxis, :], uv_ct_tar[np.newaxis, :]), axis=0),
                                 kernel_size=20, sigma=10.0)
        while True:
            t_0=time.time()
            wgT_tar = env.wgT_tar
            wgT = env.wgT
            rz = rmat2euler_rz_degree(wgT)
            dT=np.eye(4)
            img=env.observation()
            del_tr = wgT_tar[0:2, 3] - wgT[0:2, 3]
            abs_del_tr = np.linalg.norm(del_tr)

            uv_ct = project_XYZw_to_uv(intr, env.cwT, XYZw_ct)
            uv_kpt = project_XYZw_to_uv(intr, env.cwT, XYZw_keypt)
            # cv2.circle(img, (int(uv_kpt[0]),int(uv_kpt[1])), radius=5, color=255, thickness=-1)
            # cv2.circle(img, (int(uv_ct[0]),int(uv_ct[1])), radius=5, color=255, thickness=-1)
            # cv2.imshow('img', img)
            # cv2.waitKey(1)
            gauss_img = gaussian_img([480,640],np.concatenate((uv_kpt[np.newaxis, :], uv_ct[np.newaxis, :]), axis=0),
                                     kernel_size=20, sigma=10.0)
            if cut_to_square:
                img=clip_image(img,npy_size)
            else:
                img=cv2.resize(img, (npy_size, npy_size))
            if npy_size!= img_w or npy_size!= img_h:
                img=cv2.resize(img,(img_w,img_h))
            gauss_img=clip_image(gauss_img, 280)
            gauss_img_goal=clip_image(gauss_img_goal, 280)
            gauss_img=gauss_img.reshape(280,280,1)
            gauss_img_goal=gauss_img_goal.reshape(280,280,1)
            obs_dict={
                "robot0_eye_in_hand_image": img,
                "robot0_eye_in_hand_image_goal": img_goal,
                "abs_rot": np.array([rz]),
                'gaussian_img': np.asarray(gauss_img, dtype=np.float32),
                'gaussian_img_goal': np.asarray(gauss_img_goal, dtype=np.float32),
            }
            obs_dict=input_dict_preprocess(obs_dict,rollout=True)
            pred1, pred2, x1, x2 = model(obs_dict)
            pred1 = pred1.detach().cpu().numpy()
            pred2 = pred2.detach().cpu().numpy()
            x1 = x1.detach().cpu().numpy()
            x2 = x2.detach().cpu().numpy()
            pred1=pred1.reshape(256,256)
            pred2=pred2.reshape(256,256)
            x1=x1.reshape(256,256)
            x2=x2.reshape(256,256)
            #================action========================
            del_rot_mat = np.linalg.inv(wgT[0:3, 0:3]) @ wgT_tar[0:3, 0:3]  # 绕夹爪自己的轴
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
            env.action(dT)
            # time.sleep(0.1)

            #================visualize======================
            combined_img = np.hstack((pred1,x1))
            combined_img_goal = np.hstack((pred2,x2))
            all_combined_img = np.vstack((combined_img, combined_img_goal))
            cv2.imshow('Images', all_combined_img*255)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 1ms
                env.init()
                break
            if env.reinit():  #close_enough
                break
            if time.time()-t_0 > time_threshold:
                env.init()
                break











