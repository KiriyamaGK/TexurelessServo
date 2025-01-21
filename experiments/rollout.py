import os
import numpy as np
import json
from sim.environment import Environment
from utils.transform import rotation_matrix_z,rmat2euler_rz_degree
from utils.perception import CameraIntrinsic
from utils.input_process import input_dict_preprocess
from datetime import datetime
import pybullet as p
import time
import cv2
import torch
from networks.helpers import get_network_cls
from utils.input_process import clip_image


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _setup_model(model_config: dict):
    """
    Set up the model.
    """
    model = get_network_cls(model_config["algorithm"]["policy"]["name"])
    return model(
        input_low_dim=model_config["dataset"]["input_low_dim"],
        output_dim=model_config["dataset"]["output_dim"],
        obs_keys=model_config["dataset"]["specific_obs_keys"],
        batch_size=1,
        seq_length=1,
        training=False,
        **model_config["algorithm"]["policy"]["params"]
    )#**动态传参，字典中的键与函数参数名完全匹配

if __name__ == '__main__':
    config_dir= "../configs/rollout.json"

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
    cv2_visualize=config['cv2_visualize']
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

    for idx in range(eval_epoch_num):
        init_transform_dict = env.return_cur_pos_info()
        env.act_to_goal()
        img_goal = env.observation()
        if cv2_visualize:
            img_goal_vis = cv2.cvtColor(img_goal.copy(), cv2.COLOR_BGR2RGB)
        if cut_to_square:
            img_goal = clip_image(img_goal, npy_size)
        else:
            img_goal = cv2.resize(img_goal, (npy_size, npy_size))
        if npy_size!= img_w and npy_size!= img_h:
            img_goal = cv2.resize(img_goal, (img_w, img_h))
        # cv2.imwrite('/media/kiriyamagk/One Touch/AlignAnything/imgs/{}.png'.format(idx+1),img_goal_vis)
        env.act_with_abs_dict(init_transform_dict)
        print("[INFO] start rollout_ ...".format(idx))
        while True:
            t_0=time.time()
            wgT = env.wgT
            rz = rmat2euler_rz_degree(wgT)
            dT=np.eye(4)
            img=env.observation()
            if cv2_visualize:
                img_vis = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                combined_img = np.hstack((img_vis, img_goal_vis))
                cv2.imshow('Images', combined_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):     #1ms
                    env.init()
                    break
            if cut_to_square:
                img=clip_image(img,npy_size)
            else:
                img=cv2.resize(img, (npy_size, npy_size))
            if npy_size!= img_w and npy_size!= img_h:
                img=cv2.resize(img,(img_w,img_h))
            obs_dict={
                "robot0_eye_in_hand_image": img,
                "robot0_eye_in_hand_image_goal": img_goal,
                "abs_rot": np.array([rz]),
            }
            obs_dict=input_dict_preprocess(obs_dict,rollout=True)
            predictions=model(obs_dict).detach().cpu().numpy().reshape(-1,)
            print("pred:",predictions)
            # predictions/=4
            vel_tr=predictions[0:2]
            vel_rot=predictions[-1]
            dT[0:2,3]=vel_tr
            dT[0:3,0:3]=rotation_matrix_z(vel_rot/180*np.pi)
            env.action(dT)
            # time.sleep(0.1)
            if env.reinit():
                break
            if time.time()-t_0 > time_threshold:
                env.init()
                break











