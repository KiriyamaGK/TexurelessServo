import os
import numpy as np
import json
from sim.environment import Environment
from utils.transform import rotation_matrix_z,rmat2euler_rz_degree
from utils.perception import CameraIntrinsic
from utils.input_process import input_dict_preprocess
import datetime
import pybullet as p
import time
import cv2
import torch
from networks.helpers import get_network_cls
from utils.input_process import clip_image
from utils.plot import plot_rot_and_trans,plot_trajs,plot_vel,plot_time
from utils.statistics import calculate_success_rate,visualize_final_error
import atexit


def cleanup():
    # print("success list:",success_list)
    if eval_metrics["success_rate"]["utilized"]:
        calculate_success_rate(success_list, os.path.join(save_base_pth, "success_rate.json"))
    visualize_final_error(final_error_list, os.path.join(save_base_pth, "final_error.json"))
    plot_time(time_list,save_base_pth)

# 注册退出时的回调函数
atexit.register(cleanup)

def ensure_dir_with_timestamp(base_dir):
    """
    创建一个包含时间戳的子文件夹
    :param base_dir: 基础文件夹路径（例如 'eval_results'）
    :return: 创建的子文件夹的完整路径
    """
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    subfolder_name = f"{timestamp}"
    full_path = os.path.join(base_dir, subfolder_name)
    os.makedirs(full_path, exist_ok=True)

    print(f"Created directory: {full_path}")
    return full_path


def _setup_model(model_config: dict):
    """
    Set up the model.
    """
    model,_ = get_network_cls(model_config["algorithm"]["policy"]["name"])
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
    ckpt_base = os.path.dirname(ckpts_dir)
    objs_descriptor=config['objs_descriptor']
    npy_size=config['npy_img_size']
    eval_epoch_num=config['eval_epoch_num']
    cv2_visualize=config['cv2_visualize']

    stop_policy = config['stop_policy']
    eval_metrics=config['eval_metrics']
    succ_tr=eval_metrics['success_rate']['trans_threshold']
    succ_rot = eval_metrics['success_rate']['rot_threshold']

    expert_motion_type=config['expert_motion_type']

    rgb_key = [n for n in model_config["dataset"]['specific_obs_keys'] if ('image' in n or "img" in n)]
    low_dim_key = [n for n in model_config["dataset"]['specific_obs_keys'] if n not in rgb_key]

    assert low_dim_key == ['abs_rot']
    # assert rgb_key == ["robot0_eye_in_hand_image", "robot0_eye_in_hand_image_goal"]

    save_base_pth=ensure_dir_with_timestamp(os.path.join(ckpt_base,'eval_results'))
    with open(os.path.join(save_base_pth,"config.json"), "w")as f:
        json.dump(config, f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _setup_model(model_config)
    state_dict = torch.load(ckpts_dir, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
    env=Environment(camera_intrinsic,objs_descriptor=objs_descriptor)
    env.init()
    env.setup_stop_policy(stop_policy)
    success_list = []
    final_error_list = []
    time_list = []
    for idx in range(eval_epoch_num):
        model.buffer=[]
        error_rot_lst=[]
        error_trans_lst=[]
        wgT_list=[]
        vel_tr_lst=[]
        vel_rot_lst=[]

        init_transform_dict = env.return_cur_pos_info()
        env.act_to_goal()
        img_goal = env.observation()
        if cv2_visualize:
            img_goal_vis = cv2.cvtColor(img_goal.copy(), cv2.COLOR_BGR2RGB)
        if cut_to_square:
            img_goal = clip_image(img_goal, npy_size)
        else:
            img_goal = cv2.resize(img_goal, (npy_size, npy_size))
        if npy_size!= img_w or npy_size!= img_h:
            img_goal = cv2.resize(img_goal, (img_w, img_h))
        # cv2.imwrite('/media/kiriyamagk/One Touch/AlignAnything/imgs/{}.png'.format(idx+1),img_goal_vis)
        env.act_with_abs_dict(init_transform_dict)
        print("==============================")
        print("[INFO] start rollout_{}...".format(idx))
        obj_id=env.obj_idx
        obj_pth=os.path.join(save_base_pth, str(obj_id))
        os.makedirs(obj_pth, exist_ok=True)
        t_0 = time.time()
        # try:
        while True:
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
            if npy_size!= img_w or npy_size!= img_h:
                img=cv2.resize(img,(img_w,img_h))
            obs_dict={
                "robot0_eye_in_hand_image": img,
                "robot0_eye_in_hand_image_goal": img_goal,
                'gaussian_img_kpt': np.zeros((img_w//4, img_h//4,1)),
                'gaussian_img_kpt_goal': np.zeros((img_w // 4, img_h // 4, 1)),
                "abs_rot": np.array([rz]),
            }
            obs_dict=input_dict_preprocess(obs_dict,rollout=True)
            pred=model(obs_dict)
            if isinstance(pred, dict):
                predictions=pred['pred_act']
            else:
                predictions=pred
            predictions=predictions.detach().cpu().numpy().reshape(-1,)
            # print("pred:",predictions)
            # predictions/=4
            vel_tr=predictions[0:2]
            vel_rot=predictions[-1]
            dT[0:2,3]=vel_tr
            dT[0:3,0:3]=rotation_matrix_z(vel_rot/180*np.pi)
            env.action(dT)
            env.determine_vel_in_threshold(vel_tr=np.linalg.norm(vel_tr), vel_rot=abs(vel_rot))
            # time.sleep(0.1)
            rtn_dict=env.reinit_eval()
            trans_error=rtn_dict['dist']
            rot_error=rtn_dict['angle']
            error_rot_lst.append(rot_error)          #deg
            error_trans_lst.append(trans_error*1000) #m to mm
            vel_tr_lst.append(np.linalg.norm(vel_tr)*1000) #mm
            vel_rot_lst.append(abs(vel_rot))
            wgT_list.append(wgT)
            # print("time:",time.time()-env.task_timer)
            if rtn_dict["need_reinit"]:
                use_time=time.time()-t_0
                if eval_metrics["error_curve"]["utilized"]:
                    error_pth = os.path.join(obj_pth, "error_curve")
                    os.makedirs(error_pth, exist_ok=True)
                    plot_rot_and_trans(error_rot_lst=error_rot_lst, error_trans_lst=error_trans_lst, use_time=use_time,obj_pth=error_pth)
                    print("last rot error: {}".format(error_rot_lst[-1]))
                    print("last trans error: {}".format(error_trans_lst[-1]))

                if eval_metrics["success_rate"]["utilized"]:
                    success=1 if (error_rot_lst[-1]<=succ_rot and error_trans_lst[-1]<=succ_tr*1000) else 0
                    success_list.append([obj_id,success])
                if eval_metrics["trajectory"]["utilized"]:
                    traj_pth=os.path.join(obj_pth, "traj")
                    os.makedirs(traj_pth, exist_ok=True)
                    plot_trajs(wgT_list=wgT_list, wgT_tar=env.wgT_tar, motion_type=expert_motion_type, obj_path=traj_pth)
                if eval_metrics["velocity"]["utilized"]:
                    vel_pth = os.path.join(obj_pth, "vel")
                    os.makedirs(vel_pth, exist_ok=True)
                    plot_vel(vel_tr=vel_tr_lst,vel_rot=vel_rot_lst,use_time=use_time,obj_path=vel_pth)
                final_error_list.append([obj_id,error_trans_lst[-1],error_rot_lst[-1]])
                time_list.append([obj_id,use_time])
                break
        # except KeyboardInterrupt or SystemExit:
        #     pass
        #     # if eval_metrics["success_rate"]["utilized"]:
        #     #     calculate_success_rate(success_list,os.path.join(save_base_pth,"success_rate.json"))
        #     cleanup()











