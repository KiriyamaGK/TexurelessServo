import os
import numpy as np
import json
from utils.transform import rotation_matrix_z,rmat2euler_rz_degree
from utils.input_process import input_dict_preprocess
import datetime
import time
import cv2
import torch
from real.environment import Environment
from networks.helpers import get_network_cls
from utils.input_process import clip_image
from utils.plot import plot_rot_and_trans,plot_trajs,plot_vel,plot_time,plot_img_diff
from utils.statistics import calculate_success_rate,visualize_final_error
import atexit
from utils.paths import path_completion,PROJECT_ROOT_DIR,determine_ckpt_dirs
from utils.transform import rot_angle_normalization


def cleanup():
    # print("success list:",success_list)
    if eval_metrics["success_rate"]["utilized"]:
        calculate_success_rate(success_list, os.path.join(save_base_pth, "success_rate.json"))
    visualize_final_error(final_error_list, os.path.join(save_base_pth, "final_error.json"))
    plot_time(time_list,save_base_pth,show=False)

# 注册退出时的回调函数
atexit.register(cleanup)

def ensure_dir_with_timestamp(base_dir,num):
    """
    创建一个包含时间戳的子文件夹
    :param base_dir: 基础文件夹路径（例如 'eval_results'）
    :return: 创建的子文件夹的完整路径
    """
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    subfolder_name = f"{timestamp}"+'(epoch'+str(num)+')'
    full_path = os.path.join(base_dir, subfolder_name)
    os.makedirs(full_path, exist_ok=True)

    print(f"Created directory: {full_path}")
    return full_path

def get_epoch_num_from_pthname(strin):
    start_id=6
    for i in range(5):
        if strin[start_id+i+1]!='_':
            continue
        else:
            end_id=start_id+i
            break
    return strin[start_id:end_id+1]


def _setup_model(model_config: dict,return_dual_feat:bool=False):
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
        return_dual_features=return_dual_feat,
        **model_config["algorithm"]["policy"]["params"]
    )#**动态传参，字典中的键与函数参数名完全匹配

if __name__ == '__main__':
    config_dir= "../configs/rollout_real.json"
    return_dual_feat=False
    current_pt_desire = False

    with open(config_dir, "r") as j:
        config = json.load(j)

    env = Environment(robot_address=config["hardware"]["robot_address"], **config["distances"],
                      **config["hardware"]["camera"])
    cam = env.camera
    robot_ins = env.robot_ins

    desire_pt = robot_ins.get_gripper_TCP_pose()
    desire_pt[3] = -180
    desire_pt[4] = 0
    if not current_pt_desire:
        desire_pt=[-663.347412109375, -49.298927307128906, 114.417236328125, -180,0, -153.80250549316406]

    fps = 30
    vis_h, vis_w = 480, 1280
    mp4 = cv2.VideoWriter_fourcc(*'mp4v')

    model_config_dir=path_completion(config["logs_dir"],PROJECT_ROOT_DIR)
    with open(model_config_dir, "r") as j:
        model_config = json.load(j)

    img_w=model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"]
    img_h=img_w
    cut_to_square=config["cut_to_square"]

    logs_dir=path_completion(config["logs_dir"],PROJECT_ROOT_DIR)
    ckpt_base = os.path.dirname(logs_dir)
    ckpts_dirs=determine_ckpt_dirs(config["ckpts_dir"],ckpt_base)

    eval_epoch_num = config['eval_epoch_num']

    npy_size=config['npy_img_size']
    cv2_visualize=config['cv2_visualize']

    stop_policy = config['stop_policy']
    eval_metrics=config['eval_metrics']
    use_eval_metrics=eval_metrics["utilized"]
    succ_tr=eval_metrics['success_rate']['trans_threshold']
    succ_rot = eval_metrics['success_rate']['rot_threshold']
    record_video=config['record_video']

    obj_id=config['obj_id']


    rgb_key = [n for n in model_config["dataset"]['specific_obs_keys'] if ('image' in n or "img" in n)]
    low_dim_key = [n for n in model_config["dataset"]['specific_obs_keys'] if n not in rgb_key]
    if 'hdf5_img_size' in model_config["dataset"]:
        hdf5_img_size = model_config["dataset"]["hdf5_img_size"]

    assert low_dim_key == ['abs_rot']
    # assert rgb_key == ["robot0_eye_in_hand_image", "robot0_eye_in_hand_image_goal"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _setup_model(model_config,return_dual_feat)

    env.setup_stop_policy(stop_policy)
    env.setup_desire_pt(desire_pt)
    env.init()

    ckpts_idx=0
    for ckpts_dir in ckpts_dirs:
        ckpts_num=get_epoch_num_from_pthname(os.path.basename(ckpts_dir))
        save_base_pth = ensure_dir_with_timestamp(os.path.join(ckpt_base, 'eval_results'),ckpts_num)
        with open(os.path.join(save_base_pth, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        state_dict = torch.load(ckpts_dir, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device).eval()

        cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
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
            diff_list=[]
            video_flag = False

            # init_transform_dict = env.return_cur_pos_info()
            # env.act_to_goal()
            # img_goal = env.observation()
            robot_ins.move_cart(desire_pt, tool=1, user=0, vel=40)
            img_goal=cam.get_frame()["wrist"][:,:,::-1]

            if cv2_visualize:
                img_goal_vis = cv2.cvtColor(img_goal.copy(), cv2.COLOR_BGR2RGB)
            if cut_to_square:
                img_goal = clip_image(img_goal, npy_size)
            else:
                img_goal = cv2.resize(img_goal, (npy_size, npy_size))
            if 'hdf5_img_size' in model_config["dataset"]:
                if npy_size != hdf5_img_size:
                    img_goal = cv2.resize(img_goal, (hdf5_img_size, hdf5_img_size))
                if hdf5_img_size != img_w or hdf5_img_size != img_h:
                    img_goal = cv2.resize(img_goal, (img_w, img_h))
            else:
                if npy_size != img_w or npy_size != img_h:
                    img_goal = cv2.resize(img_goal, (img_w, img_h))

            # env.act_with_abs_dict(init_transform_dict)
            theta, alpha, start_pt = env.generate_motion_paras(desire_pt)
            robot_ins.move_cart(start_pt, tool=1, user=0, vel=40)

            print("==============================")
            print("[INFO] start rollout_{}...".format(idx))

            obj_pth=os.path.join(save_base_pth, str(obj_id))
            os.makedirs(obj_pth, exist_ok=True)

            video_path=os.path.join(obj_pth,str(obj_id)+'.mp4')
            if not os.path.exists(video_path):
                out = cv2.VideoWriter(video_path, mp4, fps, (vis_w, vis_h))
                video_flag=True
            t_0 = time.time()
            # try:
            while True:
                tcp = robot_ins.get_gripper_TCP_pose()
                rz=rot_angle_normalization(tcp[5])  # 轉換到0-360之間，-180和180之間有間斷點，不方便學習
                img=cam.get_frame()["wrist"][:,:,::-1]

                if cv2_visualize:
                    img_vis = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                    combined_img = np.hstack((img_vis, img_goal_vis))
                    cv2.imshow('Images', combined_img)
                    if record_video and video_flag:
                        out.write(combined_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):     #1ms
                        env.init()
                        break
                if cut_to_square:
                    img=clip_image(img,npy_size)
                else:
                    img=cv2.resize(img, (npy_size, npy_size))
                if 'hdf5_img_size' in model_config["dataset"]:
                    if npy_size != hdf5_img_size:
                        img = cv2.resize(img, (hdf5_img_size, hdf5_img_size))
                    if hdf5_img_size != img_w or hdf5_img_size != img_h:
                        img = cv2.resize(img, (img_w, img_h))
                else:
                    if npy_size != img_w or npy_size != img_h:
                        img = cv2.resize(img, (img_w, img_h))
                obs_dict={
                    "robot0_eye_in_hand_image": img,
                    "robot0_eye_in_hand_image_goal": img_goal,
                    # 'gaussian_img_kpt': np.zeros((img_w//4, img_h//4,1)),
                    # 'gaussian_img_kpt_goal': np.zeros((img_w // 4, img_h // 4, 1)),
                    "abs_rot": np.array([rz]),
                }
                obs_dict=input_dict_preprocess(obs_dict,rollout=True)
                pred=model(obs_dict)
                if isinstance(pred, dict):
                    predictions=pred['pred_act']
                else:
                    predictions=pred
                predictions=predictions.detach().cpu().numpy().reshape(-1,)
                if return_dual_feat:
                    img_feat=pred['x_img_feat'].detach().cpu().numpy().reshape(-1,)
                    img_goal_dual_feat=pred["x_img_goal_dual_feat"].detach().cpu().numpy().reshape(-1,)
                    diff=abs(np.mean(img_feat-img_goal_dual_feat))
                    diff_list.append(diff)
                # print("pred:",predictions)
                # predictions/=4
                vel_tr=predictions[0:2]
                vel_rot=predictions[-1]

                env.determine_vel_in_threshold(vel_tr=np.linalg.norm(vel_tr), vel_rot=abs(vel_rot))
                robot_ins.servo_cart(desc_pos=[vel_tr[0],vel_tr[1],0,0,0,vel_rot], mode=1, vel=10.0)
                # time.sleep(0.1)
                rtn_dict=env.reinit_eval()
                trans_error=rtn_dict['dist']
                rot_error=rtn_dict['angle']
                error_rot_lst.append(rot_error)             #deg
                error_trans_lst.append(trans_error)         # mm
                vel_tr_lst.append(np.linalg.norm(vel_tr))   #mm
                vel_rot_lst.append(abs(vel_rot))

                if rtn_dict["need_reinit"]:
                    use_time=time.time()-t_0
                    if eval_metrics["error_curve"]["utilized"] and use_eval_metrics:
                        error_pth = os.path.join(obj_pth, "error_curve")
                        os.makedirs(error_pth, exist_ok=True)
                        plot_rot_and_trans(error_rot_lst=error_rot_lst, error_trans_lst=error_trans_lst, use_time=use_time,obj_pth=error_pth,show=False)
                        print("last rot error: {}".format(error_rot_lst[-1]))
                        print("last trans error: {}".format(error_trans_lst[-1]))
                    if eval_metrics["success_rate"]["utilized"] and use_eval_metrics:
                        success=1 if (error_rot_lst[-1]<=succ_rot and error_trans_lst[-1]<=succ_tr) else 0
                        success_list.append([obj_id,success])
                    if eval_metrics["velocity"]["utilized"] and use_eval_metrics:
                        vel_pth = os.path.join(obj_pth, "vel")
                        os.makedirs(vel_pth, exist_ok=True)
                        plot_vel(vel_tr=vel_tr_lst,vel_rot=vel_rot_lst,use_time=use_time,obj_path=vel_pth,show=False)
                    if return_dual_feat:
                        dual_feat_pth = os.path.join(obj_pth, "img_diff")
                        os.makedirs(dual_feat_pth, exist_ok=True)
                        plot_img_diff(diff_list=diff_list,use_time=use_time,obj_path=dual_feat_pth,show=False)
                    final_error_list.append([obj_id,error_trans_lst[-1],error_rot_lst[-1]])
                    time_list.append([obj_id,use_time])
                    if video_flag:
                        out.release()
                    cv2.imwrite(os.path.join(obj_pth, "goal_img.png"),img_goal_vis)
                    break

        if ckpts_idx<len(ckpts_dirs)-1:
            cleanup()
        ckpts_idx+=1










