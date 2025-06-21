import os
import numpy as np
import json
from sim.environment import Environment
from utils.transform import rotation_matrix_z,rmat2euler_rz_degree,compute_pos_error,error_pos_transform
from sim.perception import CameraIntrinsic
from utils.input_process import input_dict_preprocess
import datetime
import time
import cv2
import torch
from networks.helpers import get_network_cls
from utils.input_process import clip_image,conditioned_clip_and_resize
from utils.plot import plot_rot_and_trans,plot_trajs,plot_vel,plot_time,plot_img_diff,plot_error_pose
from utils.statistics import calculate_success_rate,visualize_final_error
from utils.policy import get_cur_goal_deltapos
import atexit
from utils.paths import path_completion,PROJECT_ROOT_DIR,determine_ckpt_dirs
from scipy.spatial.transform import Rotation as R


def cleanup():
    # print("success list:",success_list)
    if eval_metrics["success_rate"]["utilized"]:
        calculate_success_rate(success_list, os.path.join(save_base_pth, "success_rate.json"))
    visualize_final_error(final_error_list, os.path.join(save_base_pth, "final_error.json"))
    np.save(os.path.join(save_base_pth, "even_distributed_successrate.npy"),np.array(success_even_distributed_list))
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

def compute_uniform_evaluation(cfg):
    x_inteval = cfg["trans"]["x"]["inteval"]
    x_min = cfg["trans"]["x"]["range"][0]
    x_max = cfg["trans"]["x"]["range"][1]

    y_inteval = cfg["trans"]["y"]["inteval"]
    y_min = cfg["trans"]["y"]["range"][0]
    y_max = cfg["trans"]["y"]["range"][1]

    z_inteval = cfg["trans"]["z"]["inteval"]
    z_min = cfg["trans"]["z"]["range"][0]
    z_max = cfg["trans"]["z"]["range"][1]

    rx_inteval = cfg["rot"]["rx"]["inteval"]
    rx_min = cfg["rot"]["rx"]["range"][0]
    rx_max = cfg["rot"]["rx"]["range"][1]

    ry_inteval = cfg["rot"]["ry"]["inteval"]
    ry_min = cfg["rot"]["ry"]["range"][0]
    ry_max = cfg["rot"]["ry"]["range"][1]

    rz_inteval = cfg["rot"]["rz"]["inteval"]
    rz_min = cfg["rot"]["rz"]["range"][0]
    rz_max = cfg["rot"]["rz"]["range"][1]
    x_res=(x_max-x_min)//x_inteval+1
    y_res=(y_max-y_min)//y_inteval+1
    z_res=(z_max-z_min)//z_inteval+1
    rx_res=(rx_max-rx_min)//rx_inteval+1
    ry_res=(ry_max-ry_min)//ry_inteval+1
    rz_res=(rz_max-rz_min)//rz_inteval+1
    return int(x_res*y_res*z_res*rx_res*ry_res*rz_res)
if __name__ == '__main__':
    config_dir= "../configs/rollout.json"

    fps = 30
    vis_h, vis_w = 480, 640
    mp4 = cv2.VideoWriter_fourcc(*'mp4v')

    depth_info = {
        "normalize_scaler": 1}

    with open(config_dir, "r") as j:
        config = json.load(j)
    model_config_dir=path_completion( config["logs_dir"],PROJECT_ROOT_DIR)
    with open(model_config_dir, "r") as j:
        model_config = json.load(j)

    img_w=model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"]
    img_h=img_w

    logs_dir=path_completion(config["logs_dir"],PROJECT_ROOT_DIR)
    ckpt_base = os.path.dirname(logs_dir)
    ckpts_dirs=determine_ckpt_dirs(config["ckpts_dir"],ckpt_base)

    freq_per_pos=config["init"]["uniform_evaluation"]["freq_per_pos"] if (config["init"]["uniform_evaluation"]["utilized"]) else 0
    eval_epoch_num = config['eval_epoch_num'] if (not config["init"]["uniform_evaluation"]["utilized"]) else compute_uniform_evaluation(config["init"]["uniform_evaluation"])*freq_per_pos
    print(eval_epoch_num)
    if isinstance(config['objs_descriptor'],list):
        if eval_epoch_num<len(config['objs_descriptor']):
            objs_descriptor=config['objs_descriptor'][:eval_epoch_num]
        else:
            objs_descriptor=config['objs_descriptor']
    elif isinstance(config['objs_descriptor'],int):
        if eval_epoch_num<20:
            objs_descriptor=eval_epoch_num
        else:
            objs_descriptor=config['objs_descriptor']
    else:
        raise RuntimeError("objs_descriptor must be an int or list")

    cv2_visualize=config['cv2_visualize']
    third_view_camera = config['third_view_camera']

    stop_policy = config['stop_policy']
    eval_metrics=config['eval_metrics']
    use_eval_metrics=eval_metrics["utilized"]
    succ_tr=eval_metrics['success_rate']['trans_threshold']
    succ_rot = eval_metrics['success_rate']['rot_threshold']

    dof = config["dof"]

    init_horizon_trans = config["init"]['init_horizon_trans']["value"]
    init_vertical_trans = config["init"]['init_vertical_trans']["value"]
    init_rot = config["init"]['init_rot']['value']
    using_max_v_trans=config["init"]['init_vertical_trans']["using_max_v_trans"]
    using_minus_vertical = config["init"]['init_vertical_trans']["using_minus"]
    use_max_trans = config["init"]['init_horizon_trans']["use_max_trans"]
    use_max_rot = config["init"]['init_rot']['use_max_rot']
    init_transform_frame = config["init"]['init_transform_frame'] if 'init_transform_frame' in config["init"] else "grip"
    uniform_evaluation=config["init"]['uniform_evaluation']

    expert_motion_type=config['expert_motion_type']
    record_video=config['record_video']
    random_light=config['random_light']

    rgb_key = [n for n in model_config["dataset"]['specific_obs_keys'] if ('image' in n or "img" in n)]
    low_dim_key = [n for n in model_config["dataset"]['specific_obs_keys'] if n not in rgb_key]
    assert 'hdf5_img_size' in model_config["dataset"]
    hdf5_img_size = model_config["dataset"]["hdf5_img_size"]
    pose_and_orientations=model_config["dataset"]["additional_demo_info"]["pose_and_orientations"] if "additional_demo_info" in model_config["dataset"] and "pose_and_orientations" in model_config["dataset"]["additional_demo_info"] else None
    depth_info["utilized"]=model_config["algorithm"]["policy"]["params"]["encoder"]["using_depth"] if "using_depth" in model_config["algorithm"]["policy"]["params"]["encoder"] else False
    using_pose_estm=model_config["algorithm"]["policy"]["params"]["encoder"]["using_pose_estimation"] if "using_pose_estimation" in model_config["algorithm"]["policy"]["params"]["encoder"] else False
    num_cams=model_config["algorithm"]["policy"]["params"]["encoder"]["num_cameras"] if "num_cameras" in model_config["algorithm"]["policy"]["params"]["encoder"] else 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _setup_model(model_config)
    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    env = Environment(camera_config=camera_intrinsic, objs_descriptor=objs_descriptor,use_max_rot=use_max_rot,use_max_trans=use_max_trans,init_horizon_trans=init_horizon_trans,init_vertical_trans=init_vertical_trans,init_transform_frame=init_transform_frame,uniform_evaluation=uniform_evaluation,using_max_v_trans=using_max_v_trans,using_minus_vertical=using_minus_vertical,init_rot=init_rot,dof=dof,depth_info=depth_info,pose_and_orientations=pose_and_orientations,conditioned_sampling=True,trans_vel=[0.00012,0.0012],rot_vel=0.5,third_view_camera=third_view_camera)
    # env = Environment(camera_config=camera_intrinsic, objs_descriptor=objs_descriptor, use_max_rot=use_max_rot,
    #                   use_max_trans=use_max_trans, init_horizon_trans=init_horizon_trans,
    #                   init_vertical_trans=init_vertical_trans, init_transform_frame=init_transform_frame,
    #                   using_max_v_trans=using_max_v_trans, using_minus_vertical=using_minus_vertical, init_rot=init_rot,
    #                   dof=dof, depth_info=depth_info, pose_and_orientations=pose_and_orientations)
    env.init()
    env.setup_stop_policy(stop_policy)

    ckpts_idx=0
    for ckpts_dir in ckpts_dirs:
        ckpts_num=get_epoch_num_from_pthname(os.path.basename(ckpts_dir))
        save_base_pth = ensure_dir_with_timestamp(os.path.join(ckpt_base, 'eval_results'),ckpts_num)
        with open(os.path.join(save_base_pth, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        state_dict = torch.load(ckpts_dir, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device).eval()

        # cv2.namedWindow('Images', cv2.WINDOW_NORMAL)
        success_even_distributed_list = []
        success_list = []
        final_error_list = []
        time_list = []
        for idx in range(eval_epoch_num):
            # print("=========================")
            # print(idx)
            final_error_info_dict = {}
            model.buffer=[]
            error_rot_lst=[]
            error_trans_lst=[]
            error_pos_list=[]
            z_error_lst=[]
            wgT_list=[]
            vel_tr_lst=[]
            vel_rot_lst=[]
            diff_list=[]
            video_flag = False

            init_transform_dict = env.return_cur_pos_info()
            env.act_to_goal()

            im_goal_dict = env.observation()
            img_goal=im_goal_dict['img_1']
            img_goal2 = im_goal_dict['img_2'] if 'img_2' in im_goal_dict and num_cams==2 else None
            img_dep_goal = im_goal_dict["img_1_depth"] if "img_1_depth" in im_goal_dict else None  #[h,w]->[h,w,1]
            img_dep_goal2 = im_goal_dict["img_2_depth"] if "img_2_depth" in im_goal_dict and num_cams==2 else None

            if cv2_visualize:
                img_goal_vis = cv2.cvtColor(img_goal.copy(), cv2.COLOR_BGR2RGB) if img_goal2 is None else cv2.cvtColor(np.vstack((img_goal.copy(), img_goal2.copy())), cv2.COLOR_BGR2RGB)

            img_goal=conditioned_clip_and_resize(img=img_goal, img_h=img_h, img_w=img_w, hdf5_img_size=hdf5_img_size)
            img_goal2 = conditioned_clip_and_resize(img=img_goal2, img_h=img_h, img_w=img_w, hdf5_img_size=hdf5_img_size) if img_goal2 is not None else None
            img_dep_goal = conditioned_clip_and_resize(img=img_dep_goal, img_h=img_h, img_w=img_w,hdf5_img_size=hdf5_img_size) if img_dep_goal is not None else None
            img_dep_goal2 = conditioned_clip_and_resize(img=img_dep_goal2, img_h=img_h, img_w=img_w,hdf5_img_size=hdf5_img_size) if img_dep_goal2 is not None else None

            env.act_with_abs_dict(init_transform_dict)
            print("==============================")
            print("[INFO] start rollout_{}...".format(idx))
            obj_id=env.obj_idx
            obj_pth=os.path.join(save_base_pth, str(obj_id))
            os.makedirs(obj_pth, exist_ok=True)

            video_path=os.path.join(obj_pth,str(obj_id)+'.mp4')
            if not os.path.exists(video_path):
                out = cv2.VideoWriter(video_path, mp4, fps, (vis_w*2, vis_h)) if num_cams==1 else cv2.VideoWriter(video_path, mp4, fps, (vis_w*2, vis_h*2))
                video_flag=True
            t_0 = time.time()
            # try:
            while True:
                wgT,wgT_tar = env.wgT,env.wgT_tar
                if dof == 3:
                    rz = rmat2euler_rz_degree(wgT)
                dT=np.eye(4)

                im_dict=env.observation() if not random_light else env.observation(random_light_dir=True)
                img = im_dict['img_1']
                img2 = im_dict['img_2'] if 'img_2' in im_dict and num_cams==2 else None
                img_dep = im_dict["img_1_depth"] if "img_1_depth" in im_dict else None
                img_dep2 = im_dict["img_2_depth"] if "img_2_depth" in im_dict and num_cams==2 else None

                if cv2_visualize:
                    img_vis = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB) if img2 is None else cv2.cvtColor(np.vstack((img.copy(), img2.copy())), cv2.COLOR_BGR2RGB)
                    combined_img = np.hstack((img_vis, img_goal_vis))
                    cv2.imshow('Images:cur|goal', combined_img)
                    if record_video and video_flag:
                        out.write(combined_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):     #1ms
                        env.init()
                        break
                img=conditioned_clip_and_resize(img=img, img_h=img_h, img_w=img_w, hdf5_img_size=hdf5_img_size)
                img2 = conditioned_clip_and_resize(img=img2, img_h=img_h, img_w=img_w,hdf5_img_size=hdf5_img_size) if img2 is not None else None
                img_dep = conditioned_clip_and_resize(img=img_dep, img_h=img_h, img_w=img_w, hdf5_img_size=hdf5_img_size) if img_dep is not None else None
                img_dep2 = conditioned_clip_and_resize(img=img_dep2, img_h=img_h, img_w=img_w, hdf5_img_size=hdf5_img_size) if img_dep2 is not None else None
                obs_dict={
                    "robot0_eye_in_hand_image": img,
                    "robot0_eye_in_hand_image_goal": img_goal
                }
                if img_dep is not None:
                    obs_dict["depth_image"] = img_dep[..., np.newaxis]
                    obs_dict["depth_image_goal"] = img_dep_goal[..., np.newaxis]
                if img_dep2 is not None:
                    obs_dict["depth_image_2"] = img_dep2[..., np.newaxis]
                    obs_dict["depth_image_2_goal"] = img_dep_goal2[..., np.newaxis]

                if dof == 3:
                    obs_dict["abs_rot"]=np.array([rz])

                if img2 is not None:
                    obs_dict["robot0_eye_in_hand_image_2"]=img2
                    obs_dict["robot0_eye_in_hand_image_2_goal"]=img_goal2

                obs_dict=input_dict_preprocess(obs_dict,rollout=True)
                pred=model(obs_dict)
                if isinstance(pred, dict):
                    predictions=pred["output_tensor"].detach().cpu().numpy().reshape(-1,)
                    delta_pos = pred["pred_delta_pos"].detach().cpu().numpy().reshape(-1,) if using_pose_estm else None
                else:
                    predictions=pred.detach().cpu().numpy().reshape(-1,)
                    delta_pos = None
                # print("pred:",predictions)
                # predictions/=4
                vel_tr=predictions[0:2] if dof == 3 else predictions[0:3]
                vel_rot=predictions[-1] if dof == 3 else predictions[3:]
                dT[0:3,3]=np.concatenate((vel_tr,np.array[0]),axis=0) if dof == 3 else vel_tr
                dT[0:3,0:3]=rotation_matrix_z(vel_rot/180*np.pi) if dof==3 else R.from_rotvec(vel_rot/180*np.pi).as_matrix()
                # print("vel_rot:{}".format(vel_rot))
                env.action(dT)
                env.determine_vel_in_threshold(vel_tr=np.linalg.norm(vel_tr), vel_rot=abs(vel_rot) if dof == 3 else np.linalg.norm(vel_rot))
                # time.sleep(0.1)
                rtn_dict=env.reinit_eval(all_epochs_num=eval_epoch_num,cur_epoch=idx,freq_per_pos=freq_per_pos)
                trans_error=rtn_dict['dist']
                rot_error=rtn_dict['angle']
                z_error=rtn_dict["z_error"]
                error_rot_lst.append(rot_error)          #deg
                error_trans_lst.append(trans_error*1000) #m to mm
                if dof == 6:
                    z_error_lst.append(z_error*1000) #m to mm

                vel_tr_lst.append(np.linalg.norm(vel_tr)*1000) #mm
                vel_rot_lst.append(abs(vel_rot) if dof == 3 else np.linalg.norm(vel_rot))
                wgT_list.append(wgT)
                if delta_pos is not None:
                    delta_pos_gt=get_cur_goal_deltapos(wgT,wgT_tar)["delta_pose"] #mm,deg
                    error_pos=compute_pos_error(pos_cur=delta_pos,pos_tar=delta_pos_gt)  #[6,]
                    error_pos_list.append(error_pos_transform(error_pos))      #[delta_xyz,delta_z,delta_theta][3,]

                if rtn_dict["need_reinit"]:
                    use_time=time.time()-t_0
                    #eval metrics
                    if eval_metrics["error_curve"]["utilized"] and use_eval_metrics:
                        error_pth = os.path.join(obj_pth, "error_curve")
                        os.makedirs(error_pth, exist_ok=True)
                        plot_rot_and_trans(error_rot_lst=error_rot_lst, error_trans_lst=error_trans_lst, use_time=use_time,obj_pth=error_pth,z_error_lst=z_error_lst,show=False)
                        print("last rot error: {}".format(error_rot_lst[-1]))
                        print("last trans error: {}".format(error_trans_lst[-1]))
                        if dof == 6:
                            print("last z error: {}".format(z_error_lst[-1]))

                    if eval_metrics["error_delta_pose"]["utilized"] and use_eval_metrics:
                        error_pose_pth = os.path.join(obj_pth, "error_pose")
                        os.makedirs(error_pose_pth, exist_ok=True)
                        plot_error_pose(error_pos_list=error_pos_list, use_time=use_time,obj_pth=error_pose_pth,show=False)
                        print("last trans pose XYZ estimation error: {}".format(error_pos_list[-1][0]))
                        print("last trans pose Z estimation error: {}".format(error_pos_list[-1][1]))
                        print("last rot pose estimation error: {}".format(error_pos_list[-1][2]))

                    if eval_metrics["success_rate"]["utilized"] and use_eval_metrics:
                        success=1 if (error_rot_lst[-1]<=succ_rot and error_trans_lst[-1]<=succ_tr*1000) else 0
                        success_list.append([obj_id,success])
                        #均匀分布绘图
                        if idx % freq_per_pos == 0:
                            tmp_dict = env.all_even_poses[env.evenly_posid]
                            tmp_dict['success'] = success
                            tmp_dict["error_trans"]=[error_trans_lst[-1]]
                            tmp_dict["error_transz"] = [z_error_lst[-1]]
                            tmp_dict["error_transxy"] = [np.sqrt(error_trans_lst[-1]**2-z_error_lst[-1]**2)]
                            tmp_dict["error_rot"]=[error_rot_lst[-1]]
                            success_even_distributed_list.append(tmp_dict)

                        else:
                            success_even_distributed_list[-1]["success"] += success
                            success_even_distributed_list[-1]["error_trans"].append(error_trans_lst[-1])
                            success_even_distributed_list[-1]["error_transz"].append(z_error_lst[-1])
                            success_even_distributed_list[-1]["error_transxy"].append(np.sqrt(error_trans_lst[-1]**2-z_error_lst[-1]**2))
                            success_even_distributed_list[-1]["error_rot"].append(error_rot_lst[-1])

                    if eval_metrics["trajectory"]["utilized"] and use_eval_metrics:
                        traj_pth=os.path.join(obj_pth, "traj")
                        os.makedirs(traj_pth, exist_ok=True)
                        plot_trajs(wgT_list=wgT_list, wgT_tar=env.wgT_tar, motion_type=expert_motion_type, obj_path=traj_pth,show=False)
                    if eval_metrics["velocity"]["utilized"] and use_eval_metrics:
                        vel_pth = os.path.join(obj_pth, "vel")
                        os.makedirs(vel_pth, exist_ok=True)
                        plot_vel(vel_tr=vel_tr_lst,vel_rot=vel_rot_lst,use_time=use_time,obj_path=vel_pth,show=False)

                    final_error_info_dict["obj_id"]=obj_id
                    final_error_info_dict["final_trans_error"]=error_trans_lst[-1]
                    final_error_info_dict["final_rot_error"]=error_rot_lst[-1]
                    final_error_info_dict["final_z_error"]=z_error_lst[-1] if dof == 6 else None
                    final_error_info_dict["final_pos_xyz_error"]=error_pos_list[-1][0] if len(error_pos_list)!=0 else None
                    final_error_info_dict["final_pos_z_error"] = error_pos_list[-1][1] if len(
                        error_pos_list) != 0 else None
                    final_error_info_dict["final_pos_rot_error"] = error_pos_list[-1][2] if len(
                        error_pos_list) != 0 else None
                    final_error_list.append(final_error_info_dict)
                    time_list.append([obj_id,use_time])
                    if video_flag:
                        out.release()
                    break
            # except KeyboardInterrupt or SystemExit:
            #     pass
            #     # if eval_metrics["success_rate"]["utilized"]:
            #     #     calculate_success_rate(success_list,os.path.join(save_base_pth,"success_rate.json"))
            #     cleanup()
        if ckpts_idx<len(ckpts_dirs)-1:
            cleanup()
        ckpts_idx+=1










