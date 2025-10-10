import os
from collections import deque
from scipy.spatial.transform import Rotation as R
from utils.hdf5 import add_useless_things, split_train_val_from_hdf5, add_env_meta, compute_num_samples, add_config, create_hdf5_filter_key
import h5py
import numpy as np
import json
from sim.environment import Environment
from utils.paths import return_disc_route
from utils.transform import rmat2euler_rz_degree,construct_dT_from_action
from sim.perception import CameraIntrinsic
import time
import cv2
from utils.input_process import clip_image
from utils.policy import get_expert_policy
from utils.dagger import compute_position_distance_sim, get_policy_action, train_policy, setup_policy_model, prepare_observation_for_policy


def greedy_state_selection(states, student_actions,
                           expert_actions, num_selected_pts, w1=1.0, w2=1.0,s1=1.0,s2=1.0,a1=1.0,a2=1.0):
    """
    使用贪心策略选择需要咨询专家的状态

    Args:
        student_trajectory_states: 学生轨迹中的状态列表 [s1, s2, ..., sT]
        student_trajectory_actions: 学生轨迹中的动作列表 [a1, a2, ..., aT]
        expert_policy: 专家策略函数，输入状态返回专家动作
        num_selected_pts: 需要选择的状态数量
        w1, w2: 误差和隔离度的权重参数

    Returns:
        selected_states: 选中的状态列表
    """
    if len(states) <= num_selected_pts:
        return states.copy()

    # 计算每个状态的动作误差
    action_errors = []
    for i, state in enumerate(states):
        expert_action = expert_actions[i]
        student_action = student_actions[i]
        error = determine_action_error(student_action, expert_action, a1=a1, a2=a2)
        action_errors.append(error)

    action_errors = np.array(action_errors)

    # 贪心选择过程
    selected_states = []
    selected_indices = []

    for _ in range(num_selected_pts):
        best_score = -np.inf
        best_idx = -1

        for i, state in enumerate(states):
            if i in selected_indices:
                continue

            # 计算误差项
            error_term = w1 * action_errors[i]

            # 计算隔离度项
            isolation_term = 0
            if selected_states:
                # 计算当前状态与所有已选状态的最小距离
                distances = [determine_state_error(state,selected_state,s1=s1, s2=s2) for selected_state in selected_states]
                isolation_term = w2 * np.min(distances)
            else:
                isolation_term = w2 * 0  # 第一个状态给一个基础值

            # 综合得分
            score = error_term + isolation_term

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx != -1:
            selected_states.append(states[best_idx])
            selected_indices.append(best_idx)

    return selected_states

def filter_translation(input,thres):
    assert thres>0
    input=np.array(input)
    return np.where(np.abs(input) < thres, 0, input)

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

def get_goal_info(env):
    env.act_to_goal()
    if not use_light_key:
        rtn_dict = env.observation(random_light_dir=random_light_dir, use_prob=True)
        img_light = None
        img2_light = None
    else:
        rtn_dict = env.observation(random_light_dir=False)
        rtn_light_dict = env.observation(random_light_dir=True, use_prob=False)
        img_light = rtn_light_dict['img_1']
        img2_light = rtn_light_dict['img_2'] if 'img_2' in rtn_light_dict else None

    img = rtn_dict['img_1']
    img2 = rtn_dict['img_2'] if 'img_2' in rtn_dict else None
    im_dep = rtn_dict["img_1_depth"] if "img_1_depth" in rtn_dict else None
    im_dep2 = rtn_dict["img_2_depth"] if "img_2_depth" in rtn_dict else None

    return {"img_goal":img,"img_goal2":img2,"img_light_goal":img_light,"img_light_goal2":img2_light,"img_dep_goal":im_dep,"img_dep_goal2":im_dep2}

def process_and_append_observations(img, img2, img_light, img2_light, im_dep, im_dep2, img_h, 
                             img_lst, img2_lst, img_light_list, img2_light_list, im_dep_lst, im_dep2_lst, 
                             use_light_key=True, show_images=True):
 
    # 处理主相机图像
    img_vis = img.copy()
    img = clip_image(img, img_h)
    img_lst.append(img)
    
    # 处理主相机光照图像
    if use_light_key and img_light is not None:
        img_light = clip_image(img_light, img_h)
        img_light_list.append(img_light)
    
    # 处理第二相机图像
    if img2 is not None:
        img2_vis = img2.copy()
        img2 = clip_image(img2, img_h)
        img2_lst.append(img2)
        
        # 显示组合图像
        if show_images:
            combined_img = np.hstack((img_vis, img2_vis))
            cv2.imshow("Combined Image", combined_img)
            cv2.waitKey(1)
    
    # 处理第二相机光照图像
    if img2_light is not None:
        img2_light = clip_image(img2_light, img_h)
        img2_light_list.append(img2_light)
    
    # 处理主相机深度图像
    if im_dep is not None:
        im_dep = clip_image(im_dep, img_h)
        im_dep_lst.append(im_dep[..., np.newaxis])  # [h,w,1]
    
    # 处理第二相机深度图像
    if im_dep2 is not None:
        im_dep2 = clip_image(im_dep2, img_h)
        im_dep2_lst.append(im_dep2[..., np.newaxis])

def determine_mat_error(T1:np.ndarray,T2:np.ndarray,a1,a2):
    dR = np.linalg.inv(T1[0:3,0:3]) @ T2[0:3,0:3]
    rot_error = R.from_matrix(dR).as_rotvec() #rad
    trans_error = np.linalg.norm(T1[0:3,3] - T2[0:3,3])
    return a1 * trans_error + a2 * rot_error

def determine_state_error(T1:np.ndarray,T2:np.ndarray,s1,s2):
    return determine_mat_error(T1,T2,s1,s2)

def determine_action_error(arr1,arr2,a1,a2):
    T1 = np.eye(4)
    T2 = np.eye(4)
    T1[0:3,3] = arr1[0:3]
    T2[0:3,3] = arr2[0:3]
    T1[0:3,0:3] = R.from_rotvec(arr1[3:6]).as_matrix()
    T2[0:3,0:3] = R.from_rotvec(arr2[3:6]).as_matrix()
    return determine_mat_error(T1,T2,a1,a2)

if __name__ == '__main__':
    img_w=300
    img_h=300
    visualize_img = False

    config_dir= "../configs/demo_collection.json"

    with open(config_dir, "r") as j:
        config = json.load(j)

    #overall setting
    base_dir = return_disc_route("One Touch")
    # base_dir = config['overall_setting']["dataset_base_dir"]
    objs_descriptor=config['overall_setting']['objs_descriptor']
    current_date=config['overall_setting']['file_name']
    demo_total_num = config['overall_setting']['demo_total_num']
    replace_existed_hdf5=config["overall_setting"]["replace_existed_hdf5"]

    #demo collection
    dof = config["demo_collection"]["dof"]
    motion_type=config["demo_collection"]['trans_and_rot_type']
    conditioned_sampling=config["demo_collection"]['conditioned_sampling']
    random_light_dir = config["demo_collection"]['random_light_dir']
    use_light_key = config["demo_collection"]["use_random_light_img_key"] if random_light_dir else False
    depth_info=config["demo_collection"]['depth']
    record_pose=config["demo_collection"]['record_pose']
    third_view_camera=config["demo_collection"]['third_view_camera']

    #================================dagger===============================
    # dagger settings
    use_dagger = False
    is_hdf_open = True
    dagger_config = {}
    if "dagger" in config["demo_collection"]:
        dagger_config = config["demo_collection"]["dagger"]
        use_dagger = dagger_config["utilized"]

    # print("use_dagger: ", use_dagger)

    # min_position_threshold: list，表示在DAgger模式下，当夹爪与目标物体之间的最小位置范围。
    # check_frame_interval: 整数，表示在DAgger模式下每隔多少帧检查一次夹爪与物体的距离。

    
    # 如果使用DAgger，设置策略模型
    policy_model = None
    optimizer = None
    criterion = None
    model_config = None
    if use_dagger:
        policy_model, optimizer, criterion, model_config = setup_policy_model(  #事实上model_cfg就是train_mlp
            config_path="../configs/train_mlp.json",
            checkpoint_path=dagger_config.get("model_path", None)
        )
        # 确保模型处于评估模式
        policy_model.eval()
        save_img_size = model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"]

        min_position_threshold = dagger_config["task_termination"]["min_position_threshold"]

    #================================dagger===============================

    trans_vel=config["demo_collection"]["velocity"]['trans_vel'] #m
    rot_vel=config["demo_collection"]["velocity"]['rot_vel']    #deg
    uniform_vel=config["demo_collection"]["velocity"]['uniform_vel']

    init_horizon_trans=config["demo_collection"]["init"]['init_horizon_trans']["value"]
    init_vertical_trans = config["demo_collection"]["init"]['init_vertical_trans']["value"]
    init_rot=config["demo_collection"]["init"]['init_rot']["value"]
    use_max_rot = config["demo_collection"]["init"]['init_rot']['use_max_rot']
    use_max_trans=config["demo_collection"]["init"]['init_horizon_trans']["use_max_h_trans"]
    use_max_v_trans = config["demo_collection"]["init"]['init_vertical_trans']["using_max_v_trans"]
    using_minus_vertical = config["demo_collection"]["init"]['init_vertical_trans']["using_minus"]
    pose_and_orientations=config["demo_collection"]["init"]['pose_and_orientations']
    init_transform_frame=config["demo_collection"]["init"]['init_transform_frame'] if 'init_transform_frame' in config["demo_collection"]["init"] else "grip"

    angle_eps =config["demo_collection"]["stop_policy"]['angle_eps']
    dist_eps = config["demo_collection"]["stop_policy"]['dist_eps']

    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    env=Environment(camera_config=camera_intrinsic,objs_descriptor=objs_descriptor,use_max_rot=use_max_rot,use_max_trans=use_max_trans,using_max_v_trans = use_max_v_trans,init_horizon_trans=init_horizon_trans,init_vertical_trans=init_vertical_trans,using_minus_vertical=using_minus_vertical,init_rot=init_rot,init_transform_frame=init_transform_frame,dof=dof,angle_eps=angle_eps,dist_eps=dist_eps,depth_info=depth_info,pose_and_orientations=pose_and_orientations,_is_collect=True,conditioned_sampling=conditioned_sampling,trans_vel=trans_vel["value"],rot_vel=rot_vel["value"],third_view_camera=third_view_camera,uniform_evaluation={"utilized":False},manually_init=True)
    env.init()

    database_dir = os.path.join(base_dir, 'AlignAnything', current_date, 'hdf5')
    ensure_dir(database_dir)
    dataset_dir = os.path.join(database_dir, 'mimic.hdf5')

    if replace_existed_hdf5:
        new_f_out = h5py.File(dataset_dir, "w")
    else:
        if os.path.exists(dataset_dir):
            new_f_out = h5py.File(dataset_dir, "r+")
        else:
            new_f_out = h5py.File(dataset_dir, "w")

    existed_demo_num = 0

    #================================dagger===============================
    # DAgger varibles
    end_dagger_traj = False
    first_in_error = False
    num_rollout_trajs = 0
    num_expert_trajs = 0
    total_fail_pool_size = 0
    rest_fail_pool_size = 100000
    cur_fail_pool = deque()

    # pre-defined params
    n_base_expert_trajs = 100
    num_selected_pts = 10
    n_fail_pool_size = 100
    s_tr = 1
    s_rot = 1
    a_tr = 1
    a_rot = 1
    w1 = 1
    w2 = 1

    #================================dagger===============================
    
    for idx in range(demo_total_num):
        assert rest_fail_pool_size >= 0

        if not is_hdf_open:
            new_f_out = h5py.File(dataset_dir, "r+")
            is_hdf_open = True

        print("====================start collecting demo_{} ====================".format(idx))
        
        #================================dagger===============================
        first_in_error = False
        is_dagger_traj_iter = False
        train_epochs = 3 #todo: set constant for now,can be self-adaptive in the future
                
        if is_dagger_traj_iter:
            print("[INFO] Using DAgger strategy for iteration {}".format(idx))
        #================================dagger===============================

        if idx==0:
            if 'data' in new_f_out and not replace_existed_hdf5:
                existed_demo_num=len(new_f_out["data"])

        if existed_demo_num>=1:#根据existed_demo_num的数量整体偏移
            obs_path = 'data/demo_{}/obs'.format(idx+existed_demo_num)
            action_path = 'data/demo_{}/actions'.format(idx+existed_demo_num)
            pos_path = 'data/demo_{}/delta_pos_curgoal'.format(idx + existed_demo_num)
        else:
            obs_path = 'data/demo_{}/obs'.format(idx)
            action_path = 'data/demo_{}/actions'.format(idx)
            pos_path = 'data/demo_{}/delta_pos_curgoal'.format(idx)

        wgT_list = []
        action_list=[]
        expert_action_list=[] # 专家动作列表（用于DAgger的标签）
        img_lst=[]
        img_light_list=[]
        im_dep_lst=[]
        img2_lst = []
        img2_light_list = []
        im_dep2_lst=[]
        rz_list=[]
        delta_pose_list=[]
        end_dagger_traj = False
        first_dagger_print = False

        #get goal info
        init_transform_dict = env.return_cur_pos_info()
        goal_dict=get_goal_info(env)
        env.act_with_abs_dict(init_transform_dict)

        #================================dagger===============================
        frame_counter = 0
        task_start_time = time.time()
        #================================dagger===============================
        
        while True:
            frame_counter += 1
            
            if not use_light_key:
                rtn_dict=env.observation(random_light_dir=random_light_dir,use_prob=True)
                img_light = None
                img2_light = None
            else:
                rtn_dict=env.observation(random_light_dir=False)
                rtn_light_dict=env.observation(random_light_dir=True,use_prob=False)
                img_light = rtn_light_dict['img_1']
                img2_light = rtn_light_dict['img_2'] if 'img_2' in rtn_light_dict else None

            img = rtn_dict['img_1']
            img2 = rtn_dict['img_2'] if 'img_2' in rtn_dict else None
            im_dep = rtn_dict["img_1_depth"] if "img_1_depth" in rtn_dict else None
            im_dep2 = rtn_dict["img_2_depth"] if "img_2_depth" in rtn_dict else None

            wgT_tar=env.wgT_tar
            wgT=env.wgT
            
            # 获取专家策略的动作（总是计算，因为DAgger需要专家标签）
            expert_act_dict=get_expert_policy(wgT_tar=wgT_tar,wgT=wgT,trans_vel=trans_vel,rot_vel=rot_vel,uniform_vel=uniform_vel,dist_eps=env.dist_eps,angle_eps=env.angle_eps,motion_type=motion_type,dof=dof)
            
            vel_tr=filter_translation(expert_act_dict['vel_tr'],thres=1e-7)
            vel_rot=expert_act_dict['vel_rot'] #3dof:绕世界系 6dof:绕夹爪系
            expert_dT=expert_act_dict["dT"]
            expert_action = np.concatenate((vel_tr,vel_rot))
            
            # 默认使用专家动作
            dT = expert_dT
            action = expert_action
            
            if is_dagger_traj_iter and policy_model is not None:
                obs_dict = prepare_observation_for_policy(
                    img_size=save_img_size,
                    hdf_img_size=img_h,
                    img=img,
                    img_goal=goal_dict["img_goal"],
                    img2=img2,
                    img2_goal=goal_dict["img_goal2"] if goal_dict["img_goal2"] is not None else None,
                    img_light=img_light if use_light_key else None,
                    img_light_goal=goal_dict["img_light_goal"] if use_light_key else None,
                    img2_light=img2_light if (use_light_key and img2_light is not None) else None,
                    img2_light_goal=goal_dict["img_light_goal2"] if (use_light_key and goal_dict["img_light_goal2"] is not None) else None
                )

                policy_action = get_policy_action(policy_model, obs_dict)
                action = policy_action
                # 从策略动作构建变换矩阵dT
                dT = construct_dT_from_action(policy_action, dof=dof)
                # 打印策略动作和专家动作的差异
                # print(f"[DAgger] Policy Action: {action}, Expert Action: {expert_action}")
            
            # 保存动作（对于普通收集是实际执行的动作，对于DAgger是专家动作）
            action_list.append(action)
            if is_dagger_traj_iter:
                expert_action_list.append(expert_action)
            wgT_list.append(env.wgT)
            
            if record_pose:
                delta_pose_list.append(expert_act_dict['cur_goal_delta_pose'])

            if dof==3:
                rz = rmat2euler_rz_degree(wgT)
                rz_list.append(rz)

            #postprocess
            process_and_append_observations(img, img2, img_light, img2_light, im_dep, im_dep2, img_h, 
                             img_lst, img2_lst, img_light_list, img2_light_list, im_dep_lst, im_dep2_lst, 
                             use_light_key=use_light_key, show_images=visualize_img)
            
            #================================dagger===============================
            # DAgger策略检查物体和夹爪位置
            if is_dagger_traj_iter and frame_counter % dagger_config["check_frame_interval"] == 0:
                collision_res = compute_position_distance_sim(env.objId, env.gripId)
                distance = collision_res["min_distance"]
                contact_flag = collision_res["is_colliding"]
                # print(f"distance: {distance}")
                if  distance < min_position_threshold[0]:
                    if not first_dagger_print:
                        print(f"[DAgger] Collision detected, distance: {distance}")
                    end_dagger_traj = True
                    first_dagger_print = True
            #================================dagger===============================

            if end_dagger_traj:
                dT = construct_dT_from_action(expert_action, dof=dof)
            # 正常的移动
            env.action(dT)
            reinit_res = env.reinit()
            
            if is_dagger_traj_iter: #out of distribution
                if env.wgT[2,3] < env.wgT_tar[2,3] - 0.02: #touch ground
                    end_dagger_traj = True
                tar_mat = env.wgT_tar
                cur_mat = env.wgT
                error_dT = np.linalg.inv(tar_mat) @ cur_mat
                error_trans_xy = np.linalg.norm(tar_mat[0:2,3]-cur_mat[0:2,3])
                error_trans_z = abs(tar_mat[2,2]-cur_mat[2,2])
                error_rotvec = R.from_matrix(error_dT[0:3,0:3]).as_rotvec()
                if not end_dagger_traj and (error_trans_xy > 1.5 * env.init_horizon_trans or error_trans_z>env.init_vertical_trans or any([abs(th)>1.5*env.init_rot[i] for i,th in enumerate(error_rotvec)])):
                    end_dagger_traj = True

            if reinit_res["close_enough"] or end_dagger_traj:
                # add the obs-action pair of the last frame
                if is_dagger_traj_iter:
                    print("Final distance between gripper and object:", distance)
                    print(f"Final error:trans:{reinit_res['dist']},rot:{reinit_res['angle']}")
                    action_list.append(np.array([0,0,0]) if dof==3 else np.array([0,0,0,0,0,0]))
                    expert_action_list.append(np.array([0,0,0]) if dof==3 else np.array([0,0,0,0,0,0]))
                else:
                    action_list.append(np.array([0,0,0]) if dof==3 else np.array([0,0,0,0,0,0]))
                wgT_list.append(np.eye(4))

                process_and_append_observations(
                    img=goal_dict["img_goal"],
                    img2=goal_dict["img_goal2"], 
                    img_light=goal_dict["img_light_goal"], 
                    img2_light=goal_dict["img_light_goal2"], 
                    im_dep=goal_dict["img_dep_goal"], 
                    im_dep2=goal_dict["img_dep_goal2"], 
                    img_h=img_h, 
                    img_lst=img_lst, 
                    img2_lst=img2_lst, 
                    img_light_list=img_light_list, 
                    img2_light_list=img2_light_list, 
                    im_dep_lst=im_dep_lst, 
                    im_dep2_lst=im_dep2_lst, 
                    use_light_key=use_light_key, 
                    show_images= visualize_img
                )

                if dof==3:
                    rz_list.append(0)
                if record_pose:
                    delta_pose_list.append(np.zeros(6))
                
                #save hdf5
                epi_length=len(img_lst)
                assert epi_length==len(action_list)
                if existed_demo_num>=1:
                    add_useless_things(new_f_out=new_f_out,demo_ind=idx+existed_demo_num,epi_len=epi_length)
                else:
                    add_useless_things(new_f_out=new_f_out, demo_ind=idx, epi_len=epi_length)
                new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image', data=img_lst)

                if use_light_key:
                    new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_light', data=img_light_list)
                if len(img2_lst)!=0:
                    new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_2', data=img2_lst)
                if len(img2_light_list)!=0:
                    new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_2_light', data=img2_light_list)
                if len(im_dep_lst)!=0:
                    new_f_out.create_dataset(obs_path + '/depth_image', data=im_dep_lst)
                if len(im_dep2_lst)!=0:
                    new_f_out.create_dataset(obs_path + '/depth_image_2', data=im_dep2_lst)
                if dof==3:
                    new_f_out.create_dataset(obs_path + '/abs_rot', data=rz_list)
                if len(delta_pose_list)!=0:
                    new_f_out.create_dataset(pos_path, data=delta_pose_list)

                # 在DAgger中，保存的动作取决于是否是DAgger模式
                if is_dagger_traj_iter:
                    new_f_out.create_dataset(action_path, data=expert_action_list)
                    print("expert_action_lst-1:", expert_action_list[-1])
                else:
                    new_f_out.create_dataset(action_path, data=action_list)
                    print("action_lst-1:", action_list[-1])
                
                print("[INFO] demo_{} collected successfully.".format(idx))
                #================================dagger===============================
                if is_dagger_traj_iter:
                    print(f"[DAgger] Iteration {idx} completed successfully")
                # ================================dagger===============================

                #===============================policy training=============================

                # old iteration setting
                if is_dagger_traj_iter:
                    num_rollout_trajs += 1
                    selected_states = greedy_state_selection(wgT_list,action_list,expert_action_list,num_selected_pts,w1,w2,s_tr,s_rot,a_tr,a_rot)
                    rest_fail_pool_size += len(selected_states)
                    total_fail_pool_size += len(selected_states)
                    for state in selected_states:
                        cur_fail_pool.append(state)

                else:
                    num_expert_trajs += 1
                    rest_fail_pool_size -= 1

                should_train = (total_fail_pool_size >= n_fail_pool_size and rest_fail_pool_size == 0 and not is_dagger_traj_iter) or idx == n_base_expert_trajs -1

                if should_train and policy_model is not None:
                    print(f"[DAgger] Training policy model at iteration {idx} with {train_epochs} epochs")

                    # 训练模型
                    new_f_out.close()
                    is_hdf_open = False
                    data_cfg = model_config["dataset"]
                    data_cfg["hdf5_path"] = dataset_dir
                    train_cfg = model_config["training"]
                    # 保存新模型
                    model_path = os.path.join(base_dir, 'AlignAnything', current_date, 'models')
                    ensure_dir(model_path)
                    
                    # 构建过滤键
                    filter_key = "train"
                    f_tmp = h5py.File(dataset_dir, 'r')
                    all_demos = sorted(list(f_tmp['data'].keys()))
                    f_tmp.close()
                    create_hdf5_filter_key(hdf5_path=dataset_dir, demo_keys=all_demos, key_name=filter_key,return_length=False)

                    #训练
                    policy_model = train_policy(
                        img_size=model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"],
                        num_train_steps=model_config["training"]["num_train_steps_per_epoch"],
                        model=policy_model,
                        optimizer=optimizer,
                        criterion=criterion,
                        num_epochs=train_epochs,
                        batch_size=model_config["training"]["batch_size"],
                        train_cfg=train_cfg,
                        data_cfg=data_cfg,
                        save_path=model_path,
                        episode_idx=idx,
                        filter_by_attribute=filter_key,
                    )
                #===============================policy training===============================

                #switch to new iteration
                if idx < n_base_expert_trajs - 1:
                    is_dagger_traj_iter = False
                elif idx == n_base_expert_trajs -1:
                    is_dagger_traj_iter = True
                else:
                    if is_dagger_traj_iter:
                        if total_fail_pool_size >= n_fail_pool_size:
                            is_dagger_traj_iter = False
                            print(f"[DAgger] Policy switched form dagger to expert in iteration {idx}.")
                    if should_train and idx >= n_base_expert_trajs:
                        is_dagger_traj_iter = True
                        total_fail_pool_size = 0  #rest_fail_pool_size is set 0 above
                        assert len(cur_fail_pool) == 0
                        print(f"[DAgger] Policy switched from expert to dagger in iteration {idx}.")

                # new iteration setting
                pos = None
                if not is_dagger_traj_iter:
                    pos = cur_fail_pool.popleft()
                env.init(pos)
                break

    # add_env_meta(new_f_out,additional_itms={"pose_and_orientations":pose_and_orientations})
    if not is_hdf_open:
        new_f_out = h5py.File(dataset_dir, "r+")
        is_hdf_open = True
    add_config(new_f_out, config)
    new_f_out.close()
    compute_num_samples(dataset_dir)
    split_train_val_from_hdf5(dataset_dir, val_ratio=0.1)








