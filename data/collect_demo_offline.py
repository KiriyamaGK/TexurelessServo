import os
from utils.input_process import input_dict_preprocess
from utils.hdf5 import add_useless_things, split_train_val_from_hdf5, add_env_meta, compute_num_samples, add_config
import h5py
import numpy as np
import json
from sim.environment import Environment
from utils.paths import return_disc_route
from utils.transform import rmat2euler_rz_degree
from sim.perception import CameraIntrinsic
import time
import cv2
from utils.input_process import clip_image,conditioned_clip_and_resize
from utils.policy import get_expert_policy
from data.process_hdf5 import _disturb_abs_rot,_portion_last_episode,_add_end_episode,_add_medium_episode,insert_imgs
from utils.dagger import compute_position_distance, load_policy_model, get_policy_action, aggregate_dataset, train_policy
import torch
from networks.helpers import get_loss_fn, get_optimizer_cls, get_network_cls
from torch.utils.data import DataLoader
from dataset.dataset import dataset_factory

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
        rtn_dict = env.observation(random_light_dir=random_light_dir, use_prob=True)  # TODO:记得修改
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

def prepare_observation_for_policy(img_size, hdf_img_size,img, img_goal, img2=None, img2_goal=None):
    """
    准备输入给策略模型的观察数据
    """
    img = conditioned_clip_and_resize(img, img_size, img_size, hdf_img_size)
    img_goal = conditioned_clip_and_resize(img_goal, img_size, img_size, hdf_img_size)
    if img2 is not None:
        img2 = conditioned_clip_and_resize(img2, img_size, img_size, hdf_img_size)
        img2_goal = conditioned_clip_and_resize(img2_goal, img_size, img_size, hdf_img_size)  
    obs_dict = {
        "robot0_eye_in_hand_image": img,
        "robot0_eye_in_hand_image_goal": img_goal
    }
    if img2 is not None:
        obs_dict["robot0_eye_in_hand_image_2"] = img2
        obs_dict["robot0_eye_in_hand_image_2_goal"] = img2_goal
    
    obs_dict=input_dict_preprocess(obs_dict,rollout=True)
        
    return obs_dict

def setup_policy_model(config_path="../configs/train_mlp.json", checkpoint_path=None):
    """
    设置策略模型
    """
    # 加载配置
    with open(config_path, "r") as j:
        config = json.load(j)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置模型
    model_cls, need_init_params = get_network_cls(config["algorithm"]["policy"]["name"])
    if need_init_params:
        model = model_cls(
            input_low_dim=config["dataset"]["input_low_dim"],
            output_dim=config["dataset"]["output_dim"],
            obs_keys=config["dataset"]["specific_obs_keys"],
            batch_size=config["training"]["batch_size"],
            seq_length=config["dataset"]["seq_length"],
            training=False,  # 推理模式
            **config["algorithm"]["policy"]["params"]
        )
    else:
        model = model_cls()
    
    # 加载预训练权重
    if checkpoint_path:
        model = load_policy_model(checkpoint_path, model, device)
    
    # 设置优化器和损失函数，用于在线学习
    optimizer = get_optimizer_cls(config["algorithm"]["optimizer"]["name"])(
        model.parameters(), **config["algorithm"]["optimizer"]["params"]
    )
    
    criterion = get_loss_fn(
        config["algorithm"]["loss"]["name"],
        config["algorithm"]["loss"]["weight"],
        config["dataset"]["seq_length"],
        config["dataset"]["output_dim"]
    )
    
    return model, optimizer, criterion, config

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

if __name__ == '__main__':
    img_w=220
    img_h=220
    visualize_img = False

    config_dir= "../configs/demo_collection.json"

    with open(config_dir, "r") as j:
        config = json.load(j)

    #overall setting
    # base_dir = return_disc_route("One Touch")
    base_dir = config['overall_setting']["dataset_base_dir"]
    objs_descriptor=config['overall_setting']['objs_descriptor']
    current_date=config['overall_setting']['file_name']
    demo_total_num = config['overall_setting']['demo_total_num']
    replace_existed_hdf5=config["overall_setting"]["replace_existed_hdf5"] #TODO:remember to use

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
    
    # start_episode: 整数，表示从第几个episode开始使用DAgger策略。在此之前的episode会使用普通的行为克隆方式收集数据。
    # frequency: 整数，表示每多少个非DAgger的episode后执行一次DAgger收集过程。例如，值为5表示每5个正常episode后执行一次DAgger。
    # episodes_per_dagger: 整数，表示每次触发DAgger时连续执行的DAgger episode数量。例如，值为10表示每次触发DAgger时会连续收集10个使用策略模型的episodes。
    
    # position_threshold: 浮点数，表示在DAgger模式下，当夹爪与目标物体之间的位置距离小于此值时，当前episode会提前结束。单位是米。
    # check_frame_interval: 整数，表示在DAgger模式下每隔多少帧检查一次夹爪与物体的距离。
    
    # train_frequency: 整数，表示每完成多少个DAgger episodes后进行一次模型训练。
    # train_epochs: 整数，表示每次训练模型时执行的轮数。
    
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
    #================================dagger===============================

    trans_vel=config["demo_collection"]["velocity"]['trans_vel'] #m
    rot_vel=config["demo_collection"]["velocity"]['rot_vel']    #deg
    uniform_vel=config["demo_collection"]["velocity"]['uniform_vel']

    init_horizon_trans=config["demo_collection"]["init"]['init_horizon_trans']["value"]
    init_vertical_trans = config["demo_collection"]["init"]['init_vertical_trans']["value"]
    init_rot=config["demo_collection"]["init"]['init_rot']["value"]
    use_high_proportion_x=config["demo_collection"]["init"]['init_horizon_trans']["use_high_proportion_x"]
    use_max_rot = config["demo_collection"]["init"]['init_rot']['use_max_rot']
    use_max_trans=config["demo_collection"]["init"]['init_horizon_trans']["use_max_trans"]
    using_minus_vertical = config["demo_collection"]["init"]['init_vertical_trans']["using_minus"]
    pose_and_orientations=config["demo_collection"]["init"]['pose_and_orientations']
    init_transform_frame=config["demo_collection"]["init"]['init_transform_frame'] if 'init_transform_frame' in config["demo_collection"]["init"] else "grip"

    angle_eps =config["demo_collection"]["stop_policy"]['angle_eps']
    dist_eps = config["demo_collection"]["stop_policy"]['dist_eps']

    #post process
    disturb_abs_rot = config['post_process']['disturb_abs_rot']
    portion_last_episode = config['post_process']['portion_last_episode']
    add_end_episode = config['post_process']['add_end_episode']
    add_medium_episode = config['post_process']['add_medium_episode']
    assert (not portion_last_episode["utilized"]) or (not add_end_episode["utilized"])

    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    env=Environment(camera_config=camera_intrinsic,objs_descriptor=objs_descriptor,use_max_rot=use_max_rot,use_max_trans=use_max_trans,init_horizon_trans=init_horizon_trans,init_vertical_trans=init_vertical_trans,using_minus_vertical=using_minus_vertical,use_high_proportion_x=use_high_proportion_x,init_rot=init_rot,init_transform_frame=init_transform_frame,dof=dof,angle_eps=angle_eps,dist_eps=dist_eps,depth_info=depth_info,pose_and_orientations=pose_and_orientations,_is_collect=True,conditioned_sampling=conditioned_sampling,trans_vel=trans_vel["value"],rot_vel=rot_vel["value"],third_view_camera=third_view_camera,uniform_evaluation={"utilized":False})
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
    # DAgger相关计数器
    dagger_episodes_done = 0
    non_dagger_episodes_count = 0
    current_dagger_episodes = 0
    end_episode = False
    
    # 用于存储DAgger收集的新数据
    dagger_new_data = {
        "obs": [],
        "actions": [],
        "delta_pos_curgoal": []
    }
    #================================dagger===============================
    
    for idx in range(demo_total_num):
        print("====================start collecting demo_{} ====================".format(idx))
        
        #================================dagger===============================
        is_dagger_episode = False
        if use_dagger and idx >= dagger_config["start_episode"]:
            if non_dagger_episodes_count >= dagger_config["frequency"]:
                is_dagger_episode = True
                current_dagger_episodes += 1
                if current_dagger_episodes >= dagger_config["episodes_per_dagger"]:
                    current_dagger_episodes = 0
                    non_dagger_episodes_count = 0
            else:
                non_dagger_episodes_count += 1
            if not is_hdf_open:
                new_f_out = h5py.File(dataset_dir, "r+")
                is_hdf_open = True
                
        if is_dagger_episode:
            print("[INFO] Using DAgger strategy for episode {}".format(idx))
        #================================dagger===============================

        if idx==0:
            if 'data' in new_f_out and not replace_existed_hdf5:
                existed_demo_num=len(new_f_out["data"])

        if existed_demo_num>=1:#根据existed_demo_num和dagger_episodes_done的数量整体偏移
            obs_path = 'data/demo_{}/obs'.format(idx+existed_demo_num)
            action_path = 'data/demo_{}/actions'.format(idx+existed_demo_num)
            pos_path = 'data/demo_{}/delta_pos_curgoal'.format(idx + existed_demo_num)
        else:
            obs_path = 'data/demo_{}/obs'.format(idx)
            action_path = 'data/demo_{}/actions'.format(idx)
            pos_path = 'data/demo_{}/delta_pos_curgoal'.format(idx)

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
        end_episode = False

        #get goal info
        init_transform_dict = env.return_cur_pos_info()
        goal_dict=get_goal_info(env)
        env.act_with_abs_dict(init_transform_dict)

        #================================dagger===============================
        frame_counter = 0
        #================================dagger===============================
        
        while True:
            frame_counter += 1
            
            if not use_light_key:
                rtn_dict=env.observation(random_light_dir=random_light_dir,use_prob=True) #TODO:记得修改
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
            
            if is_dagger_episode and policy_model is not None:
                obs_dict = prepare_observation_for_policy(img_size = save_img_size, hdf_img_size = img_h,
                    img=img, img_goal=goal_dict["img_goal"], 
                    img2=img2, img2_goal=goal_dict["img_goal2"] if goal_dict["img_goal2"] is not None else None
                )

                policy_action = get_policy_action(policy_model, obs_dict)
                action = policy_action
                
                # 从策略动作构建变换矩阵dT
                if dof == 3:
                    dT = np.eye(4)
                    dT[0:2, 3] = policy_action[:2]  # 前两个元素是平移
                    dT[0:3, 0:3] = rotation_matrix_z(policy_action[2] / 180 * np.pi)  # 最后一个元素是旋转
                else:
                    dT = np.eye(4)
                    dT[0:3, 3] = policy_action[:3]  # 前三个元素是平移
                    rot_vec = policy_action[3:] / 180 * np.pi  # 后三个元素是旋转向量
                    from scipy.spatial.transform import Rotation as R
                    dT[0:3, 0:3] = R.from_rotvec(rot_vec).as_matrix()
                
                # 打印策略动作和专家动作的差异
                # print(f"[DAgger] Policy Action: {action}, Expert Action: {expert_action}")
            
            # 保存动作（对于普通收集是实际执行的动作，对于DAgger是专家动作）
            action_list.append(action)
            if is_dagger_episode:
                expert_action_list.append(expert_action)
            
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
            if is_dagger_episode and frame_counter % dagger_config["check_frame_interval"] == 0:
                collision_res = compute_position_distance(env.objId, env.gripId)
                distance = collision_res["min_distance"]
                contact_flag = collision_res["is_colliding"]
                print(f"distance: {distance}")
                if distance < dagger_config["position_threshold"][0] or distance > dagger_config["position_threshold"][1] or contact_flag:
                    print(f"[DAgger] Position threshold reached at frame {frame_counter}, distance: {distance}")
                    end_episode = True
            #================================dagger===============================
            
            # 正常的移动
            env.action(dT)
            reinit_res = env.reinit()
            if reinit_res or end_episode:
                if not reinit_res:
                    env.init()
                # add the obs-action pair of the last frame
                if is_dagger_episode:
                    print("Final distance between gripper and object:", distance)
                    action_list.append(np.array([0,0,0]) if dof==3 else np.array([0,0,0,0,0,0]))
                    expert_action_list.append(np.array([0,0,0]) if dof==3 else np.array([0,0,0,0,0,0]))
                else:
                    action_list.append(np.array([0,0,0]) if dof==3 else np.array([0,0,0,0,0,0]))

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


                #post process
                if disturb_abs_rot["utilized"]:
                    rz_list,_=_disturb_abs_rot(rz_list,action_list)

                if portion_last_episode["utilized"]:
                    action_list,_=_portion_last_episode(action_list,portion_last_episode["portion_last_num"],dof)
                    if is_dagger_episode:
                        expert_action_list,_=_portion_last_episode(expert_action_list,portion_last_episode["portion_last_num"],dof)

                if add_end_episode["utilized"]:
                    pick_id=len(img_lst)-1
                    insert_id=len(img_lst)-1
                    add_num=add_end_episode["add_num"]

                    rz_list, action_list,delta_pose_list=_add_end_episode(add_num=add_num,disturb_abs_rot=disturb_abs_rot["utilized"],abs_rot_list=rz_list,act_lst=action_list,pose_list=delta_pose_list)
                    if is_dagger_episode:
                        # 对专家动作列表也进行相同的处理
                        _, expert_action_list, _ = _add_end_episode(add_num=add_num,disturb_abs_rot=disturb_abs_rot["utilized"],abs_rot_list=rz_list,act_lst=expert_action_list,pose_list=delta_pose_list)
                    
                    img_lst=insert_imgs(img_lst,pick_id,insert_id,add_num)
                    if len(img_light_list) != 0:
                        img_light_list=insert_imgs(img_light_list,pick_id,insert_id,add_num)
                    if len(img2_lst) != 0:
                        img2_lst=insert_imgs(img2_lst,pick_id,insert_id,add_num)
                    if len(img2_light_list) != 0:
                        img2_light_list=insert_imgs(img2_light_list,pick_id,insert_id,add_num)
                    if len(im_dep_lst) != 0:
                        im_dep_lst=insert_imgs(im_dep_lst,pick_id,insert_id,add_num)
                    if len(im_dep2_lst) != 0:
                        im_dep2_lst=insert_imgs(im_dep2_lst,pick_id,insert_id,add_num)

                if add_medium_episode["utilized"]:
                    action_list, rz_list, delta_pose_list,need_add_medium, trans_id, rot_id=_add_medium_episode(act_lst=action_list, abs_rot_list=rz_list, ac_dim=dof,add_num=add_medium_episode["add_num"],pose_list=delta_pose_list)
                    if is_dagger_episode and need_add_medium:
                        # 对专家动作列表也进行相同的处理
                        expert_action_list, _, _, _, _, _ = _add_medium_episode(act_lst=expert_action_list, abs_rot_list=rz_list, ac_dim=dof, add_num=add_medium_episode["add_num"], pose_list=delta_pose_list)
                    
                    if need_add_medium:
                        print("+++++++++++++++++++++++++++++++++++++++++")
                        pick_id = trans_id + 1
                        insert_id = rot_id
                        add_num = add_medium_episode["add_num"]

                        img_lst = insert_imgs(img_lst, pick_id, insert_id, add_num)
                        if len(img_light_list) != 0:
                            img_light_list = insert_imgs(img_light_list, pick_id, insert_id, add_num)
                        if len(img2_lst) != 0:
                            img2_lst = insert_imgs(img2_lst, pick_id, insert_id, add_num)
                        if len(img2_light_list) != 0:
                            img2_light_list = insert_imgs(img2_light_list, pick_id, insert_id, add_num)
                        if len(im_dep_lst) != 0:
                            im_dep_lst = insert_imgs(im_dep_lst, pick_id, insert_id, add_num)
                        if len(im_dep2_lst) != 0:
                            im_dep2_lst = insert_imgs(im_dep2_lst, pick_id, insert_id, add_num)
                
                # 如果是DAgger模式，保存到临时数据中
                if is_dagger_episode:
                    # 创建该episode的数据字典
                    episode_data = {
                        "obs": {
                            "robot0_eye_in_hand_image": np.array(img_lst)
                        },
                        "actions": np.array(expert_action_list),  # 使用专家动作作为标签
                    }
                    
                    if len(img2_lst)!=0:
                        episode_data["obs"]["robot0_eye_in_hand_image_2"] = np.array(img2_lst)
                    if use_light_key:
                        episode_data["obs"]["robot0_eye_in_hand_image_light"] = np.array(img_light_list)
                    if len(img2_light_list)!=0:
                        episode_data["obs"]["robot0_eye_in_hand_image_2_light"] = np.array(img2_light_list)
                    if len(im_dep_lst)!=0:
                        episode_data["obs"]["depth_image"] = np.array(im_dep_lst)
                    if len(im_dep2_lst)!=0:
                        episode_data["obs"]["depth_image_2"] = np.array(im_dep2_lst)
                    if len(delta_pose_list)!=0:
                        episode_data["delta_pos_curgoal"] = np.array(delta_pose_list)
                    
                    # 添加到DAgger数据集
                    dagger_new_data["obs"].append(episode_data["obs"])
                    dagger_new_data["actions"].append(episode_data["actions"])
                    if "delta_pos_curgoal" in episode_data:
                        dagger_new_data["delta_pos_curgoal"].append(episode_data["delta_pos_curgoal"])
                
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
                if is_dagger_episode:
                    new_f_out.create_dataset(action_path, data=expert_action_list)
                    print("expert_action_lst-1:", expert_action_list[-1])
                else:
                    new_f_out.create_dataset(action_path, data=action_list)
                    print("action_lst-1:", action_list[-1])
                
                print("[INFO] demo_{} collected successfully.".format(idx))
                #================================dagger===============================
                if is_dagger_episode:
                    dagger_episodes_done += 1
                    print(f"[DAgger] Episode completed. Total DAgger episodes: {dagger_episodes_done}")
                    
                    # 每完成一定数量的DAgger episodes后，进行一次在线训练
                    if dagger_episodes_done % dagger_config.get("train_frequency", 5) == 0 and policy_model is not None:
                        print(f"[DAgger] Training policy model after {dagger_episodes_done} episodes")
                        
                        # 聚合数据集并训练模型
                        # aggregate_dataset(dagger_new_data, dataset_dir)
                        # aggregate_dataset(dagger_new_data, new_f_out)
                        
                        # 训练模型
                        new_f_out.close()
                        is_hdf_open = False
                        tr_model_cfg = model_config["dataset"]
                        # tr_model_cfg["hdf5_file"] = new_f_out
                        tr_model_cfg["hdf5_path"] = dataset_dir

                        policy_model = train_policy( #这个函数的img_size填写的是对的
                            img_size = model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"],
                            num_train_steps = model_config["training"]["num_train_steps_per_epoch"],
                            model=policy_model,
                            optimizer=optimizer,
                            criterion=criterion,
                            num_epochs=dagger_config.get("train_epochs", 5),
                            batch_size=model_config["training"]["batch_size"],
                            config = tr_model_cfg
                        )
                        
                        # 保存新模型
                        model_path = os.path.join(base_dir, 'AlignAnything', current_date, 'models')
                        ensure_dir(model_path)
                        model_file = os.path.join(model_path, f'dagger_episode_{dagger_episodes_done}.pth')
                        torch.save(policy_model.state_dict(), model_file)
                        print(f"[DAgger] Model saved to {model_file}")
                        
                        # 清空新数据缓冲区
                        dagger_new_data = {
                            "obs": [],
                            "actions": [],
                            "delta_pos_curgoal": []
                        }
                #================================dagger===============================
                break
    # add_env_meta(new_f_out,additional_itms={"pose_and_orientations":pose_and_orientations})
    if not is_hdf_open:
        new_f_out = h5py.File(dataset_dir, "r+")
        is_hdf_open = True
    add_config(new_f_out, config)
    new_f_out.close()
    compute_num_samples(dataset_dir)
    split_train_val_from_hdf5(dataset_dir, val_ratio=0.1)








