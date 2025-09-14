import os
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
from data.process_hdf5 import _disturb_abs_rot,_portion_last_episode,_add_end_episode,_add_medium_episode,insert_imgs
from utils.dagger import compute_position_distance_sim, get_policy_action, train_policy, setup_policy_model, prepare_observation_for_policy
from utils.dagger_params import is_in_dagger_episode, should_train_policy, print_dagger_status

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
    
    # min_position_threshold: list，表示在DAgger模式下，当夹爪与目标物体之间的最小位置距离不在此范围时，当前episode会提前结束。单位是米。
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

        min_position_threshold = dagger_config["task_termination"]["min_position_threshold"]
        pose_error_threshold = dagger_config["task_termination"]["pose_error_threshold"]  # m,deg,sec
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
    # DAgger相关变量
    end_episode = False
    first_in_error = False
    #================================dagger===============================
    
    for idx in range(demo_total_num):
        if not is_hdf_open:
            new_f_out = h5py.File(dataset_dir, "r+")
            is_hdf_open = True

        print("====================start collecting demo_{} ====================".format(idx))
        
        #================================dagger===============================
        first_in_error = False
        is_dagger_episode = False
        should_train = False
        train_epochs = 0
        
        if use_dagger:
            # 判断是否应该使用DAgger策略
            is_dagger_episode = is_in_dagger_episode(idx, dagger_config)
            # 判断是否应该训练策略，并获得训练时的DAgger数据比例
            should_train, train_epochs, dagger_proportion = should_train_policy(idx, dagger_config)
            
            # 打印DAgger状态
            print_dagger_status(idx, is_dagger_episode, should_train, train_epochs)
                
        if is_dagger_episode:
            print("[INFO] Using DAgger strategy for episode {}".format(idx))
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
                collision_res = compute_position_distance_sim(env.objId, env.gripId)
                distance = collision_res["min_distance"]
                contact_flag = collision_res["is_colliding"]
                print(f"distance: {distance}")
                
                if distance < min_position_threshold[0] or distance > min_position_threshold[1] or contact_flag:
                    print(f"[DAgger] Position minimum threshold reached at frame {frame_counter}, distance: {distance}, threshold: {min_position_threshold}")
                    end_episode = True
            #================================dagger===============================
            
            # 正常的移动
            env.action(dT)
            reinit_res = env.reinit()
            
            if is_dagger_episode:
                tr = reinit_res["dist"]
                rot = reinit_res["angle"]
                print(f"------------error:trans:{tr},rot:{rot}-----------------")
                if tr > pose_error_threshold["trans"] or rot > pose_error_threshold["rot"]:
                    if first_in_error:
                        dt = time.time() - error_timer
                        if dt > pose_error_threshold["time"]:
                            print(
                                f"[DAgger] Pose error reached at frame {frame_counter}, trans: {tr}, rot: {rot} for over {dt} seconds.")
                            end_episode = True
                    else:
                        error_timer = time.time()
                        first_in_error = True
                else:
                    first_in_error = False


            if reinit_res["close_enough"] or end_episode:
                if not reinit_res["close_enough"]:
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
                    print(f"[DAgger] Episode {idx} completed successfully")
                # ================================dagger===============================

                #===============================policy training=============================
                # 使用之前计算的训练状态
                if should_train and policy_model is not None:
                    print(f"[DAgger] Training policy model at episode {idx} with {train_epochs} epochs")

                    # 训练模型
                    new_f_out.close()
                    is_hdf_open = False
                    data_cfg = model_config["dataset"]
                    data_cfg["hdf5_path"] = dataset_dir
                    train_cfg = model_config["training"]
                    # 保存新模型
                    model_path = os.path.join(base_dir, 'AlignAnything', current_date, 'models')
                    ensure_dir(model_path)
                    
                    # 根据 dagger_proportion 构建过滤键
                    filter_key = None
                    if 'dagger' in config['demo_collection'] and dagger_proportion is not None:
                        # 构建 dagger/non-dagger 划分
                        f_tmp = h5py.File(dataset_dir, 'r')
                        all_demos = sorted(list(f_tmp['data'].keys()))
                        f_tmp.close()
                        dagger_ranges = dagger_config.get('dagger_episodes', {}).get('use_type', [])
                        dagger_set = set()
                        for s, e in dagger_ranges:
                            for ep in range(s, e + 1):
                                demo_id = f"demo_{ep+existed_demo_num}"  #crucial improvement
                                if demo_id in all_demos:
                                    dagger_set.add(demo_id)
                        dagger_demos = [d for d in all_demos if d in dagger_set]
                        non_dagger_demos = [d for d in all_demos if d not in dagger_set]

                        num_total = len(all_demos)
                        num_dagger_target = int(round(dagger_proportion * num_total))
                        num_non_dagger_target = max(0, num_total - num_dagger_target)

                        rng = np.random.default_rng(seed=idx)
                        chosen_dagger = rng.choice(dagger_demos, size=min(len(dagger_demos), num_dagger_target), replace=False).tolist()
                        chosen_non_dagger = rng.choice(non_dagger_demos, size=min(len(non_dagger_demos), num_non_dagger_target), replace=False).tolist()
                        mixed = sorted(chosen_dagger + chosen_non_dagger)
                        tmp_key = f"dagger_mix_ep_{idx}"
                        create_hdf5_filter_key(hdf5_path=dataset_dir, demo_keys=mixed, key_name=tmp_key,return_length=False)
                        filter_key = tmp_key
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
                        filter_by_attribute=filter_key
                    )


                #===============================policy training===============================
                break
    # add_env_meta(new_f_out,additional_itms={"pose_and_orientations":pose_and_orientations})
    if not is_hdf_open:
        new_f_out = h5py.File(dataset_dir, "r+")
        is_hdf_open = True
    add_config(new_f_out, config)
    new_f_out.close()
    compute_num_samples(dataset_dir)
    split_train_val_from_hdf5(dataset_dir, val_ratio=0.1)








