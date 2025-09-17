import time
import cv2
import numpy as np
import os
import random
import threading
import json
import torch
from utils.paths import PROJECT_ROOT_DIR
from utils.paths import return_disc_route
from utils.file import ensure_dir
from utils.policy import get_expert_policy
from utils.hdf5 import add_useless_things, split_train_val_from_hdf5, add_env_meta, compute_num_samples, add_config, create_hdf5_filter_key
import h5py
from utils.input_process import clip_image
from real.environment import Environment
from real.teleop_with_joystick import Teleop
from data.process_hdf5 import _disturb_abs_rot,_portion_last_episode,_add_end_episode,_add_medium_episode,insert_imgs
from utils.transform import rotation_matrix_z, rmat2euler_rz_degree,construct_dT_from_action
from utils.dagger_params import is_in_dagger_episode, should_train_policy, print_dagger_status
from utils.dagger import  get_policy_action, aggregate_dataset, train_policy, setup_policy_model, prepare_observation_for_policy
from real.collision_detection.sdf_collision import CollisionDetector
from utils.augmentation import AugmentationModule

import atexit
from pynput import keyboard

##TODO:1.modify mesh loading
def filter_translation(input,thres):
    assert thres>0
    input=np.array(input)
    return np.where(np.abs(input) < thres, 0, input)

def get_goal_info(env):
    env.act_to_goal()
    rtn_dict = env.observation()
    img = rtn_dict['img_1']
    img2 = rtn_dict['img_2'] if 'img_2' in rtn_dict else None

    return {"img_goal": img, "img_goal2": img2}

def filter_pos(pos):
    for i in range(6):
        if abs(pos[i])<1e-10:
            pos[i]=0
    return pos


def cleanup():
    cam.release()
    add_env_meta(new_f_out)
    add_config(new_f_out, config)
    new_f_out.close()
    compute_num_samples(dataset_dir)
    split_train_val_from_hdf5(dataset_dir, val_ratio=0.1)

##TODO : Python的atexit注册的函数会在主线程退出且仅剩守护线程时触发,因此必须要让其他线程设置为守护线程（daemon = True）
atexit.register(cleanup)

global_start = False  # 全局变量


def _on_key_press(key):
    global global_start  # 声明修改全局变量
    try:
        if key.char == 's':
            print("============teleosperation started,press F for finish==========")
            global_start = True
            time.sleep(0.1)

    except AttributeError:
        pass

def teleop_and_pic(img_gt_1, img_gt_2,img_size):
    global global_start  # 声明使用全局变量
    global_start = False  # 重置状态

    Teleop_ins = Teleop(robot_ins, trans_coeff=0.2, rot_coeff=0.1, use_rxry=True, use_z=True, use_camera=False,
                        ctrl_freq=100, listen_finish=True)

    print("============teleoperation process,print S for start============")
    keyboard_listener = keyboard.Listener(on_press=_on_key_press)
    keyboard_listener.start()

    while not global_start:
        pass
    print("started")
    keyboard_listener.stop()

    teleop_thread = threading.Thread(target=Teleop_ins.operation, daemon=True)
    teleop_thread.start()
    while not Teleop_ins.stop_teleop:
        rtn_dict = env.observation()
        img = rtn_dict['img_1']
        img2 = rtn_dict['img_2'] if 'img_2' in rtn_dict else None

        # postprocess
        img = clip_image(img, img_size, keep_right=True)
        new_img=(0.5*img+0.5*img_gt_1).astype(np.uint8)
        if img2 is not None:
            img2 = clip_image(img2, img_size, keep_right=True)
            new_img2 = (0.5 * img2 + 0.5 * img_gt_2).astype(np.uint8)
            combined_img = np.hstack((new_img, new_img2))

        else:
            combined_img = new_img
        cv2.imshow("Combined Image", combined_img)
        cv2.waitKey(1)

    time.sleep(0.1)
    teleop_thread.join()





if __name__=='__main__':
    ##TODO:数据采集和测试阶段一共算了四种类型的dT，其中env.sample_init_pos算了g_tar_gT；get_action算了g_gtar(和缩短后的dT)；need_init和need_init_eval在compute_error的时候各自算了一次（见env相关函数）
    # ===================================manually_set_info===================================
    initial_teleop = False
    init_pos = np.array(
        [-510.449,-106.808,147.588,-179.2,-0.534,-162.46])
    goal_img_base_dir = "/media/kiriyamagk/One Touch/AlignAnything_real/25.06.22/hdf5/goal_images"
    goal_idx=1999
    origin_color_type = "bgr"
    # ===================================manually_set_info===================================


    config_dir = "../configs/demo_collection_real.json"
    with open(config_dir, "r") as j:
        config = json.load(j)

    env=Environment(robot_address=config["hardware"]["robot_address"],**config["demo_collection"]["env"],**config["hardware"]["camera"])

    cam=env.camera
    robot_ins=env.robot_ins

    if not initial_teleop:
        # init_pos = robot_ins.get_gripper_TCP_pose()
        in_desire_pt = init_pos
        env.robot_ins.move_cart(filter_pos(in_desire_pt),tool=1, user=0, vel=40)
    else:
        img_1_gt = cv2.imread(os.path.join(goal_img_base_dir,"img1", f"{goal_idx}.png"))
        img_2_gt = cv2.imread(os.path.join(goal_img_base_dir,"img2", f"{goal_idx}.png"))  # /255
        teleop_and_pic(img_1_gt, img_2_gt,config["demo_collection"]["img"]["size"])

    env.set_target_coordinate(use_cur=True)
    env.init()

    # overall setting
    base_dir = return_disc_route("One Touch")
    desire_pt_change_cycle=config["overall_setting"]["desire_pt_change_cycle"]
    current_date = config['overall_setting']['file_name']
    demo_total_num = config['overall_setting']['demo_total_num']
    replace_existed_hdf5 = config["overall_setting"]["replace_existed_hdf5"]  # TODO:remember to use
    delete_last_demo = config["overall_setting"]["delete_last_demo"]


    #velocity
    trans_vel = config["demo_collection"]["env"]["velocity"]['trans_vel']  # mm
    rot_vel = config["demo_collection"]["env"]["velocity"]['rot_vel']  # deg
    uniform_vel = config["demo_collection"]["env"]["velocity"]['uniform_vel']

    #demo_collection:
      ##收集数据频率
    data_collect_freq = config["demo_collection"]["data_collect_freq"]
    ctrl_freq = config["demo_collection"]["ctrl_freq"]

      ##img相关
    img_save_type = config["demo_collection"]["img"]["save_type"]
    assert img_save_type in ["rgb", "bgr"]
    img_size = config["demo_collection"]["img"]["size"]
    use_augmentation = config["demo_collection"]["img"]["augmentation"]["utilized"]
    pretrained_model_pth = config["demo_collection"]["img"]["augmentation"]["pretrained_model_pth"]
    scale_range_min = config["demo_collection"]["img"]["augmentation"]["scale_range_min"]
    scale_range_max = config["demo_collection"]["img"]["augmentation"]["scale_range_max"]
    offset_range_min = config["demo_collection"]["img"]["augmentation"]["offset_range_min"]
    offset_range_max = config["demo_collection"]["img"]["augmentation"]["offset_range_max"]
    noise_std = config["demo_collection"]["img"]["augmentation"]["noise_std"]

    if use_augmentation:
        augmentation_module = AugmentationModule(
            pretrained_model_pth=pretrained_model_pth,
            scale_range_min=scale_range_min,
            scale_range_max=scale_range_max,
            offset_range_min=offset_range_min,
            offset_range_max=offset_range_max,
            noise_std=noise_std,
        )

      ##record pose
    record_pose = config["demo_collection"]['record_pose']

      ##post process
    disturb_abs_rot = config["demo_collection"]['post_process']['disturb_abs_rot']
    portion_last_episode = config["demo_collection"]['post_process']['portion_last_episode']
    add_end_episode = config["demo_collection"]['post_process']['add_end_episode']
    add_medium_episode = config["demo_collection"]['post_process']['add_medium_episode']
    assert (not portion_last_episode["utilized"]) or (not add_end_episode["utilized"])

    database_dir = os.path.join(base_dir, 'AlignAnything_real', current_date, 'hdf5')
    ensure_dir(database_dir)
    dataset_dir=os.path.join(database_dir, 'mimic.hdf5')

    if replace_existed_hdf5:
        new_f_out = h5py.File(dataset_dir, "w")
    else:
        if os.path.exists(dataset_dir):
            new_f_out = h5py.File(dataset_dir, "r+")
        else:
            new_f_out = h5py.File(dataset_dir, "w")

    existed_demo_num=0
    
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
    # check_freq: 检查频率
    
    # train_frequency: 整数，表示每完成多少个DAgger episodes后进行一次模型训练。
    # train_epochs: 整数，表示每次训练模型时执行的轮数。
    
    # 如果使用DAgger，设置策略模型
    policy_model = None
    optimizer = None
    criterion = None
    model_config = None
    if use_dagger:
        policy_model, optimizer, criterion, model_config = setup_policy_model(
            config_path="../configs/train_mlp.json",
            checkpoint_path=dagger_config.get("model_path", None)
        )
        # 确保模型处于评估模式
        policy_model.eval()
        save_img_size = model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"]

        min_position_threshold = dagger_config["task_termination"]["min_position_threshold"]
        pose_error_threshold = dagger_config["task_termination"]["pose_error_threshold"]  # m,deg,sec
        collision_check_freq = dagger_config["check_freq"]

        gripper_path = os.path.join(PROJECT_ROOT_DIR, "meshes/zhixing/crt_ctag2f120.urdf")
        object_path = os.path.join(PROJECT_ROOT_DIR, "meshes/classical_part.STL")
        cali_T = np.eye(4)
        cali_T[0, 0] *= -1
        cali_T[2, 2] *= -1
        cali_T[2, 3] = 0.06 #todo:remember to calibrate
        collision_detector = CollisionDetector(gripper_path,object_path,scalar_1=1.0,scalar_2=0.001,use_convex_hull_1=False,use_convex_hull_2=False,cali_T = cali_T)
    #================================dagger===============================

    #================================dagger===============================
    # DAgger相关变量
    end_episode = False
    first_in_error = False
    #================================dagger===============================

    for uu in range(demo_total_num):
        if not is_hdf_open:
            new_f_out = h5py.File(dataset_dir, "r+")
            is_hdf_open = True
            
        print("=====================collecting demo_{}=====================".format(uu))
        
        #================================dagger===============================
        first_in_error = False
        is_dagger_episode = False
        should_train = False
        train_epochs = 0
        
        if use_dagger:
            # 判断是否应该使用DAgger策略
            is_dagger_episode = is_in_dagger_episode(uu, dagger_config)
            # 判断是否应该训练策略，并获得训练时的DAgger数据比例
            should_train, train_epochs, dagger_proportion = should_train_policy(uu, dagger_config)
            
            # 打印DAgger状态
            print_dagger_status(uu, is_dagger_episode, should_train, train_epochs)
                
        if is_dagger_episode:
            print("[INFO] Using DAgger strategy for episode {}".format(uu))
        #================================dagger===============================
        
        #preprocess
        if uu==0:
            if 'data' in new_f_out and not replace_existed_hdf5:
                existed_demo_num=len(new_f_out["data"])
                if existed_demo_num>=1 and delete_last_demo:
                    del new_f_out['data/demo_{}'.format(existed_demo_num-1)]
        #根据existed_demo_num的数量整体偏移
        if existed_demo_num>=1:
            if delete_last_demo:
                obs_path = 'data/demo_{}/obs'.format(uu+existed_demo_num-1)
                action_path = 'data/demo_{}/actions'.format(uu+existed_demo_num-1)
                pos_path = 'data/demo_{}/delta_pos_curgoal'.format(uu+existed_demo_num-1)
            else:
                obs_path = 'data/demo_{}/obs'.format(uu+existed_demo_num)
                action_path = 'data/demo_{}/actions'.format(uu+existed_demo_num)
                pos_path = 'data/demo_{}/delta_pos_curgoal'.format(uu + existed_demo_num)
        else:
            obs_path = 'data/demo_{}/obs'.format(uu)
            action_path = 'data/demo_{}/actions'.format(uu)
            pos_path = 'data/demo_{}/delta_pos_curgoal'.format(uu)

        if uu!=0 and uu % desire_pt_change_cycle == 0:
            teleop_and_pic(img_lst[-1], img2_lst[-1],img_size)
            env.set_target_coordinate(use_cur=True)

        action_list = []
        expert_action_list=[] # 专家动作列表（用于DAgger的标签）
        img_lst=[]
        img2_lst = []
        img_light_list = []
        img2_light_list = []
        rz_list=[]
        delta_pose_list=[]

        quit = False
        flag_tr = False
        flag_rot = False

        # get goal info
        goal_dict = get_goal_info(env)

        # action back to init T
        env.action_abs_T(env.wgT_tar@env.g_tar_g_init_T)


        def robo_operator():
            #global
            global env
            global trans_vel
            global rot_vel
            global uniform_vel
            global quit
            global ctrl_freq

            while not quit:
                tt=time.time()
                act_dict = get_expert_policy(wgT_tar=env.wgT_tar, wgT=env.wgT, trans_vel=trans_vel, rot_vel=rot_vel,
                                             uniform_vel=uniform_vel, dist_eps=env.dist_eps, angle_eps=env.angle_eps,
                                             motion_type="simultaneously", dof=6, need_trans_unit_transform=False,fine_print=False,real=True)
                # print("vel_rot: ",act_dict["vel_rot"])
                # print("vel_trans: ",act_dict["vel_tr"])
                # print("delta_pos: ",act_dict["cur_goal_delta_pose"])
                env.action_dT(act_dict["dT"])
                dt=time.time()-tt
                time.sleep(max(0,1/ctrl_freq-dt))
                # print("actual_ctrl_freq: ",1/(time.time()-tt))
            # operate.join()

        # robo_operator()
        operate = threading.Thread(target=robo_operator, daemon = True)
        ##TODO daemon = True的含义为：设置target为守护线程，即主程序中断，robo_operator也会中断。此处设置daemon=True是必须的，因为这么做是为了配合atexit()函数(参见atexit注释)

        operate.start()
        
        #================================dagger===============================
        def collision_detection():
            global end_episode
            global frame_counter
            global min_position_threshold
            global is_dagger_episode
            global collision_detector
            global collision_check_freq
            global env
            global distance

            first_frame = True
            while True:
                # DAgger策略检查物体和夹爪位置
                if is_dagger_episode and frame_counter:
                    t_start = time.time()

                    #apply transform to meshes first
                    if not first_frame:
                        last_wgT = wgT.copy()
                        wgT = env.wgT.copy()
                        dT = np.linalg.inv(last_wgT)@wgT
                    else:
                        wgT = env.wgT.copy()
                        wgT_tar = env.wgT_tar.copy()
                        dT = np.linalg.inv(wgT_tar)@wgT
                        first_frame = False
                    collision_detector.update_pos(dT)

                    #check collision
                    contact_flag ,distance = collision_detector.check_collision(num_sample_points=500,threshold=min_position_threshold[0])
                    print(f"distance: {distance}")
                    
                    if distance < min_position_threshold[0] or distance > min_position_threshold[1] or contact_flag:
                        print(f"[DAgger] Position minimum threshold reached at frame {frame_counter}, distance: {distance}, threshold: {min_position_threshold}")
                        end_episode = True

                    
                    dt = time.time()-t_start
                    if dt < 1/collision_check_freq:
                        time.sleep(1/collision_check_freq - dt)

        coll_det_thread = threading.Thread(target=collision_detection, daemon = True)
        coll_det_thread.start()
                    
        #================================dagger===============================
                    
                    

        tt = time.time()

        t0 = time.time()
        # gkkk=0

        #================================dagger===============================
        frame_counter = 0
        #================================dagger===============================

        while True:
            frame_counter += 1
            if time.time() - t0 > 1/data_collect_freq:    #此程序运行约0.01s（10hz）,因此循环频率需要低于10hz
                # print("camera circulation_time:", time.time() - t0)
                t0 = time.time()
                # 读取图像帧，包括RGB图
                rtn_dict = env.observation()
                img = rtn_dict['img_1']
                img2 = rtn_dict['img_2'] if 'img_2' in rtn_dict else None


                wgT_tar = env.wgT_tar
                wgT = env.wgT
                # 获取专家策略的动作（总是计算，因为DAgger需要专家标签）
                expert_act_dict = get_expert_policy(wgT_tar=wgT_tar, wgT=wgT, trans_vel=trans_vel, rot_vel=rot_vel,
                                             uniform_vel=uniform_vel, dist_eps=env.dist_eps, angle_eps=env.angle_eps,
                                             motion_type="simultaneously", dof=6,need_trans_unit_transform=False,fine_print=False,real=True)  ##TODO:need to be reused in robo_operator

                vel_tr = filter_translation(expert_act_dict['vel_tr'], thres=1e-5)
                vel_rot = expert_act_dict['vel_rot']  # 3dof:绕世界系 6dof:绕夹爪系
                expert_dT = expert_act_dict["dT"]
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
                        img_light = None,
                        img_light_goal = None,
                        img2_light = None,
                        img2_light_goal = None
                    )
                    policy_action = get_policy_action(policy_model, obs_dict)
                    action = policy_action
                    # 从策略动作构建变换矩阵dT
                    dT = construct_dT_from_action(policy_action, dof=6)
                    # 打印策略动作和专家动作的差异
                    # print(f"[DAgger] Policy Action: {action}, Expert Action: {expert_action}")
            
                # 保存动作（对于普通收集是实际执行的动作，对于DAgger是专家动作）
                action_list.append(action)
                if is_dagger_episode:
                    expert_action_list.append(expert_action)

                if record_pose:
                    delta_pose_list.append(expert_act_dict['cur_goal_delta_pose'])

                # postprocess
                img_vis = img.copy()
                img = clip_image(img, img_size,keep_right=True)
                if img_save_type == "rgb":
                    img = img[:, :, ::-1]
                img_lst.append(img)

                if img2 is not None:
                    img2_vis = img2.copy()
                    img2 = clip_image(img2, img_size,keep_right=True)
                    if img_save_type == "rgb":
                        img2 = img2[:, :, ::-1]
                    img2_lst.append(img2)

                    combined_img = np.hstack((img_vis, img2_vis))
                else:
                    combined_img = img_vis

                cv2.imshow("Combined Image", combined_img)
                cv2.waitKey(1)


                #env.action_dT(dT) ##TODO:应该在robo_operator被使用

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
                    quit = True
                    operate.join()
                    coll_det_thread.join()
                    cv2.destroyAllWindows()
                    print('...录制结束，数据处理中...')

                    if is_dagger_episode:
                        print("Final distance between gripper and object:", distance)
                        action_list.append(np.array([0,0,0,0,0,0]))
                        expert_action_list.append(np.array([0,0,0,0,0,0]))
                    else:
                        action_list.append(np.array([0,0,0,0,0,0]))

                    img_goal = clip_image(goal_dict["img_goal"], img_size,keep_right=True)
                    if img_save_type == "rgb":
                        img_goal = img_goal[:, :, ::-1]
                    img_lst.append(img_goal)

                    if goal_dict["img_goal2"] is not None:
                        im_goal2 = clip_image(goal_dict["img_goal2"], img_size,keep_right=True)
                        if img_save_type == "rgb":
                            im_goal2 = im_goal2[:, :, ::-1]
                        img2_lst.append(im_goal2)

                    if record_pose:
                        delta_pose_list.append(np.zeros(6))

                    # post process
                    if disturb_abs_rot["utilized"]:
                        rz_list, _ = _disturb_abs_rot(rz_list, action_list)

                    if portion_last_episode["utilized"]:
                        action_list, _ = _portion_last_episode(action_list, portion_last_episode["portion_last_num"],
                                                               ac_dim=6)
                        if is_dagger_episode:
                            expert_action_list, _ = _portion_last_episode(expert_action_list, portion_last_episode["portion_last_num"],
                                                               ac_dim=6)

                    if add_end_episode["utilized"]:
                        pick_id = len(img_lst) - 1
                        insert_id = len(img_lst) - 1
                        add_num = add_end_episode["add_num"]

                        rz_list, action_list, delta_pose_list = _add_end_episode(add_num=add_num,
                                                                                 disturb_abs_rot=disturb_abs_rot[
                                                                                     "utilized"], abs_rot_list=rz_list,
                                                                                 act_lst=action_list,
                                                                                 pose_list=delta_pose_list)

                        if is_dagger_episode:
                            # 对专家动作列表也进行相同的处理
                            _,expert_action_list,_ = _add_end_episode(add_num=add_num,
                                                                                 disturb_abs_rot=disturb_abs_rot[
                                                                                     "utilized"], abs_rot_list=rz_list,
                                                                                 act_lst=expert_action_list,
                                                                                 pose_list=delta_pose_list)
                        img_lst = insert_imgs(img_lst, pick_id, insert_id, add_num)
                        if len(img2_lst) != 0:
                            img2_lst = insert_imgs(img2_lst, pick_id, insert_id, add_num)

                    if add_medium_episode["utilized"]:
                        action_list, rz_list, delta_pose_list, need_add_medium, trans_id, rot_id = _add_medium_episode(
                            act_lst=action_list, abs_rot_list=rz_list, ac_dim=6,
                            add_num=add_medium_episode["add_num"], pose_list=delta_pose_list)
                        if is_dagger_episode and need_add_medium:
                            # 对专家动作列表也进行相同的处理
                            expert_action_list,_,_,_,_,_ = _add_medium_episode(
                            act_lst=expert_action_list, abs_rot_list=rz_list, ac_dim=6,
                            add_num=add_medium_episode["add_num"], pose_list=delta_pose_list)

                        if need_add_medium:
                            print("+++++++++++++++++++++++++++++++++++++++++")
                            pick_id = trans_id + 1
                            insert_id = rot_id
                            add_num = add_medium_episode["add_num"]

                            img_lst = insert_imgs(img_lst, pick_id, insert_id, add_num)
                            if len(img2_lst) != 0:
                                img2_lst = insert_imgs(img2_lst, pick_id, insert_id, add_num)

                    #================main augmentation================
                    if use_augmentation:
                        for img in img_lst:
                            img_light = augmentation_module.augment_image(img, False)
                            img_light_list.append(img_light)
                        for img2 in img2_lst:
                            img2_light = augmentation_module.augment_image(img2, False)
                            img2_light_list.append(img2_light)
                    # ================main augmentation================

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
                        if len(img_light_list) != 0:
                            episode_data["obs"]["robot0_eye_in_hand_image_light"] = np.array(img_light_list)
                        if len(img2_light_list) != 0:
                            episode_data["obs"]["robot0_eye_in_hand_image_2_light"] = np.array(img2_light_list)

                    # save hdf5
                    epi_length = len(img_lst)
                    assert epi_length == len(action_list)
                    if existed_demo_num >= 1:
                        if delete_last_demo:
                            add_useless_things(new_f_out=new_f_out, demo_ind=uu + existed_demo_num-1, epi_len=epi_length)
                        else:
                            add_useless_things(new_f_out=new_f_out, demo_ind=uu + existed_demo_num, epi_len=epi_length)
                    else:
                        add_useless_things(new_f_out=new_f_out, demo_ind=uu, epi_len=epi_length)
                    new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image', data=img_lst)

                    if len(img2_lst) != 0:
                        new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_2', data=img2_lst)
                    if len(img_light_list)!=0:
                        new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_light', data=img_light_list)
                    if len(img2_light_list)!= 0:
                        new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_2_light', data=img2_light_list)
                    if len(delta_pose_list) != 0:
                        new_f_out.create_dataset(pos_path, data=delta_pose_list)

                    # 在DAgger中，保存的动作取决于是否是DAgger模式
                    if is_dagger_episode:
                        new_f_out.create_dataset(action_path, data=expert_action_list)
                        print("expert_action_lst-1:", expert_action_list[-1])
                    else:
                        new_f_out.create_dataset(action_path, data=action_list)
                        print("action_lst-1:", action_list[-1])

                    print("[INFO] demo_{} collected successfully.".format(uu))
                    #================================dagger===============================
                    if is_dagger_episode:
                        print(f"[DAgger] Episode {uu} completed successfully")
                    # ================================dagger===============================
                        
                    #===============================policy training===============================
                    # 使用之前计算的训练状态
                    if should_train and policy_model is not None:
                        print(f'[DAgger] Training policy model at episode {uu} with {train_epochs} epochs')

                        # 关闭并重新打开HDF5文件
                        new_f_out.close()
                        is_hdf_open = False
                        
                        # 准备训练配置
                        data_cfg = model_config["dataset"].copy()
                        data_cfg["hdf5_path"] = dataset_dir
                        train_cfg = model_config["training"].copy()
                        
                        # 创建模型保存路径
                        model_path = os.path.join(base_dir, 'AlignAnything_real', current_date, 'models')
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

                            rng = np.random.default_rng(seed=uu)
                            chosen_dagger = rng.choice(dagger_demos, size=min(len(dagger_demos), num_dagger_target), replace=False).tolist()
                            chosen_non_dagger = rng.choice(non_dagger_demos, size=min(len(non_dagger_demos), num_non_dagger_target), replace=False).tolist()
                            mixed = sorted(chosen_dagger + chosen_non_dagger)
                            tmp_key = f"dagger_mix_ep_{uu}"
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
                            episode_idx=uu,
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





