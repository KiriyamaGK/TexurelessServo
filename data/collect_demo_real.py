import time
import cv2
import numpy as np
import os
import random
import threading
import json
from utils.paths import return_disc_route
from utils.file import ensure_dir
from utils.policy import get_expert_policy
from utils.hdf5 import add_useless_things, split_train_val_from_hdf5, add_env_meta, compute_num_samples, add_config
import h5py
from utils.input_process import clip_image
from real.environment import Environment
from data.process_hdf5 import _disturb_abs_rot,_portion_last_episode,_add_end_episode,_add_medium_episode,insert_imgs


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



if __name__=='__main__':
    ##TODO:数据采集和测试阶段一共算了四种类型的dT，其中env.sample_init_pos算了g_tar_gT；get_action算了g_gtar(和缩短后的dT)；need_init和need_init_eval在compute_error的时候各自算了一次（见env相关函数）

    current_pt_desire=True

    config_dir = "../configs/demo_collection_real.json"
    with open(config_dir, "r") as j:
        config = json.load(j)

    env=Environment(robot_address=config["hardware"]["robot_address"],**config["demo_collection"]["env"],**config["hardware"]["camera"])
    cam=env.camera
    robot_ins=env.robot_ins

    init_pos = robot_ins.get_gripper_TCP_pose()
    init_pos[3] = -180
    init_pos[4] = 0
    in_desire_pt = init_pos
    env.robot_ins.move_cart(filter_pos(in_desire_pt),tool=1, user=0, vel=40)
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

      ##img相关
    img_save_type = config["demo_collection"]["img"]["save_type"]
    assert img_save_type in ["rgb", "bgr"]
    img_size = config["demo_collection"]["img"]["size"]

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

    for uu in range(demo_total_num):
        print("=====================collecting demo_{}=====================".format(uu))
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

        if uu == 0:
            desire_pt = in_desire_pt
        elif uu % desire_pt_change_cycle == 0:
            desire_pt = env.place(p_0=desire_pt)
            env.set_target_coordinate(use_cur=True)  #TODO: 这部分和sample init pos配合逻辑有些问题

        action_list = []
        img_lst=[]
        img2_lst = []
        rz_list=[]
        delta_pose_list=[]

        quit = False
        flag_tr = False
        flag_rot = False

        # # move to initial T,and get info
        # env.action_abs_T(env.wgT_tar@env.g_tar_g_init_T)
        # init_transform_dict = env.return_cur_pos_info()

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

            while not quit:
                act_dict = get_expert_policy(wgT_tar=env.wgT_tar, wgT=env.wgT, trans_vel=trans_vel, rot_vel=rot_vel,
                                             uniform_vel=uniform_vel, dist_eps=env.dist_eps, angle_eps=env.angle_eps,
                                             motion_type="simultaneously", dof=6, need_trans_unit_transform=False,fine_print=False,real=True)
                # print("vel_rot: ",act_dict["vel_rot"])
                # print("vel_trans: ",act_dict["vel_tr"])
                # print("delta_pos: ",act_dict["cur_goal_delta_pose"])
                env.action_dT(act_dict["dT"])
                time.sleep(0.008)
            # operate.join()

        # robo_operator()
        operate = threading.Thread(target=robo_operator, daemon=False)
        operate.start()
        tt = time.time()

        t0 = time.time()
        flag_flag = False  # 表示平动刚完成
        flag_flag_flag = False
        # gkkk=0

        while True:
            if time.time() - t0 > 1/data_collect_freq:    #此程序运行约0.01s（10hz）,因此循环频率需要低于10hz
                # print("camera circulation_time:", time.time() - t0)
                t0 = time.time()
                # 读取图像帧，包括RGB图
                rtn_dict = env.observation()
                img = rtn_dict['img_1']
                img2 = rtn_dict['img_2'] if 'img_2' in rtn_dict else None


                wgT_tar = env.wgT_tar
                wgT = env.wgT
                act_dict = get_expert_policy(wgT_tar=wgT_tar, wgT=wgT, trans_vel=trans_vel, rot_vel=rot_vel,
                                             uniform_vel=uniform_vel, dist_eps=env.dist_eps, angle_eps=env.angle_eps,
                                             motion_type="simultaneously", dof=6,need_trans_unit_transform=False,fine_print=False,real=True)  ##TODO:need to be reused in robo_operator

                vel_tr = filter_translation(act_dict['vel_tr'], thres=1e-5)
                vel_rot = act_dict['vel_rot']  # 3dof:绕世界系 6dof:绕夹爪系
                dT = act_dict["dT"]
                action_list.append(np.concatenate((vel_tr, vel_rot)))

                if record_pose:
                    delta_pose_list.append(act_dict['cur_goal_delta_pose'])

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

                if env.reinit():
                    quit = True
                    operate.join()
                    cv2.destroyAllWindows()
                    print('...录制结束，数据处理中...')

                    action_list.append(np.array([0, 0, 0, 0, 0, 0]))

                    img_goal = clip_image(goal_dict["img_goal"], img_size,keep_right=True)
                    img_lst.append(img_goal)

                    if goal_dict["img_goal2"] is not None:
                        im_goal2 = clip_image(goal_dict["img_goal2"], img_size,keep_right=True)
                        img2_lst.append(im_goal2)
                    if record_pose:
                        delta_pose_list.append(np.zeros(6))

                        # post process
                        if disturb_abs_rot["utilized"]:
                            rz_list, _ = _disturb_abs_rot(rz_list, action_list)

                        if portion_last_episode["utilized"]:
                            action_list, _ = _portion_last_episode(action_list, portion_last_episode["portion_last_num"],
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
                            img_lst = insert_imgs(img_lst, pick_id, insert_id, add_num)
                            if len(img2_lst) != 0:
                                img2_lst = insert_imgs(img2_lst, pick_id, insert_id, add_num)

                        if add_medium_episode["utilized"]:
                            action_list, rz_list, delta_pose_list, need_add_medium, trans_id, rot_id = _add_medium_episode(
                                act_lst=action_list, abs_rot_list=rz_list, ac_dim=6,
                                add_num=add_medium_episode["add_num"], pose_list=delta_pose_list)
                            if need_add_medium:
                                print("+++++++++++++++++++++++++++++++++++++++++")
                                pick_id = trans_id + 1
                                insert_id = rot_id
                                add_num = add_medium_episode["add_num"]

                                img_lst = insert_imgs(img_lst, pick_id, insert_id, add_num)
                                if len(img2_lst) != 0:
                                    img2_lst = insert_imgs(img2_lst, pick_id, insert_id, add_num)

                        # save hdf5
                        epi_length = len(img_lst)
                        assert epi_length == len(action_list)
                        if existed_demo_num >= 1:
                            add_useless_things(new_f_out=new_f_out, demo_ind=uu + existed_demo_num, epi_len=epi_length)
                        else:
                            add_useless_things(new_f_out=new_f_out, demo_ind=uu, epi_len=epi_length)
                        new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image', data=img_lst)

                        if len(img2_lst) != 0:
                            new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_2', data=img2_lst)

                        if len(delta_pose_list) != 0:
                            new_f_out.create_dataset(pos_path, data=delta_pose_list)

                        new_f_out.create_dataset(action_path, data=action_list)
                        print("action_lst-1:", action_list[-1])
                        print("[INFO] demo_{} collected successfully.".format(uu))
                    break

    cam.release()

    add_env_meta(new_f_out)
    add_config(new_f_out,config)
    new_f_out.close()
    compute_num_samples(dataset_dir)
    split_train_val_from_hdf5(dataset_dir, val_ratio=0.1)




