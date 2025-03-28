import os
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
from utils.input_process import clip_image
from utils.policy import get_expert_policy
from data.process_hdf5 import _disturb_abs_rot,_portion_last_episode,_add_end_episode,_add_medium_episode,insert_imgs

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

if __name__ == '__main__':
    img_w=220
    img_h=220

    config_dir= "../configs/demo_collection.json"

    with open(config_dir, "r") as j:
        config = json.load(j)

    #overall setting
    objs_descriptor=config['overall_setting']['objs_descriptor']
    current_date=config['overall_setting']['file_name']
    demo_total_num = config['overall_setting']['demo_total_num']
    replace_existed_hdf5=config["overall_setting"]["replace_existed_hdf5"] #TODO:remember to use

    #demo_collection
    dof = config["demo_collection"]["dof"]
    motion_type=config["demo_collection"]['trans_and_rot_type']
    trans_vel_norm=config["demo_collection"]['trans_vel'] #m
    rot_vel_norm=config["demo_collection"]['rot_vel']    #deg
    init_horizon_trans=config["demo_collection"]['init_horizon_trans']
    init_vertical_trans = config["demo_collection"]['init_vertical_trans']
    init_rot=config["demo_collection"]['init_rot']
    use_max_rot=config["demo_collection"]['use_max_rot']
    random_light_dir = config["demo_collection"]['random_light_dir']
    use_light_key=config["demo_collection"]["use_random_light_img_key"] if random_light_dir else False

    #post process
    disturb_abs_rot = config['post_process']['disturb_abs_rot']
    portion_last_episode = config['post_process']['portion_last_episode']
    add_end_episode = config['post_process']['add_end_episode']
    add_medium_episode = config['post_process']['add_medium_episode']
    assert (not portion_last_episode["utilized"]) or (not add_end_episode["utilized"])

    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    env=Environment(camera_intrinsic,objs_descriptor=objs_descriptor,use_max_rot=use_max_rot,init_horizon_trans=init_horizon_trans,init_vertical_trans=init_vertical_trans,init_rot=init_rot,dof=dof)
    env.init()

    base_dir = return_disc_route("One Touch")

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
    for idx in range(demo_total_num):
        print("[INFO] start collecting demo_{} ...".format(idx))

        if idx==0:
            if 'data' in new_f_out and not replace_existed_hdf5:
                existed_demo_num=len(new_f_out["data"])

        if existed_demo_num>=1:#根据existed_demo_num的数量整体偏移
            obs_path = 'data/demo_{}/obs'.format(idx+existed_demo_num)
            action_path = 'data/demo_{}/actions'.format(idx+existed_demo_num)
        else:
            obs_path = 'data/demo_{}/obs'.format(idx)
            action_path = 'data/demo_{}/actions'.format(idx)

        action_list=[]
        img_lst=[]
        img_light_list=[]
        img2_lst = []
        img2_light_list = []
        rz_list=[]
        gaussian_img_ct_lst=[]
        gaussian_img_kpt_lst = []
        # iii=0
        while True:
            if not use_light_key:
                rtn_dict=env.observation(random_light_dir=random_light_dir,use_prob=True)

                img=rtn_dict['img_1']
                img2=rtn_dict['img_2'] if 'img_2' in rtn_dict else None
                img_light = None
                img2_light = None
            else:
                rtn_dict=env.observation(random_light_dir=False)
                rtn_light_dict=env.observation(random_light_dir=True,use_prob=False)

                img = rtn_dict['img_1']
                img2 = rtn_dict['img_2'] if 'img_2' in rtn_dict else None
                img_light = rtn_light_dict['img_1']
                img2_light = rtn_light_dict['img_2'] if 'img_2' in rtn_light_dict else None

            wgT_tar=env.wgT_tar
            wgT=env.wgT
            act_dict=get_expert_policy(wgT_tar=wgT_tar,wgT=wgT,trans_vel_norm=trans_vel_norm,rot_vel_norm=rot_vel_norm,dist_eps=env.dist_eps,angle_eps=env.angle_eps,motion_type=motion_type,dof=dof)

            vel_tr=filter_translation(act_dict['vel_tr'],thres=1e-7)
            vel_rot=act_dict['vel_rot'] #3dof:绕世界系 6dof:绕夹爪系
            dT=act_dict["dT"]
            # vel = np.concatenate((vel_tr,np.array([vel_rot]))) if not isinstance(vel_rot,np.ndarray) else  np.concatenate((vel_tr,vel_rot))
            action_list.append(np.concatenate((vel_tr,vel_rot)))

            if dof==3:
                rz = rmat2euler_rz_degree(wgT)
                rz_list.append(rz)

            #postprocess
            img_vis=img.copy()
            img=clip_image(img,img_h)
            img_lst.append(img)
            if use_light_key:
                img_light=clip_image(img_light,img_h)
                img_light_list.append(img_light)
            if img2 is not None:
                img2_vis=img2.copy()
                img2=clip_image(img2,img_h)
                img2_lst.append(img2)
                combined_img = np.hstack((img_vis, img2_vis))
                # cv2.imshow("Combined Image", combined_img)
                # cv2.waitKey(1)
            if img2_light is not None:
                img2_light=clip_image(img2_light,img_h)
                img2_light_list.append(img2_light)
            # print(iii)
            # iii+=1
            env.action(dT)
            if env.reinit():

                #post process
                if disturb_abs_rot["utilized"]:
                    rz_list,_=_disturb_abs_rot(rz_list,action_list)

                if portion_last_episode["utilized"]:
                    action_list,_=_portion_last_episode(action_list,portion_last_episode["portion_last_num"],dof)

                if add_end_episode["utilized"]:
                    pick_id=len(img_lst)-1
                    insert_id=len(img_lst)-1
                    add_num=add_end_episode["add_num"]

                    rz_list, action_list=_add_end_episode(add_num,disturb_abs_rot["utilized"],rz_list,action_list)
                    img_lst=insert_imgs(img_lst,pick_id,insert_id,add_num)
                    if len(img_light_list) != 0:
                        img_light_list=insert_imgs(img_light_list,pick_id,insert_id,add_num)
                    if len(img2_lst) != 0:
                        img2_lst=insert_imgs(img2_lst,pick_id,insert_id,add_num)
                    if len(img2_light_list) != 0:
                        img2_light_list=insert_imgs(img2_light_list,pick_id,insert_id,add_num)

                if add_medium_episode["utilized"]:
                    action_list, rz_list, need_add_medium, trans_id, rot_id=_add_medium_episode(action_list, rz_list, dof,add_medium_episode["add_num"])
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
                if dof==3:
                    new_f_out.create_dataset(obs_path + '/abs_rot', data=rz_list)
                new_f_out.create_dataset(action_path, data=action_list)
                print("action_lst-1:",action_list[-1])
                print("[INFO] demo_{} collected successfully.".format(idx))
                break
    add_env_meta(new_f_out)
    add_config(new_f_out, config)
    new_f_out.close()
    compute_num_samples(dataset_dir)
    split_train_val_from_hdf5(dataset_dir, val_ratio=0.1)








