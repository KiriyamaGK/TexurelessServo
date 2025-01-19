import numpy as np
import h5py
import os
import cv2
import json
import datetime
from utils.hdf5 import split_train_val_from_hdf5

def return_folders(dir):
    num_list = []
    for num in os.listdir(dir):
        npy = os.path.join(dir, str(num), 'demo.npy')
        if os.path.exists(npy):
            num_list.append(int(num))
        else:
            raise RuntimeError('npy: {} not found'.format(npy))
    num_list.sort()
    return num_list

def trans_imgs_to_square(imgs,img_size,cut2square):
    assert len(imgs.shape) in [3,4]
    if len(imgs.shape) == 4:
        height, width = imgs[0].shape[0], imgs[0].shape[1]
    elif len(imgs.shape) == 3:
        height, width = imgs.shape[0], imgs.shape[1]
    if height<img_size or width<img_size:
        raise RuntimeError('Upsampling is forbidden.')
    elif height!=img_size or width!=img_size:
        if cut2square:
            if width > height:
                left = (width - height) // 2
                top = 0
                right = (width + height) // 2
                bottom = height
            else:
                left = 0
                top = (height - width) // 2
                right = width
                bottom = (height + width) // 2
            if len(imgs.shape) == 4:
                imgs = imgs[:,top:bottom, left:right,:]  # 使用 NumPy 数组进行裁剪
            elif len(imgs.shape) == 3:
                imgs = imgs[top:bottom,left:right,:]

        if len(imgs.shape) == 4:
            length = imgs.shape[0]
            img_cropped=np.array([np.stack(cv2.resize(imgs[j],(img_size,img_size))) for j in range(length)])
        elif len(imgs.shape) == 3:
            img_cropped = cv2.resize(imgs,(img_size,img_size))
        return img_cropped
    else:
        return imgs

if __name__ == '__main__':
    img_size=120
    # current_date = datetime.datetime.now()
    # hdf_date = current_date.strftime('%Y.%m.%d')
    date='25.01.17'
    replace_exist=True
    cut_to_square=True
    # formatted_date='2024.11.07'
    # npy_date=hdf_date[2:]
    exact_abs_rot_list=True
    add_goal_image=True
    base='/media/kiriyamagk/One Touch/AlignAnything'


    base_dir=os.path.join(base,date)
    hdf_base=os.path.join(base_dir, 'hdf5')
    npy_base=os.path.join(base_dir, 'npys')
    hdf5_path=os.path.join(hdf_base,'mimic.hdf5')
    os.makedirs(hdf_base,exist_ok=True)

    if replace_exist:
        new_f_out = h5py.File(hdf5_path, "w")
    else:
        if os.path.exists(hdf5_path):
            new_f_out = h5py.File(hdf5_path, "r+")
        else:
            new_f_out = h5py.File(hdf5_path, "w")

    num_list=return_folders(npy_base)
    num_train=len(num_list)

    for i in range(num_train):
        almost_npy_dir=os.path.join(npy_base,str(num_list[i]))
        if 'data' in new_f_out:
            if 'demo_{}'.format(i) in new_f_out['data']:
                if 'agentview_image' in new_f_out['data/demo_{}/obs'.format(i)] or 'robot0_eye_in_hand_image' in new_f_out['data/demo_{}/obs'.format(i)]:
                    print("demo_{} already exists".format(i))
                    continue
        dir=os.path.join(almost_npy_dir,'demo.npy')
        print('processing {}'.format(dir))
        data=np.load(dir,allow_pickle=True).item()
        data_img=data['img']
        data_act=data['action_list']
        if exact_abs_rot_list:
            data_rot=data['abs_rot']
        row_1=len(data_img)

        if row_1==0:
            print("Empty demonstrations in demo_{}".format(i-1))
            almost_npy_dir = os.path.join(npy_base, str(num_list[i-1]))
            dir = os.path.join(almost_npy_dir, 'demo.npy')
            print('processing {}'.format(dir))
            data = np.load(dir, allow_pickle=True).item()
            data_img = data['img']
            data_act = data['action_list']
            if exact_abs_rot_list:
                data_rot = data['abs_rot']
            row_1 = len(data_img)

        new_f_out.create_dataset('data/demo_{}/dones'.format(i), data=np.zeros((row_1 - 1)))
        new_f_out.create_dataset('data/demo_{}/interventions'.format(i), data=np.zeros((row_1, 1)))
        new_f_out.create_dataset('data/demo_{}/policy_acting'.format(i), data=np.zeros((row_1)))
        new_f_out.create_dataset('data/demo_{}/rewards'.format(i), data=np.zeros((row_1 - 1)))
        new_f_out.create_dataset('data/demo_{}/states'.format(i), data=np.zeros((0)))
        new_f_out.create_dataset('data/demo_{}/user_acting'.format(i), data=np.zeros((row_1, 1)))

        ori_h,ori_w=data_img[0].shape[0:2]
        if ori_h!=img_size or ori_w!=img_size:
            data_img=trans_imgs_to_square(data_img,img_size,cut_to_square)

        obs_path = 'data/demo_{}/obs'.format(i)
        action_path = 'data/demo_{}/actions'.format(i)

        new_f_out.create_dataset(action_path, data=data_act)
        new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image', data=data_img)

        if add_goal_image:
            data_goal=np.array([np.stack(data_img[-1]) for kk in range(row_1)])
            new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_goal', data=data_goal)
        if exact_abs_rot_list:
            new_f_out.create_dataset(obs_path + '/abs_rot', data=data_rot)

    env_meta = {
                "env_name": "Libero_Kitchen_Tabletop_Manipulation",
                "env_version": "1.4.1",
                "type": 1,
                "env_kwargs": {
                    "robots": [
                        "Panda"
                    ],
                    "controller_configs": {
                        "type": "OSC_POSE",
                        "input_max": 1,
                        "input_min": -1,
                        "output_max": [
                            0.05,
                            0.05,
                            0.05,
                            0.5,
                            0.5,
                            0.5
                        ],
                        "output_min": [
                            -0.05,
                            -0.05,
                            -0.05,
                            -0.5,
                            -0.5,
                            -0.5
                        ],
                        "kp": 150,
                        "damping_ratio": 1,
                        "impedance_mode": "fixed",
                        "kp_limits": [
                            0,
                            300
                        ],
                        "damping_ratio_limits": [
                            0,
                            10
                        ],
                        "position_limits": None,
                        "orientation_limits": None,
                        "uncouple_pos_ori": True,
                        "control_delta": True,
                        "interpolation": None,
                        "ramp_ratio": 0.2
                    },
                    "bddl_file_name": None,
                    "reward_shaping": False,
                    "camera_names": [
                        "agentview",
                        "robot0_eye_in_hand"
                    ],
                    "camera_heights": 84,
                    "camera_widths": 84,
                    "has_renderer": False,
                    "has_offscreen_renderer": True,
                    "ignore_done": True,
                    "use_object_obs": True,
                    "use_camera_obs": True,
                    "camera_depths": False,
                    "render_gpu_device_id": 0
                }
            }

    dat = new_f_out['data']
    dat.attrs['env_args'] = json.dumps(env_meta, indent=4)  # data增加属性
    new_f_out.close()

    total_samples = 0

    f = h5py.File(hdf5_path, "r+")
    for ep in f['data']:
         num=len(f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)])
         if num==1:
             print("{} len is only 1,adding......".format(ep))
             row_1=num+1
             del f['data/{}/dones'.format(ep)]
             del f['data/{}/interventions'.format(ep)]
             del f['data/{}/policy_acting'.format(ep)]
             del f['data/{}/rewards'.format(ep)]
             del f['data/{}/states'.format(ep)]
             del f['data/{}/user_acting'.format(ep)]

             f.create_dataset('data/{}/dones'.format(ep), data=np.zeros((row_1 - 1)))
             f.create_dataset('data/{}/interventions'.format(ep), data=np.zeros((row_1, 1)))
             f.create_dataset('data/{}/policy_acting'.format(ep), data=np.zeros((row_1)))
             f.create_dataset('data/{}/rewards'.format(ep), data=np.zeros((row_1 - 1)))
             f.create_dataset('data/{}/states'.format(ep), data=np.zeros((0)))
             f.create_dataset('data/{}/user_acting'.format(ep), data=np.zeros((row_1, 1)))

             act=f['data/{}/actions'.format(ep)][0].copy()

             del f['data/{}/actions'.format(ep)]

             f['data/{}/actions'.format(ep)]=[act,act]


             wrist_img = f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)][0].copy()
             del f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)]
             f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)] = [wrist_img, wrist_img]

             if 'robot0_eye_in_hand_image_goal' in f['data/{}/obs'.format(ep)]:
                 wrist_img = f['data/{}/obs/robot0_eye_in_hand_image_goal'.format(ep)][0].copy()
                 del f['data/{}/obs/robot0_eye_in_hand_image_goal'.format(ep)]
                 f['data/{}/obs/robot0_eye_in_hand_image_goal'.format(ep)] = [wrist_img, wrist_img]

             if 'abs_rot' in f['data/{}/obs'.format(ep)]:
                 abs_rot = f['data/{}/obs/abs_rot'.format(ep)][0].copy()
                 del f['data/{}/obs/abs_rot'.format(ep)]
                 f['data/{}/obs/abs_rot'.format(ep)] = [abs_rot, abs_rot]

    f.close()

    f = h5py.File(hdf5_path, "a")  # edit mode
    for ep in f["data"]:
        # add "num_samples" into per-episode metadata
        if "num_samples" in f["data/{}".format(ep)].attrs:
            del f["data/{}".format(ep)].attrs["num_samples"]
        n_sample = f["data/{}/actions".format(ep)].shape[0] - 1
        f["data/{}".format(ep)].attrs["num_samples"] = n_sample
        total_samples += n_sample

        # print("num_samples:",n_sample)
    # add total samples to global metadata
    if "total" in f["data"].attrs:
        del f["data"].attrs["total"]
    f["data"].attrs["total"] = total_samples
    split_train_val_from_hdf5(hdf5_path)