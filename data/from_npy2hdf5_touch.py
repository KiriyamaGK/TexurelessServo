import numpy as np
import h5py
import os
import cv2
import json
import datetime
from utils.hdf5 import split_train_val_from_hdf5,add_env_meta,compute_num_samples
from utils.paths import return_disc_route

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
    img_size=220
    gau_size=64
    # current_date = datetime.datetime.now()
    # hdf_date = current_date.strftime('%Y.%m.%d')
    date='25.01.24'
    replace_exist=True
    cut_to_square=True
    use_data_light=True
    # formatted_date='2024.11.07'
    # npy_date=hdf_date[2:]
    exact_gauss_img=False
    exact_abs_rot_list=True
    base=return_disc_route('One Touch/AlignAnything')
    val_ratio = 0.1


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
        if use_data_light:
            data_img_light=data['img_light']

        data_act=data['action_list']
        if exact_abs_rot_list:
            data_rot=data['abs_rot']
        if exact_gauss_img:
            data_gauss_img_ct=data['gauss_img_ct'][0]
            data_gauss_img_ct=data_gauss_img_ct.reshape(-1,gau_size,gau_size,1)
            data_gauss_img_kpt = data['gauss_img_kpt'][0]
            data_gauss_img_kpt = data_gauss_img_kpt.reshape(-1, gau_size, gau_size, 1)
        row_1=len(data_img)

        if row_1==0:
            print("Empty demonstrations in demo_{}".format(i-1))
            almost_npy_dir = os.path.join(npy_base, str(num_list[i-1]))
            dir = os.path.join(almost_npy_dir, 'demo.npy')
            print('processing {}'.format(dir))
            data = np.load(dir, allow_pickle=True).item()
            data_img = data['img']
            if use_data_light:
                data_img_light = data['img_light']
            data_act = data['action_list']
            if exact_abs_rot_list:
                data_rot = data['abs_rot']
            if exact_gauss_img:
                data_gauss_img_ct = data['gauss_img_ct'][0]
                data_gauss_img_ct = data_gauss_img_ct.reshape(-1, gau_size, gau_size, 1)
                data_gauss_img_kpt = data['gauss_img_kpt'][0]
                data_gauss_img_kpt = data_gauss_img_kpt.reshape(-1, gau_size, gau_size, 1)
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
            if use_data_light:
                data_img_light=trans_imgs_to_square(data_img_light,img_size,cut_to_square)

        obs_path = 'data/demo_{}/obs'.format(i)
        action_path = 'data/demo_{}/actions'.format(i)

        new_f_out.create_dataset(action_path, data=data_act)
        new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image', data=data_img)
        if use_data_light:
            new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image_light', data=data_img_light)
        if exact_gauss_img:
            new_f_out.create_dataset(obs_path + '/gaussian_img_kpt', data=data_gauss_img_kpt)
            new_f_out.create_dataset(obs_path + '/gaussian_img_ct', data=data_gauss_img_ct)
        if exact_abs_rot_list:
            new_f_out.create_dataset(obs_path + '/abs_rot', data=data_rot)

    add_env_meta(new_f_out)
    new_f_out.close()

    compute_num_samples(hdf5_path)
    split_train_val_from_hdf5(hdf5_path,val_ratio)