import h5py
import numpy as np
import os
import random
from utils.hdf5 import split_train_val_from_hdf5,compute_num_samples,delete_useless_things,add_useless_things
from utils.paths import return_disc_route

def _disturb_abs_rot(abs_rot_list,action_list):
    need_disturb = True
    num = len(abs_rot_list)
    rot_id = num + 1
    for uu in range(num):
        rot = action_list[uu][2]
        if rot != 0:
            rot_id = uu + 1
            break
    if rot_id < 2 or rot_id > num:  # rot_id <2:纯旋转，rot_id > num：纯平移
        need_disturb = False
    else:
        del abs_rot_list[rot_id:]
        for i in range(rot_id, num):
            abs_rot_list.append(random.uniform(0, 360.0))
        assert len(abs_rot_list) == num
    return abs_rot_list,need_disturb

def _portion_last_episode(action_list,portion_last_num,ac_dim):
    need_portion = True
    num = len(action_list)
    # rot_id = num + 1
    # for uu in range(num):
    #     rot = action_list[uu][2] if ac_dim == 3 else np.linalg.norm(
    #         np.array(action_list[uu][3:]))
    #     if rot != 0:
    #         rot_id = uu + 1
    #         break
    # if rot_id > num:  # 纯平移
    #     need_portion = False
    # else:
    # cvt_last_num = min(portion_last_num,num - rot_id)  # eg: 0 0 0 0 0 1 1 num=7 ,rot_id=5+1=6 -> cvt_last_num=7-6=1
    cvt_last_num=portion_last_num
    for i in range(cvt_last_num):
        if ac_dim == 3:
            action_list[i + num - cvt_last_num][2] *= (cvt_last_num - 1 - i) / cvt_last_num
        else:
            for kk in range(6):
                action_list[i + num - cvt_last_num][kk] *= (cvt_last_num - 1 - i) / cvt_last_num
    return action_list,need_portion

def _add_end_episode(add_num,disturb_abs_rot,abs_rot_list,act_lst,pose_list):
    act = act_lst[-1].copy()

    abs_rot_list=None if (isinstance(abs_rot_list,list) and len(abs_rot_list) == 0) else abs_rot_list
    abs_rot = abs_rot_list[-1].copy() if abs_rot_list is not None else None

    pose_list=None if (isinstance(pose_list,list) and len(pose_list) == 0) else pose_list
    pose = pose_list[-1].copy() if pose_list is not None else None

    disturb_abs_rot=False if abs_rot_list is None else disturb_abs_rot
    for i in range(add_num):
        act_lst.append(act)

        if pose_list is not None:
            pose_list.append(pose)
        if disturb_abs_rot:
            abs_rot_list.append(random.uniform(1.0, 10.0))
        else:
            if abs_rot_list is not None:
                abs_rot_list.append(abs_rot)
    return abs_rot_list,act_lst,pose_list

def _add_medium_episode(act_lst, abs_rot_list, ac_dim, add_num,pose_list):
    abs_rot_list=None if (isinstance(abs_rot_list,list) and len(abs_rot_list) == 0) else abs_rot_list
    pose_list = None if (isinstance(pose_list, list) and len(pose_list) == 0) else pose_list

    need_add = True
    num = len(act_lst)
    rot_id = num
    trans_id=num
    for uu in range(num):
        rot = act_lst[uu][2:] if ac_dim == 3 else act_lst[uu][3:]
        rot_flag = (np.linalg.norm(rot) != 0)
        if rot_flag:
            rot_id = uu
            break
    if rot_id < 2 or rot_id >= num:  # 纯平移、几乎纯旋转
        need_add = False
    else:
        row_1 = num + add_num
        trans_id = rot_id - 2
        trans = np.array(act_lst[trans_id].copy()[0:2]) if ac_dim == 3 else np.array(
            act_lst[trans_id].copy()[0:3])
        trans *= 0.5
        #abs rot
        if abs_rot_list is not None:
            abs_rot = np.array(abs_rot_list[trans_id + 1].copy())
            abs_rot_list = np.concatenate(
                (abs_rot_list[0:rot_id].copy(), np.repeat(abs_rot.copy(), add_num, axis=0),
                 abs_rot_list[rot_id:].copy()))
            assert abs_rot_list.shape[0] == row_1
        #delta pose
        if pose_list is not None:
            pose = np.array(pose_list[trans_id + 1].copy())
            pose_list = np.concatenate(
                (pose_list[0:rot_id].copy(), np.repeat(pose.copy(), add_num, axis=0),
                 pose_list[rot_id:].copy()))
            assert pose_list.shape[0] == row_1
        #action
        act = np.concatenate((trans, np.array(rot)), axis=0)
        act_lst = np.concatenate(
            (act_lst[0:rot_id].copy(), np.repeat(act[np.newaxis, :].copy(), add_num, axis=0), act_lst[rot_id:].copy()))
        assert act_lst.shape[0] == row_1
    return act_lst, abs_rot_list,pose_list, need_add,trans_id, rot_id

def hdf_insert_images_for_medium(f, ep_key, dataset_name, trans_id, rot_id, add_num, row_1):
    if dataset_name in f[ep_key]:
        # 获取数据集的形状和dtype
        dset = f[f'{ep_key}/{dataset_name}']
        shape = dset.shape
        dtype = dset.dtype

        # 创建一个新的数据集来存储修改后的数据
        new_dset = f[ep_key].create_dataset(f'{dataset_name}_temp', shape=(row_1, *shape[1:]), dtype=dtype)

        # 将原数据集的数据复制到新数据集的相应位置
        new_dset[:rot_id] = dset[:rot_id].copy()
        new_dset[rot_id:rot_id+add_num] = np.repeat(dset[trans_id+1][np.newaxis, :].copy(), add_num, axis=0)
        new_dset[rot_id+add_num:] = dset[rot_id:row_1-add_num].copy()

        # 删除原数据集并重命名新数据集
        del f[ep_key][dataset_name]
        f[ep_key].move(f'{dataset_name}_temp', dataset_name)

def hdf_insert_images_for_end(f,ep_key,img_key):
    img = f['{}/'.format(ep_key)+img_key][-1].copy()
    lst = f['{}/'.format(ep_key)+img_key][:]
    lst = np.concatenate((lst.copy(), np.repeat(img[np.newaxis, :].copy(), add_num, axis=0)))
    del f[ep_key][img_key]
    f[ep_key].create_dataset(img_key, data=lst)

def insert_imgs(img_lst:np.array,insert_id,pick_id,insert_num):
    '''
    对img_lst在insert_id插入insert_num张img_lst[pick_id]
    '''
    num=len(img_lst)
    assert insert_id >=0 and insert_id < num
    assert pick_id >= 0 and pick_id < num

    img=img_lst[pick_id].copy()

    if insert_id==0:
        return np.concatenate((np.repeat(img[np.newaxis, :], insert_num, axis=0),img_lst))
    elif insert_id==num-1:
        return np.concatenate((img_lst,np.repeat(img[np.newaxis, :], insert_num, axis=0)))
    else:
        return np.concatenate((img_lst[:insert_id],np.repeat(img[np.newaxis, :], insert_num, axis=0), img_lst[insert_id:]))

if __name__ == '__main__':
    date='25.06.22'
    fn=os.path.join(return_disc_route('One Touch/AlignAnything_real'),date,'hdf5/mimic.hdf5')

    f = h5py.File(fn, "r+")
    for ep in f["data"]:
        print("processing {}".format(ep))

        im_key=f"data/{ep}/obs/robot0_eye_in_hand_image"
        img_lst=f[im_key][:].copy()
        length=len(img_lst)
        goal=img_lst[-1:].copy()
        goal=goal[:,:,:,::-1]
        img_lst = np.concatenate((img_lst[0:length-1],goal),axis=0)
        del f[im_key]
        f.create_dataset(im_key, data=img_lst)

        im_key2 = f"data/{ep}/obs/robot0_eye_in_hand_image_2"
        img_lst2 = f[im_key2][:].copy()
        length2 = len(img_lst2)
        goal2 = img_lst2[-1:].copy()
        goal2 = goal2[:, :, :, ::-1]
        img_lst2 = np.concatenate((img_lst2[0:length2 - 1], goal2), axis=0)
        del f[im_key2]
        f.create_dataset(im_key2, data=img_lst2)

    f.close()
    compute_num_samples(fn)
    split_train_val_from_hdf5(fn)