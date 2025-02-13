import h5py
import numpy as np
import os
import random
from utils.hdf5 import split_train_val_from_hdf5

def insert_images(f, ep_key, dataset_name, trans_id, rot_id, add_num, row_1):
    if dataset_name in f[ep_key]:
        # 获取数据集的形状和dtype
        dset = f[f'{ep_key}/{dataset_name}']
        shape = dset.shape
        dtype = dset.dtype

        # 创建一个新的数据集来存储修改后的数据
        new_dset = f[ep_key].create_dataset(f'{dataset_name}_temp', shape=(row_1, *shape[1:]), dtype=dtype)

        # 将原数据集的数据复制到新数据集的相应位置
        new_dset[:rot_id] = dset[:rot_id]
        new_dset[rot_id:rot_id+add_num] = np.repeat(dset[trans_id+1][np.newaxis, :], add_num, axis=0)
        new_dset[rot_id+add_num:] = dset[rot_id:row_1-add_num]

        # 删除原数据集并重命名新数据集
        del f[ep_key][dataset_name]
        f[ep_key].move(f'{dataset_name}_temp', dataset_name)

if __name__ == '__main__':
    date='25.01.23'
    fn=os.path.join('/media/kiriyamagk/One Touch/AlignAnything',date,'hdf5/mimic.hdf5')

    disturb_abs_rot=True

    portion_last_episode=True
    portion_last_num=10

    add_end_episode=False
    add_end_num = 5

    add_medium_episode = True
    add_medium_num=2

    assert (not portion_last_episode) or (not add_end_episode)
    if disturb_abs_rot:               #机器人绕末端z轴旋转时，把记录的绝对欧拉角rz替换成随机噪声
        total_samples = 0
        f = h5py.File(fn, "r+")
        assert 'abs_rot' in f['data/demo_0/obs']
        for ep in f['data']:
            print("processing", ep)
            num = len(f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)])
            rot_id=num+1
            for uu in range(num):
                rot = f['data/{}/actions'.format(ep)][uu][2]
                if rot!=0:
                    rot_id=uu+1
                    break
            if rot_id <2  or rot_id > num:  #rot_id <2:纯旋转，rot_id > num：纯平移
                continue
            else:
                abs_rot_list = f['data/{}/obs/abs_rot'.format(ep)][0:rot_id].tolist()
                for i in range(rot_id,num):
                    abs_rot_list.append(random.uniform(0, 360.0))
                assert len(abs_rot_list)==num
                del f['data/{}/obs/abs_rot'.format(ep)]
                f['data/{}/obs/abs_rot'.format(ep)] = abs_rot_list

        f.close()

    if portion_last_episode:    #把最后portion_last_num个数据的action_rot按比例衰减
        f = h5py.File(fn, "r+")
        for id in range(len(f['data'])):
            print('processing demo ', id)
            action_list=f['data/demo_{}/actions'.format(id)][:]
            num=len(action_list)
            rot_id = num+1
            for uu in range(num):
                rot = f['data/{}/actions'.format(ep)][uu][2]
                if rot!=0:
                    rot_id=uu+1
                    break
            if rot_id > num:    #纯平移
                continue
            cvt_last_num=min(portion_last_num,num-rot_id)   #eg: 0 0 0 0 0 1 1 num=7 ,rot_id=5+1=6 -> cvt_last_num=1
            for i in range(cvt_last_num):
                action_list[i+num-cvt_last_num][2]*=(cvt_last_num-1-i)/cvt_last_num
            del f['data/demo_{}/actions'.format(id)]
            f['data/demo_{}/actions'.format(id)]=action_list
        f.close()

    if add_end_episode:  #把最后一帧数据复制add_end_num帧
        add_num=add_end_num
        total_samples = 0
        f = h5py.File(fn, "r+")
        for ep in f['data']:
            print("processing", ep)
            num = len(f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)])

            row_1 = num + add_num
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

            act = f['data/{}/actions'.format(ep)][-1].copy()

            if 'abs_rot' in f['data/{}/obs'.format(ep)]:
                abs_rot = f['data/{}/obs/abs_rot'.format(ep)][-1].copy()
                abs_rot_list=f['data/{}/obs/abs_rot'.format(ep)][:].tolist()
                for i in range(add_num):
                    if disturb_abs_rot:
                        abs_rot_list.append(random.uniform(1.0, 10.0))
                    else:
                        abs_rot_list.append(abs_rot)
                del f['data/{}/obs/abs_rot'.format(ep)]
                f['data/{}/obs/abs_rot'.format(ep)] = abs_rot_list

            act_lst=f['data/{}/actions'.format(ep)][:].tolist()

            for i in range(add_num):
                act_lst.append(act)

            del f['data/{}/actions'.format(ep)]
            f['data/{}/actions'.format(ep)] = act_lst

            ep_key = 'data/{}/obs'.format(ep)
            if 'robot0_eye_in_hand_image' in f[ep_key]:
                wrist_img = f['{}/robot0_eye_in_hand_image'.format(ep_key)][-1].copy()
                wrist_lst = f['{}/robot0_eye_in_hand_image'.format(ep_key)][:]
                wrist_lst = np.concatenate((wrist_lst, np.repeat(wrist_img[np.newaxis, :], add_num, axis=0)))
                del f[ep_key]['robot0_eye_in_hand_image']
                f[ep_key].create_dataset('robot0_eye_in_hand_image', data=wrist_lst)

            if 'robot0_eye_in_hand_image_goal' in f[ep_key]:
                wrist_img_goal = f['{}/robot0_eye_in_hand_image_goal'.format(ep_key)][-1].copy()
                wrist_goal_lst = f['{}/robot0_eye_in_hand_image_goal'.format(ep_key)][:]
                wrist_goal_lst = np.concatenate((wrist_goal_lst, np.repeat(wrist_img_goal[np.newaxis, :], add_num, axis=0)))
                del f[ep_key]['robot0_eye_in_hand_image_goal']
                f[ep_key].create_dataset('robot0_eye_in_hand_image_goal', data=wrist_goal_lst)

            if 'gaussian_img_kpt' in f[ep_key]:
                gauss_kpt = f['{}/gaussian_img_kpt'.format(ep_key)][-1].copy()
                gauss_kpt_lst = f['{}/gaussian_img_kpt'.format(ep_key)][:]
                gauss_kpt_lst = np.concatenate((gauss_kpt_lst, np.repeat(gauss_kpt[np.newaxis, :], add_num, axis=0)))
                del f[ep_key]['gaussian_img_kpt']
                f[ep_key].create_dataset('gaussian_img_kpt', data=gauss_kpt_lst)

            if 'gaussian_img_ct' in f[ep_key]:
                gauss_ct = f['{}/gaussian_img_ct'.format(ep_key)][-1].copy()
                gauss_ct_lst = f['{}/gaussian_img_ct'.format(ep_key)][:]
                gauss_ct_lst = np.concatenate((gauss_ct_lst, np.repeat(gauss_ct[np.newaxis, :], add_num, axis=0)))
                del f[ep_key]['gaussian_img_ct']
                f[ep_key].create_dataset('gaussian_img_ct', data=gauss_ct_lst)
        f.close()

    if add_medium_episode:         #增加过渡数据集add_medium_num帧，每帧action平移量、旋转量各取一半
        add_num=add_medium_num
        f = h5py.File(fn, "r+")
        for ep in f['data']:
            print("processing", ep)
            num = len(f['data/{}/obs/robot0_eye_in_hand_image'.format(ep)])
            rot_id=num
            for uu in range(num):
                rot = f['data/{}/actions'.format(ep)][uu][2]
                if rot!=0:
                    rot_id=uu
                    break
            if rot_id <2 or rot_id >= num: #纯平移、几乎纯旋转
                continue
            else:
                trans_id=rot_id-2
                trans=f['data/{}/actions'.format(ep)][trans_id].copy()[0:2]*0.5

                row_1 = num + add_num
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


                if 'abs_rot' in f['data/{}/obs'.format(ep)]:
                    abs_rot = f['data/{}/obs/abs_rot'.format(ep)][trans_id+1].copy()
                    abs_rot=np.array([abs_rot])
                    abs_rot_list = f['data/{}/obs/abs_rot'.format(ep)][:]
                    abs_rot_list=np.concatenate((abs_rot_list[0:rot_id], np.repeat(abs_rot, add_num, axis=0),abs_rot_list[rot_id:]))
                    assert abs_rot_list.shape[0] == row_1
                    del f['data/{}/obs/abs_rot'.format(ep)]
                    f['data/{}/obs/abs_rot'.format(ep)] = abs_rot_list


                act_lst = f['data/{}/actions'.format(ep)][:]
                act = np.array([trans[0], trans[1], rot])
                act_lst = np.concatenate(
                    (act_lst[0:rot_id], np.repeat(act[np.newaxis, :], add_num, axis=0), act_lst[rot_id:]))
                assert act_lst.shape[0] == row_1
                del f['data/{}/actions'.format(ep)]
                f['data/{}/actions'.format(ep)] = act_lst

                ep_key = 'data/{}/obs'.format(ep)
                if 'robot0_eye_in_hand_image' in f[ep_key]:
                    # wrist_img = f['{}/robot0_eye_in_hand_image'.format(ep_key)][trans_id+1].copy()
                    # wrist_lst = f['{}/robot0_eye_in_hand_image'.format(ep_key)][:]
                    # wrist_lst = np.concatenate((wrist_lst[0:rot_id], np.repeat(wrist_img[np.newaxis, :], add_num, axis=0),wrist_lst[rot_id:]))
                    # assert wrist_lst.shape[0] == row_1
                    # del f[ep_key]['robot0_eye_in_hand_image']
                    # f[ep_key].create_dataset('robot0_eye_in_hand_image', data=wrist_lst)
                    insert_images(f, ep_key, 'robot0_eye_in_hand_image', trans_id, rot_id, add_num, row_1)

                if 'robot0_eye_in_hand_image_goal' in f[ep_key]:
                    # wrist_img_goal = f['{}/robot0_eye_in_hand_image_goal'.format(ep_key)][trans_id+1].copy()
                    # wrist_goal_lst = f['{}/robot0_eye_in_hand_image_goal'.format(ep_key)][:]
                    # wrist_goal_lst = np.concatenate((wrist_goal_lst[0:rot_id], np.repeat(wrist_img_goal[np.newaxis, :], add_num, axis=0),wrist_goal_lst[rot_id:]))
                    # assert wrist_goal_lst.shape[0] == row_1
                    # del f[ep_key]['robot0_eye_in_hand_image_goal']
                    # f[ep_key].create_dataset('robot0_eye_in_hand_image_goal', data=wrist_goal_lst)
                    insert_images(f, ep_key, 'robot0_eye_in_hand_image_goal', trans_id, rot_id, add_num, row_1)
                if 'gaussian_img_kpt' in f[ep_key]:
                    insert_images(f, ep_key, 'gaussian_img_kpt', trans_id, rot_id, add_num, row_1)
                if 'gaussian_img_ct' in f[ep_key]:
                    insert_images(f, ep_key, 'gaussian_img_ct', trans_id, rot_id, add_num, row_1)
        f.close()

    total_samples = 0
    f = h5py.File(fn, "a")  # edit mode
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
    split_train_val_from_hdf5(fn)