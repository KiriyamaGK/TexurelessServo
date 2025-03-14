import time

import cv2
import h5py
import os
import random
from utils.file import ensure_dir
from utils.paths import return_disc_route

def tuple_to_int_list(itm):
    '''

    :param itm: tuple  eg.:(1,3)
    :return:list eg.:[1,2,3]
    '''
    assert isinstance(itm, tuple) and len(itm) == 2
    output=[]
    for i in range(itm[1]-itm[0]+1):
        if i ==0:
            output.append(itm[0])
        else:
            output.append(itm[0]+i)
    return output

def create_part_type_dict(idx_list):
    '''

    :param idx_list: eg.:[(0,50),[51,52],[100]],tuple代表闭区间，list代表一个数
    :return:
        dict:eg.:{"part_0":[0,1,2],
        "part_1":[4,5,6]}
    '''
    part_type_dict = {}
    for idx,itm in enumerate(idx_list):
        if isinstance(itm,tuple):
            part_type_dict["part_{}".format(idx)] = tuple_to_int_list(itm)
        elif isinstance(itm,list):
            part_type_dict["part_{}".format(idx)] = itm
        else:
            raise RuntimeError("elements in idx_list must be tuple or list")
    return part_type_dict


if __name__ == '__main__':
    date_name = "25.03.11"
    num_parts=8
    num_pic_per_part=200
    max_demos_per_part=10
    idx_list=[(0,149),(150,269),(270,389),(390,509),(510,629),(630,748),(749,868),(869,987)] #tuple代表闭区间，list代表一个数

    assert len(idx_list)==num_parts

    part_type_dict = create_part_type_dict(idx_list) #list(list)

    base_dir = return_disc_route("One Touch")
    hdf_path=os.path.join(base_dir, 'AlignAnything_real', date_name, 'hdf5/mimic.hdf5')
    img_base_dir = os.path.join(base_dir, 'AlignAnything_real', date_name, 'cycle_gan')
    new_f_out = h5py.File(hdf_path, 'r')

    for part_idx,demo_itms in part_type_dict.items():
        if len(demo_itms)>max_demos_per_part:
            demo_itms = demo_itms[:max_demos_per_part]
        part_idx=int(part_idx[5:])
        print("processing part {}......".format(part_idx+1))
        part_name='part_{}'.format(part_idx+1)
        demo_name="demo_{}".format(part_idx)

        part_base_dir = os.path.join(img_base_dir, part_name)
        trainA_dir = os.path.join(part_base_dir, "trainA")
        # testA_dir = os.path.join(part_base_dir, "testA")
        ensure_dir(trainA_dir)
        # ensure_dir(testA_dir)

        #得到余数和商，用于分配每个demo的图片数
        quo=num_pic_per_part//len(demo_itms)
        rem=num_pic_per_part%len(demo_itms)

        img_idx = 0

        for idx,demo_itm in enumerate(demo_itms):
            num_pic=quo+1 if idx<=rem-1 else quo
            num_epi=len(new_f_out["data/demo_{}/obs/robot0_eye_in_hand_image".format(demo_itm)])
            interval=num_epi//num_pic
            for i in range(num_pic):
                img=new_f_out['data/demo_{}/obs/robot0_eye_in_hand_image'.format(demo_itm)][i*interval-1][:,:,::-1]#bgr
                img_save_dir=os.path.join(trainA_dir, "{}.png".format(img_idx))
                cv2.imwrite(img_save_dir,img)
                img_idx+=1
                # print(img_idx)










