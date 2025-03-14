import random

import numpy as np
import cv2
from utils.paths import return_disc_route
from utils.file import ensure_dir
import os

if __name__ == '__main__':
    part_idx_list=[1,2,3,4,5,6,7,8]
    num_imgs_per_part=30
    base_dir = return_disc_route("One Touch")
    date_name="25.03.11"
    part_name="all_parts"

    img_base_dir = os.path.join(base_dir, 'AlignAnything_real', date_name, 'cycle_gan')
    part_base_dir = os.path.join(img_base_dir, part_name)
    dir_types=["trainA", "trainB", "testB"]
    for d in dir_types:
        ensure_dir(os.path.join(part_base_dir, d))

    for part_idx in part_idx_list:#all parts
        ori_part_dir = os.path.join(img_base_dir, "part_{}".format(part_idx))
        for dir_type in dir_types: #all types
            all_nms=os.listdir(ori_part_dir+"/"+dir_type)
            selected_nms = random.sample(all_nms, num_imgs_per_part)#select imgs
            idx=0
            for nm in selected_nms:
                im=cv2.imread(os.path.join(ori_part_dir,dir_type, nm))
                save_base= os.path.join(part_base_dir, dir_type)
                cv2.imwrite(os.path.join(save_base,"part_"+str(part_idx)+"_"+str(idx).zfill(3)+".png"), im)
                idx+=1



