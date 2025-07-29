import os
import random
import numpy as np
import h5py
import cv2

if __name__ == '__main__':
    dataset_path = "/media/kiriyamagk/One Touch/AlignAnything_real/25.06.23/hdf5/merged.hdf5"
    img_key_names = ["robot0_eye_in_hand_image","robot0_eye_in_hand_image_2"]
    tgt_dirname = "augmentation/raw_imgs"
    num_images = 60  #per img type in img_key_names
    color_channel_inverse = True
    select_probability = 0.014

    collected = 0
    mid_dir = os.path.dirname(dataset_path)
    tgt_dirs = [os.path.join(mid_dir, tgt_dirname,itm) for itm in img_key_names]
    for tgt_dir in tgt_dirs:
        os.makedirs(tgt_dir, exist_ok=True)
    with h5py.File(dataset_path, "r") as f:
        demo_num = len(f["data"])
        for i in range(demo_num):
            if collected >= num_images:
                break
            obs_paths = [f'data/demo_{i}/obs/' + n for n in img_key_names]
            if random.random() < select_probability:
                img_name = str(collected) + ".png"
                for img_idx , obs_path in enumerate(obs_paths):
                    imgs = f[obs_path][:]
                    demo_idx = random.randint(0, imgs.shape[0]-1)
                    img = imgs[demo_idx] if not color_channel_inverse else imgs[demo_idx][:,:,::-1].copy()
                    cv2.imwrite(tgt_dirs[img_idx] + "/" + img_name, img)
                collected += 1
                print(f"Demo {i} selected,collected {collected} imgs")
    print(f"Raw dataset completed,{collected} imgs in total.")

