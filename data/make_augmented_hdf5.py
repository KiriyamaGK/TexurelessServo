import os
import time
from ultralytics import YOLO
import cv2
import numpy as np
import h5py
from utils.augmentation import AugmentationModule


if __name__ == '__main__':
    augmentation_module = AugmentationModule(
        pretrained_model_pth="/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train/weights/best.pt",
        scale_range_min=0.87,
        scale_range_max=1.15,
        offset_range_min=-0.1,
        offset_range_max=0.1,
        noise_std=0.07,
        draw_box=False,
        box_color=(0, 255, 0),
        box_thickness=2
    )
    dataset_path = "/media/kiriyamagk/One Touch/AlignAnything_real/25.06.23/hdf5/merged.hdf5"
    img_key_names = ["robot0_eye_in_hand_image", "robot0_eye_in_hand_image_2"]
    color_channel_inverse = True

    with h5py.File(dataset_path, "r+") as f:
        demo_num = len(f["data"])
        for i in range(demo_num):
            print(f"Processing demo {i}......")
            obs_paths = [f'data/demo_{i}/obs/' + n for n in img_key_names]
            for obs_path in obs_paths:
                imgs = f[obs_path][:]
                augmented_imgs = []
                for idx in range(imgs.shape[0]):
                    img = imgs[idx]
                    augmented_img = augmentation_module.augment_image(img, color_channel_inverse)
                    augmented_imgs.append(augmented_img)

                    # #debug
                    # cv2.imshow("img",augmented_img[:,:,::-1])
                    # cv2.waitKey(0)

                # Convert list to numpy array
                augmented_imgs = np.array(augmented_imgs)
                aug_path = obs_path + "_light"
                if aug_path in f:
                    del f[aug_path]
                f.create_dataset(aug_path, data=augmented_imgs)