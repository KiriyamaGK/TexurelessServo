import os

import cv2
import h5py
import numpy as np

def save_imglst(base_dir,img_name,img_lst,demo_key,bgr2rgb=True):
    if img_lst is not None:
        save_path = os.path.join(base_dir,demo_key,img_name)
        os.makedirs(save_path,exist_ok=True)
        print("Saving images in {}".format(save_path))
        assert isinstance(img_lst,(list,np.ndarray))
        idx = 0
        for img in img_lst:
            if bgr2rgb:
                img = img[:,:,::-1]
            cv2.imwrite(save_path + f"/{str(idx).zfill(3)}.png",img)
            idx += 1
        print("Images saved successfully.")

if __name__ == '__main__':
    use_date = "25.06.23"
    hdf_name = "merged.hdf5"
    _is_real_dataset = True
    load_type = ["default","aug"] # default/aug
    demo_idxs = [100] #list of integer or string "all"
    output_dir = "imgs"

    hdf_base = "H://AlignAnything_real" if _is_real_dataset else "H://AlignAnything"
    hdf_pth = os.path.join(hdf_base, use_date, "hdf5",hdf_name)
    f = h5py.File(hdf_pth,"r")
    assert isinstance(demo_idxs, (list, str))
    if isinstance(demo_idxs, str):
        assert demo_idxs == "all"
        demo_keys = f["data"].keys()
    else:
        demo_keys = ["demo_" + str(demo_idx) for demo_idx in demo_idxs]
    for demo_key in demo_keys:
        img1_lst = f["data"][demo_key]["obs/robot0_eye_in_hand_image"][:] if "default" in load_type else None
        img2_lst = f["data"][demo_key]["obs/robot0_eye_in_hand_image_2"][:] if "default" in load_type and "robot0_eye_in_hand_image_2" \
                                                                               in f["data"][demo_key]["obs"].keys() else None
        img1_aug_lst = f["data"][demo_key]["obs/robot0_eye_in_hand_image_light"][:] if "aug" in load_type else None
        img2_aug_lst = f["data"][demo_key]["obs/robot0_eye_in_hand_image_2_light"][:] if ("aug" in load_type and "robot0_eye_in_hand_image_2_light"
                                                                                          in f["data"][demo_key]["obs"].keys()) else None
        base_dir = os.path.join(os.getcwd(),output_dir)
        save_imglst(base_dir,"img1",img1_lst,demo_key,bgr2rgb=True)
        save_imglst(base_dir,"img2",img2_lst,demo_key,bgr2rgb=True)
        save_imglst(base_dir, "img1_aug", img1_aug_lst,demo_key, bgr2rgb=True)
        save_imglst(base_dir, "img2_aug", img2_aug_lst, demo_key,bgr2rgb=True)

