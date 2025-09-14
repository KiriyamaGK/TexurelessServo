import pyrealsense2 as rs
import os
import random
import json
import numpy as np
from math import sin, cos, pi
from scipy.spatial.transform import Rotation as R
import cv2

base_dir= "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/goal_images"
goal_img_num=2000
bgr2rgb = True

img_1_gt = cv2.imread(os.path.join(base_dir,"img1",f"{0}.png"))
img_2_gt = cv2.imread(os.path.join(base_dir,"img2",f"{0}.png"))#/255

for i in range(goal_img_num):
    if i>=1:
        img_1=cv2.imread(os.path.join(base_dir,"img1",f"{i}.png"))
        img_2=cv2.imread(os.path.join(base_dir,"img2",f"{i}.png"))#/255
        img_1=img_1[:,:,::-1]
        img_2=img_2[:,:,::-1]


        img_1_combined=(0.5*img_1+0.5*img_1_gt).astype(np.uint8)
        img_2_combined=(0.5*img_2+0.5*img_2_gt).astype(np.uint8)
        if bgr2rgb:
            img_1_combined=img_1_combined[:,:,::-1]
            img_2_combined=img_2_combined[:,:,::-1]
        all_imgs=np.hstack((img_1_combined,img_2_combined))
        # cv2.imshow("img_1_combined_DEMO_{}".format(i),img_1_combined)
        # cv2.imshow("img_2_combined_DEMO_{}".format(i),img_2_combined)
        cv2.imshow("img_combined".format(i),all_imgs)
        cv2.waitKey(100)
        # cv2.destroyAllWindows()