import pyrealsense2 as rs
import os
import random
import json
import numpy as np
from math import sin, cos, pi
from scipy.spatial.transform import Rotation as R
import cv2
from utils.input_process import clip_image

base_dir= "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/goal_image1"
bgr2rgb = True


img_1 = cv2.imread(os.path.join(base_dir,f"img_1.jpg"))
img_1_badpos = cv2.imread(os.path.join(base_dir,f"img_1_badpos.jpg"))

img_2=cv2.imread(os.path.join(base_dir,f"img_2.jpg"))#/255
img_2_badpos = cv2.imread(os.path.join(base_dir,f"img_2_badpos.jpg"))
# img_1 =img_1[:,:,::-1]
# img_2 =img_2[:,:,::-1]


img_1_combined=(0.5*img_1+0.5*img_1_badpos).astype(np.uint8)
img_2_combined=(0.5*img_2+0.5*img_2_badpos).astype(np.uint8)

cv2.imshow("img_1_combined",img_1_combined)
cv2.imshow("img_2_combined",img_2_combined)
# cv2.imshow("img_1",img_1)
# cv2.imshow("img_2",img_2)
# cv2.imshow("img1_badpos",img_1_badpos)
# cv2.imshow("img2_badpos",img_2_badpos)
cv2.waitKey(0)