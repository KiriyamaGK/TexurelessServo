import pyrealsense2 as rs
import os
import random
import json
import numpy as np
from math import sin, cos, pi
from scipy.spatial.transform import Rotation as R
import cv2
from utils.input_process import clip_image

base_dir= "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/goal_image"
goal_img_num=2000
bgr2rgb = True


img_1 = cv2.imread(os.path.join(base_dir,f"img_1.jpg"))
img_1=clip_image(img_1,300,keep_right=True)
img_1_badpos = cv2.imread(os.path.join(base_dir+"s","img1",f"{1999}.png"))#/255
img_1_badpos = cv2.resize(img_1_badpos,(300,300))

img_2=cv2.imread(os.path.join(base_dir,f"img_2.jpg"))#/255
img_2=clip_image(img_2,300,keep_right=True)
img_2_badpos = cv2.imread(os.path.join(base_dir+"s","img2",f"{0}.png"))#/255
img_2_badpos = cv2.resize(img_2_badpos,(300,300))


img_1_combined=(0.5*img_1+0.5*img_1_badpos).astype(np.uint8)
img_2_combined=(0.5*img_2+0.5*img_2_badpos).astype(np.uint8)

cv2.imshow("img_1_combined",img_1_combined)
cv2.imshow("img_2_combined",img_2_combined)
cv2.waitKey(0)