import cv2
import numpy as np
import h5py
import Robot

robot=Robot.RPC("192.168.58.2")

# desire_pt=[-565.1048583984375, -236.511032104492, 150,-180,0, -173.2628784179687]
# robot.MoveCart(desire_pt, tool=1, user=0, vel=20)

ret,version= robot.GetSDKVersion()
if ret ==0:
    #查 询 SDK版 本 号
    print("SDK版 本 号 为", version )
else:
    print(" 查 询 失 败 ， 错 误 码 为 ",ret)