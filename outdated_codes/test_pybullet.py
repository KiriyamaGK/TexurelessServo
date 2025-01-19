import cv2
import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation as R

width=240
height=240
fov = 60
aspect = width / height
near = 0.02
far = 1
obj_scale_factor=0.001
table_scale_factor=3


physicsClient = p.connect(p.GUI)  # 或者使用p.DIRECT来非可视化地连接

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
a=pybullet_data.getDataPath()

planeId = p.loadURDF("plane.urdf")
ObjFileName = "meshes/objs/4.urdf"
TableFileName = "meshes/table/table.urdf"
GripFileName = "meshes/gripper/custom_wsg50_with_r2d2_gripper.sdf"


objId = p.loadURDF(ObjFileName,globalScaling=obj_scale_factor,useFixedBase=True)
tableId = p.loadURDF(TableFileName,globalScaling=table_scale_factor,useFixedBase=True)
gripId = p.loadSDF(GripFileName,globalScaling=1)

objStartPos = [0, 0, 1.88]
objStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
p.resetBasePositionAndOrientation(objId, objStartPos, objStartOrientation)

gripStartPos = [0, 0, 2.3]
gripStartOrientation = p.getQuaternionFromEuler([np.pi, 0, 0])
p.resetBasePositionAndOrientation(gripId[0], gripStartPos, gripStartOrientation)

projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

cwT=np.array([[-1,0,0,0],  #外参，左上角c右下角w，前三列是c系在w系中的表示,按列排列，最后一列是从c的原点指向w的原点并在c系中表示
              [0,1,0,-0.02],
              [0,0,-1,2.2],
              [0,0,0,1]])
wcT=np.linalg.inv(cwT)
dTx=np.eye(4)

#
# images = p.getCameraImage(width,
#                               height,
#                               viewMatrix=view_mat,                  #float tuple
#                               projectionMatrix=projection_matrix,   #float tuple
#                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
#                               )
# # cv2.imshow('img',np.ascontiguousarray(images[2][:, :, :3]))
# cv2.waitKey(0)

# p.resetDebugVisualizerCamera(cameraDistance=2.7, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])

origin = [0, 0, 0]
xAxis = [1, 0, 0]
yAxis = [0, 1, 0]
zAxis = [0, 0, 1]
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=10, cameraPitch=-45, cameraTargetPosition=objStartPos)
p.addUserDebugLine(origin, xAxis, lineColorRGB=[1, 0, 0], lineWidth=2, lifeTime=0)
p.addUserDebugLine(origin, yAxis, lineColorRGB=[0, 1, 0], lineWidth=2, lifeTime=0)
p.addUserDebugLine(origin, zAxis, lineColorRGB=[0, 0, 1], lineWidth=2, lifeTime=0)
i=0
# 进入仿真循环
while (1):
    # gripStartPos[2]+=-0.01
    # p.resetBasePositionAndOrientation(gripId[0], gripStartPos, gripStartOrientation)
    if i==0:
        dTx[0:3, 0:3] = R.from_rotvec(np.array([1, 0, 0]) * 4 / 180 * np.pi).as_matrix()
        wcT = wcT @ dTx
        cwT = np.linalg.inv(wcT)
        print(cwT)

    view_mat = cwT.copy()
    view_mat[2, :] *= -1
    view_mat = view_mat.flatten(order='F')
    images = p.getCameraImage(width,
                              height,
                              viewMatrix=view_mat,                  #float tuple
                              projectionMatrix=projection_matrix,   #float tuple
                              renderer=p.ER_BULLET_HARDWARE_OPENGL,
                              )
    cam_pos = wcT[0:3, 3]
    cam_x = cwT[0:3, 0]
    cam_y = cwT[0:3, 1]
    cam_z = cwT[0:3, 2]
    # 计算线条的终点位置
    cam_x_end = cam_pos + cam_x
    cam_y_end = cam_pos + cam_y
    cam_z_end = cam_pos + cam_z

    p.addUserDebugLine(cam_pos, cam_x_end, lineColorRGB=[1, 0, 0], lineWidth=1, lifeTime=0)
    p.addUserDebugLine(cam_pos, cam_y_end, lineColorRGB=[0, 1, 0], lineWidth=1, lifeTime=0)
    p.addUserDebugLine(cam_pos, cam_z_end, lineColorRGB=[0, 0, 1], lineWidth=1, lifeTime=0)
    p.stepSimulation()
    i+=1
    time.sleep(1/10)
  # 这里可以添加代码来控制机器人或进行其他仿真操作