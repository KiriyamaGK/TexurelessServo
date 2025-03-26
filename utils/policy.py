import numpy as np
from utils.transform import rotation_matrix_z
from scipy.spatial.transform import Rotation as R

def get_expert_policy(wgT_tar,wgT,trans_vel_norm,rot_vel_norm,dist_eps,angle_eps,motion_type,dof):
    dT=np.eye(4)
    if dof==3: #dT用于左乘
        del_tr = wgT_tar[0:2, 3] - wgT[0:2, 3]
        abs_del_tr = np.linalg.norm(del_tr)
        if motion_type != "simultaneously":
            if abs_del_tr > dist_eps:
                vel_tr = del_tr / abs_del_tr * trans_vel_norm
                dT[0:2, 3] = vel_tr
                vel_rot = 0
            else:
                vel_tr = np.array([0, 0])
                del_rot_mat = np.linalg.inv(wgT[0:3, 0:3]) @ wgT_tar[0:3, 0:3]  # 绕夹爪自己的轴
                # abs_del_rot=abs(asin(del_rot_mat[0,1]))
                if del_rot_mat[0, 1] > 0:
                    vel_rot = rot_vel_norm
                elif del_rot_mat[0, 1] < 0:
                    vel_rot = -rot_vel_norm
                else:
                    vel_rot = 0
                dT[0:3, 0:3] = rotation_matrix_z(vel_rot / 180 * np.pi)
        else:
            del_rot_mat = np.linalg.inv(wgT[0:3, 0:3]) @ wgT_tar[0:3, 0:3]  # 绕夹爪自己的轴
            if abs_del_tr > dist_eps:
                vel_tr = del_tr / abs_del_tr * trans_vel_norm
            else:
                vel_tr = np.array([0, 0])
            if del_rot_mat[0, 1] > 0:
                vel_rot = rot_vel_norm
            elif del_rot_mat[0, 1] < 0:
                vel_rot = -rot_vel_norm
            else:
                vel_rot = 0
            dT[0:2, 3] = vel_tr
            dT[0:3, 0:3] = rotation_matrix_z(vel_rot / 180 * np.pi)
        vel_rot = np.array([vel_rot])

    else:  #dT用于右乘
        del_T = np.linalg.inv(wgT) @ wgT_tar
        w = R.from_matrix(del_T[:3, :3]).as_rotvec()
        v = del_T[:3, 3]  # 以上v,w的计算等价于，先求g_gtarT,对前三列旋转矩阵直接求轴角计算w,最后一列直接作为v
        abs_del_tr = np.linalg.norm(v) #mm
        abs_del_angle = np.linalg.norm(w) / np.pi * 180  # degree
        vel_tr = v / np.linalg.norm(v) * trans_vel_norm if abs_del_tr > dist_eps else np.array([0, 0, 0])

        if motion_type == "simultaneously":
            vel_rot=w/np.linalg.norm(w)*rot_vel_norm if abs_del_angle>angle_eps else np.array([0, 0, 0])
        else:
            vel_rot = w / np.linalg.norm(w) * rot_vel_norm if abs_del_tr <= dist_eps and abs_del_angle > angle_eps else np.array([0, 0, 0])
        dT[0:3, 3] = vel_tr
        dT[0:3, 0:3] = R.from_rotvec(vel_rot/180*np.pi).as_matrix()
    return {
        "dT":dT,
        "vel_rot":vel_rot,
        "vel_tr":vel_tr
    }