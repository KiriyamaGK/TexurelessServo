import numpy as np
from utils.transform import rotation_matrix_z

def get_expert_policy(wgT_tar,wgT,trans_vel_norm,rot_vel_norm,dist_eps,angle_eps,motion_type):
    dT=np.eye(4)
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
    return {
        "dT":dT,
        "vel_rot":vel_rot,
        "vel_tr":vel_tr
    }