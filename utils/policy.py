import numpy as np
from utils.transform import rotation_matrix_z
from scipy.spatial.transform import Rotation as R


def get_cur_goal_deltapos(wgT,wgT_tar,need_trans_unit_transform=True):
    '''
    mm,deg
    '''
    del_T = np.linalg.inv(wgT) @ wgT_tar
    delta_pose = np.zeros(6)  # current2goal
    if need_trans_unit_transform:
        delta_pose[0:3] = del_T[0:3, 3] * 1000  # mm
    else:
        delta_pose[0:3] = del_T[0:3, 3]         # mm
    delta_pose[3:] = R.from_matrix(del_T[:3, :3]).as_rotvec() / np.pi * 180  # deg
    return {"del_T": del_T, "delta_pose": delta_pose}

def get_expert_policy(wgT_tar,wgT,trans_vel,rot_vel,uniform_vel,dist_eps,angle_eps,motion_type,dof,need_trans_unit_transform=True,fine_print=False,real=False):
    if uniform_vel["utilized"]:
        assert (not trans_vel["utilized"]) and (not rot_vel["utilized"])
    else:
        assert trans_vel["utilized"] and rot_vel["utilized"]
    trans_vel_norm = trans_vel["value"] if trans_vel["utilized"] else uniform_vel["value"]
    rot_vel_norm = rot_vel["value"] if rot_vel["utilized"] else uniform_vel["value"]

    rtn_dict = get_cur_goal_deltapos(wgT,wgT_tar,need_trans_unit_transform=need_trans_unit_transform)
    del_T,delta_pose = rtn_dict["del_T"],rtn_dict["delta_pose"]

    dT = np.eye(4)

    if dof==3: #dT用于左乘
        assert isinstance(trans_vel_norm,(int,float))
        del_tr = wgT_tar[0:2, 3].copy() - wgT[0:2, 3].copy()
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
                vel_tr = np.array([0, 0])   #m
            if del_rot_mat[0, 1] > 0:
                vel_rot = rot_vel_norm
            elif del_rot_mat[0, 1] < 0:
                vel_rot = -rot_vel_norm     #degree
            else:
                vel_rot = 0
            dT[0:2, 3] = vel_tr
            dT[0:3, 0:3] = rotation_matrix_z(vel_rot / 180 * np.pi)
        vel_rot = np.array([vel_rot])

    else:  #dT用于右乘
        w = R.from_matrix(del_T[:3, :3].copy()).as_rotvec()
        v = del_T[:3, 3].copy()  # 以上v,w的计算等价于，先求g_gtarT,对前三列旋转矩阵直接求轴角计算w,最后一列直接作为v
        abs_del_tr = np.linalg.norm(v) #m/mm
        # print("==================================")
        abs_del_angle = np.linalg.norm(w) / np.pi * 180  # degree
        # if fine_print:
        #     print(f"policy abs_del_tr:{abs_del_tr}, abs_del_angle:{abs_del_angle}")
        if not uniform_vel["utilized"]:
            if isinstance(trans_vel_norm, (int, float)):
                vel_tr = v /  np.linalg.norm(v) * trans_vel_norm if trans_vel_norm < abs_del_tr else v #m/mm
            else:
                assert isinstance(trans_vel_norm, list)
                abs_del_tr_xy = np.linalg.norm(del_T[:2, 3].copy())
                abs_del_tr_z = abs(del_T[2, 3].copy())
                # print("================================")
                # print("abs_del_tr_xy", abs_del_tr_xy)
                # print("abs_del_tr_z", abs_del_tr_z)
                if  trans_vel_norm[0] >= abs_del_tr_xy:
                    if 0.1 * trans_vel_norm[0] < abs_del_tr_xy:
                        vel_tr_xy = v[0:2]
                    else:
                        vel_tr_xy = np.array([0, 0])
                else:
                    vel_tr_xy=v[0:2] / np.linalg.norm(v[0:2]) * trans_vel_norm[0]

                if  trans_vel_norm[1] >= abs_del_tr_z:
                    if 0.1 * trans_vel_norm[1] < abs_del_tr_z:
                        vel_tr_z = v[2:]
                    else:
                        vel_tr_z = np.array([0])
                else:
                    vel_tr_z=v[2:] / np.linalg.norm(v[2:]) * trans_vel_norm[1]
                # vel_tr_xy = v[0:2] / np.linalg.norm(v[0:2]) * trans_vel_norm[0] if trans_vel_norm[0] < abs_del_tr_xy else v[0:2]  # m
                # vel_tr_z = v[2:] / np.linalg.norm(v[2:]) * trans_vel_norm[1] if trans_vel_norm[1] < abs_del_tr_z else v[2:]  # m
                # print("vel_tr_xy", vel_tr_xy)
                # if vel_tr_xy[0]!=0:
                    # print(vel_tr_xy[1]/vel_tr_xy[0])
                vel_tr=np.concatenate((vel_tr_xy,vel_tr_z),axis=0)
            if fine_print:
                print(f"rot_vel_norm:{rot_vel_norm},abs_del_angle:{abs_del_angle}")

            if not real:
                if rot_vel_norm >= abs_del_angle:
                    if not 0.5*rot_vel_norm >= abs_del_angle:
                        vel_rot = w/ np.pi * 180
                    else:
                        vel_rot=np.array([0, 0,0])
                else:
                    vel_rot = w / np.linalg.norm(w) * rot_vel_norm

            else:
                # vel_rot = np.array([0, 0, 0])
                if abs_del_angle<min(0.8*angle_eps,rot_vel_norm):
                    vel_rot = np.array([0, 0, 0])
                    if fine_print:
                        print(1)
                else:
                    if abs_del_angle>max(angle_eps,rot_vel_norm):
                        vel_rot = w / np.linalg.norm(w) * rot_vel_norm
                        if fine_print:
                            print(2)
                    else:
                        vel_rot = w / np.linalg.norm(w) * min(rot_vel_norm,0.5*angle_eps)
                        if fine_print:
                            print(3)


        else:
            raise RuntimeError("UNIFORM VELOCITY SHOULD NOT BE UTILIZED")
            # print("distance:", dis)
        # print("dis_tr:",abs_del_tr*1000)
        # print("dis_angle:", abs_del_angle)
        # print("vel_tr:", np.linalg.norm(vel_tr))
        # print("vel_rot:", np.linalg.norm(vel_rot))
        if motion_type != "simultaneously":
            vel_rot =np.array([0, 0, 0]) if abs_del_tr > dist_eps else vel_rot
        dT[0:3, 3] = vel_tr
        dT[0:3, 0:3] = R.from_rotvec(vel_rot/180*np.pi).as_matrix()
    return {
        "dT":dT,
        "vel_rot":vel_rot,  #deg
        "vel_tr":vel_tr,    #m/mm
        "cur_goal_delta_pose":delta_pose #mm,degree
    }