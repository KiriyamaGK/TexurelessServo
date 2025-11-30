from scipy.spatial.transform import Rotation as R
import numpy as np

def greedy_state_selection(states, student_actions,
                           expert_actions, num_selected_pts, w1=1.0, w2=1.0,s1=1.0,s2=1.0,a1=1.0,a2=1.0,action_error_threshold = 0.01,is_mm = False):
    """
    使用贪心策略选择需要咨询专家的状态

    Args:
        student_trajectory_states: 学生轨迹中的状态列表 [s1, s2, ..., sT] ,4*4 matrix (m/mm)
        student_trajectory_actions: 学生轨迹中的动作列表 [a1, a2, ..., aT] 6d (m/mm , deg)
        expert_policy: 专家策略函数，输入状态返回专家动作
        num_selected_pts: 需要选择的状态数量
        w1, w2: 误差和隔离度的权重参数

    Returns:
        selected_states: 选中的状态列表
    """
    # 计算每个状态的动作误差
    action_errors = []
    # =================only for printing=================
    state_6ds = []
    action_error_trs = []
    action_error_rots = []
    # =================only for printing=================
    # print("states_len:",len(states))
    # print("plc_act_len:",len(student_actions))
    # print("exp_act_len:", len(expert_actions))
    assert len(student_actions) == len(expert_actions)
    assert len(student_actions) == len(states)
    for i, state in enumerate(states):
        expert_action = expert_actions[i].astype(np.float64)
        expert_action[3:6] *= np.pi/180.

        student_action = student_actions[i].astype(np.float64)
        student_action[3:6] *= np.pi/180.

        error = determine_action_error(student_action, expert_action, a1=a1, a2=a2)
        action_errors.append(error)

        #=================only for printing=================
        state_6d = np.zeros(6)
        state_6d[0:3] = state[0:3,3].copy()
        if not is_mm:
            state_6d[0:3] *= 1000
        state_6d[3:6] = R.from_matrix(state[0:3,0:3].copy()).as_rotvec() /np.pi*180
        state_6ds.append(state_6d)

        error_tr = (student_action[0:3].copy() - expert_action[0:3].copy())
        if not is_mm:
            error_tr *= 1000
        error_rot = np.linalg.norm((R.from_rotvec(student_action[3:6]).inv() * R.from_rotvec(expert_action[3:6])).as_rotvec())/np.pi*180
        action_error_trs.append(error_tr)
        action_error_rots.append(error_rot)
        # =================only for printing=================

    action_errors = np.array(action_errors)

    # 贪心选择过程
    selected_states = []
    selected_indices = []
    selected_errors = []
    selected_distances = []
    print(f"[Greedy Selection] Greedy selection started.")

    for selection_round in range(min(num_selected_pts,len(states))):
        best_score = -np.inf
        best_idx = -1

        for i, state in enumerate(states):
            if i in selected_indices or action_errors[i] <= action_error_threshold:
                continue

            # 计算误差项
            error_term = w1 * action_errors[i]

            # 计算隔离度项
            isolation_term = 0
            if selected_states:
                # 计算当前状态与所有已选状态的最小距离
                distances = [determine_state_error(state,selected_state,s1=s1, s2=s2) for selected_state in selected_states]
                isolation_term = w2 * np.min(distances)
            else:
                isolation_term = w2 * 0  # 第一个状态给一个基础值

            # 综合得分
            score = error_term + isolation_term

            if score > best_score:
                best_score = score
                best_idx = i
                best_isolation_term = isolation_term

        if best_idx != -1:
            selected_states.append(states[best_idx])
            selected_indices.append(best_idx)
            selected_errors.append(action_errors[best_idx])
            selected_distances.append(best_isolation_term / w2 if w2 > 0 else 0)  # 还原实际距离

            print(f"Round {selection_round + 1}: selected state {best_idx}, "
                  f"err_act_tr(mm) = [{action_error_trs[best_idx][0]:.3f}, {action_error_trs[best_idx][1]:.3f}, {action_error_trs[best_idx][2]:.3f}], "
                  f"err_act_rot(deg) = {action_error_rots[best_idx]:.3f}, "
                  f"min_distance(mm,deg) = {selected_distances[-1]:.3f}, "
                  f"score = {best_score:.3f}, "
                  f"state_6d(mm,deg) = [{state_6ds[best_idx][0]:.1f}, {state_6ds[best_idx][1]:.1f}, {state_6ds[best_idx][2]:.1f}, {state_6ds[best_idx][3]:.1f}, {state_6ds[best_idx][4]:.1f}, {state_6ds[best_idx][5]:.1f}]")

    print(f"[Greedy Selection] Selected {len(selected_states)} in total.")
    return selected_states

def determine_mat_error(T1:np.ndarray,T2:np.ndarray,wtr,wrot):
    dR = np.linalg.inv(T1[0:3,0:3]) @ T2[0:3,0:3]
    rot_error = np.linalg.norm(R.from_matrix(dR).as_rotvec()) #rad
    trans_error = np.linalg.norm(T1[0:3,3] - T2[0:3,3]) #m/mm
    return wtr * trans_error + wrot * rot_error

def determine_state_error(T1:np.ndarray,T2:np.ndarray,s1,s2):
    return determine_mat_error(T1,T2,s1,s2)

def determine_action_error(arr1,arr2,a1,a2):
    T1 = np.eye(4)
    T2 = np.eye(4)
    T1[0:3,3] = arr1[0:3]
    T2[0:3,3] = arr2[0:3]
    T1[0:3,0:3] = R.from_rotvec(arr1[3:6]).as_matrix()
    T2[0:3,0:3] = R.from_rotvec(arr2[3:6]).as_matrix()
    return determine_mat_error(T1,T2,a1,a2)

