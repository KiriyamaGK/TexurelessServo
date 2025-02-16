import os.path
import time
import matplotlib.pyplot as plt
import numpy as np
from utils.transform import rmat2euler_rz_degree
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_rot_and_trans(error_rot_lst,error_trans_lst,use_time=10,obj_pth=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(error_rot_lst, label='Rotation Error', marker='o', linestyle='-', color='blue')
    ax1.set_title('Rotation Error')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Error(°)')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(error_trans_lst, label='Translation Error', marker='x', linestyle='--', color='red')
    ax2.set_title('Translation Error')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error(mm)')
    ax2.grid(True)
    ax2.legend()

    final_rot_error = error_rot_lst[-1]
    final_trans_error = error_trans_lst[-1]

    ax1.annotate(f'Final Rot Error: {final_rot_error:.2f}(°)',
                 xy=(len(error_rot_lst) - 1, final_rot_error),
                 xytext=(len(error_rot_lst) - 1, final_rot_error + 0.05),
                 # arrowprops=dict(facecolor='blue', shrink=0.05),
                 color='blue')

    ax2.annotate(f'Final Trans Error: {final_trans_error:.2f}(mm)',
                 xy=(len(error_trans_lst) - 1, final_trans_error),
                 xytext=(len(error_trans_lst) - 1, final_trans_error + 0.05),
                 # arrowprops=dict(facecolor='red', shrink=0.05),
                 color='red')
    plt.annotate(f'Use Time: {use_time:.2f}(s)',
                 xy=(len(error_trans_lst) - 1, final_trans_error),
                 xytext=(len(error_trans_lst), 0),
                 # arrowprops=dict(facecolor='red', shrink=0.05),
                 color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(obj_pth,'{}.png'.format({int(time.time())})), dpi=300, bbox_inches='tight')
    plt.show()


def plot_trajs(wgT_list, wgT_tar, motion_type, obj_path=None):
    wgT_list = np.array(wgT_list)
    assert len(wgT_list.shape) == 3, "wgT_list must be a 3D array"

    xy_tar = wgT_tar[0:2, 3]
    xy_0 = wgT_list[0, 0:2, 3]
    rz_tar = rmat2euler_rz_degree(wgT_tar)
    rz_0 = rmat2euler_rz_degree(wgT_list[0])

    xys = []
    rzs = []

    for wgT in wgT_list:
        xy = wgT[0:2, 3]
        rz = rmat2euler_rz_degree(wgT)
        xys.append(xy)
        rzs.append(rz)

    xys = np.array(xys)
    rzs = np.array(rzs)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹
    ax.plot(xys[:, 0], xys[:, 1], rzs, label='Predicted Trajectory', color='blue', marker='o', markersize=4)

    if motion_type != 'simultaneously':
        ax.plot([xy_0[0], xy_tar[0]], [xy_0[1], xy_tar[1]], [rz_0, rz_0],
                label='Expert Trajectory', color='red', linestyle='--', marker='o', markersize=4)
        ax.plot([xy_tar[0], xy_tar[0]], [xy_tar[1], xy_tar[1]], [rz_0, rz_tar],
                color='red', linestyle='--', marker='o', markersize=4)
    else:
        ax.plot([xy_0[0], xy_tar[0]], [xy_0[1], xy_tar[1]], [rz_0, rz_tar],
                label='Expert Trajectory', color='red', linestyle='--', marker='o', markersize=4)

    ax.scatter(xy_0[0], xy_0[1], rz_0, color='green', label='Start', s=100, zorder=5)
    ax.scatter(xy_tar[0], xy_tar[1], rz_tar, color='red', label='Target', s=100, zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Rz (degrees)')
    ax.set_title('Trajectories')
    ax.legend()

    plt.savefig(os.path.join(obj_path, '{}.png'.format({int(time.time())})), dpi=300)
    plt.show()

def plot_vel(vel_tr,vel_rot,use_time,obj_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(vel_rot, label='Rotation Velocity', marker='o', linestyle='-', color='blue')
    ax1.set_title('Rotation Velocity')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('°')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(vel_tr, label='Translation Velocity', marker='x', linestyle='--', color='red')
    ax2.set_title('Translation Velocity')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('mm')
    ax2.grid(True)
    ax2.legend()


    plt.annotate(f'Use Time: {use_time:.2f}(s)',
                 xy=(len(vel_tr) - 1, 0),
                 xytext=(len(vel_tr), 0),
                 # arrowprops=dict(facecolor='red', shrink=0.05),
                 color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(obj_path,'{}.png'.format({int(time.time())})), dpi=300, bbox_inches='tight')
    plt.show()

def plot_time(time_list,save_base_pth):
    # 提取每个 obj_id 的数据
    obj_time_dict = {}
    for obj_id, use_time in time_list:
        if obj_id not in obj_time_dict:
            obj_time_dict[obj_id] = []
        obj_time_dict[obj_id].append(use_time)

    for obj_id, times in obj_time_dict.items():
        # 创建柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(times)), times, color='skyblue', edgecolor='black')
        plt.xlabel("Attempt Index")
        plt.ylabel("Use Time(s)")
        plt.title(f"Use Time for Object ID: {obj_id}")
        plt.xticks(range(len(times)))  # 设置 x 轴刻度

        save_path = os.path.join(save_base_pth, str(obj_id),"use_time.png")
        plt.savefig(save_path)
        plt.close()  # 关闭当前图表，避免内存泄漏

    print(f"All plots saved to {save_base_pth}")

# 示例数据
if __name__ == "__main__":
    # 示例 wgT_list 和 wgT_tar
    wgT_list = np.array([
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]],
        [[1, 0, 0, 2], [0, 1, 0, 2], [0, 0, 1, 0]],
        [[1, 0, 0, 3], [0, 1, 0, 3], [0, 0, 1, 0]],
    ])
    wgT_tar = np.array([[1, 0, 0, 4], [0, 1, 0, 4], [0, 0, 1, 0]])

    plot_trajs(wgT_list, wgT_tar, "example")
