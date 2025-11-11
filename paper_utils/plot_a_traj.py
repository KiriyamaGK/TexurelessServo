import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.font_manager import FontProperties
from utils.transform import normalize_rotvec

# 设置中文字体
font_pth = "/usr/share/fonts/truetype/custom/SimHei.ttf"
font = FontProperties(fname=font_pth)

def get_tar(traj_data, real_base_dir=None, npy_file=None):
    """
    计算轨迹数据的误差
    """
    if not isinstance(traj_data, (list, np.ndarray)):
        return None

    # 获取最后一个元素
    last_element = traj_data[-1]

    # 检查最后一个元素是否是字典且包含wgT_tar
    wgT_file = npy_file.split('_')[0] + "_wgT_tar.npy"
    if os.path.exists(os.path.join(real_base_dir, wgT_file)):
        wgT_tar = np.load(os.path.join(real_base_dir, wgT_file), allow_pickle=True)
    else:
        # 如果不满足条件，设置wgT_tar为单位阵，T为最后一个元素
        print("simulation env target pos was not saved,using predefined...")
        wgT_tar = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00], [-0.00000000e+00, -1.00000000e+00 , 1.22464680e-16, -2.32682892e-16], [-0.00000000e+00, -1.22464680e-16 ,-1.00000000e+00 , 1.90000000e+00], [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    return wgT_tar


def transform_to_6d_pose(T):
    """
    将4x4齐次变换矩阵转换为6维位姿 [x, y, z, rx, ry, rz]
    其中旋转部分转换为轴角表示（单位：度）
    """
    # 平移部分
    translation = T[:3, 3]

    # 旋转部分转换为轴角（单位：度）
    rotation_matrix = T[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    axis_angle = rotation.as_rotvec()
    axis_angle_deg = axis_angle * 180 / np.pi

    return np.concatenate([translation, axis_angle_deg])


def plot_trajectory(traj_dir, real_base_dir, npy_file,is_sim,save_dir_name):
    """
    绘制轨迹的6维位姿图
    """
    try:
        # 加载轨迹数据
        traj_data = np.load(traj_dir, allow_pickle=True)
        print(f"轨迹数据形状: {len(traj_data)} 个变换矩阵")

        # 获取目标位姿
        wgT_tar = get_tar(traj_data, real_base_dir=real_base_dir, npy_file=npy_file)
        target_pose = transform_to_6d_pose(wgT_tar)
        if is_sim:
            target_pose[0:3] *= 1000
        if target_pose[3] > 0:
            ori_norm = np.linalg.norm(target_pose[3:6])
            target_pose[3:6] *= -(360 - ori_norm) / ori_norm

        # 转换所有轨迹点到6维位姿
        poses = []
        i = 0
        for T in traj_data:
            i+=1
            if isinstance(T, np.ndarray) and T.shape == (4, 4):
                pose = transform_to_6d_pose(T)
                if is_sim:
                    pose[0:3] *= 1000
                if pose[3]>0:
                    ori_norm = np.linalg.norm(pose[3:6])
                    pose[3:6]*= -(360 - ori_norm)/ori_norm
                poses.append(pose)

        poses = np.array(poses)
        print(f"转换后的位姿数据形状: {poses.shape}")

        # 创建6个子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # 坐标轴标签
        labels = ['X (mm)', 'Y (mm)', 'Z (mm)', 'theta1 (°)', 'theta2 (°)', 'theta3 (°)']

        # 绘制每个维度的轨迹
        for i in range(6):
            ax = axes[i]
            time_steps = np.arange(len(poses))

            # 绘制轨迹
            ax.plot(time_steps, poses[:, i], 'b-', linewidth=2, label='实际轨迹')

            # 绘制目标位置虚线
            ax.axhline(y=target_pose[i], color='r', linestyle='--', linewidth=2, label='目标位置')

            ax.set_xlabel('时间步', fontproperties=font, fontsize=12)
            ax.set_ylabel(labels[i], fontproperties=font, fontsize=12)
            ax.set_title(f'{labels[i]} 轨迹', fontproperties=font, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(prop=font)

        plt.tight_layout()

        # 保存图像
        os.makedirs(save_dir_name, exist_ok=True)
        output_file = os.path.join(save_dir_name,'trajectory_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"轨迹图已保存到: {output_file}")

        # 显示图像
        plt.show()

        return True

    except Exception as e:
        print(f"绘制轨迹时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # 配置参数
    base_dir = "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/好的结果/2025-11-08_00-00-00"
    # part_idx = 5
    part_idx = 1
    save_dir_name = "real_with_dagger"
    sub_dir = "2025-11-09_21-37-11(epoch599)"
    timestamp = "1762700975"

    is_sim = False

    # 构建文件路径
    real_base_dir = os.path.join(base_dir, sub_dir, str(part_idx), "traj")
    npy_file = timestamp + "_traj.npy"
    traj_dir = os.path.join(real_base_dir, npy_file)

    print(f"轨迹文件: {traj_dir}")

    # 检查文件是否存在
    if not os.path.exists(traj_dir):
        print(f"轨迹文件不存在: {traj_dir}")
    else:
        # 绘制轨迹
        plot_trajectory(traj_dir, real_base_dir, npy_file,is_sim,save_dir_name)

