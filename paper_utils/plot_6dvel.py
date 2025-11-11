import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.font_manager import FontProperties

# 设置中文字体
font_pth = "/usr/share/fonts/truetype/custom/SimHei.ttf"
font = FontProperties(fname=font_pth)


def plot_vel6d(vel6d_dir,save_dir_name):
    """
    绘制6D速度图
    """
    try:
        # 加载速度数据
        vel6d_data = np.load(vel6d_dir)
        print(f"6D速度数据形状: {vel6d_data.shape}")

        # 创建图形，上下两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 时间步
        time_steps = np.arange(len(vel6d_data))

        # 绘制前三维（线速度）
        ax1.plot(time_steps, vel6d_data[:, 0], 'r-', linewidth=2, label='X方向')
        ax1.plot(time_steps, vel6d_data[:, 1], 'g-', linewidth=2, label='Y方向')
        ax1.plot(time_steps, vel6d_data[:, 2], 'b-', linewidth=2, label='Z方向')
        ax1.set_xlabel('时间步', fontproperties=font, fontsize=12)
        ax1.set_ylabel('平移运动量 (mm)', fontproperties=font, fontsize=12)
        ax1.set_title('平移运动曲线', fontproperties=font, fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(prop=font)

        # 绘制后三维（角速度）
        ax2.plot(time_steps, vel6d_data[:, 3], 'r-', linewidth=2, label='theta_X方向')
        ax2.plot(time_steps, vel6d_data[:, 4], 'g-', linewidth=2, label='theta_X方向')
        ax2.plot(time_steps, vel6d_data[:, 5], 'b-', linewidth=2, label='theta_X方向')
        ax2.set_xlabel('时间步', fontproperties=font, fontsize=12)
        ax2.set_ylabel('旋转运动量 (°)', fontproperties=font, fontsize=12)
        ax2.set_title('旋转运动曲线', fontproperties=font, fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(prop=font)

        plt.tight_layout()

        # 保存图像
        os.makedirs(save_dir_name, exist_ok=True)
        output_file = os.path.join(save_dir_name,f'vel6d_plot1.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"6D速度图已保存到: {output_file}")

        # 显示图像
        plt.show()

        return True

    except Exception as e:
        print(f"绘制6D速度图时出错: {e}")
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
    timestamp = "1762699554"

    # 构建文件路径
    real_base_dir = os.path.join(base_dir, sub_dir, str(part_idx), "vel6d")
    npy_file = timestamp + "_vel6d.npy"
    vel6d_dir = os.path.join(real_base_dir, npy_file)

    print(f"轨迹文件: {vel6d_dir}")

    # 检查文件是否存在
    if not os.path.exists(vel6d_dir):
        print(f"vel6d文件不存在: {vel6d_dir}")
    else:
        # 绘制6D速度图
        plot_vel6d(vel6d_dir,save_dir_name)

