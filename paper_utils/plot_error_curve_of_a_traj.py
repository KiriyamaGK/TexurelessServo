import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.font_manager import FontProperties


def plot_error_curves(trans_dir, rot_dir, save_dir_name):
    """
    绘制平移和旋转误差曲线
    """
    try:
        # 加载数据
        trans_errors = np.load(trans_dir)  # 平移误差，单位mm
        rot_errors = np.load(rot_dir)  # 旋转误差，单位°

        print(f"平移误差数据形状: {trans_errors.shape}")
        print(f"旋转误差数据形状: {rot_errors.shape}")

        # 设置中文字体
        font_pth = "/usr/share/fonts/truetype/custom/SimHei.ttf"
        font = FontProperties(fname=font_pth)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 绘制平移误差曲线（单位：mm）
        x_points = np.arange(len(trans_errors))
        ax1.plot(x_points, trans_errors, 'b-', linewidth=2)
        ax1.set_xlabel('时间步', fontproperties=font, fontsize=12)
        ax1.set_ylabel('平移误差 (mm)', fontproperties=font, fontsize=12)
        ax1.set_title('平移误差曲线', fontproperties=font, fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 绘制旋转误差曲线（单位：°）
        x_points_rot = np.arange(len(rot_errors))
        ax2.plot(x_points_rot, rot_errors, 'r-', linewidth=2)
        ax2.set_xlabel('时间步', fontproperties=font, fontsize=12)
        ax2.set_ylabel('旋转误差 (°)', fontproperties=font, fontsize=12)
        ax2.set_title('旋转误差曲线', fontproperties=font, fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        os.makedirs(save_dir_name, exist_ok=True)
        output_file = os.path.join(save_dir_name,'error_curves1.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"误差曲线图已保存到: {output_file}")

        # 显示图像
        plt.show()

        return True

    except Exception as e:
        print(f"绘制误差曲线时出错: {e}")
        return None


if __name__ == '__main__':
    # 配置参数
    base_dir = "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/好的结果/2025-11-08_00-00-00"
    # part_idx = 5
    part_idx = 1
    save_dir_name = "real_with_dagger"
    sub_dir = "2025-11-09_21-37-11(epoch599)"
    timestamp = "1762698393"
    # 构建文件路径
    real_base_dir = os.path.join(base_dir, sub_dir, str(part_idx), "error_curve")
    trans_dir = os.path.join(real_base_dir, timestamp + "_error_curve_trans.npy")
    rot_dir = os.path.join(real_base_dir, timestamp + "_error_curve_rot.npy")

    print(f"平移误差文件: {trans_dir}")
    print(f"旋转误差文件: {rot_dir}")

    # 检查文件是否存在
    if not os.path.exists(trans_dir):
        print(f"平移误差文件不存在: {trans_dir}")
    if not os.path.exists(rot_dir):
        print(f"旋转误差文件不存在: {rot_dir}")

    if os.path.exists(trans_dir) and os.path.exists(rot_dir):
        # 绘制误差曲线
        plot_error_curves(trans_dir, rot_dir, save_dir_name)
    else:
        print("无法找到误差曲线文件，请检查文件路径")
