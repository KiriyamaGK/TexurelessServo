import numpy as np
from scipy.spatial.transform import Rotation


def geodesic_distance_rad(R1, R2):
    """计算两个旋转矩阵之间的测地线距离（弧度）"""
    R_rel = R1.T @ R2
    theta = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
    return theta


def max_euler_axis_angle_geodesic_error():
    max_error_rad = 0
    for rx in np.linspace(-8, 8, 100):  # 角度制采样
        for ry in np.linspace(-8, 8, 100):
            for rz in np.linspace(-40, 40, 100):
                # 转换为弧度制
                euler_angles_deg = np.array([rx, ry, rz])
                euler_angles_rad = np.radians(euler_angles_deg)

                #欧拉角对应的旋转矩阵（使用弧度制）
                R_original = Rotation.from_euler('xyz', euler_angles_rad).as_matrix()

                # 轴角旋转矩阵（弧度制）
                R_recovered = Rotation.from_rotvec(euler_angles_rad).as_matrix()

                # 计算测地线距离（弧度）
                error_rad = geodesic_distance_rad(R_original, R_recovered)

                if error_rad > max_error_rad:
                    max_error_rad = error_rad

    return np.degrees(max_error_rad)  # 最终结果转换为角度制显示


max_error_deg = max_euler_axis_angle_geodesic_error()
print(f"Max geodesic error: {max_error_deg:.6f} degrees")