import numpy as np


def generate_cylinder_points(R, H, n):
    """
    在圆柱体内生成严格均匀的n个点。

    参数:
        R: 圆柱体半径
        H: 圆柱体高度
        n: 点数量

    返回:
        points: 形状为(n, 3)的数组，每行是(x, y, z)坐标
    """
    # 步骤1: 分解n为n_s, n_theta, n_z (简单分解，可优化)
    # 初始化参数
    n_theta = max(8, int(np.round(np.sqrt(n * 2 * np.pi * R / H))))  # 启发式选择n_theta
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            factors.append(n // i)
    factors = sorted(set(factors))  # 获取因数

    # 选择n_s和n_z，使n_s ≈ n_z
    n_remaining = n // n_theta
    if n_remaining == 0:
        n_theta = max(1, n_theta - 1)
        n_remaining = n // n_theta
    n_s = min(factors, key=lambda x: abs(x - np.sqrt(n_remaining)))
    n_z = n_remaining // n_s
    if n_s * n_theta * n_z != n:
        # 调整以确保n_s * n_theta * n_z = n
        n_s = factors[-1]
        n_z = n_remaining // n_s
        if n_s * n_z * n_theta != n:
            n_z = n_remaining // n_s
            n_theta = n // (n_s * n_z)  # 最后调整n_theta

    # 步骤2: 生成网格
    s = (np.arange(n_s) + 0.5) / n_s  # s坐标 (0.5/n_s 到 1-0.5/n_s)
    r = R * np.sqrt(s)  # 径向坐标
    theta = np.arange(n_theta) * (2 * np.pi / n_theta)  # 角向坐标
    z = (np.arange(n_z) + 0.5) * (H / n_z)  # 轴向坐标

    # 步骤3: 计算笛卡尔坐标
    points = []
    for k in range(n_z):
        for i in range(n_s):
            for j in range(n_theta):
                x = r[i] * np.cos(theta[j])
                y = r[i] * np.sin(theta[j])
                points.append([x, y, z[k]])

    return np.array(points)


# 使用示例
R = 1.0  # 半径
H = 2.0  # 高度
n = 100  # 点数
points = generate_cylinder_points(R, H, n)
print(f"Generated {len(points)} points.")
print("First 5 points:")
print(points[:5])