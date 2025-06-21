import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

def euler_to_rotvec(rx, ry, rz):
    # 转换为弧度
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)

    # 计算旋转矩阵
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    R = Rz @ Ry @ Rx

    # 计算旋转向量
    theta = np.arccos((np.trace(R) - 1) / 2)
    if np.isclose(theta, 0):
        return np.zeros(3)

    n = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(theta))

    return n * theta  # 旋转向量


# 生成样本
num_samples = 1000
rx_samples = np.random.uniform(-5, 5, num_samples)
ry_samples = np.random.uniform(-5, 5, num_samples)
rz_samples = np.random.uniform(-20, 20, num_samples)

# 计算旋转向量
rotvecs = np.array([euler_to_rotvec(rx, ry, rz)
                    for rx, ry, rz in zip(rx_samples, ry_samples, rz_samples)])

# 转换为角度制可视化
rotvecs_deg = np.degrees(rotvecs)

# 计算每个点与k个最近邻的平均距离
def compute_neighbor_distances(points, k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)  # +1 包含自身
    distances, _ = nbrs.kneighbors(points)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身（距离=0）
    return avg_distances

# 计算密度（平均距离的倒数）
avg_distances = compute_neighbor_distances(rotvecs_deg)
density = 1 / (avg_distances + 1e-6)  # 避免除零

# 可视化密度
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(
    rotvecs_deg[:, 0], rotvecs_deg[:, 1], rotvecs_deg[:, 2],
    c=density, cmap='plasma', s=20, alpha=0.7
)

max_val = 25
ax.set_xlim([-0.25*max_val, 0.25*max_val])
ax.set_ylim([-0.25*max_val, 0.25*max_val])
ax.set_zlim([-max_val, max_val])

ax.set_xlabel('X (deg)'); ax.set_ylabel('Y (deg)'); ax.set_zlabel('Z (deg)')
plt.colorbar(sc, label='Density (1/avg distance)')
plt.title('Rotation Vector Density (Nearest Neighbor)')
plt.show()