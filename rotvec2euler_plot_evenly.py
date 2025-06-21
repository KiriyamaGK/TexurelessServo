import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# 参数设置
n_samples = 15  # 每个维度的离散点数
u_range = np.radians(8)  # ±8度
v_range = np.radians(8)  # ±8度
w_range = np.radians(40)  # ±40度

# 创建离散点
u = np.linspace(-u_range, u_range, n_samples)
v = np.linspace(-v_range, v_range, n_samples)
w = np.linspace(-w_range, w_range, n_samples)

# 生成所有组合的网格点
U, V, W = np.meshgrid(u, v, w)
points = np.stack([U.ravel(), V.ravel(), W.ravel()], axis=1)

# 转换为欧拉角(ZYX顺序)
eulers = np.zeros((len(points), 3))
for i, axis in enumerate(points):
    theta = np.linalg.norm(axis)
    if theta < 1e-6:
        eulers[i] = [0, 0, 0]
    else:
        rot = R.from_rotvec(axis)
        eulers[i] = rot.as_euler('zyx', degrees=True)

# 创建3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制离散点
sc = ax.scatter(eulers[:, 0], eulers[:, 1], eulers[:, 2],
                c=eulers[:, 2], cmap='viridis', s=50, alpha=0.7)

# 添加标签和标题
ax.set_title('Euler Angles (ZYX) Distribution from Axis-Angle\n'
             f'Discrete Points (n={n_samples} per axis)', pad=20)
ax.set_xlabel('Z rotation (deg)')
ax.set_ylabel('Y rotation (deg)')
ax.set_zlabel('X rotation (deg)')
fig.colorbar(sc, ax=ax, label='X rotation (deg)')

# 调整视角以便更好地观察
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()