import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

# 参数设置
n_samples = 15
u_range_deg = 8  # ±8度
v_range_deg = 8  # ±8度
w_range_deg = 40  # ±40度

# 创建离散网格（直接使用角度值）
u = np.linspace(-u_range_deg, u_range_deg, n_samples)
v = np.linspace(-v_range_deg, v_range_deg, n_samples)
w = np.linspace(-w_range_deg, w_range_deg, n_samples)
U, V, W = np.meshgrid(u, v, w)
points_deg = np.stack([U.ravel(), V.ravel(), W.ravel()], axis=1)  # 角度制

# 转换为弧度用于计算
points_rad = np.radians(points_deg)

# 转换为欧拉角(ZYX顺序)
eulers = np.zeros((len(points_rad), 3))
for i, axis in enumerate(points_rad):
    theta = np.linalg.norm(axis)
    if theta < 1e-6:
        eulers[i] = [0, 0, 0]
    else:
        rot = R.from_rotvec(axis)
        eulers[i] = rot.as_euler('zyx', degrees=True)  # 输出保持角度制

# 创建可视化图形
fig = plt.figure(figsize=(18, 8))
fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.1, wspace=0.3)

# 轴角空间（但按照欧拉角的视觉顺序排列）
ax1 = fig.add_subplot(121, projection='3d')
# 注意：这里我们保持轴角数据不变，只是调整了显示的坐标轴顺序
sc1 = ax1.scatter(points_deg[:,2], points_deg[:,1], points_deg[:,0],
                 c=eulers[:,2], cmap='viridis', s=30, alpha=0.7)
#c=eulers[:,2]取每一行的最后一列（rx）, cmap='viridis'决定了颜色渐变式样，深紫色（低值）→ 蓝色 → 绿色 → 亮黄色（高值）
#颜色映射规则是：c的最大值作为终止色，最小值作为起始色，其余颜色也相应映射

ax1.set_title('Axis-Angle Space\n(Color shows X rotation)', pad=15)
ax1.set_xlabel('Z rotation (deg)', labelpad=10)
ax1.set_ylabel('Y rotation (deg)', labelpad=10)
ax1.set_zlabel('X rotation (deg)', labelpad=10)
ax1.view_init(elev=20, azim=-45)

# 欧拉角空间 (保持原始右图方式)
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(eulers[:,0], eulers[:,1], eulers[:,2],  # ZYX顺序
                 c=eulers[:,2], cmap='viridis', s=30, alpha=0.7)
ax2.set_title('Euler Angle Space (ZYX)\n(Color shows X rotation)', pad=15)
ax2.set_xlabel('Z rotation (deg)', labelpad=10)
ax2.set_ylabel('Y rotation (deg)', labelpad=10)
ax2.set_zlabel('X rotation (deg)', labelpad=10)
ax2.view_init(elev=20, azim=-45)

# # 添加颜色条
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 手动调整颜色条位置
# cbar = fig.colorbar(sc2, cax=cbar_ax)
# cbar.set_label('X Rotation (deg)')

plt.tight_layout()
plt.show()