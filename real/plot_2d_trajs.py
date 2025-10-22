import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_2d_trajectory(points = None, labels=None, title="2D Trajectory", show_animation=True):
    """
    绘制二维轨迹图，显示点的先后顺序

    参数:
    - points: 点列表，格式 [[x1, y1], [x2, y2], ...]
    - labels: 每个点的标签（可选）
    - title: 图表标题
    - show_animation: 是否显示动画效果
    """
    if points is None:
        print("错误：点列表为空")
        return
    if isinstance(points, list) and len(points) == 0:
        print("错误：点列表为空")
        return

    # 转换为numpy数组便于处理
    flags = points[:, -1]
    points = np.array(points[:,:2])
    x_coords = points[:, 0]
    y_coords = points[:, 1]


    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制完整的轨迹线
    ax.plot(x_coords, y_coords, 'b--', alpha=0.3, linewidth=1, label='轨迹')

    # 绘制点并显示顺序
    colors = plt.cm.viridis(np.linspace(0, 1, len(points)))

    for i, (x, y) in enumerate(points):
        # 绘制点
        color = 'red' if flags[i] == 1 else [colors[i]]
        ax.scatter(x, y, c=color, s=100, alpha=0.7, zorder=5)



    # 设置图表属性
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 自动调整坐标轴范围
    margin_x = (max(x_coords) - min(x_coords)) * 0.1
    margin_y = (max(y_coords) - min(y_coords)) * 0.1
    ax.set_xlim(min(x_coords) - margin_x, max(x_coords) + margin_x)
    ax.set_ylim(min(y_coords) - margin_y, max(y_coords) + margin_y)

    plt.tight_layout()

    plt.show()

    return fig, ax


if __name__ == '__main__':
    # 定义轨迹点
    npy_pth = "/home/kiriyamagk/桌面/AlignAnything/real/coarse_locolization_results/1761060488.npy"
    # trajectory_points = [
    #     [0, 0],  # 起点
    #     [2, 3],  # 点2
    #     [4, 1],  # 点3
    #     [6, 4],  # 点4
    #     [3, 5]  # 终点
    # ]
    trajectory_points = np.load(npy_pth,allow_pickle=True)

    # 绘制轨迹
    plot_2d_trajectory(trajectory_points,
                       title="轨迹")