from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
import h5py
from scipy.spatial.transform import Rotation as R
import matplotlib.cm as cm


def plot_6d_pts(X: np.ndarray, dim=2, method=None, color_list=None, point_size=5, alpha=0.6):
    """
    可视化6维向量，支持按颜色分类画轨迹

    参数:
    X: 输入数据，形状为 (n_samples, 6)
    dim: 降维后的维度 (2 或 3)
    method: 降维方法 ('pca' 或 'tsne')
    color_list: 颜色列表，长度与X相同，用于分类着色
    point_size: 点的大小
    alpha: 透明度 (0-1之间，0完全透明，1完全不透明)
    """
    if method == "tsne":
        tech = TSNE(n_components=dim, random_state=42, perplexity=30)
    else:
        if not method == "pca":
            raise ValueError("method must be 'pca' or 'tsne'")
        tech = PCA(n_components=dim)

    X_tech = tech.fit_transform(X)

    # 设置图形大小
    fig = plt.figure(figsize=(12, 8))

    if color_list is not None:
        # 按照颜色分类绘制
        unique_colors = list(set(color_list))
        color_groups = {color: [] for color in unique_colors}

        # 按颜色分组数据点
        for i, color in enumerate(color_list):
            color_groups[color].append(X_tech[i])

        if dim == 2:
            for color, points in color_groups.items():
                points = np.array(points)
                if color == 'g':  # Dagger轨迹使用渐变色
                    # 使用viridis色彩映射
                    colors = cm.viridis(np.linspace(0, 1, len(points)))
                    scatter = plt.scatter(points[:, 0], points[:, 1],
                                          c=colors, alpha=alpha, s=point_size, label="Dagger Traj")
                else:
                    label = "Base Expert" if color == "r" else "Aggregated Expert Traj"
                    plt.scatter(points[:, 0], points[:, 1],
                                c=color, alpha=alpha, s=point_size, label=label)

            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')

        else:  # dim == 3
            ax = fig.add_subplot(111, projection='3d')

            for color, points in color_groups.items():
                points = np.array(points)
                if color == 'g':  # Dagger轨迹使用渐变色
                    # 使用viridis色彩映射
                    colors = cm.viridis(np.linspace(0, 1, len(points)))
                    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                         c=colors, alpha=alpha, s=point_size, label="Dagger Traj")
                else:
                    label = "Base Expert" if color == "r" else "Aggregated Expert Traj"
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                               c=color, alpha=alpha, s=point_size, label=label)

            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_zlabel(f'{method.upper()} Component 3')

        # 添加图例
        plt.legend()

    else:
        # 如果没有颜色列表，使用默认的单色绘制
        if dim == 2:
            plt.scatter(X_tech[:, 0], X_tech[:, 1], alpha=alpha, s=point_size)
            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_tech[:, 0], X_tech[:, 1], X_tech[:, 2], alpha=alpha, s=point_size)
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_zlabel(f'{method.upper()} Component 3')

    plt.title(f'{method.upper()} Projection of 6D Vectors')
    plt.tight_layout()
    plt.savefig("res.png", dpi=300, bbox_inches='tight')
    plt.show()

    return X_tech


def generate_poses_from_hdf5(hdf_pth: str, n_base_expert_demos=1000000, traj_vis_gap=1, traj_vis_type="all"):
    if isinstance(traj_vis_type, list):
        assert len(traj_vis_type)
    else:
        assert isinstance(traj_vis_type, str) and traj_vis_type == "all"
    vis_base_expert = traj_vis_type == "all" or "base_expert" in traj_vis_type
    vis_dagger = traj_vis_type == "all" or "dagger" in traj_vis_type
    vis_aggregated_expert = traj_vis_type == "all" or "aggregated_expert" in traj_vis_type

    X = []
    color_lst = []
    dagger_traj_indices = []  # 记录每个点属于哪个dagger轨迹
    current_traj_id = 0

    f = h5py.File(hdf_pth, "r")
    for i in range(len(f['data'])):
        print(f"reading demo_{i}")
        pos_list = f[f'data/demo_{i}/delta_pos_curgoal']  # 6d_pos(mm,deg) of the matrix g_gtar_T
        is_base_expert_epi = i < n_base_expert_demos and vis_base_expert
        is_dagger_epi = (np.linalg.norm(pos_list[-2][0:3]) > 2 or np.linalg.norm(
            pos_list[-2][3:6]) > 2) and i >= n_base_expert_demos and vis_dagger
        is_aggregated_expert_epi = (not is_base_expert_epi) and (not is_dagger_epi) and vis_aggregated_expert

        for idx, raw_pose in enumerate(pos_list):
            if idx == 0 or idx % traj_vis_gap == 0:
                raw_pose = np.array(raw_pose)
                inv_T = np.eye(4)
                inv_T[:3, 3] = raw_pose[:3]
                inv_T[:3, :3] = R.from_rotvec(raw_pose[3:6] * np.pi / 180).as_matrix()
                T = np.linalg.inv(inv_T)
                new_pose = np.zeros(6)
                new_pose[:3] = T[:3, 3] / 1000  # mm2m
                new_pose[3:6] = R.from_matrix(T[:3, :3]).as_rotvec() / np.pi * 180  # rad2deg

                if is_base_expert_epi or is_aggregated_expert_epi or is_dagger_epi:
                    X.append(new_pose)
                    if is_base_expert_epi:
                        color_lst.append('r')
                        dagger_traj_indices.append(-1)  # 非dagger轨迹标记为-1
                    elif is_dagger_epi:
                        color_lst.append('g')
                        dagger_traj_indices.append(current_traj_id)  # 记录轨迹ID
                    elif is_aggregated_expert_epi:
                        color_lst.append('b')
                        dagger_traj_indices.append(-1)  # 非dagger轨迹标记为-1

        # 如果是dagger轨迹，增加轨迹ID
        if is_dagger_epi:
            current_traj_id += 1

    X = np.array(X)
    dagger_traj_indices = np.array(dagger_traj_indices)
    f.close()
    return X, color_lst, dagger_traj_indices


def plot_6d_pts_with_dagger_gradient(X: np.ndarray, dagger_traj_indices: np.ndarray, dim=2, method=None,
                                     color_list=None, point_size=5, alpha=0.6):
    """
    可视化6维向量，Dagger轨迹使用渐变色

    参数:
    X: 输入数据
    dagger_traj_indices: 每个点对应的dagger轨迹索引，-1表示非dagger轨迹
    """
    if method == "tsne":
        tech = TSNE(n_components=dim, random_state=42, perplexity=30)
    else:
        if not method == "pca":
            raise ValueError("method must be 'pca' or 'tsne'")
        tech = PCA(n_components=dim)

    X_tech = tech.fit_transform(X)

    # 设置图形大小
    fig = plt.figure(figsize=(12, 8))

    if color_list is not None:
        # 分离不同类型的数据
        base_expert_mask = np.array(color_list) == 'r'
        dagger_mask = np.array(color_list) == 'g'
        aggregated_expert_mask = np.array(color_list) == 'b'

        if dim == 2:
            # 绘制Base Expert (红色)
            if np.any(base_expert_mask):
                plt.scatter(X_tech[base_expert_mask, 0], X_tech[base_expert_mask, 1],
                            c='r', alpha=alpha, s=point_size, label="Base Expert")

            # 绘制Aggregated Expert (蓝色)
            if np.any(aggregated_expert_mask):
                plt.scatter(X_tech[aggregated_expert_mask, 0], X_tech[aggregated_expert_mask, 1],
                            c='b', alpha=alpha, s=point_size, label="Aggregated Expert Traj")

            # 绘制Dagger轨迹，使用渐变色
            if np.any(dagger_mask):
                dagger_points = X_tech[dagger_mask]
                dagger_traj_ids = dagger_traj_indices[dagger_mask]

                # 为每个轨迹分配颜色
                unique_traj_ids = np.unique(dagger_traj_ids)
                colormap = cm.viridis

                for traj_id in unique_traj_ids:
                    if traj_id != -1:  # 跳过非dagger轨迹标记
                        traj_mask = dagger_traj_ids == traj_id
                        color = colormap(traj_id / max(1, len(unique_traj_ids) - 1))
                        plt.scatter(dagger_points[traj_mask, 0], dagger_points[traj_mask, 1],
                                    c=[color], alpha=alpha, s=point_size,
                                    label=f"Dagger Traj {traj_id}" if traj_id == 0 else "")

            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')

        else:  # dim == 3
            ax = fig.add_subplot(111, projection='3d')

            # 绘制Base Expert (红色)
            if np.any(base_expert_mask):
                ax.scatter(X_tech[base_expert_mask, 0], X_tech[base_expert_mask, 1], X_tech[base_expert_mask, 2],
                           c='r', alpha=alpha, s=point_size, label="Base Expert")

            # 绘制Aggregated Expert (蓝色)
            if np.any(aggregated_expert_mask):
                ax.scatter(X_tech[aggregated_expert_mask, 0], X_tech[aggregated_expert_mask, 1],
                           X_tech[aggregated_expert_mask, 2],
                           c='b', alpha=alpha, s=point_size, label="Aggregated Expert Traj")

            # 绘制Dagger轨迹，使用渐变色
            if np.any(dagger_mask):
                dagger_points = X_tech[dagger_mask]
                dagger_traj_ids = dagger_traj_indices[dagger_mask]

                # 为每个轨迹分配颜色
                unique_traj_ids = np.unique(dagger_traj_ids)
                colormap = cm.viridis

                for traj_id in unique_traj_ids:
                    if traj_id != -1:  # 跳过非dagger轨迹标记
                        traj_mask = dagger_traj_ids == traj_id
                        color = colormap(traj_id / max(1, len(unique_traj_ids) - 1))
                        ax.scatter(dagger_points[traj_mask, 0], dagger_points[traj_mask, 1],
                                   dagger_points[traj_mask, 2],
                                   c=[color], alpha=alpha, s=point_size,
                                   )

            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_zlabel(f'{method.upper()} Component 3')

        # 添加图例
        plt.legend()

        # 添加渐变色条
        if np.any(dagger_mask):
            # 创建ScalarMappable对象
            unique_traj_ids = np.unique(dagger_traj_ids[dagger_traj_ids != -1])
            if len(unique_traj_ids) > 0:
                norm = plt.Normalize(vmin=min(unique_traj_ids), vmax=max(unique_traj_ids))
                sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
                sm.set_array([])

                # 添加色条
                cbar = plt.colorbar(sm, ax=plt.gca() if dim == 2 else ax,
                                    label='Dagger Trajectory Index')
                cbar.set_label('Dagger Trajectory Index', rotation=270, labelpad=15)

    else:
        # 如果没有颜色列表，使用默认的单色绘制
        if dim == 2:
            plt.scatter(X_tech[:, 0], X_tech[:, 1], alpha=alpha, s=point_size)
            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_tech[:, 0], X_tech[:, 1], X_tech[:, 2], alpha=alpha, s=point_size)
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_zlabel(f'{method.upper()} Component 3')

    plt.title(f'{method.upper()} Projection of 6D Vectors (Dagger Trajectories Colored by Episode)')
    plt.tight_layout()
    plt.savefig("res.png", dpi=300, bbox_inches='tight')
    plt.show()

    return X_tech


if __name__ == "__main__":
    _random_gen = False
    vis_dim = 3
    vis_method = "pca"  # "pca" or "tsne"
    hdf_pth = "/media/kiriyamagk/One Touch/AlignAnything/25.10.30/hdf5/mimic.hdf5"

    n_base_expert_demos = 200
    traj_vis_gap = 1  # interval of points to sample within one traj

    # traj_vis_type can be "all" or [(opt)"base_expert", (opt)"dagger", (opt)"aggregated_expert"]
    # traj_vis_type = ["dagger", "aggregated_expert"]
    traj_vis_type = ["dagger"]

    # 点的大小和透明度参数
    point_size = 5
    alpha_value = 0.6

    if _random_gen:
        X = generate_random_poses(n_rand_samples, _range)
        plot_6d_pts(X, dim=vis_dim, method=vis_method, color_list=None,
                    point_size=point_size, alpha=alpha_value)
    else:
        X, color_list, dagger_traj_indices = generate_poses_from_hdf5(hdf_pth, n_base_expert_demos, traj_vis_gap,
                                                                      traj_vis_type)

        # 使用新的绘图函数，支持dagger轨迹渐变色
        plot_6d_pts_with_dagger_gradient(X, dagger_traj_indices, dim=vis_dim, method=vis_method,
                                         color_list=color_list, point_size=point_size, alpha=alpha_value)