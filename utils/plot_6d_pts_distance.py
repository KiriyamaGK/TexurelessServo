from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random
import h5py
from scipy.spatial.transform import Rotation as R


def plot_6d_pts(X: np.ndarray, dim=2, method=None, color_list=None,point_size = 5):
    """
    可视化6维向量，支持按颜色分类画轨迹

    参数:
    X: 输入数据，形状为 (n_samples, 6)
    dim: 降维后的维度 (2 或 3)
    method: 降维方法 ('pca' 或 'tsne')
    color_list: 颜色列表，长度与X相同，用于分类着色
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
                label = "Base Expert" if color == "r" else "Dagger Traj" if color == "g" else "Aggregated Expert Traj"
                alpha = 0.6 if "Expert" in label else 0.6
                ax.scatter(points[:, 0], points[:, 1],
                           c=color, alpha=alpha, s=point_size, label=label)

            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')

        else:  # dim == 3
            ax = fig.add_subplot(111, projection='3d')

            for color, points in color_groups.items():
                points = np.array(points)
                label = "Base Expert" if color == "r" else "Dagger Traj" if color == "g" else "Aggregated Expert Traj"
                alpha = 0.6 if "Expert" in label else 0.6
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
            plt.scatter(X_tech[:, 0], X_tech[:, 1], alpha=0.5, s=30)
            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_tech[:, 0], X_tech[:, 1], X_tech[:, 2], alpha=0.5, s=30)
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_zlabel(f'{method.upper()} Component 3')

    plt.title(f'{method.upper()} Projection of 6D Vectors')
    plt.tight_layout()
    plt.savefig("res.png", dpi=300, bbox_inches='tight')
    plt.show()

    return X_tech

def generate_random_poses(n_samples:int,_range:list[list]):
    X = []
    for i in range(n_samples):
        print(f"generating sample_{i}")
        trans_xy = np.sqrt(random.uniform(0, _range[0][1] ** 2))
        ori_xy = random.uniform(0, 2 * np.pi)
        trans_z = random.uniform(_range[1][0], _range[1][1])
        th1 = random.uniform(_range[2][0], _range[2][1])
        th2 = random.uniform(_range[3][0], _range[3][1])
        th3 = random.uniform(-_range[4][0], _range[4][1])
        arr = np.array([trans_xy * np.cos(ori_xy), trans_xy * np.sin(ori_xy), trans_z, th1, th2, th3])
        X.append(arr)
    X = np.array(X)
    return X

def generate_poses_from_hdf5(hdf_pth:str,n_base_expert_demos = 1000000,traj_vis_gap = 1,traj_vis_type = "all",num_selected_pts = 0,n_fail_pool_size = 0):
    if isinstance(traj_vis_type,list):
        assert len(traj_vis_type)
    else:
        assert isinstance(traj_vis_type,str) and traj_vis_type == "all"
    vis_base_expert = traj_vis_type == "all" or "base_expert" in traj_vis_type
    vis_dagger = traj_vis_type == "all" or "dagger" in traj_vis_type
    vis_aggregated_expert = traj_vis_type == "all" or "aggregated_expert" in traj_vis_type
    X = []
    color_lst = []

    n_delta_trajs_gap = n_fail_pool_size//num_selected_pts

    f = h5py.File(hdf_pth, "r")
    for i in range(len(f['data'])):
        if i>=-1 and i<2800000:
            print(f"reading demo_{i}")
            pos_list = f[f'data/demo_{i}/delta_pos_curgoal']  # 6d_pos(mm,deg) of the matrix g_gtar_T
            is_base_expert_epi = i < n_base_expert_demos and vis_base_expert
            is_dagger_epi = i >= n_base_expert_demos and (i + 1 - n_base_expert_demos) % (
                        n_delta_trajs_gap + n_fail_pool_size) <= n_delta_trajs_gap and vis_dagger
            is_aggregated_expert_epi = i >= n_base_expert_demos and (i + 1 - n_base_expert_demos) % (
                        n_delta_trajs_gap + n_fail_pool_size) > n_delta_trajs_gap and vis_aggregated_expert
            for idx,raw_pose in enumerate(pos_list):
                if idx == 0 or idx % traj_vis_gap == 0:
                    raw_pose = np.array(raw_pose)
                    inv_T = np.eye(4)
                    inv_T[:3, 3] = raw_pose[:3]
                    inv_T[:3, :3] = R.from_rotvec(raw_pose[3:6] * np.pi / 180).as_matrix()
                    T = np.linalg.inv(inv_T)
                    new_pose = np.zeros(6)
                    new_pose[:3] = T[:3, 3] / 1000 #mm2m
                    new_pose[3:6] = R.from_matrix(T[:3, :3]).as_rotvec() / np.pi * 180 #rad2deg
                    if is_base_expert_epi or is_aggregated_expert_epi or is_dagger_epi:
                        X.append(new_pose)
                        if is_base_expert_epi:
                            color_lst.append('r')
                        if is_dagger_epi:
                            color_lst.append('g')
                        if is_aggregated_expert_epi:
                            color_lst.append('b')

    X = np.array(X)
    f.close()
    return X,color_lst

if __name__ == "__main__":
    _random_gen = False
    vis_dim = 3
    vis_method = "pca" # "pca" or "tsne
    # hdf_pth = "/media/kiriyamagk/One Touch/AlignAnything/25.10.30/hdf5/mimic.hdf5"
    hdf_pth = "/media/kiriyamagk/One Touch/AlignAnything_real/25.11.21/hdf5/mimic.hdf5"
    # if not random_gen
    n_base_expert_demos = 176
    num_selected_pts = 5
    n_fail_pool_size = 60
    traj_vis_gap = 1 # interval of points to sample within one traj
    point_size = 5

    #if random_gen
    n_rand_samples = 3000
    _range = [[-0.2, 0.2], [-0.2, 0.2], [-8, 8], [-8, 8], [-40, 40]]  # range of gtar_g_T (m,deg)
    #else:
    # traj_vis_type can be "all" or [(opt)"base_expert", (opt)"dagger", (opt)"aggregated_expert"]
    # traj_vis_type = ["dagger", "aggregated_expert"]
    # traj_vis_type = ["dagger"]
    # traj_vis_type = [ "aggregated_expert"]
    # traj_vis_type = [ "base_expert"]
    traj_vis_type = "all"


    if _random_gen:
        X = generate_random_poses(n_rand_samples,_range)
    else:
        X, color_list = generate_poses_from_hdf5(hdf_pth,n_base_expert_demos,traj_vis_gap,traj_vis_type,num_selected_pts,n_fail_pool_size)
    plot_6d_pts(X,dim=vis_dim,method = vis_method,color_list = color_list,point_size = point_size)
    #要画的图：
    # 1.从dagger轨迹引出若干个专家轨迹，这种图画几张
    # 2.展现出dagger轨迹在episode加大时不断聚集在目标位置附近，做法是在图片中只出现dagger轨迹，但是颜色根据episode渐变。

