import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.font_manager import FontProperties

from matplotlib.font_manager import FontProperties

font_pth = "/usr/share/fonts/truetype/custom/SimHei.ttf"
font = FontProperties(fname=font_pth)

def calculate_errors(traj_data,is_sim = True,real_base_dir=None, npy_file=None):
    """
    计算轨迹数据的误差
    """
    if not isinstance(traj_data, (list,np.ndarray)):
        return None

    # 获取最后一个元素
    last_element = traj_data[-1]

    # 检查最后一个元素是否是字典且包含wgT_tar
    wgT_file = npy_file.split('_')[0]+"_wgT_tar.npy"
    if os.path.exists(os.path.join(real_base_dir,wgT_file)):
        wgT_tar =np.load(os.path.join(real_base_dir,wgT_file),allow_pickle=True)
    else:
        # 如果不满足条件，设置wgT_tar为单位阵，T为最后一个元素
        print("simulation env target pos was not saved,using predefined...")
        wgT_tar = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00], [-0.00000000e+00, -1.00000000e+00 , 1.22464680e-16, -2.32682892e-16], [-0.00000000e+00, -1.22464680e-16 ,-1.00000000e+00 , 1.90000000e+00], [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])

    T = np.array(last_element)

    # 计算相对变换
    dT = np.linalg.inv(wgT_tar) @ T
    if is_sim:
        dT[:3, 3]*=1000
    # 计算平移误差
    translation_error = np.linalg.norm(dT[:3, 3])

    # 计算Z轴误差
    z_error = abs(dT[2, 3])

    # 计算旋转误差（轴角表示，转换为角度）
    rotation_matrix = dT[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    axis_angle = rotation.as_rotvec()
    rotation_error_deg = np.linalg.norm(axis_angle) * 180 / np.pi

    # 计算XYZ各方向的平移误差
    x_error = abs(dT[0, 3])
    y_error = abs(dT[1, 3])

    return {
        'translation_error': float(translation_error),
        'rotation_error_deg': float(rotation_error_deg),
        'x_error': float(x_error),
        'y_error': float(y_error),
        'z_error': float(z_error)
    }


def analyze_trajectory_errors(base_dir, sub_dir, part_idx,is_sim =True,save_dir_name = None):
    """
    分析轨迹误差
    """
    real_base_dir = os.path.join(base_dir, sub_dir, str(part_idx), "traj")

    if not os.path.exists(real_base_dir):
        print(f"目录不存在: {real_base_dir}")
        return None

    # 获取所有npy文件
    npy_files = [f for f in os.listdir(real_base_dir) if f.endswith('.npy')]

    if not npy_files:
        print(f"在目录 {real_base_dir} 中没有找到npy文件")
        return None

    print(f"找到 {len(npy_files)} 个npy文件")

    # 存储所有误差数据
    all_errors = {
        'translation_errors': [],
        'rotation_errors_deg': [],
        'x_errors': [],
        'y_errors': [],
        'z_errors': []
    }

    # 处理每个npy文件
    for npy_file in npy_files:
        file_path = os.path.join(real_base_dir, npy_file)
        try:
            # 加载npy文件
            traj_data = np.load(file_path, allow_pickle=True)


            # 计算误差
            errors = calculate_errors(traj_data,is_sim,real_base_dir, npy_file)

            if errors is not None:
                all_errors['translation_errors'].append(errors['translation_error'])
                all_errors['rotation_errors_deg'].append(errors['rotation_error_deg'])
                all_errors['x_errors'].append(errors['x_error'])
                all_errors['y_errors'].append(errors['y_error'])
                all_errors['z_errors'].append(errors['z_error'])
            else:
                print(f"文件 {npy_file} 数据格式不符合要求")

        except Exception as e:
            print(f"处理文件 {npy_file} 时出错: {e}")

    if not all_errors['translation_errors']:
        print("没有成功处理任何文件")
        return None

    # 计算统计信息
    stats = {}
    for error_type, error_values in all_errors.items():
        if error_values:  # 确保列表不为空
            stats[error_type] = {
                'mean': float(np.mean(error_values)),
                'std': float(np.std(error_values)),
                'max': float(np.max(error_values)),
                'min': float(np.min(error_values)),
                'median': float(np.median(error_values))
            }

    os.makedirs(save_dir_name, exist_ok=True)
    output_file = os.path.join(save_dir_name,'trajectory_error_statistics.json')

    # 创建可序列化的结果字典
    result = {
        'statistics': stats,
        'file_count': len(npy_files),
        'processed_count': len(all_errors['translation_errors']),
        # 'all_errors': all_errors  # 包含所有原始误差数据  #可选
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"统计结果已保存到: {output_file}")

    # 打印统计摘要
    print("\n=== 误差统计摘要 ===")
    print(f"处理文件数量: {result['processed_count']}/{result['file_count']}")

    for error_type, stat in stats.items():
        print(f"\n{error_type}:")
        print(f"  均值: {stat['mean']:.6f}")
        print(f"  标准差: {stat['std']:.6f}")
        print(f"  最大值: {stat['max']:.6f}")
        print(f"  最小值: {stat['min']:.6f}")
        print(f"  中位数: {stat['median']:.6f}")

    return result



if __name__ == '__main__':
    # 配置参数
    base_dir = "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/好的结果/2025-11-08_00-00-00"
    # part_idx = 5
    part_idx = 1
    save_dir_name = "real_with_dagger_light"
    sub_dir = "2025-11-09_23-31-05(epoch599)"

    is_sim = False

    # 分析轨迹误差
    result = analyze_trajectory_errors(base_dir, sub_dir, part_idx,is_sim,save_dir_name)
