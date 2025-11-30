import os
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datetime import datetime, timedelta

font_pth = "/usr/share/fonts/truetype/custom/SimHei.ttf"
font = FontProperties(fname=font_pth)

# 设置图表风格
try:
    plt.style.use('seaborn-whitegrid')
except:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linestyle'] = '--'


def find_chinese_fonts():
    """查找系统中可用的中文字体"""
    import matplotlib.font_manager as fm

    all_fonts = [(f.name, f.fname) for f in fm.fontManager.ttflist]

    # 查找可能包含中文的字体
    chinese_keywords = ['sim']

    chinese_fonts = []
    for name, path in all_fonts:
        if any(keyword in name for keyword in chinese_keywords):
            chinese_fonts.append((name, path))

    print("找到的中文字体:")
    for name, path in sorted(chinese_fonts)[:10]:  # 显示前10个
        print(f"  {name}: {path}")

    return chinese_fonts


# 查找并使用最佳字体
chinese_fonts = find_chinese_fonts()


def parse_time_data(time_data,str_date,episode_bias):
    """解析时间数据"""

    # 计算每个episode的累计时间（从开始时间11-01_22.39计算）
    _mon,_dhm = str_date.split("-")
    _day,_hm =  _dhm.split("_")
    _hour,_min = _hm.split(".")
    start_time = datetime(2023, int(_mon), int(_day), int(_hour), int(_min))  # 假设年份为2023
    time_deltas = {}

    for episode, time_str in time_data.items():
        # 解析时间字符串
        parts = time_str.split('_')
        date_part = parts[0]
        time_part = parts[1].replace('.', ':')

        month, day = map(int, date_part.split('-'))
        hour, minute = map(int, time_part.split(':'))

        # 创建datetime对象
        current_time = datetime(2023, month, day, hour, minute)

        # 计算时间差
        time_delta = current_time - start_time
        time_deltas[episode+episode_bias] = time_delta

    return time_deltas


def load_error_data(base_dir, part_idx=5, epochs_to_process="all"):
    """加载误差数据"""
    res_list = []
    sub_dirs = os.listdir(base_dir)
    print("找到的目录:", sub_dirs)

    for sub_dir in sub_dirs:
        if "epoch" not in sub_dir:
            continue

        # 更健壮的 epoch 提取
        if "epoch" in sub_dir and ")" in sub_dir:
            epoch = sub_dir.split("epoch")[1].split(")")[0]
        else:
            continue

        if not epoch.isdigit():
            continue

        epoch = int(epoch)
        if epochs_to_process != "all" and epoch not in epochs_to_process:
            continue

        json_path = os.path.join(base_dir, sub_dir, "final_error.json")
        if not os.path.exists(json_path):
            print(f"警告: {json_path} 不存在")
            continue

        try:
            with open(json_path, "r") as j:
                data = json.load(j)
                value_tr = data[str(part_idx)]["translation_xyz"]["mean"]
                value_rot = data[str(part_idx)]["rotation"]["mean"]
                itm = {"epoch": epoch, "value": value_tr, "rot": value_rot}
                res_list.append(itm)
        except Exception as e:
            print(f"读取 {json_path} 时出错: {e}")

    # 按epoch排序
    res_list.sort(key=lambda x: x["epoch"])
    return res_list


def plot_errors(res_list, fig_title="在线学习方法的平均误差", save_path="online_learning_errors", save_dir_name=None,time_data = None,str_date = None,episode_bias = 0):
    """绘制误差曲线图"""
    if not res_list:
        print("没有数据可绘制")
        return

    # 提取数据
    epochs = [item["epoch"] + episode_bias for item in res_list]
    tr_errors = [item["value"] for item in res_list]
    rot_errors = [item["rot"] for item in res_list]

    # 获取时间数据
    time_deltas = parse_time_data(time_data,str_date,episode_bias)

    # 创建更宽的图表
    fig, ax = plt.subplots(figsize=(14, 7))  # 宽度从12增加到14

    # 检查是否需要双y轴
    need_dual_axis = False
    if max(rot_errors) > 0 and max(tr_errors) > 0:
        need_dual_axis = (max(tr_errors) / max(rot_errors) > 10) or (max(rot_errors) / max(tr_errors) > 10)

    if need_dual_axis:
        # 双y轴
        color1 = '#2E86AB'
        ax.set_xlabel('轨迹回合 (Episode)', fontsize=14, fontweight='bold', fontproperties=font)
        ax.set_ylabel('平移误差 (mm)', color=color1, fontsize=14, fontweight='bold', fontproperties=font)
        line1 = ax.plot(epochs, tr_errors, 'o-', color=color1, linewidth=2.5,
                        markersize=6, label='平移误差 (mm)', markerfacecolor='white', markeredgewidth=2)
        ax.tick_params(axis='y', labelcolor=color1)

        # 在平移误差数据点上标注数值（黑色，大字体）
        for i, (epoch, tr_error) in enumerate(zip(epochs, tr_errors)):
            ax.annotate(f'{tr_error:.3f}',
                        (epoch, tr_error),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha='center',
                        fontsize=10,
                        color='black',
                        fontweight='bold',
                        fontproperties=font)

        ax2 = ax.twinx()
        color2 = '#A23B72'
        ax2.set_ylabel('旋转误差 (°)', color=color2, fontsize=14, fontweight='bold', fontproperties=font)
        line2 = ax2.plot(epochs, rot_errors, 's-', color=color2, linewidth=2.5,
                         markersize=6, label='旋转误差 (°)', markerfacecolor='white', markeredgewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # 在旋转误差数据点上标注数值（黑色，大字体）
        for i, (epoch, rot_error) in enumerate(zip(epochs, rot_errors)):
            ax2.annotate(f'{rot_error:.3f}',
                         (epoch, rot_error),
                         textcoords="offset points",
                         xytext=(0, -18),
                         ha='center',
                         fontsize=10,
                         color='black',
                         fontweight='bold',
                         fontproperties=font)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                  fontsize=12, framealpha=0.9, shadow=True, prop=font)
    else:
        # 单y轴
        line1 = ax.plot(epochs, tr_errors, 'o-', color='#2E86AB', linewidth=2.5,
                        markersize=6, label='平移误差 (mm)', markerfacecolor='white', markeredgewidth=2)
        line2 = ax.plot(epochs, rot_errors, 's-', color='#A23B72', linewidth=2.5,
                        markersize=6, label='旋转误差 (°)', markerfacecolor='white', markeredgewidth=2)
        ax.set_ylabel('误差值', fontsize=14, fontweight='bold', fontproperties=font)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9, shadow=True, prop=font)

        # 在数据点上标注数值（黑色，大字体）
        for i, (epoch, tr_error, rot_error) in enumerate(zip(epochs, tr_errors, rot_errors)):
            # 平移误差标注在上方
            ax.annotate(f'{tr_error:.3f}',
                        (epoch, tr_error),
                        textcoords="offset points",
                        xytext=(0, -18),
                        ha='center',
                        fontsize=10,
                        color='black',
                        fontweight='bold',
                        fontproperties=font)
            # 旋转误差标注在下方
            ax.annotate(f'{rot_error:.3f}',
                        (epoch, rot_error),
                        textcoords="offset points",
                        xytext=(0, 18),
                        ha='center',
                        fontsize=10,
                        color='black',
                        fontweight='bold',
                        fontproperties=font)

    # 添加时间信息（gradient风格）
    time_annotation_epochs = [epoch for epoch in epochs if epoch in time_deltas]
    # 如果数据点太多，每隔两个标注一个
    if len(time_annotation_epochs) > 0:
        time_annotation_epochs = time_annotation_epochs[::2]  # 每隔1个点取一个，所以步长为

    # 获取y轴范围用于时间标注的垂直位置
    y_min, y_max = ax.get_ylim()
    time_label_y = y_min - 0.08 * (y_max - y_min)

    # 创建时间渐变背景
    all_hours = [time_deltas[epoch].total_seconds() / 3600 for epoch in time_annotation_epochs]
    max_hours = max(all_hours) if all_hours else 1

    for epoch in time_annotation_epochs:
        time_delta = time_deltas[epoch]
        hours = time_delta.total_seconds() / 3600
        time_text = f'{hours:.1f}h'

        # 根据时间长度计算颜色深浅
        color_intensity = hours / max_hours
        bg_color = plt.cm.Blues(0.3 + 0.3 * color_intensity)

        ax.annotate(time_text,
                    (epoch, time_label_y),
                    textcoords="data",
                    ha='center',
                    va='top',
                    fontsize=10,
                    color='white',  # 白色文字在深色背景上
                    fontweight='bold',
                    fontproperties=font,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color,
                              alpha=0.8, edgecolor='none'))

    # 设置通用属性
    ax.set_xlabel('轨迹回合 (Episode)', fontsize=14, fontweight='bold', fontproperties=font)
    ax.set_title(fig_title, fontsize=16, fontweight='bold', pad=30, fontproperties=font)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 调整x轴范围，为左右边界留出更多空间
    ax.set_xlim(min(epochs) - 10, max(epochs) + 10)  # 增加左右边距

    # 调整y轴范围，为时间标注留出空间
    ax.set_ylim(y_min - 0.12 * (y_max - y_min), y_max* 1.1)

    # 保存图片时明确指定字体
    fig.tight_layout()

    # 方法1: 保存为PNG（确保字体嵌入）
    os.makedirs(save_dir_name, exist_ok=True)
    plt.savefig(save_dir_name + '/' + f'{save_path}.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                transparent=False)
    # 显示图表
    plt.show()

    return fig


def print_statistics(res_list):
    """打印统计信息"""
    if not res_list:
        return

    tr_errors = [item["value"] for item in res_list]
    rot_errors = [item["rot"] for item in res_list]

    print(f"\n统计信息:")
    print(f"平移误差 - 最小值: {min(tr_errors):.4f}, 最大值: {max(tr_errors):.4f}, 最终值: {tr_errors[-1]:.4f}")
    print(f"旋转误差 - 最小值: {min(rot_errors):.4f}, 最大值: {max(rot_errors):.4f}, 最终值: {rot_errors[-1]:.4f}")
    print(f"数据点数: {len(res_list)}")


if __name__ == '__main__':
    # 配置参数
    base_dir = "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/好的结果/2025-11-23_00-00-00"
    # part_idx = 5
    part_idx = 1
    save_dir_name = "real_with_dagger_new"
    episode_bias = 149
    time_data = {
        0: "11-22_14.49",
        72: "11-22_15.19",
        144: "11-22_15.51",
        216: "11-22_16.25",
        288: "11-22_16.58",
        360: "11-22_17.32",
        432: "11-22_18.05",
        504: "11-22_18.39",
        576: "11-22_19.13",
        648: "11-22_19.47",
    }
    str_date = "11-22_14.14"

    save_path = "online_learning_errors"
    fig_title = "在线学习方法的平均误差"
    epochs_to_process = "all"

    # 加载数据
    res_list = load_error_data(base_dir, part_idx, epochs_to_process)

    # 绘制图表
    if res_list:
        plot_errors(res_list, fig_title, save_path, save_dir_name,time_data,str_date,episode_bias)
        print_statistics(res_list)
    else:
        print("没有找到有效数据")