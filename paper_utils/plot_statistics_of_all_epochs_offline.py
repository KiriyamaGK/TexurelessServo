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


def parse_time_data():
    """解析时间数据"""
    time_data = {
        15: "11.02 23:37",
        20: "11.03 0:47",
        25: "11.03 1:57",
        30: "11.03 3:06",
        35: "11.03 4:16",
        40: "11.03 5:25",
        45: "11.03 6:35",
        50: "11.03 7:46",
        55: "11.03 8:56",
        60: "11.03 10:06",
        65: "11.03 11:17",
        70: "11.03 12:28",
        75: "11.03 13:39"
    }

    # 计算每个epoch的累计时间（从开始时间11.02 20:20计算）
    start_time = datetime(2023, 11, 2, 20, 20)  # 假设年份为2023
    time_deltas = {}

    for epoch, time_str in time_data.items():
        # 解析时间字符串
        parts = time_str.split()
        date_part = parts[0]
        time_part = parts[1]

        month, day = map(int, date_part.split('.'))
        hour, minute = map(int, time_part.split(':'))

        # 创建datetime对象
        current_time = datetime(2023, month, day, hour, minute)

        # 计算时间差
        time_delta = current_time - start_time
        time_deltas[epoch] = time_delta

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


def plot_errors(res_list, fig_title="离线学习方法的平均误差", save_path="offline_learning_errors", save_dir_name=None,
                time_style="elegant"):
    """绘制误差曲线图
    time_style选项:
    - "elegant": 优雅风格，黑色文字带浅色背景框
    - "minimal": 极简风格，直接在x轴刻度下方
    - "timeline": 时间轴风格，用线条连接
    - "gradient": 渐变背景色表示时间流逝
    """
    if not res_list:
        print("没有数据可绘制")
        return

    # 提取数据
    epochs = [item["epoch"] for item in res_list]
    tr_errors = [item["value"] for item in res_list]
    rot_errors = [item["rot"] for item in res_list]

    # 获取时间数据
    time_deltas = parse_time_data()

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 检查是否需要双y轴
    need_dual_axis = False
    if max(rot_errors) > 0 and max(tr_errors) > 0:
        need_dual_axis = (max(tr_errors) / max(rot_errors) > 10) or (max(rot_errors) / max(tr_errors) > 10)

    if need_dual_axis:
        # 双y轴
        color1 = '#2E86AB'
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontweight='bold', fontproperties=font)
        ax.set_ylabel('平移误差 (mm)', color=color1, fontsize=14, fontweight='bold', fontproperties=font)
        line1 = ax.plot(epochs, tr_errors, 'o-', color=color1, linewidth=2.5,
                        markersize=6, label='平移误差 (mm)', markerfacecolor='white', markeredgewidth=2)
        ax.tick_params(axis='y', labelcolor=color1)

        # 在平移误差数据点上标注数值（黑色，大两号字体）
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

        # 在旋转误差数据点上标注数值（黑色，大两号字体）
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

        # 在数据点上标注数值（黑色，大两号字体）
        for i, (epoch, tr_error, rot_error) in enumerate(zip(epochs, tr_errors, rot_errors)):
            # 平移误差标注在上方
            ax.annotate(f'{tr_error:.3f}',
                        (epoch, tr_error),
                        textcoords="offset points",
                        xytext=(0, 12),
                        ha='center',
                        fontsize=10,
                        color='black',
                        fontweight='bold',
                        fontproperties=font)
            # 旋转误差标注在下方
            ax.annotate(f'{rot_error:.3f}',
                        (epoch, rot_error),
                        textcoords="offset points",
                        xytext=(0, -18),
                        ha='center',
                        fontsize=10,
                        color='black',
                        fontweight='bold',
                        fontproperties=font)

    # 添加时间信息（每隔两个epoch标注一次）
    time_annotation_epochs = [epoch for i, epoch in enumerate(epochs) if i % 2 == 0 and epoch in time_deltas]

    # 获取y轴范围用于时间标注的垂直位置
    y_min, y_max = ax.get_ylim()

    if time_style == "elegant":
        # 优雅风格：黑色文字带浅色背景框
        time_label_y = y_min - 0.08 * (y_max - y_min)

        for epoch in time_annotation_epochs:
            time_delta = time_deltas[epoch]
            hours = time_delta.total_seconds() / 3600
            time_text = f'{hours:.1f}h'

            ax.annotate(time_text,
                        (epoch, time_label_y),
                        textcoords="data",
                        ha='center',
                        va='top',
                        fontsize=10,
                        color='black',  # 改为黑色
                        fontweight='bold',
                        fontproperties=font,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#F8F9FA',
                                  alpha=0.9, edgecolor='#DEE2E6', linewidth=1))

        # 调整y轴范围
        ax.set_ylim(y_min - 0.12 * (y_max - y_min), y_max)

    elif time_style == "minimal":
        # 极简风格：直接在x轴刻度下方
        for epoch in time_annotation_epochs:
            time_delta = time_deltas[epoch]
            hours = time_delta.total_seconds() / 3600
            time_text = f'{hours:.1f}h'

            ax.text(epoch, y_min - 0.02 * (y_max - y_min), time_text,
                    ha='center', va='top', fontsize=9, color='black',
                    fontweight='bold', fontproperties=font)

        ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max)

    elif time_style == "timeline":
        # 时间轴风格：用线条连接时间点
        time_label_y = y_min - 0.1 * (y_max - y_min)

        # 画时间轴基线
        ax.axhline(y=time_label_y, color='#6C757D', linewidth=0.8, alpha=0.6)

        for epoch in time_annotation_epochs:
            time_delta = time_deltas[epoch]
            hours = time_delta.total_seconds() / 3600
            time_text = f'{hours:.1f}h'

            # 画垂直线连接到时间轴
            ax.plot([epoch, epoch], [y_min, time_label_y],
                    color='#6C757D', linewidth=0.8, alpha=0.4, linestyle='--')

            # 时间标注
            ax.annotate(time_text,
                        (epoch, time_label_y),
                        textcoords="offset points",
                        xytext=(0, -5),
                        ha='center',
                        va='top',
                        fontsize=9,
                        color='black',
                        fontweight='bold',
                        fontproperties=font,
                        bbox=dict(boxstyle="circle,pad=0.2", facecolor='#E9ECEF',
                                  alpha=0.9, edgecolor='#ADB5BD'))

        ax.set_ylim(y_min - 0.15 * (y_max - y_min), y_max)

    elif time_style == "gradient":
        # 渐变背景色表示时间流逝
        time_label_y = y_min - 0.08 * (y_max - y_min)

        # 创建时间渐变背景
        all_hours = [time_deltas[epoch].total_seconds() / 3600 for epoch in time_annotation_epochs]
        max_hours = max(all_hours)

        for i, epoch in enumerate(time_annotation_epochs):
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

        ax.set_ylim(y_min - 0.12 * (y_max - y_min), y_max)

    # 设置通用属性
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontweight='bold', fontproperties=font)
    ax.set_title(fig_title, fontsize=16, fontweight='bold', pad=25, fontproperties=font)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(min(epochs) - 1, max(epochs) + 1)

    # 保存图片时明确指定字体
    fig.tight_layout()

    # 方法1: 保存为PNG（确保字体嵌入）
    if save_dir_name:
        os.makedirs(save_dir_name, exist_ok=True)
        plt.savefig(os.path.join(save_dir_name, f'{save_path}_{time_style}.png'), dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)
    else:
        plt.savefig(f'{save_path}_{time_style}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', transparent=False)

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
    base_dir = "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/好的结果/2025-10_31_00-00-00"
    part_idx = 5
    save_dir_name = "sim_without_dagger"
    save_path = "offline_learning_errors"
    fig_title = "离线学习方法的平均误差"
    epochs_to_process = "all"

    # 加载数据
    res_list = load_error_data(base_dir, part_idx, epochs_to_process)

    # 绘制不同风格的图表
    if res_list:
        # 测试不同的时间标注风格
        time_styles = ["elegant", "minimal", "timeline", "gradient"]

        for style in time_styles:
            print(f"\n正在生成 {style} 风格的图表...")
            plot_errors(res_list, fig_title, save_path, save_dir_name, time_style=style)

        print_statistics(res_list)
    else:
        print("没有找到有效数据")