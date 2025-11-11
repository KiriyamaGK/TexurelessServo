import os
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

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


def plot_errors(res_list, fig_title="离线学习方法的平均误差", save_path="offline_learning_errors",save_dir_name=None):
    """绘制误差曲线图"""
    if not res_list:
        print("没有数据可绘制")
        return

    # 提取数据
    epochs = [item["epoch"] for item in res_list]
    tr_errors = [item["value"] for item in res_list]
    rot_errors = [item["rot"] for item in res_list]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 检查是否需要双y轴
    need_dual_axis = False
    if max(rot_errors) > 0 and max(tr_errors) > 0:
        need_dual_axis = (max(tr_errors) / max(rot_errors) > 10) or (max(rot_errors) / max(tr_errors) > 10)

    if need_dual_axis:
        # 双y轴
        color1 = '#2E86AB'
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontweight='bold',fontproperties=font)
        ax.set_ylabel('平移误差 (mm)', color=color1, fontsize=14, fontweight='bold',fontproperties=font)
        line1 = ax.plot(epochs, tr_errors, 'o-', color=color1, linewidth=2.5,
                        markersize=6, label='平移误差 (mm)', markerfacecolor='white', markeredgewidth=2)
        ax.tick_params(axis='y', labelcolor=color1)

        ax2 = ax.twinx()
        color2 = '#A23B72'
        ax2.set_ylabel('旋转误差 (°)', color=color2, fontsize=14, fontweight='bold',fontproperties=font)
        line2 = ax2.plot(epochs, rot_errors, 's-', color=color2, linewidth=2.5,
                         markersize=6, label='旋转误差 (°)', markerfacecolor='white', markeredgewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                  fontsize=12, framealpha=0.9, shadow=True,prop=font)
    else:
        # 单y轴
        line1 = ax.plot(epochs, tr_errors, 'o-', color='#2E86AB', linewidth=2.5,
                        markersize=6, label='平移误差 (mm)', markerfacecolor='white', markeredgewidth=2)
        line2 = ax.plot(epochs, rot_errors, 's-', color='#A23B72', linewidth=2.5,
                        markersize=6, label='旋转误差 (°)', markerfacecolor='white', markeredgewidth=2)
        ax.set_ylabel('误差值', fontsize=14, fontweight='bold',fontproperties=font)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9, shadow=True,prop=font)

    # 设置通用属性
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=14, fontweight='bold',fontproperties=font)
    ax.set_title(fig_title, fontsize=16, fontweight='bold', pad=20,fontproperties=font)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(min(epochs) - 1, max(epochs) + 1)

    # 保存图片时明确指定字体
    fig.tight_layout()

    # 方法1: 保存为PNG（确保字体嵌入）
    os.makedirs(save_dir_name, exist_ok=True)
    plt.savefig(save_dir_name+'/'+f'{save_path}.png', dpi=300, bbox_inches='tight',
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
    base_dir = "/home/kiriyamagk/桌面/0628_FORMAL_RESULTS/好的结果/2025-10_31_00-00-00"
    part_idx = 5
    # part_idx = 1
    save_dir_name = "sim_without_dagger"

    save_path = "offline_learning_errors"
    fig_title = "离线学习方法的平均误差"
    epochs_to_process = "all"

    # 加载数据
    res_list = load_error_data(base_dir, part_idx, epochs_to_process)

    # 绘制图表
    if res_list:
        plot_errors(res_list, fig_title,save_path,save_dir_name)
        print_statistics(res_list)
    else:
        print("没有找到有效数据")