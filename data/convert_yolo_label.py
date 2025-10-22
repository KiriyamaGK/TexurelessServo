import os
import glob
import yaml


def convert_to_single_class(data_yaml_path, label_dir, keep_class_id):
    """
    将多类别数据集转换为单类别数据集

    Args:
        data_yaml_path: data.yaml文件路径
        label_dir: 标注文件目录
        keep_class_id: 要保留的原始类别ID
    """

    # 1. 读取并修改data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # 获取原始类别名称
    original_names = data['names']
    if keep_class_id >= len(original_names):
        print(f"错误: keep_class_id {keep_class_id} 超出范围 (0-{len(original_names) - 1})")
        return

    # 确定新的类别名称
    kept_class_name = original_names[keep_class_id]

    # 更新data.yaml数据
    data['nc'] = 1
    data['names'] = [kept_class_name]

    # 保存修改后的data.yaml
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"已更新 data.yaml: nc=1, names=['{kept_class_name}']")

    # 2. 修改所有标注文件
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    print(f"processing {label_dir}....")
    for txt_file in txt_files:
        print(f"processing {txt_file}...")
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                # 只保留指定类别的标注，并将其ID改为0
                if class_id == keep_class_id:
                    new_line = "0 " + " ".join(parts[1:5]) + "\n"
                    new_lines.append(new_line)

        # 写回文件
        with open(txt_file, 'w') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    # 使用示例：只保留原来的类别1（在数据集中可能是第二个类别）
    base_dir = "/media/kiriyamagk/One Touch/AlignAnything_real/1021yolovn.v3i.yolov9"
    convert_key_lst = ["train","test","valid"]
    for convert_key in convert_key_lst:
        label_pth = os.path.join(base_dir, convert_key,"labels")
        yaml_pth = os.path.join(base_dir ,"data.yaml")
        convert_to_single_class(yaml_pth, label_pth, keep_class_id=0)