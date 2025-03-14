import os

def find_largest_num_of_dir(parent_dir):
    max_number = 0
    # 遍历所有文件夹
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)

        # 确保它是文件夹
        if os.path.isdir(folder_path):
            try:
                # 将文件夹名称转换为数字
                current_number = int(folder_name)

                # 检查数字是否是当前最大的
                if current_number > max_number:
                    max_number = current_number
            except ValueError:
                # 如果转换失败，忽略非数字文件夹名称
                continue
    return os.path.join(parent_dir, str(max_number))

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)