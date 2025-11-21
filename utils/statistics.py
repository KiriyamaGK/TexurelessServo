import json


def determine_key_and_append(dic,lst,key):
    '''
    if key in dic:
    lst.append(key)
    '''
    if key in dic.keys() and dic[key] is not None:
        lst.append(dic[key])
    return lst

def calculate_statistics(dic):
    '''
    :param dic: key:metrics value:final_error
    :return:
    '''
    summary = {}
    for k,v in dic.items():
        if v is None:
            continue
        else:
            if k not in summary.keys():
                summary[k]={}
            summary[k]["mean"] = sum(v)/len(v)
            summary[k]["max"] = max(v)
            summary[k]["min"] = min(v)

    return summary

def calculate_success_rate(data, output_file="success_rate.json"):
    obj_stats = {}
    total_success = 0
    total_attempts = 0

    # 计算每个对象的成功次数和尝试次数
    for idx in range(len(data)):
        obj_id = str(data[idx][0])
        success = data[idx][1]
        if obj_id not in obj_stats:
            obj_stats[obj_id] = {"total": 0, "success": 0}

        obj_stats[obj_id]["total"] += 1
        obj_stats[obj_id]["success"] += success

        total_success += success
        total_attempts += 1

    # 构建最终结果
    result_dict = {}
    for obj_id, stats in obj_stats.items():
        result_dict[obj_id] = {
            "success_num": stats["success"],
            "attempts": stats["total"],
            "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
        }

    result_dict["total"] = {
        "success_num": total_success,
        "attempts": total_attempts,
        "success_rate": total_success / total_attempts if total_attempts > 0 else 0,
    }

    print("==============================")
    print("Success Rate Report:")
    print(f"{'Object ID':<10} {'Success Num':<12} {'Attempts':<9} {'Success Rate':<12}")
    print("-" * 45)
    for obj_id, stats in result_dict.items():
        print(f"{obj_id:<10} {stats['success_num']:<12} {stats['attempts']:<9} {stats['success_rate']:.2f}")

    # 保存结果到文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)

    print(f"\nResults have been saved to {output_file}")

    return result_dict

def calculate_plug_success_rate(data, output_file="plug_success_rate.json"):
    obj_stats = {}
    total_success = 0
    total_attempts = 0

    # 计算每个对象的成功次数和尝试次数
    for idx in range(len(data)):
        obj_id = str(data[idx][0])
        success = data[idx][1]
        if obj_id not in obj_stats:
            obj_stats[obj_id] = {"total": 0, "success": 0}

        obj_stats[obj_id]["total"] += 1
        obj_stats[obj_id]["success"] += success

        total_success += success
        total_attempts += 1

    # 构建最终结果
    result_dict = {}
    for obj_id, stats in obj_stats.items():
        result_dict[obj_id] = {
            "success_num": stats["success"],
            "attempts": stats["total"],
            "success_rate": stats["success"] / stats["total"] if stats["total"] > 0 else 0,
        }

    result_dict["total"] = {
        "success_num": total_success,
        "attempts": total_attempts,
        "success_rate": total_success / total_attempts if total_attempts > 0 else 0,
    }

    print("==============================")
    print("Plog Success Rate Report:")
    print(f"{'Object ID':<10} {'Success Num':<12} {'Attempts':<9} {'Success Rate':<12}")
    print("-" * 45)
    for obj_id, stats in result_dict.items():
        print(f"{obj_id:<10} {stats['success_num']:<12} {stats['attempts']:<9} {stats['success_rate']:.2f}")

    # 保存结果到文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)

    print(f"\nResults have been saved to {output_file}")

    return result_dict

def visualize_final_error(data, output_file="final_error.json"):
    obj_trans_stats = {}
    obj_rot_stats = {}
    obj_z_error_stats = {}

    obj_pos_xyz_stats = {}
    obj_pos_rot_stats = {}
    obj_pos_z_stats = {}

    for idx in range(len(data)):
        obj_id = data[idx]["obj_id"]
        if obj_id not in obj_trans_stats:
            obj_trans_stats[obj_id] = []
            obj_rot_stats[obj_id] = []
            obj_z_error_stats[obj_id] = []
            obj_pos_xyz_stats[obj_id] = []
            obj_pos_z_stats[obj_id] = []
            obj_pos_rot_stats[obj_id] = []

        #lists appending
        obj_trans_stats[obj_id] = determine_key_and_append(dic=data[idx],lst=obj_trans_stats[obj_id],key="final_trans_error")
        obj_rot_stats[obj_id] = determine_key_and_append(dic=data[idx],lst=obj_rot_stats[obj_id],key="final_rot_error")
        obj_z_error_stats[obj_id] = determine_key_and_append(dic=data[idx],lst=obj_z_error_stats[obj_id],key="final_z_error")
        obj_pos_xyz_stats[obj_id] = determine_key_and_append(dic=data[idx], lst=obj_pos_xyz_stats[obj_id],key="final_pos_xyz_error")
        obj_pos_z_stats[obj_id] = determine_key_and_append(dic=data[idx], lst=obj_pos_z_stats[obj_id],key="final_pos_z_error")
        obj_pos_rot_stats[obj_id] = determine_key_and_append(dic=data[idx], lst=obj_pos_rot_stats[obj_id],key="final_pos_rot_error")

    stats_summary = {}
    for obj_id in obj_trans_stats:
        stats_summary[obj_id] =calculate_statistics({  # stats_summary[obj_id] does NOT contain none
            "translation_xyz":obj_trans_stats[obj_id],
            "rotation":obj_rot_stats[obj_id],
            "translation_z":obj_z_error_stats[obj_id] if len(obj_z_error_stats[obj_id]) > 0 else None,
            "pose_estimation_xyz":obj_pos_xyz_stats[obj_id] if len(obj_pos_xyz_stats[obj_id]) > 0 else None,
            "pose_estimation_z": obj_pos_rot_stats[obj_id] if len(obj_pos_rot_stats[obj_id]) > 0 else None, # xxx_data may contain none
            "pose_estimation_rot": obj_pos_rot_stats[obj_id] if len(obj_pos_rot_stats[obj_id]) > 0 else None ,
        })
        stats_summary[obj_id]["attempts"]=len(obj_trans_stats[obj_id])


    print("==============================")
    print("Final Error Statistics:")
    for obj_id, stats in stats_summary.items():
        print(f"Object ID: {obj_id}")
        print(f"  Attempts: {stats['attempts']}")
        for key in stats.keys():
            if key=="attempts":
                continue
            else:
                print(f"  {key} - Mean: {stats[key]['mean']:.4f}, "
                      f"Max: {stats[key]['max']:.4f}, "
                      f"Min: {stats[key]['min']:.4f}")
    with open(output_file, "w") as f:
        json.dump(stats_summary, f, indent=4)

    print(f"Statistics saved to {output_file}")


if __name__ == "__main__":
    # 示例数据
    data = [
        [1, 1],
        [1, 0],
        [2, 1],
        [2, 1],
        [3, 0],
        [3, 0],
        [3, 1]
    ]

    # 调用函数
    calculate_success_rate(data)