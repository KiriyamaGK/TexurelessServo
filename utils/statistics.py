import json
import csv

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

    # 打印结果
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
def visualize_final_error(data, output_file="final_error.json"):
    obj_trans_stats = {}
    obj_rot_stats = {}
    for idx in range(len(data)):
        obj_id = str(data[idx][0])
        if obj_id not in obj_trans_stats:
            assert obj_id not in obj_rot_stats
            obj_trans_stats[obj_id] = []
            obj_rot_stats[obj_id] = []
        obj_trans_stats[obj_id].append(data[idx][1])
        obj_rot_stats[obj_id].append(data[idx][2])

    stats_summary = {}
    for obj_id in obj_trans_stats:
        trans_data = obj_trans_stats[obj_id]
        rot_data = obj_rot_stats[obj_id]

        stats_summary[obj_id] = {
            "translation": {
                "mean": sum(trans_data) / len(trans_data),
                "max": max(trans_data),
                "min": min(trans_data),
            },
            "rotation": {
                "mean": sum(rot_data) / len(rot_data),
                "max": max(rot_data),
                "min": min(rot_data),
            },
            "attempts": len(trans_data),
        }
    print("Final Error Statistics:")
    for obj_id, stats in stats_summary.items():
        print(f"Object ID: {obj_id}")
        print(f"Attempts: {stats['attempts']}")
        print(f"  Translation - Mean: {stats['translation']['mean']:.4f}, "
              f"Max: {stats['translation']['max']:.4f}, "
              f"Min: {stats['translation']['min']:.4f}")
        print(f"  Rotation - Mean: {stats['rotation']['mean']:.4f}, "
              f"Max: {stats['rotation']['max']:.4f}, "
              f"Min: {stats['rotation']['min']:.4f}")

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