from ultralytics import YOLO
import numpy as np
from utils.augmentation import draw_bounding_box

def get_detect_result(detect_model:YOLO, img, color_channel_inv=True, tracker_enabled:bool=False,verbose=False,area_proportion = 0.0):
    assert detect_model is not None

    nms_config = {
        'conf': 0.25,  # 置信度阈值，过滤低置信度检测
        'iou': 0.45,  # NMS IoU阈值，值越小越严格
        'classes': None,  # 指定类别，None表示所有类别
        'agnostic_nms': False,  # 是否类别无关的NMS
        'max_det': 100,  # 每张图最大检测数量
    }

    # 使用跟踪模式或普通检测模式
    if tracker_enabled:
        results = detect_model.track(img, persist=True,**nms_config,verbose = verbose) if not color_channel_inv else detect_model.track(
            img[:, :, ::-1], persist=True,**nms_config,verbose = verbose)
    else:
        results = detect_model(img.copy(),**nms_config,verbose = verbose) if not color_channel_inv else detect_model(img.copy()[:, :, ::-1],**nms_config,verbose = verbose)

    res_dict = {
        "res_img": None,
        "bbox": [],
        "confidences": [],  # 新增：置信度
        "class_ids": [],  # 新增：类别ID
        "class_names": []  # 新增：类别名称
    }
    aug_img = img.copy()
    result = results[0]

    # 获取类别名称
    class_names = result.names

    for idx in range(len(result.boxes)):
        # 获取边界框坐标
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy()

        # 获取置信度和类别
        confidence = result.boxes.conf[idx].cpu().numpy()
        class_id = int(result.boxes.cls[idx].cpu().numpy())
        class_name = class_names[class_id]

        # 在图像上绘制边界框和标签
        label = f"{class_name} {confidence:.2f}"
        # 计算每个bbox的面积
        bbox_area = abs((x2 - x1) * (y2 - y1))
        img_area = img.shape[0] * img.shape[1]
        min_area_threshold = img_area * area_proportion  # 图像面积的10%

        # 有效性掩码：置信度阈值和bbox大小阈值
        if bbox_area > min_area_threshold:
            aug_img = draw_bounding_box(aug_img, bbox=[x1, y1, x2, y2], label=label)
            # 存储检测结果
            res_dict["bbox"].append([x1, y1, x2, y2])
            res_dict["confidences"].append(confidence)
            res_dict["class_ids"].append(class_id)
            res_dict["class_names"].append(class_name)

    res_dict["res_img"] = aug_img
    return res_dict