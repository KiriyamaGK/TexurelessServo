def manual_nms(boxes, scores, iou_threshold=0.5):
    """
    手动实现NMS
    """
    if len(boxes) == 0:
        return []

    # 按置信度排序
    sorted_indices = np.argsort(scores)[::-1]

    keep = []
    while sorted_indices.size > 0:
        # 取当前置信度最高的框
        current_idx = sorted_indices[0]
        keep.append(current_idx)

        if sorted_indices.size == 1:
            break

        # 计算当前框与剩余框的IoU
        current_box = boxes[current_idx]
        remaining_boxes = boxes[sorted_indices[1:]]

        # 计算IoU
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
                    remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_current + area_remaining - intersection

        iou = intersection / union

        # 保留IoU低于阈值的框
        remaining_indices = np.where(iou < iou_threshold)[0]
        sorted_indices = sorted_indices[remaining_indices + 1]

    return keep