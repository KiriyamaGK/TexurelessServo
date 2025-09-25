import numpy as np
import torch
import cv2
import random
import os


def augment_lighting_for_image_np(
    img_np,
    scale_range_min=0.3,
    scale_range_max=1.8,
    offset_range_min=-0.3,
    offset_range_max=0.3,
    noise_std=0.1
):
    # 确保输入是 float32 类型，范围在 [0, 1]
    img_float = img_np.astype(np.float32) / 255.0  # [H, W, C]

    # 生成随机的缩放因子（每个通道独立）
    scale = np.random.uniform(
        scale_range_min,
        scale_range_max,
        size=(3,)
    ).reshape(1, 1, 3)  # [1, 1, 3]

    # 生成随机的偏移量（每个通道独立）
    offset = np.random.uniform(
        offset_range_min,
        offset_range_max,
        size=(3,)
    ).reshape(1, 1, 3)  # [1, 1, 3]

    # 生成高斯噪声（均值为0，标准差=noise_std）
    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=img_float.shape
    )

    # 应用光照增强
    img_augmented = img_float * scale + offset + noise

    # 限制在 [0, 1] 范围内
    img_augmented = np.clip(img_augmented, 0.0, 1.0)

    # 转换回 uint8 [0, 255]
    img_augmented = (img_augmented * 255).astype(np.uint8)

    return img_augmented

def augment_lighting_for_image(img_np, scale_range_min=0.3, scale_range_max=1.8,
                               offset_range_min=-0.3, offset_range_max=0.3, noise_std=0.1):

    # 转换为PyTorch张量并移到GPU
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().cuda() / 255.0 # [3,H,W]

    # 生成随机的缩放因子(每个通道独立)
    scale = torch.tensor([
        np.random.uniform(scale_range_min, scale_range_max),
        np.random.uniform(scale_range_min, scale_range_max),
        np.random.uniform(scale_range_min, scale_range_max)
    ]).cuda().view(3, 1, 1)

    # 生成随机的偏移量(每个通道独立)
    offset = torch.tensor([
        np.random.uniform(offset_range_min, offset_range_max),
        np.random.uniform(offset_range_min, offset_range_max),
        np.random.uniform(offset_range_min, offset_range_max)
    ]).cuda().view(3, 1, 1)

    # 生成高斯噪声
    noise = torch.normal(mean=0, std=noise_std, size=img_tensor.shape).cuda()

    # 应用光照增强
    img_tensor = torch.clamp(img_tensor * scale + offset + noise, 0.0, 1.0)

    # 转换回NumPy数组
    img_augmented = img_tensor.permute(1, 2, 0).cpu().numpy()*255
    img_augmented=img_augmented.astype(np.uint8)
    return img_augmented

def simple_retinex(img, sigma=80):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = cv2.addWeighted(img, 1.0, blurred, -1.0, 128)
    return retinex


def apply_solid_color_rectangle_non_target(img, target_mask, max_attempts=50):
    """在非目标区域添加纯色矩形"""
    h, w = img.shape[:2]
    non_target_mask = get_non_target_mask(target_mask)
    integral = cv2.integral(non_target_mask)

    for _ in range(max_attempts):
        rect_h = np.random.randint(20, min(100, h // 4))
        rect_w = np.random.randint(20, min(100, w // 4))

        pos = find_valid_rect_position(integral, h, w, rect_h, rect_w)
        if pos is not None:
            y, x = pos
            color = np.random.randint(0, 256, 3)
            img_copy = img.copy()
            img_copy[y:y + rect_h, x:x + rect_w] = color
            return img_copy, True

    return img, False

def get_non_target_mask(target_mask):
    """生成标准的0-255非目标掩膜（255=非目标区）"""
    return ((1 - target_mask) * 255).astype(np.uint8)

def swap_random_rectangles_non_target(img, target_mask, max_attempts=50):
    """
    严格在非目标区域交换两个同尺寸矩形
    返回：增强后的图像，是否成功交换
    """
    h, w = img.shape[:2]
    non_target_mask = get_non_target_mask(target_mask)
    integral = cv2.integral(non_target_mask)

    for _ in range(max_attempts):
        # 随机生成矩形尺寸（确保足够大）
        rect_h = np.random.randint(20, min(100, h // 4))
        rect_w = np.random.randint(20, min(100, w // 4))

        # 寻找第一个有效矩形位置
        pos1 = find_valid_rect_position(integral, h, w, rect_h, rect_w)
        if pos1 is None: continue

        # 寻找第二个有效位置（不与第一个重叠）
        pos2 = find_valid_rect_position(
            integral, h, w, rect_h, rect_w,
            exclude_pos=pos1 + (rect_h, rect_w)
        )
        if pos2 is None: continue

        # 执行交换
        y1, x1 = pos1
        y2, x2 = pos2
        img_copy = img.copy()
        rect1 = img[y1:y1 + rect_h, x1:x1 + rect_w].copy()
        rect2 = img[y2:y2 + rect_h, x2:x2 + rect_w].copy()

        img_copy[y1:y1 + rect_h, x1:x1 + rect_w] = rect2
        img_copy[y2:y2 + rect_h, x2:x2 + rect_w] = rect1
        return img_copy, True

    return img, False


def find_valid_rect_position(integral, img_h, img_w, rect_h, rect_w, exclude_pos=None, max_attempts=1000):
    """
    改进版有效位置查找（解决总是返回None的问题）

    参数：
        integral: 积分图（必须由0-255的掩膜生成）
        img_h, img_w: 图像高宽
        rect_h, rect_w: 矩形高宽
        exclude_pos: 要避开的区域 (y, x, h, w)
        max_attempts: 最大随机尝试次数

    返回：
        (y, x) 或 None（找不到时）
    """
    # 方法1：随机采样+验证（更高效）
    for _ in range(max_attempts):
        y = np.random.randint(0, img_h - rect_h)
        x = np.random.randint(0, img_w - rect_w)

        # 检查与排除区域的重叠
        if exclude_pos and check_overlap(y, x, rect_h, rect_w, *exclude_pos):
            continue

        # 验证是否完全在非目标区
        total = integral[y + rect_h, x + rect_w] - integral[y, x + rect_w] - integral[y + rect_h, x] + integral[y, x]
        if total == rect_h * rect_w * 255:
            return (y, x)

    # 方法2：保底的全扫描（确保一定能找到）
    valid_positions = []
    for y in range(img_h - rect_h):
        for x in range(img_w - rect_w):
            if exclude_pos and check_overlap(y, x, rect_h, rect_w, *exclude_pos):
                continue

            total = integral[y + rect_h, x + rect_w] - integral[y, x + rect_w] - integral[y + rect_h, x] + integral[
                y, x]
            if total == rect_h * rect_w * 255:
                valid_positions.append((y, x))

    return random.choice(valid_positions) if valid_positions else None

def check_overlap(y1, x1, h1, w1, y2, x2, h2, w2):
    """检查两个矩形是否重叠"""
    return not (x1 + w1 <= x2 or
               x2 + w2 <= x1 or
               y1 + h1 <= y2 or
               y2 + h2 <= y1)

def process_image_with_yolo(img, model,color_channel_inv=False):
    """
    Process image with YOLO and return target mask (binary mask of highest confidence detection)
    """
    # if color_channel_inv:
    #     img = img[:, :, ::-1].copy()

    results = model(img) if not color_channel_inv else model(img[:, :, ::-1])

    if len(results) == 0:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    # Get the result with highest confidence
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    # Get the highest confidence detection
    confidences = result.boxes.conf.cpu().numpy()
    best_idx = np.argmax(confidences)

    # Get mask for the highest confidence detection
    if result.masks is not None and len(result.masks) > best_idx:
        mask = result.masks[best_idx].data.cpu().numpy()[0]  # [H, W]
        return (mask > 0).astype(np.uint8)
    else:
        # If no mask, use bounding box
        x1, y1, x2, y2 = result.boxes.xyxy[best_idx].cpu().numpy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[int(y1):int(y2), int(x1):int(x2)] = 1
        return mask


def draw_bounding_box(img, mask, color=(0, 255, 0), thickness=2):
    """
    Draw the bounding box of the mask on the image
    """
    # Find the bounding box coordinates
    y, x = np.where(mask > 0)
    if len(y) == 0 or len(x) == 0:
        return img  # No mask to draw

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    img_with_bbox = img.copy()
    cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), color, thickness)
    return img_with_bbox


class AugmentationModule():
    def __init__(self, 
                 pretrained_model_pth,
                 scale_range_min=0.3, scale_range_max=1.8,
                 offset_range_min=-0.3, offset_range_max=0.3,
                 noise_std=0.1, draw_box=False, box_color=(0, 255, 0), box_thickness=2):
        os.environ['YOLO_VERBOSE'] = 'False' #abandon yolo logger printing
        from ultralytics import YOLO

        self.model = YOLO(pretrained_model_pth)
        self.scale_range_min = scale_range_min
        self.scale_range_max = scale_range_max
        self.offset_range_min = offset_range_min
        self.offset_range_max = offset_range_max
        self.noise_std = noise_std
        self.draw_box = draw_box
        self.box_color = box_color
        self.box_thickness = box_thickness

    def augment_image(self,img, color_channel_inv=False):
        """
        Apply all augmentations to an image with safety checks and visualization options

        Args:
            img: Input image (numpy array)
            model: YOLO model for target detection
            color_channel_inv: Whether to invert color channels
            scale_range_min/max: Lighting scale range
            offset_range_min/max: Lighting offset range
            noise_std: Noise standard deviation
            draw_box: Whether to draw bounding box
            box_color: Color for bounding box (BGR format)
            box_thickness: Thickness of bounding box lines

        Returns:
            Augmented image with optional bounding box visualization
        """
        # 1. Get target mask from YOLO
        target_mask = process_image_with_yolo(img, self.model,color_channel_inv)

        # 2. Apply lighting augmentation (preserves target area)
        img_aug = augment_lighting_for_image_np(
            img,
            scale_range_min=self.scale_range_min,
            scale_range_max=self.scale_range_max,
            offset_range_min=self.offset_range_min,
            offset_range_max=self.offset_range_max,
            noise_std=self.noise_std
        )

        # 3. Apply rectangle swap augmentation to non-target area (with safety checks)
        img_aug, swap_success = swap_random_rectangles_non_target(img_aug, target_mask)

        # 4. Apply solid color rectangle to non-target area (with safety checks)
        img_aug, solid_success = apply_solid_color_rectangle_non_target(img_aug, target_mask)

        # 5. Draw the bounding box if requested
        if self.draw_box:
            img_aug = draw_bounding_box(
                img_aug,
                target_mask,
                color=self.box_color,
                thickness=self.box_thickness
            )
        return img_aug


if __name__ == '__main__':
    img = cv2.imread('/media/noematrix/One Touch/AlignAnything_real/25.06.22/hdf5/goal_images/img1/0.png')
    img=img[:,:,::-1].copy()
    # img_aug = augment_lighting_for_image(img_np=img)
    cv2.imshow("img",img)
    cv2.waitKey(0)

    from skimage import exposure
    # adjusted = exposure.adjust_gam ma(img, gamma=0.5)  # 伽马校正
    adjusted = simple_retinex(img)
    cv2.imshow('adjusted', adjusted)
    cv2.waitKey(0)
