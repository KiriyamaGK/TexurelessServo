import numpy as np
import torch
import cv2
import random

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
