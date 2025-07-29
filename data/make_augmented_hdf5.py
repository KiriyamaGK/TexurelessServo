import os
import time
from ultralytics import YOLO
import cv2
import numpy as np
import h5py
from utils.augmentation import apply_solid_color_rectangle_non_target,swap_random_rectangles_non_target,augment_lighting_for_image_np


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


def augment_image(img, model, color_channel_inv=False,
                 scale_range_min=0.3, scale_range_max=1.8,
                 offset_range_min=-0.3, offset_range_max=0.3,
                 noise_std=0.1, draw_box=False, box_color=(0, 255, 0), box_thickness=2):
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
    target_mask = process_image_with_yolo(img, model,color_channel_inv)

    # 2. Apply lighting augmentation (preserves target area)
    img_aug = augment_lighting_for_image_np(
        img,
        scale_range_min=scale_range_min,
        scale_range_max=scale_range_max,
        offset_range_min=offset_range_min,
        offset_range_max=offset_range_max,
        noise_std=noise_std
    )

    # 3. Apply rectangle swap augmentation to non-target area (with safety checks)
    img_aug, swap_success = swap_random_rectangles_non_target(img_aug, target_mask)

    # 4. Apply solid color rectangle to non-target area (with safety checks)
    img_aug, solid_success = apply_solid_color_rectangle_non_target(img_aug, target_mask)

    # 5. Draw the bounding box if requested
    if draw_box:
        img_aug = draw_bounding_box(
            img_aug,
            target_mask,
            color=box_color,
            thickness=box_thickness
        )
    return img_aug

if __name__ == '__main__':
    model = YOLO("/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train/weights/best.pt")
    dataset_path = "/media/kiriyamagk/One Touch/AlignAnything_real/25.06.23/hdf5/merged.hdf5"
    img_key_names = ["robot0_eye_in_hand_image", "robot0_eye_in_hand_image_2"]
    color_channel_inverse = True

    with h5py.File(dataset_path, "r+") as f:
        demo_num = len(f["data"])
        for i in range(demo_num):
            print(f"Processing demo {i}......")
            obs_paths = [f'data/demo_{i}/obs/' + n for n in img_key_names]
            for obs_path in obs_paths:
                imgs = f[obs_path][:]
                augmented_imgs = []
                for idx in range(imgs.shape[0]):
                    img = imgs[idx]
                    augmented_img = augment_image(img, model,color_channel_inverse,scale_range_min=0.87, scale_range_max=1.15,
                              offset_range_min=-0.1, offset_range_max=0.1, noise_std=0.07, draw_box=False, box_color=(0, 255, 0), box_thickness=2)
                    augmented_imgs.append(augmented_img)

                    # #debug
                    # cv2.imshow("img",augmented_img[:,:,::-1])
                    # cv2.waitKey(0)

                # Convert list to numpy array
                augmented_imgs = np.array(augmented_imgs)
                aug_path = obs_path + "_light"
                if aug_path in f:
                    del f[aug_path]
                f.create_dataset(aug_path, data=augmented_imgs)