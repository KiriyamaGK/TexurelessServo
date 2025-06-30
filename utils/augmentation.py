import numpy as np
import torch
import cv2

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

if __name__ == '__main__':
    img = cv2.imread('/media/noematrix/One Touch/AlignAnything_real/25.06.22/hdf5/goal_images/img1/0.png')
    img_n=img[:,:,::-1].copy()
    img_aug = augment_lighting_for_image(img_np=img_n)
    cv2.imshow("img",img_aug)
    cv2.waitKey(0)
