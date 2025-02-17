import numpy as np
import torch
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clip_image(img,img_size):
    height,width = img.shape[0],img.shape[1]
    # 确定裁剪区域
    if width > height:
        left = (width - height) // 2
        top = 0
        right = (width + height) // 2
        bottom = height
    else:
        left = 0
        top = (height - width) // 2
        right = width
        bottom = (height + width) // 2
    # 裁剪图片
    img_cropped = img[top:bottom, left:right]  # 使用 NumPy 数组进行裁剪
    img_cropped=cv2.resize(img_cropped,(img_size,img_size))
    return img_cropped

def image_preprocess(img:np.ndarray,bgr2rgb:bool=False,img_size=None):
    if img_size is not None:
        if img_size>img.shape[0]:
            raise RuntimeError("Upsampling from hdf5 when training is forbidden!")
        elif img_size<img.shape[0]:
            img=cv2.resize(img,(img_size,img_size))
            if len(img.shape)!=3:
                img=img[:,:,np.newaxis]
    assert len(img.shape)==3 and img.shape[-1] in [1, 3]
    if bgr2rgb:
        img=img[:,:,::-1]
    img=torch.from_numpy(img).type(torch.float32)
    if img.shape[-1] == 3:
        img = img / 255.
    img = img.permute(2, 0, 1)
    return img


def input_dict_preprocess(dic:dict,bgr2rgb:bool=False,rollout=False,img_size=None):
    for k,v in dic.items():
        if 'img' in k or "image" in k:
            if rollout:
                assert len(dic[k].shape)==3
                dic[k] = image_preprocess(v, bgr2rgb=bgr2rgb,img_size=img_size)
                dic[k]=dic[k].unsqueeze(0).unsqueeze(0) #[1,1,h,w,c]
            else:
                assert len(dic[k].shape) == 4
                t,h,w,c=dic[k].shape[0:4]
                dic[k] = torch.stack([image_preprocess(dic[k][i],bgr2rgb=bgr2rgb,img_size=img_size) for i in range(t)])
        else:
            if rollout:
                assert len(dic[k].shape) == 1  # [t=1,]
                dic[k] = dic[k][np.newaxis, np.newaxis, :]  # [1,1,1,]
            else:
                pass                                        # [t,]
            dic[k] = torch.from_numpy(dic[k]).type(torch.float32)
    return dic

def make_scaled_img(img: torch.Tensor):
    alpha = 1.1 + (1.5 - 1.1) * torch.rand(1).item()
    beta = 3 + (10 - 3) * torch.rand(1).item()
    beta/=255.
    img = img.float()
    adjusted_image = torch.clamp(alpha * img + beta, 0, 1)
    return adjusted_image

def generate_gaussian_spot(size, sigma):
    """
    生成一个高斯斑点
    :param size: 高斯斑点的大小（正方形的边长）
    :param sigma: 高斯分布的标准差
    :return: 高斯斑点张量
    """
    x = torch.linspace(-size // 2, size // 2, size)
    y = torch.linspace(-size // 2, size // 2, size)
    x, y = torch.meshgrid(x, y, indexing='ij')
    spot = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return spot / spot.max()  # [0, 1]

def add_gaussian_spot_to_image(img: torch.Tensor, size:int,sigma:int, position,to_device=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #position:左上角点
    C, H, W = img.shape
    spot = generate_gaussian_spot(size, sigma)
    spot_size = spot.shape[0]
    x, y = position

    # 确保斑点位置在图像范围内
    x = max(0, min(x, W - spot_size))
    y = max(0, min(y, H - spot_size))

    if to_device:
        spot = spot.to(device)
    # 将斑点添加到图像上
    for c in range(C):
        img[c, y:y+spot_size, x:x+spot_size] += spot

    # 裁剪结果，确保值在 [0, 1] 范围内
    img = torch.clamp(img, 0, 1)
    return img
