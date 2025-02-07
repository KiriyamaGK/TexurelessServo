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

def image_preprocess(img:np.ndarray,bgr2rgb:bool=False):
    assert img.shape[-1] in [1,3]
    if bgr2rgb:
        img=img[:,:,::-1]
    img=torch.from_numpy(img).type(torch.float32)
    if img.shape[-1] == 3:
        img = img / 255.
    img = img.permute(2, 0, 1)
    return img


def input_dict_preprocess(dic:dict,bgr2rgb:bool=False,rollout=False):
    for k,v in dic.items():
        if 'img' in k or "image" in k:
            if rollout:
                assert len(dic[k].shape)==3
                dic[k] = image_preprocess(v, bgr2rgb=bgr2rgb)
                dic[k]=dic[k].unsqueeze(0).unsqueeze(0) #[1,1,h,w,c]
            else:
                assert len(dic[k].shape) == 4
                t,h,w,c=dic[k].shape[0:4]
                dic[k] = torch.stack([image_preprocess(dic[k][i],bgr2rgb=bgr2rgb) for i in range(t)])
        else:
            if rollout:
                assert len(dic[k].shape) == 1  # [t=1,]
                dic[k] = dic[k][np.newaxis, np.newaxis, :]  # [1,1,1,]
            else:
                pass                                        # [t,]
            dic[k] = torch.from_numpy(dic[k]).type(torch.float32)
    return dic

