import numpy as np
import matplotlib.pyplot as plt

# 定义高斯核函数
def gaussian_kernel(size, sigma=1.0):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return g


def gaussian_img(image_size, pts, kernel_size=100, sigma=2.0):
    image = np.zeros((image_size[0], image_size[1]))
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel_center = kernel_size // 2
    assert len(pts.shape) in [1,2]
    if len(pts.shape) == 1:
        pts = pts[np.newaxis,:]
    for i in range(pts.shape[0]):
        # 将点的坐标转换为整数
        pt_x, pt_y = int(pts[i, 0]), int(pts[i, 1])

        # 计算目标区域的边界
        x_start = max(0, pt_x - kernel_center)
        x_end = min(image_size[1]-1, pt_x + kernel_center)
        y_start = max(0, pt_y - kernel_center)
        y_end = min(image_size[0]-1, pt_y + kernel_center)

        # 计算高斯核的裁剪边界
        kernel_x_start = kernel_center - (pt_x - x_start)
        kernel_x_end = kernel_center + (x_end - pt_x)
        kernel_y_start = kernel_center - (pt_y - y_start)
        kernel_y_end = kernel_center + (y_end - pt_y)

        # 裁剪高斯核
        kernel_cropped = kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]

        if y_end>y_start and x_end>x_start:
            # if y_end-y_start!= kernel_cropped.shape[0] or x_end-x_start!= kernel_cropped.shape[1]:
            #     raise RuntimeError
            # print("x_start: {}, x_end: {}".format(x_start, x_end))
            # print("y_start: {}, y_end: {}".format(y_start, y_end))
            # print("pt_x: {}, pt_y: {}".format(pt_x, pt_y))
            # print("kernal:",kernel_cropped.shape)
            image[y_start:y_end, x_start:x_end] += kernel_cropped

    return image


