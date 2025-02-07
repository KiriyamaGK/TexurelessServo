import numpy as np
from stl import mesh
import os

def get_stl_geometry(file_base,obj_id):
    file_path=os.path.join(file_base,'part{}.STL'.format(obj_id))
    stl_mesh = mesh.Mesh.from_file(file_path)
    # 获取三维坐标数组
    vertices = stl_mesh.vectors
    x_min = np.min(vertices[:, :, 0])
    x_max = np.max(vertices[:, :, 0])
    y_min = np.min(vertices[:, :, 1])
    y_max = np.max(vertices[:, :, 1])
    z_min = np.min(vertices[:, :, 2])
    z_max = np.max(vertices[:, :, 2])
    assert x_max == -x_min
    return (y_max+x_max)/2,z_max-z_min

