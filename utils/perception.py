import random

import numpy as np
import open3d as o3d
import pybullet as p
from math import tan


class CameraIntrinsic(object):
    """Intrinsic parameters of a pinhole camera model.

    Attributes:
        width (int): The width in pixels of the camera.
        height(int): The height in pixels of the camera.
        K: The intrinsic camera matrix.
    """

    def __init__(self, width=None, height=None, fx=None, fy=None, cx=None, cy=None,fov=None,scale=None,near=None,far=None,useK=None):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fov = fov
        self.scale = scale
        self.near = near
        self.far = far
        self.useK = useK

    @classmethod
    def from_dict(cls, data):
        """Deserialize intrinisic parameters from a dict object."""
        if data["build_with_K"]=="True":
            intrinsic = cls(
                width=data["width"],
                height=data["height"],
                fx=data["K"][0],
                fy=data["K"][4],
                cx=data["K"][2],
                cy=data["K"][5],
                near=data["near"],
                far=data["far"],
                useK=True,
            )
            return intrinsic
        elif data["build_with_K"]=="False":
            intrinsic = cls(
                width=data["width"],
                height=data["height"],
                fov=data['fov'],
                scale=data["width"]/data["height"],
                near=data["near"],
                far=data["far"],
                useK=False,
            )
            return intrinsic
    


class Camera(object):
    """Virtual RGB-D camera based on the PyBullet camera interface.

    Attributes:
        intrinsic: The camera intrinsic parameters.
    """

    def __init__(self, intrinsic: CameraIntrinsic):
        self.intrinsic = intrinsic
        self.near = intrinsic.near
        self.far = intrinsic.far
        self.useK=intrinsic.useK
        self.gl_proj_matrix = _build_projection_matrix(intrinsic, self.near, self.far,self.useK)
        if self.useK:
            self.gl_proj_matrix = self.gl_proj_matrix.flatten(order="F")

    def render(self, extrinsic, client=0):
        """Render synthetic RGB and depth images.

        Args:
            extrinsic: Extrinsic parameters, T_cam_ref (^{cam}_{world} T).
        """
        # Construct OpenGL compatible view and projection matrices.
        gl_view_matrix = extrinsic.copy() if extrinsic is not None else np.eye(4)
        gl_view_matrix[2, :] *= -1  # flip the Z axis
        gl_view_matrix = gl_view_matrix.flatten(order="F")

        result = p.getCameraImage(
            width=self.intrinsic.width,
            height=self.intrinsic.height,
            viewMatrix=gl_view_matrix,
            projectionMatrix=self.gl_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=client
        )

        # rgb, z_buffer = np.ascontiguousarray(result[2][:, :, :3]), result[3]
        if isinstance(result[2], np.ndarray):
            rgb, z_buffer = np.ascontiguousarray(result[2][:, :, :3]), result[3]
        else:
            # fix issue #2: https://github.com/hhcaz/CNS/issues/2
            H, W = self.intrinsic.height, self.intrinsic.width
            rgb = np.ascontiguousarray(np.asarray(result[2]).reshape(H, W, -1)[:, :, :3])
            z_buffer = np.asarray(result[3]).reshape(H, W).astype(np.float32)
            rgb = rgb.astype(np.uint8)
        
        depth = (
            1.0 * self.far * self.near / (self.far - (self.far - self.near) * z_buffer)
        )

        return Frame(rgb, depth, self.intrinsic, extrinsic)


class Frame(object):
    def __init__(self, rgb, depth, intrinsic, extrinsic=None):
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(rgb),
            depth=o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )
        if intrinsic.useK:
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=intrinsic.width,
                height=intrinsic.height,
                fx=intrinsic.fx,
                fy=intrinsic.fy,
                cx=intrinsic.cx,
                cy=intrinsic.cy,
            )
        else:
            self.intrinsic =build_intrinsic_from_fovy_and_scale(intrinsic.fov,intrinsic.scale,intrinsic.width,intrinsic.height)

        self.extrinsic = extrinsic if extrinsic is not None \
            else np.eye(4)
    
    def color_image(self):
        return np.asarray(self.rgbd.color)
    
    def depth_image(self):
        return np.asarray(self.rgbd.depth)

    def point_cloud(self):
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=self.rgbd,
            intrinsic=self.intrinsic,
            extrinsic=self.extrinsic
        )

        return pc


def _build_projection_matrix(intrinsic, near, far,useK):
    if useK :
        perspective = np.array(
            [
                [intrinsic.fx, 0.0, -intrinsic.cx, 0.0],
                [0.0, intrinsic.fy, -intrinsic.cy, 0.0],
                [0.0, 0.0, near + far, near * far],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )
        ortho = _gl_ortho(0.0, intrinsic.width, intrinsic.height, 0.0, near, far)
        return np.matmul(ortho, perspective)@np.diag([-1.0, -1.0, 1.0, 1.0])
    else:
        return p.computeProjectionMatrixFOV(intrinsic.fov, intrinsic.scale, near, far)


def _gl_ortho(left, right, bottom, top, near, far):
    ortho = np.diag(
        [2.0 / (right - left), 2.0 / (top - bottom), -2.0 / (far - near), 1.0]
    )
    ortho[0, 3] = -(right + left) / (right - left)
    ortho[1, 3] = -(top + bottom) / (top - bottom)
    ortho[2, 3] = -(far + near) / (far - near)
    return ortho

def build_intrinsic_from_fovy_and_scale(fovy, scale,width,height):
    fy=height/(2*tan(fovy/2/180*np.pi))
    fx=fy/scale
    cx=width/2
    cy=height/2
    return o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,)

if __name__ == "__main__":
    import json
    import pybullet_data
    import matplotlib.pyplot as plt

    with open("../configs/demo_collection.json", "r") as j:
        config = json.load(j)
    camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])
    if config["intrinsic"]["build_with_K"] == "False":
        intr= build_intrinsic_from_fovy_and_scale(config["intrinsic"]['fov'], config["intrinsic"]['width']/config["intrinsic"]['height'],config["intrinsic"]['width'],config["intrinsic"]['height']).intrinsic_matrix
        camera_intrinsic.fx=intr[0,0]
        camera_intrinsic.fy=intr[1,1]
        camera_intrinsic.cx=intr[0,2]
        camera_intrinsic.cy=intr[1,2]

    camera = Camera(camera_intrinsic)

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf', [0, 0, 0], [0, 0, 0, 1])
    p.loadURDF("lego/lego.urdf", [0.2, 0, 0.01], [0, 0, 0, 1])
    p.loadURDF("lego/lego.urdf", [0, 0, 0.3], [0, 0, 0, 1])
    p.loadURDF("duck_vhacd.urdf", [0, 0.2, 0.01], [0, 0, 0, 1])

    cam_x = np.array([1, 0, 0])
    cam_y = np.array([0, -1, 0])
    cam_z = np.array([0, 0, -1])

    cam_extr = np.eye(4)
    cam_extr[:3, :3] = np.stack([cam_x, cam_y, cam_z], axis=1)
    cam_extr[:3, 3] = [0, 0, 1]
    frame = camera.render(np.linalg.inv(cam_extr))

    rgb = frame.color_image()
    depth = frame.depth_image()
    W, H = camera.intrinsic.width, camera.intrinsic.height
    print(np.asarray(frame.point_cloud().points).shape)
    pc = np.asarray(frame.point_cloud().points).reshape(H, W, 3)

    uv = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack(uv, axis=-1).reshape(H*W, 2)


    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(rgb)
    plt.tight_layout()
    plt.show()

    print(pc)
    print("center = {}".format(pc[H//2, W//2]))
