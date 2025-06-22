# import pyrealsense2 as rs
# import time
# import numpy as np
# import cv2

# class Camera(object):
#     '''
#     realsense相机处理类
#     '''
#
#     def __init__(self,devices,use_devices_type,width=640, height=480, fps=30):  # 图片格式可根据程序需要进行更改
#         assert isinstance(devices, dict) and isinstance(use_devices_type, list)
#
#         self.width = width
#         self.height = height
#
#         self.config = rs.config()
#
#         self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
#         self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
#
#         self.use_devices_type = use_devices_type #a list ,eg.:[wrist]
#
#         for itm in self.use_devices_type:
#             assert itm in devices.keys(), f"{itm} is not in devices"
#
#         self.devices={type: devices[type] for type in self.use_devices_type} #a dict describes device types and index numbers ,eg.:{"wrist":"215222073421"}
#
#         self.align = rs.align(rs.stream.color)
#
#         self.pipelines = {} #eg.:{"215222073421":pipeline}
#         self.profile={}     #eg.:{"215222073421":profile}
#         self.intr={}        #a dict describes camera intrinsics ,eg.:{"215222073421":K}
#
#         for type,ind in self.devices.items():  #type：相机种类,ind:序列号
#             try:
#                 pipeline = rs.pipeline()
#                 self.config.enable_device(ind)
#                 cfg=pipeline.start(self.config)
#                 self.pipelines[ind] = pipeline
#                 self.profile[ind] = cfg.get_stream(rs.stream.color)
#                 self.intr[ind] = self.profile[ind].as_video_stream_profile().get_intrinsics()
#
#                 print(f"{type} Camera {ind} started successfully.")
#             except Exception as e:
#                 print(f"Failed to start device {ind}: {e}")
#         # self.pipeline = rs.pipeline()
#         # self.config.enable_device("215222073421")
#         # self.pipeline.start(self.config)
#
#     def get_frame(self):
#         frame_dict={}
#         for device_type in self.use_devices_type:
#             try:
#                 device_index = self.devices[device_type]
#                 frameset = self.pipelines[device_index].wait_for_frames()
#                 aligned_frames = self.align.process(frameset)
#                 aligned_color_frame = aligned_frames.get_color_frame()
#
#                 if not aligned_color_frame:
#                     raise RuntimeError("Failed to get aligned color frame.")
#
#                 color_image = np.asanyarray(aligned_color_frame.get_data())
#                 frame_dict[device_type] = color_image
#
#             except Exception as e:
#                 print(f"Error in get_frame for camera {device_type}: {e}")
#                 return None
#         return frame_dict
#
#     def release(self):
#         for type,device in self.devices.items():
#             self.pipelines[device].stop()
#
#     def get_intrinsics(self,device):
#         return self.intr[device]
#
#
# if __name__ == '__main__':
#     devices = {"wrist":'215222073421',
#                "wrist2":"233622076143"}
#     camera = Camera(devices=devices,use_devices_type=["wrist"])
#     i=0
#     while True:
#
#         t_0 = time.time()
#         color_image = camera.get_frame()["wrist"]
#         color_image2=camera.get_frame()["wrist2"]
#         # cv2.namedWindow('left_img', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('left_img', color_image)
#         cv2.imshow('right_img', color_image2)
#         # if i <=5:
#         cv2.waitKey(1)
#         print(time.time()-t_0)
#         i+=1

import pyrealsense2 as rs
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


class Camera:
    """
    Realsense相机处理类，支持多相机管理

    参数:
        devices: 字典，键为设备类型，值为设备序列号
        use_devices_type: 列表，指定要使用的设备类型
        width: 图像宽度 (默认640)
        height: 图像高度 (默认480)
        fps: 帧率 (默认30)
    """

    def __init__(self,
                 devices: Dict[str, str],
                 use_devices_type: List[str],
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):

        # 参数验证
        if not isinstance(devices, dict) or not isinstance(use_devices_type, list):
            raise TypeError("devices must be a dict and use_devices_type must be a list")

        invalid_devices = [t for t in use_devices_type if t not in devices]
        if invalid_devices:
            raise ValueError(f"Device types {invalid_devices} not found in devices dict")

        self.width = width
        self.height = height
        self.fps = fps

        # 只保留要使用的设备
        self.devices = {dev_type: devices[dev_type] for dev_type in use_devices_type}

        # 初始化相机相关属性
        self.pipelines: Dict[str, rs.pipeline] = {}
        self.profiles: Dict[str, rs.video_stream_profile] = {}
        self.intrinsics: Dict[str, rs.intrinsics] = {}
        self.align = rs.align(rs.stream.color)

        # 初始化所有相机
        self._initialize_cameras()

    def _initialize_cameras(self) -> None:
        """初始化所有配置的相机"""
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        for dev_type, serial in self.devices.items():
            try:
                pipeline = rs.pipeline()
                config.enable_device(serial)
                cfg = pipeline.start(config)

                # 获取配置并存储
                self.pipelines[serial] = pipeline
                color_profile = cfg.get_stream(rs.stream.color)
                self.profiles[serial] = color_profile
                self.intrinsics[serial] = color_profile.as_video_stream_profile().get_intrinsics()

                print(f"{dev_type} Camera {serial} started successfully.")
            except Exception as e:
                print(f"Failed to start device {serial}: {e}")
                raise

    def get_frame(self) -> Optional[Dict[str, np.ndarray]]:
        """
        从所有相机获取帧

        返回:
            包含所有相机帧的字典，键为设备类型，值为BGR图像
            如果任何相机获取失败则返回None
        """
        frames = {}

        for dev_type, serial in self.devices.items():
            try:
                # 等待帧数据
                frameset = self.pipelines[serial].wait_for_frames()
                aligned_frames = self.align.process(frameset)

                # 获取彩色帧
                color_frame = aligned_frames.get_color_frame()
                if not color_frame:
                    print(f"No color frame received from {dev_type} camera")
                    return None

                frames[dev_type] = np.asanyarray(color_frame.get_data())

            except Exception as e:
                print(f"Error getting frame from {dev_type} camera: {e}")
                return None

        return frames

    def get_depth_frame(self, device_type: str) -> Optional[np.ndarray]:
        """
        获取指定相机的深度帧

        参数:
            device_type: 设备类型名称

        返回:
            深度图像 (16位) 或 None (如果失败)
        """
        if device_type not in self.devices:
            print(f"Device type {device_type} not found")
            return None

        serial = self.devices[device_type]
        try:
            frameset = self.pipelines[serial].wait_for_frames()
            aligned_frames = self.align.process(frameset)

            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                print(f"No depth frame received from {device_type} camera")
                return None

            return np.asanyarray(depth_frame.get_data())
        except Exception as e:
            print(f"Error getting depth frame: {e}")
            return None

    def get_intrinsics(self, device_type: str) -> Optional[rs.intrinsics]:
        """获取相机内参"""
        if device_type not in self.devices:
            print(f"Device type {device_type} not found")
            return None
        return self.intrinsics.get(self.devices[device_type], None)

    def get_camera_info(self, device_type: str) -> Optional[Dict]:
        """获取相机信息"""
        if device_type not in self.devices:
            return None

        serial = self.devices[device_type]
        intrinsics = self.intrinsics.get(serial)
        if not intrinsics:
            return None

        return {
            "serial": serial,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "model": str(intrinsics.model),
            "coeffs": intrinsics.coeffs
        }

    def release(self) -> None:
        """释放所有相机资源"""
        for serial in self.pipelines.values():
            try:
                serial.stop()
            except Exception as e:
                print(f"Error stopping pipeline: {e}")


if __name__ == "__main__":
    # 设备配置
    devices = {
        "wrist": "215222073421",
        "img_2": "233622076143"
    }

    try:
        # 创建相机实例
        camera = Camera(
            devices=devices,
            use_devices_type=["wrist"],
            width=1280,
            height=720,
            fps=30
        )
        while True:
            # 获取帧
            frames = camera.get_frame()
            if frames:
                for name, img in frames.items():
                    cv2.imshow(name, img)
                    cv2.waitKey(1)
            else:
                print(1)

            # 获取相机信息
            info = camera.get_camera_info("wrist")
            print(f"Camera info: {info}")

    finally:
        # 释放资源
        camera.release()
        cv2.destroyAllWindows()