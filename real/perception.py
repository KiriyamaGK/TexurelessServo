import time
import os
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

        self.print_connected_realsense_serial_numbers()
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

                # 获取传感器并设置参数
                sensor = cfg.get_device().first_color_sensor()

                # 禁用自动白平衡，设置固定值(如5500K)
                sensor.set_option(rs.option.enable_auto_white_balance, 0)
                sensor.set_option(rs.option.white_balance, 5000)  # 根据环境调整

                # 禁用自动曝光，设置固定曝光值
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, 320)  # 典型值，根据环境调整
                sensor.set_option(rs.option.gain,64)

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

    def print_connected_realsense_serial_numbers(self):
        # 创建上下文对象
        ctx = rs.context()

        # 获取所有连接的设备
        devices = ctx.query_devices()

        print(f"找到 {len(devices)} 个RealSense设备:")

        # 遍历所有设备并打印序列号
        for i, device in enumerate(devices):
            serial_number = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"设备 {i + 1}:")
            print(f"  名称: {name}")
            print(f"  序列号: {serial_number}")
            print()


if __name__ == "__main__":
    # 设备配置
    devices = {
        "img_1": "215222073421",
        "img_2": "233622076143"
    }
    # task = "make_dataset"  #"make_dataset" or "vis_detect"
    task = None
    use_tracker = True
    #主要比较yolov8,train5,midbigdown,light
    # yolo_model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train5/weights/best.pt"
    # yolo_model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train6(yolov8)/weights/best.pt"
    yolo_model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train11(yolo_midbigdown)/weights/best.pt"
    # yolo_model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train8(yolo_light)/weights/best.pt"

    # yolo_model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train12(yolo_newnew)/weights/best.pt"
    # yolo_model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train9(yolo_mid)/weights/best.pt"
    # yolo_model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train10(yolo_midbig)/weights/best.pt"
    color_channel_inv = False
    do_save_detect_results = False

    save_freq = 2
    _do_make_dataset = task == "make_dataset"
    _do_vis_detect = task == "vis_detect"
    do_save_detect_results = do_save_detect_results and _do_vis_detect

    #if make dataset
    if _do_make_dataset:
        save_base_dir = "/home/kiriyamagk/桌面/track_dataset/raw" # if do make dataset
        take_pic_inteval = 0.5 # if do make dataset, unit is second

    #if vis detect
    if _do_vis_detect:
        from utils.detection import get_detect_result
        if yolo_model_pth is not None:
            from ultralytics import YOLO
            detect_model = YOLO(yolo_model_pth)
        else:
            detect_model = None
    else:
        yolo_model_pth = None

    if _do_make_dataset:
        import os
        import datetime
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(save_base_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        last_t = time.time()
        initial_pic = True
    if do_save_detect_results:
        save_base = os.path.dirname(os.path.dirname(yolo_model_pth)) + f"/{int(time.time())}"
        save_base_raw = save_base + "/eval_results/raw"
        save_base_dect = save_base + "/eval_results/dect"
        os.makedirs(save_base_dect, exist_ok=True)
        os.makedirs(save_base_raw, exist_ok=True)

    try:
        # 创建相机实例
        camera = Camera(
            devices=devices,
            use_devices_type=["img_1","img_2"],
            width=640,
            height=480,
            fps=30,
        )
        last_save_t = time.time()
        idx = 0
        while True:
            # 获取帧
            frames = camera.get_frame()
            if frames:
                for name, img in frames.items():
                    if _do_vis_detect:
                        res_dict = get_detect_result(detect_model = detect_model,img=img,tracker_enabled=use_tracker,color_channel_inv=color_channel_inv)
                        dect_img = res_dict["res_img"]
                    # cv2.imshow(name, img)
                    cv2.imshow(name, img) if not _do_vis_detect else cv2.imshow(name, dect_img)
                    cv2.waitKey(1)

                    if do_save_detect_results and time.time() - last_save_t > 1/save_freq:
                        cv2.imwrite(save_base_dect + f"/{str(idx).zfill(5)}.png",dect_img)
                        cv2.imwrite(save_base_raw + f"/{str(idx).zfill(5)}.png", img)
                        last_save_t = time.time()
                        idx += 1

                    if _do_make_dataset and ((time.time() - last_t) >= take_pic_inteval or initial_pic):
                        if initial_pic:
                            os.makedirs(os.path.join(save_dir, name), exist_ok=True)
                        spec_save_dir = os.path.join(save_dir, name, str(idx).zfill(5)+".png")
                        cv2.imwrite(spec_save_dir, img)
                        idx += 1
                        last_t = time.time()
                        if initial_pic:
                            initial_pic = False
            else:
                print("Error occured while getting frame.")

    finally:
        # 释放资源
        camera.release()
        cv2.destroyAllWindows()