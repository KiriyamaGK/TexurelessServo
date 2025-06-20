import pyrealsense2 as rs
import time
import numpy as np
import cv2

class Camera(object):
    '''
    realsense相机处理类
    '''

    def __init__(self,devices,use_devices_type,width=640, height=480, fps=30):  # 图片格式可根据程序需要进行更改
        assert isinstance(devices, dict) and isinstance(use_devices_type, list)

        self.width = width
        self.height = height

        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)

        self.use_devices_type = use_devices_type #a list ,eg.:[wrist]

        for itm in self.use_devices_type:
            assert itm in devices.keys(), f"{itm} is not in devices"

        self.devices={type: devices[type] for type in self.use_devices_type} #a dict describes device types and index numbers ,eg.:{"wrist":"215222073421"}

        self.align = rs.align(rs.stream.color)

        self.pipelines = {} #eg.:{"215222073421":pipeline}
        self.profile={}     #eg.:{"215222073421":profile}
        self.intr={}        #a dict describes camera intrinsics ,eg.:{"215222073421":K}

        for type,ind in self.devices.items():  #type：相机种类,ind:序列号
            try:
                pipeline = rs.pipeline()
                self.config.enable_device(ind)
                cfg=pipeline.start(self.config)
                self.pipelines[ind] = pipeline
                self.profile[ind] = cfg.get_stream(rs.stream.color)
                self.intr[ind] = self.profile[ind].as_video_stream_profile().get_intrinsics()

                print(f"{type} Camera {ind} started successfully.")
            except Exception as e:
                print(f"Failed to start device {ind}: {e}")
        # self.pipeline = rs.pipeline()
        # self.config.enable_device("215222073421")
        # self.pipeline.start(self.config)

    def get_frame(self):
        frame_dict={}
        for device_type in self.use_devices_type:
            try:
                device_index = self.devices[device_type]
                frameset = self.pipelines[device_index].wait_for_frames()
                aligned_frames = self.align.process(frameset)
                aligned_color_frame = aligned_frames.get_color_frame()

                if not aligned_color_frame:
                    raise RuntimeError("Failed to get aligned color frame.")

                color_image = np.asanyarray(aligned_color_frame.get_data())
                frame_dict[device_type] = color_image

            except Exception as e:
                print(f"Error in get_frame for camera {device_type}: {e}")
                return None
        return frame_dict

    def release(self):
        for type,device in self.devices.items():
            self.pipelines[device].stop()

    def get_intrinsics(self,device):
        return self.intr[device]


if __name__ == '__main__':
    devices = {"wrist":'215222073421'}
    camera = Camera(devices=devices,use_devices_type=["wrist"])
    i=0
    while True:

        t_0 = time.time()
        color_image = camera.get_frame()["wrist"]
        # cv2.namedWindow('left_img', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('left_img', color_image)
        # if i <=5:
        cv2.waitKey(1)
        print(time.time()-t_0)
        i+=1


