import cv2
import numpy as np
from typing import Union,List,Dict,Tuple
from real.perception import Camera
from real.fr_robot import FR_Robot
import time
import os
from utils.detection import get_detect_result
from ultralytics import YOLO

class Coarse_Locolization:
    def __init__(self,cam_cfg = None,rbt_cfg = None,p_s: Union[List, np.ndarray] = None, p_e: Union[List, np.ndarray] = None, num_devs: int = 0,ctrl_freq: int = 30, model_pth:str = None,motion_vel = 0.0, wpt_radius = 0.0,bbox_center_thresh = 0,conf_thresh = 0.0, color_cn_inv = True,
                 init_pose = None,do_down_to_grasp=False,grasp_offset_x=0.0,grasp_offset_y=0.0,grasp_offset_z = 0.0,place_pose = None,record_pose = True,record_video = True):
        self.cam = Camera(**cam_cfg)
        self.rbt = FR_Robot(**rbt_cfg)
        self.wpts,self.dirr_vec = generate_waypoints(p_s, p_e, num_devs)
        self.ctrl_freq = ctrl_freq
        self.state_dict = {
            "state": "on_route", # "on_route"/"on_track"
            "nearest_wpt_idx":0 #visited waypoint with max index
        }
        self.model = YOLO(model_pth)
        self.conf_thresh = conf_thresh
        self.motion_vel = motion_vel
        self.wpt_radius = max(wpt_radius,motion_vel/1.5)
        self.bbox_center_thresh = bbox_center_thresh
        self.color_cn_inv = color_cn_inv

        self.dirr_map = { # maps reference dirr to robot base axis dirr
            "y+": [0.0,1.0] if self.dirr_vec[1]>0 else [0.0,-1.0],
            "y-": [0.0,-1.0] if self.dirr_vec[1]>0 else [0.0,1.0],
            "x+": [1.0,0.0] if self.dirr_vec[0]>0 else [-1.0,0.0],
        }

        self.dirr = self.dirr_map["y+"] #moves to y+ first
        if init_pose is not None:
            self.rbt.move_cart(init_pose,tool=2,user=0,vel=30)
        pos_in = self.rbt.get_gripper_TCP_pose()
        self.rbt.move_cart([p_s[0], p_s[1], pos_in[2], -180., 0., 0.],tool=2,user=0,vel=30)
        self.tgt_pos = [p_s[0], p_s[1], pos_in[2], -180., 0., 0.]

        #about grasping
        self.do_down_to_grasp = do_down_to_grasp
        if self.do_down_to_grasp:
            assert place_pose is not None
            from real.gripper import Gripper
            self.gripper = Gripper()
            self.grasp_offset_x = grasp_offset_x
            self.grasp_offset_y = grasp_offset_y
            self.grasp_offset_z = grasp_offset_z
            self.place_pose = place_pose
            self.gripper.move_gripper(0, 60, 60)  # open
            time.sleep(2)
        self.record_pose = record_pose
        self.record_video = record_video
        if self.record_video or self.record_pose:
            self.save_base = "coarse_positioning_results" + f"/{int(time.time())}"
            os.makedirs(self.save_base, exist_ok=True)
        if self.record_pose:
            self.record_pose_list = []
        if self.record_video:
            mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            self.out = cv2.VideoWriter(self.save_base + "/res.mp4", mp4, fps, (self.cam.width, self.cam.height * 2))

    def get_motion_vec_from_bbox(self,xc,yc):
        delta_x = float(xc-self.cam.width/2)
        delta_y = float(yc-self.cam.height/2)
        vec = np.array([delta_x,delta_y])
        if np.linalg.norm(vec) <= self.motion_vel:
            return vec
        else:
            return vec/np.linalg.norm(vec) *self.motion_vel

    def _in_bbox_center_thresh(self,xc,yc):
        ct_error = np.array([float(xc - self.cam.width / 2),float(yc - self.cam.height / 2)])
        print(f"ct_error:{np.linalg.norm(ct_error)}")
        return np.linalg.norm(ct_error) < self.bbox_center_thresh

    def update_tgt_pos(self):
        new_wpt_idx = self.state_dict["nearest_wpt_idx"]+1
        cur_2dpos_arr = np.array(self.rbt.get_gripper_TCP_pose()[0:2])
        new_wpt_pos = np.array(self.wpts[new_wpt_idx])
        if np.linalg.norm(cur_2dpos_arr-new_wpt_pos) < self.wpt_radius:
            self.tgt_pos[0] = new_wpt_pos[0]
            self.tgt_pos[1] = new_wpt_pos[1]
            self.state_dict["nearest_wpt_idx"] += 1
            if self.state_dict["nearest_wpt_idx"] % 2:
                self.dirr = self.dirr_map["x+"]
            else:
                if not self.state_dict["nearest_wpt_idx"] % 4:
                    self.dirr = self.dirr_map["y+"]
                else:
                    self.dirr = self.dirr_map["y-"]
        else:
            self.tgt_pos[0] += self.dirr[0] * self.motion_vel
            self.tgt_pos[1] += self.dirr[1] * self.motion_vel

    def _need_track(self, conf: Union[List, np.ndarray], bboxes: Union[List, np.ndarray]) -> Tuple[bool, int]:
        """
        如果最大置信度大于conf则need_track
        """
        conf_array = np.array(conf)
        bboxes_array = np.array(bboxes)

        if len(conf_array) == 0:
            print(f"conf:{conf_array},res:{0} (no detection)")
            return False, -1
        img_center = np.array([self.cam.width / 2, self.cam.height / 2])
        bbox_centers = (bboxes_array[:, :2] + bboxes_array[:, 2:]) / 2
        distances = np.linalg.norm(bbox_centers - img_center, axis=1)
        valid_mask = conf_array > self.conf_thresh
        if not np.any(valid_mask):
            print(f"conf:{conf_array},res:{0}")
            return False, -1
        valid_distances = distances[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        best_valid_idx = np.argmin(valid_distances)
        best_idx = valid_indices[best_valid_idx]
        print(f"conf:{conf_array},res:{1}")
        return True, best_idx

    def main_tracker(self):
        self.detach_pt = None
        new_need_track = False
        idx = 0
        while self.state_dict["nearest_wpt_idx"] < len(self.wpts) -1 :
            ts =  time.time()
            img = self.cam.get_frame()["img_1"]
            # img = cv2.imread("/home/kiriyamagk/桌面/AlignAnything/networks/00163.png")
            #get detect res
            detect_res = get_detect_result(self.model,img,tracker_enabled=True,color_channel_inv=self.color_cn_inv)
            confs = detect_res["confidences"]
            bboxes = detect_res["bbox"]#xyxy

            bbox_img = detect_res["res_img"]
            cv2.imshow("img",bbox_img)
            cv2.waitKey(1)
            last_need_track = new_need_track
            new_need_track, bbox_idx = self._need_track(confs,bboxes)

            #determine whether need track
            if new_need_track and not last_need_track:
                print("Start tracking!")
                self.detach_pt = self.rbt.get_gripper_TCP_pose()
                self.state_dict["state"] = "on_track"
            if last_need_track and not new_need_track:
                print("Stop tracking!")
                self.rbt.move_cart(self.detach_pt,tool=2,user=0,vel=30)
                self.state_dict["state"] = "on_route"
                self.detach_pt = None

            # motion
            pos = self.rbt.get_gripper_TCP_pose()[:]

            if self.state_dict["state"] == "on_route":
                if self.record_pose:
                    pos +=[0]
                    self.record_pose_list.append(pos)
                self.update_tgt_pos()
                self.rbt.servo_cart(self.tgt_pos[0:6],mode=0,vel=10.0)
            else:
                tgt_bbox = bboxes[bbox_idx]
                xc = (tgt_bbox[0] + tgt_bbox[2]) / 2
                yc = (tgt_bbox[1] + tgt_bbox[3]) / 2
                if self._in_bbox_center_thresh(xc,yc):
                    if self.record_pose:
                        pos += [1]
                        self.record_pose_list.append(pos)
                    if not self.do_down_to_grasp:
                        print("target found,waiting...")
                        time.sleep(5)
                    else:
                        self.pick_and_place()
                else:
                    if self.record_pose:
                        pos += [0]
                        self.record_pose_list.append(pos)
                    if self.record_video:
                        self.out.write(bbox_img.copy())
                    delta_motion = self.get_motion_vec_from_bbox(xc,yc)
                    pos[0] += delta_motion[1]
                    pos[1] += delta_motion[0]
                    self.rbt.servo_cart(pos[0:6],mode=0,vel=10.0)
            elapsed = 1/self.ctrl_freq - (time.time()-ts)
            time.sleep(elapsed) if elapsed > 0 else None
            idx+=1
        if self.record_pose:
            np.save(self.save_base + "/res.npy",self.record_pose_list)
        if self.record_video:
            self.out.release()

    def pick_and_place(self):
        leave_pose = self.rbt.get_gripper_TCP_pose()
        #grasp
        grasp_pose = leave_pose[:]
        grasp_pose[0]+=self.grasp_offset_x
        grasp_pose[1]-=self.grasp_offset_y
        grasp_pose[2]-=self.grasp_offset_z
        pre_grasp_pose = grasp_pose[:]
        pre_grasp_pose[2]+= 100
        # pre-grasp
        print("==============")
        print(pre_grasp_pose)
        self.rbt.move_cart(pre_grasp_pose, tool=2, user=0, vel=30)
        # time.sleep(100)
        print(grasp_pose)
        self.rbt.move_cart(grasp_pose,tool=2,user=0,vel=30)
        self.gripper.move_gripper(600,60,60) #close
        time.sleep(3)
        #up
        up_pose = grasp_pose[:]
        up_pose[2]+= 100
        self.rbt.move_cart(up_pose,tool=2,user=0,vel=30)
        #go upper the place place
        upper_pose = self.place_pose[:]
        upper_pose[2]+= 200
        self.rbt.move_cart(upper_pose, tool=2, user=0, vel=30)
        #place
        self.rbt.move_cart(place_pose, tool=2, user=0, vel=30)
        self.gripper.move_gripper(0, 60, 60)  #open
        time.sleep(2)
        # go upper the place pose
        self.rbt.move_cart(upper_pose, tool=2, user=0, vel=30)
        #go to leave pose
        self.rbt.move_cart(leave_pose, tool=2, user=0, vel=30)


def generate_waypoints(p_s: Union[List, np.ndarray] = None, p_e: Union[List, np.ndarray] = None,num_devs: int = 0) -> Tuple[List,List]:
    # for default, division is made in x direction , and first mov_vec is in y direction.
    res_waypoints = []
    res_waypoints.append([p_s[0], p_s[1]])
    all_vec = [p_e[0]-p_s[0], p_e[1]-p_s[1]]
    step_vec = [all_vec[0]/num_devs, all_vec[1]]
    log = []  #put y+,y-,x+ in it
    while(len(res_waypoints) < 2*(1+num_devs)):
        last_pt = res_waypoints[-1]
        if not len(log):#first step
            log.append("y+")
            new_pt = [last_pt[0], last_pt[1] + step_vec[1]]
        else:
            if "y" in log[-1]:
                log.append("x+")
                new_pt = [last_pt[0] + step_vec[0], last_pt[1]]
            else:
                if "+" in log[-2]:
                    log.append("y-")
                    new_pt = [last_pt[0], last_pt[1] - step_vec[1]]
                else:
                    assert log[-2] == "y-"
                    log.append("y+")
                    new_pt = [last_pt[0], last_pt[1] + step_vec[1]]
        res_waypoints.append(new_pt)

    return res_waypoints,all_vec

if __name__ == "__main__":
    init_pose = [-475.3507995605469, -97.18849182128906, 261.3992614746094, -178.51327514648438, -0.8906695246696472, -0.3798218071460724] #none for curr_pose
    p_s = [-410.,215.]
    p_e = [-644.,-244.]
    num_devs = 2
    ctrl_freq = 20
    model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train5/weights/best.pt"
    motion_vel = 3 #mm
    wpt_radius =  1    # if distance to a waypoint is within this value, this waypoint is considered reached
    bbox_center_thresh = 5 # if distance between image ct and bbox ct is within this value, the part is considered reached
    conf_thresh = 0.5 # the tracked obj's thresh should be above this value
    color_cn_inv = False
    record_pose = True

    #about grasping
    do_down_to_grasp = True
    grasp_offset_x = 80 #mm
    grasp_offset_y = 40  # mm
    grasp_offset_z = 110 #mm
    place_pose = [-188.1055145263672, 576.9207763671875, 30.75214195251465, -173.2011260986328, -3.958481550216675, -78.09700775146484]

    #rbt_cfg
    rbt_cfg = {}

    #cam_cfg
    cam_cfg = {
        "devices": {
            "img_1": "215222073421",
            "img_2": "233622076143"},
        "use_devices_type":["img_1"]
    }

    main_ins = Coarse_Locolization(cam_cfg = cam_cfg,rbt_cfg = rbt_cfg,p_s=p_s,p_e=p_e,num_devs=num_devs,ctrl_freq=ctrl_freq,model_pth=model_pth,motion_vel=motion_vel,wpt_radius=wpt_radius,bbox_center_thresh = bbox_center_thresh,conf_thresh = conf_thresh,color_cn_inv=color_cn_inv,init_pose = init_pose,do_down_to_grasp=do_down_to_grasp,grasp_offset_x=grasp_offset_x,grasp_offset_y=grasp_offset_y,grasp_offset_z = grasp_offset_z,place_pose = place_pose,record=record_pose)
    main_ins.main_tracker()



