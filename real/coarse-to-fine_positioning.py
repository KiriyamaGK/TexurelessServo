import cv2
import numpy as np
from typing import Union,List,Dict,Tuple

import torch
from pygments.lexer import combined
from scipy.spatial.transform import Rotation as R

from experiments.rollout_real import _setup_model
from real.perception import Camera
from real.fr_robot import FR_Robot
import time
import os
from utils.detection import get_detect_result
from ultralytics import YOLO
import json
from real.environment import Environment
from utils.input_process import conditioned_clip_and_resize, input_dict_preprocess
from utils.paths import PROJECT_ROOT_DIR, path_completion, determine_ckpt_dirs
from utils.transform import _6d_pose_to_mat, mat_to_6d_pose


class CoarseToFineLocolization:
    def __init__(self,cart_vel = None,cv2_visualize = None,img_w = None,img_h = None,hdf5_img_size = None,goal_img_pth =None,policy_model = None,robot_address = None,
                      dof = None,down_to_grasp_distance = None,init = None,
                      stop_policy = None,velocity = None,
                      hardware_cfg = None,
                      pick_and_place_from_slot = None,p_s: Union[List, np.ndarray] = None, p_e: Union[List, np.ndarray] = None, num_devs: int = 0,ctrl_freq: int = 30, model_pth:str = None,motion_vel = 0.0, wpt_radius = 0.0,bbox_center_thresh = 0,conf_thresh = 0.0, color_cn_inv = True,
                 height = None,record_pose = True,record_video = True):
        self.env = Environment(robot_address = robot_address,dof = dof,down_to_grasp_distance = down_to_grasp_distance,init = init,stop_policy = stop_policy,velocity = velocity, **hardware_cfg["camera"],pick_and_place_from_slot=pick_and_place_from_slot)
        self.cam = self.env.camera
        self.rbt = self.env.robot_ins
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

        self.cart_vel = cart_vel
        self.rbt.move_cart([p_s[0], p_s[1], height, -180., 0., 0.],tool=2,user=0,vel=self.cart_vel)
        self.tgt_pos = [p_s[0], p_s[1], height, -180., 0., 0.]

        #about grasping
        self.record_pose = record_pose
        self.record_video = record_video
        if self.record_video or self.record_pose:
            self.save_base = "coarse-to-fine_positioning_results" + f"/{int(time.time())}"
            os.makedirs(self.save_base, exist_ok=True)
        if self.record_pose:
            self.record_pose_list = []
        if self.record_video:
            mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            self.out = cv2.VideoWriter(self.save_base + "/res.mp4", mp4, fps, (self.cam.width, self.cam.height * 2))

        #model
        self.policy_model = policy_model
        self.env.set_target_coordinate(use_cur=True) #maybe useless

        #imgsz
        self.img_h = img_h
        self.img_w = img_w
        self.hdf5_img_size = hdf5_img_size
        self.img_goal = cv2.imread(goal_img_pth[0])
        self.img_goal2 = cv2.imread(goal_img_pth[1])
        self.img_goal = conditioned_clip_and_resize(img=self.img_goal, img_h=img_h, img_w=img_w, hdf5_img_size=hdf5_img_size,
                                               keep_right=True)
        self.img_goal2 = conditioned_clip_and_resize(img=self.img_goal2, img_h=img_h, img_w=img_w, hdf5_img_size=hdf5_img_size,
                                                keep_right=True)
        self.goal_obs_dict = {
            "robot0_eye_in_hand_image_goal": self.img_goal.copy(),
            "robot0_eye_in_hand_image_2_goal": self.img_goal2.copy()
        }
        self.goal_obs_dict = input_dict_preprocess(self.goal_obs_dict, rollout=True)
        self.cv2_visualize = cv2_visualize

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
        如果最大置信度大于conf则need_track，且bbox面积要大于图像画面的10%
        """
        conf_array = np.array(conf)
        bboxes_array = np.array(bboxes)

        if len(conf_array) == 0:
            print(f"conf:{conf_array},res:{0} (no detection)")
            return False, -1

        img_center = np.array([self.cam.width / 2, self.cam.height / 2])
        bbox_centers = (bboxes_array[:, :2] + bboxes_array[:, 2:]) / 2
        distances = np.linalg.norm(bbox_centers - img_center, axis=1)

        # # 计算每个bbox的面积
        # bbox_areas = (bboxes_array[:, 2] - bboxes_array[:, 0]) * (bboxes_array[:, 3] - bboxes_array[:, 1])
        # img_area = self.cam.width * self.cam.height
        # min_area_threshold = img_area * 0.1  # 图像面积的10%

        valid_mask = (conf_array > self.conf_thresh) # & (bbox_areas > min_area_threshold)

        if not np.any(valid_mask):
            # bbox_areas_percent = [f"{area / img_area * 100:.1f}%" for area in bbox_areas]
            print(f"conf:{conf_array.tolist()}") #, bbox_areas:{bbox_areas_percent}, res:{0}")
            return False, -1

        valid_distances = distances[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        best_valid_idx = np.argmin(valid_distances)
        best_idx = valid_indices[best_valid_idx]

        # best_bbox_area_percent = bbox_areas[best_idx] / img_area * 100
        print(f"conf:{conf_array.tolist()}") #, best_bbox_area:{best_bbox_area_percent:.1f}%, res:{1}")
        return True, best_idx

    def main_tracker(self):
        self.detach_pt = None
        new_need_track = False
        idx = 0
        while self.state_dict["nearest_wpt_idx"] < len(self.wpts) -1 :
            ts =  time.time()
            img_dict = self.cam.get_frame()
            img = img_dict["img_1"]
            img_2 = img_dict["img_2"]

            # img = cv2.imread("/home/kiriyamagk/桌面/AlignAnything/networks/00163.png")
            #get detect res
            detect_res = get_detect_result(self.model,img,tracker_enabled=True,color_channel_inv=self.color_cn_inv)
            confs = detect_res["confidences"]
            bboxes = detect_res["bbox"] #xyxy

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
                self.rbt.move_cart(self.detach_pt,tool=2,user=0,vel=self.cart_vel)
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
                    self.pick_and_place() #todo: grasp here
                else:
                    if self.record_pose:
                        pos += [0]
                        self.record_pose_list.append(pos)
                    if self.record_video:
                        combined_img = np.vstack((bbox_img.copy(),img_2.copy()))
                        self.out.write(combined_img)
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
        self.env.gripper.move_gripper(0, 60, 60)  # gripper is at opening 0 during data collection/rollout
        time.sleep(4)

        dev_T = np.eye(4)
        dev_T[0,3] = 30
        dev_T[1,3] = 30
        dev_T[2, 3] = 20
        cur_6d_pose = self.rbt.get_gripper_TCP_pose()
        cur_T = _6d_pose_to_mat(cur_6d_pose)
        des_6d_pose =mat_to_6d_pose(cur_T @ dev_T)
        self.rbt.move_cart(des_6d_pose, tool=2, user=0, vel=self.cart_vel)

        # reset timer
        self.env.init()

        #close-loop prediction
        while not self.env.reinit_eval()["need_reinit"]:
            self.policy_step()

        # plug into slot
        cur_pose = self.env.robot_ins.get_gripper_TCP_pose()
        T_cur = _6d_pose_to_mat(cur_pose[:])
        T_pred_place = T_cur @ np.linalg.inv(self.env.g_place_T_g_tar)
        pred_place_pose = mat_to_6d_pose(T_pred_place)
        self.env.pick_table_and_place_slot_test(pred_place_pose, T_pred_place,norm_leave = True)

        # gradually leave slot
        self.rbt.move_cart(pose=self.env.slot_wpts[0], tool=2, user=0, vel=self.env.safe_vel)
        for idx in range(self.env.num_wpts):
            self.rbt.move_cart(pose=self.env.wpts[self.env.num_wpts - 1 - idx], tool=2, user=0, vel=self.env.safe_vel)

        # go to leave pose
        self.rbt.move_cart(leave_pose, tool=2, user=0, vel=self.cart_vel)

    def policy_step(self):
        #prepare obs
        im_dict = self.env.observation()
        img = im_dict['img_1']
        img2 = im_dict['img_2']
        if cv2_visualize:
            img_vis = np.vstack((img.copy(), img2.copy()))
            cv2.imshow("img", img_vis)
            cv2.waitKey(1)
            if self.record_video:
                self.out.write(img_vis)
        img = conditioned_clip_and_resize(img=img, img_h=self.img_h, img_w=self.img_w, hdf5_img_size=self.hdf5_img_size,
                                          keep_right=True)[:, :, ::-1]
        img2 = conditioned_clip_and_resize(img=img2, img_h=self.img_h, img_w=self.img_w, hdf5_img_size=self.hdf5_img_size,
                                           keep_right=True)[:, :, ::-1]
        obs_dict = {}
        obs_dict["robot0_eye_in_hand_image"] = img.copy()
        obs_dict["robot0_eye_in_hand_image_2"] = img2.copy()
        obs_dict = input_dict_preprocess(obs_dict, rollout=True)
        for k,v in self.goal_obs_dict.items():
            obs_dict[k] = v

        #get action
        pred = self.policy_model(obs_dict)
        if isinstance(pred, dict):
            predictions = pred["output_tensor"].detach().cpu().numpy().reshape(-1, )
        else:
            predictions = pred.detach().cpu().numpy().reshape(-1, )
        vel_tr = predictions[0:3]
        vel_rot = predictions[3:]
        dT = np.eye(4)
        dT[0:3,0:3] = R.from_rotvec(vel_rot/180*np.pi).as_matrix()
        dT[0:3, 3] = vel_tr
        self.env.action_dT(dT)

        #determine vel for stop check
        self.env.determine_vel_in_threshold(vel_tr = np.linalg.norm(vel_tr),
                                       vel_rot = np.linalg.norm(vel_rot))


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
    p_s = [-410.,215.]
    p_e = [-644.,-244.]
    height = 240.
    num_devs = 2
    ctrl_freq = 20
    model_pth = "/home/kiriyamagk/桌面/AlignAnything/data/runs/detect/train5/weights/best.pt"
    motion_vel = 3 #mm
    wpt_radius =  1    # if distance to a waypoint is within this value, this waypoint is considered reached
    bbox_center_thresh = 5 # if distance between image ct and bbox ct is within this value, the part is considered reached
    conf_thresh = 0.85 # the tracked obj's thresh should be above this value
    color_cn_inv = False
    record_pose = False
    record_video = True
    cart_vel = 40

    config_dir = "../configs/rollout_coarse-to-fine.json"
    with open(config_dir, "r") as j:
        config = json.load(j)

    #setup policy
    logs_dir = path_completion(config["logs_dir"], PROJECT_ROOT_DIR)
    ckpt_base = os.path.dirname(logs_dir)
    ckpts_dir = determine_ckpt_dirs(config["ckpts_dir"], ckpt_base)
    assert len(ckpts_dir) == 1
    model_config_dir = path_completion(config["logs_dir"], PROJECT_ROOT_DIR)
    with open(model_config_dir, "r") as j:
        model_config = json.load(j)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _setup_model(model_config)
    state_dict = torch.load(ckpts_dir[0], weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    stop_policy = config["stop_policy"]

    #about imgs
    img_w = model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"]
    img_h = img_w
    hdf5_img_size = model_config["dataset"]["hdf5_img_size"]
    goal_img_pth = ["/media/kiriyamagk/One Touch/AlignAnything_real/25.11.21/hdf5/goal_images/img1/0.png","/media/kiriyamagk/One Touch/AlignAnything_real/25.11.21/hdf5/goal_images/img2/0.png"]
    cv2_visualize = config["cv2_visualize"]

    #setup env
    main_ins = CoarseToFineLocolization(cart_vel = cart_vel,cv2_visualize = cv2_visualize,img_w = img_w, img_h = img_h, hdf5_img_size = hdf5_img_size, goal_img_pth = goal_img_pth, policy_model = model, robot_address = config["hardware"]["robot_address"],
                                        dof = config["dof"], down_to_grasp_distance = None, init = config['init'],
                                        stop_policy = {"angle_eps":0,"dist_eps":0}, velocity = {"trans_vel": {"value": [0,0]},"rot_vel": {"value": 0}},
                                        hardware_cfg = config["hardware"],
                                        pick_and_place_from_slot = config["pick_and_place_from_slot"], p_s = p_s, p_e = p_e, num_devs = num_devs, ctrl_freq = ctrl_freq, model_pth = model_pth, motion_vel = motion_vel, wpt_radius = wpt_radius, bbox_center_thresh = bbox_center_thresh, conf_thresh = conf_thresh, color_cn_inv = color_cn_inv, height = height, record_pose = record_pose,record_video=record_video
                                        )
    eval_stop_policy = config["stop_policy"]
    main_ins.env.setup_stop_policy(eval_stop_policy) #important

    main_ins.main_tracker()

    # temporarily useless keys
    record_video = config["record_video"]
    bgr2rgb = config["bgr2rgb"]



