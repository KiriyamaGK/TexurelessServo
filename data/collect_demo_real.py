import time
import cv2
import numpy as np
import os
import random
import threading
import json
from utils.paths import return_disc_route
from utils.file import ensure_dir
from real.perception import Camera
from real.fr_robot import FR_Robot
from real.gripper import Gripper
from math import pi,sin,cos
from utils.transform import euler2rot,rmat2quat
from utils.hdf5 import add_useless_things, split_train_val_from_hdf5, add_env_meta, compute_num_samples, add_config
import h5py
from utils.input_process import clip_image


def unit_transform(pose):
    pose[0] = pose[0] / 1000
    pose[1] = pose[1] / 1000
    pose[2] = pose[2] / 1000
    pose[3] = pose[3] / 180 * pi
    pose[4] = pose[4] / 180 * pi
    pose[5] = pose[5] / 180 * pi
    return pose

def filter_action(x:np.array):
    assert len(x.shape)==1 and x.shape[0] in [3,6]
    if x.shape[0] == 6:
        for i in range(3):
            if x[i] >= -1e-2 and x[i] <= 1e-2:
                x[i] = 0
            if x[i+3] >= -1e-3 and x[i+3] <= 1e-3:
                x[i+3] = 0
    else:
        for i in range(3):
            if i<2:
                if x[i] >= -1e-2 and x[i] <= 1e-2:
                    x[i] = 0
            else:
                if x[i] >= -1e-3 and x[i] <= 1e-3:
                    x[i] = 0
    return x

def unit_ang(ang):
    assert len(ang)==6
    for i in range(len(ang)):
        ang[i] = ang[i] * pi / 180
    return ang

def rot_angle_normalization(ang,max_attempts_num=10):
    attempts_num = 0
    while True:
        if attempts_num >= max_attempts_num:
            raise RuntimeError("Too many attempts")
        if ang>=360:
            ang-=360
        if ang<0:
            ang+=360
        if ang>=0 and ang <360:
            return ang
        attempts_num += 1

def make_an_angle_in_180(ang,max_attempts_num=10):
    attempts_num=0
    while True:
        if attempts_num >= max_attempts_num:
            raise RuntimeError("Too many attempts")
        if abs(ang) >180:
            if ang>0:
                ang -= 360
            else:
                ang += 360
        if abs(ang) <=180:
            return ang
        attempts_num+=1


class Environment:
    def __init__(self,robot_address,w,h,fps,cam_devices,use_devices_type,trans_thres,rot_thres,down_dis):

        self.robot_ins=FR_Robot(robot_address)
        self.camera=Camera(devices=cam_devices,use_devices_type=use_devices_type,width=w, height=h, fps=fps)
        self.gripper=Gripper()

        self.corner_1 = [-405.3097229003906, 199.8372497558593]  # 工作区域左上角点
        self.corner_2 = [-707.1678466796875, -249.3986206054687]  # 工作区域右下角点

        self.x_max = max(self.corner_1[0], self.corner_2[0])
        self.x_min = min(self.corner_1[0], self.corner_2[0])
        self.y_max = max(self.corner_1[1], self.corner_2[1])
        self.y_min = min(self.corner_1[1], self.corner_2[1])

        self.trans_thres = trans_thres #最大平移距离，mm
        self.rot_thres = rot_thres  #最大rz，°
        self.down_dis = down_dis    #夹爪下移距离，mm
        print("environment initialized")



    def place(self,p_0):
        '''
        在工作空间内随机选取一点p_1,机器人从初始位置p_0抓取零件放置到p_1
        :param p_0: 初始位置
        :param down_dis: 预抓取时夹爪下移距离
        :return: p_1:放置零件的新位置，p_1与p_0仅在x,y,rz上有差异
        '''

        #开始下去，抓零件
        pose = p_0.copy()
        pose[2] -= self.down_dis
        self.gripper.move_gripper(0,60,60)
        time.sleep(2)
        self.robot_ins.move_l(pose, tool=1, user=0, vel=10)
        self.gripper.move_gripper(3000, 60, 60)
        time.sleep(2)

        #开始上来
        self.robot_ins.move_l(p_0, tool=1, user=0, vel=30)

        # 开始平移到新位置
        p_1=[0,0,0,0,0,0]
        p_1[0]=random.uniform(self.x_min,self.x_max)
        p_1[1]=random.uniform(self.y_min,self.y_max)
        p_1[2]=p_0[2]
        p_1[3]=p_0[3]
        p_1[4]=p_0[4]
        p_1[5]=random.randint(130, 175)*(random.randint(0, 1) - 0.5) * 2
        self.robot_ins.move_l(p_1, tool=1, user=0, vel=30)

        #开始下去
        pose = p_1.copy()
        pose[2] -= (self.down_dis*0.9)
        self.robot_ins.move_l(pose, tool=1, user=0, vel=10)
        self.gripper.move_gripper(0,60,60)
        time.sleep(1)

        self.robot_ins.move_l(p_1, tool=1, user=0, vel=10)
        return p_1

    def generate_motion_paras(self,desire_pt):
        '''
        给定目标位置生成运动参数，包括theta（xy方位），alpha（rz）,生成的起始点start_pt.要确保目标位姿的rz的绝对值在120°-180°，否则可能位姿无法到达
        :param desire_pt: 目标位姿
        :param rot_thres: 最大rz旋转角（°）
        :param trans_thres: 最大平移量(mm)
        :return: start_pt,alpha（°）,theta(rad)
        '''
        for i in range(100):
            theta = random.uniform(-2 * pi, 2 * pi)
            alpha = random.uniform(10, self.rot_thres) * (random.randint(0, 1) - 0.5) * 2
            assert self.rot_thres < 85
            if alpha > 0:  # 逆时针转
                if desire_pt[5] > 0:  # 腕部相机在desire_pt向left_cam偏
                    assert desire_pt[5] > 120 and desire_pt[5] <= 180
                if desire_pt[5] < 0:  # 腕部相机在desire_pt向right_cam偏
                    assert desire_pt[5] < -120 and desire_pt[5] >= -180
                    alpha = min(alpha, -120 - desire_pt[5])
            elif alpha < 0:  # 顺时针转
                if desire_pt[5] > 0:  # 腕部相机在desire_pt向left_cam偏
                    assert desire_pt[5] > 120 and desire_pt[5] <= 180
                    alpha = -1 * min(-alpha, desire_pt[5] - 120)
                if desire_pt[5] < 0:  # 腕部相机在desire_pt向right_cam偏
                    assert desire_pt[5] < -120 and desire_pt[5] >= -180

            delta = [self.trans_thres * cos(theta), self.trans_thres * sin(theta)]
            start_pt = np.array(desire_pt.copy()) + np.array([delta[0], delta[1], 0, 0, 0, alpha])
            start_pt[5]=make_an_angle_in_180(start_pt[5])
            print("start_pt", start_pt[5])
            print('desire_pt', desire_pt[5])
            print("alpha", alpha)
            assert abs(start_pt[5]) >= 120 and abs(start_pt[5]) <= 180
            ret = self.robot_ins.robot.GetInverseKin(0, start_pt, config=-1)
            if isinstance(ret, tuple) and ret[0] == 0: #如果能求出逆解
                return theta, alpha, start_pt
            else:
                continue
if __name__=='__main__':
    current_pt_desire=True

    config_dir = "../configs/demo_collection_real.json"
    with open(config_dir, "r") as j:
        config = json.load(j)

    env=Environment(robot_address=config["hardware"]["robot_address"],**config["demo_collection"]["distances"],**config["hardware"]["camera"])
    cam=env.camera
    robot_ins=env.robot_ins

    init_pos=robot_ins.get_gripper_TCP_pose()
    init_pos[3]=-180
    init_pos[4]=0
    in_desire_pt =init_pos if current_pt_desire else [-533.3317260742188, 49, 150, -180, 0, 163.76220703125]

    #img相关
    img_save_type=config["demo_collection"]["img"]["save_type"]
    assert img_save_type in ["rgb","bgr"]
    img_size=config["demo_collection"]["img"]["size"]

    #demo数量和目标位姿变化周期
    cir_num = config["overall_setting"]["cir_num"]
    desire_pt_change_cycle= config["overall_setting"]["desire_pt_change_cycle"]
    file_name=config["overall_setting"]["file_name"]

    #运动速度
    vel_tr_norm=config["demo_collection"]["velocity"]["vel_tr_norm"]
    vel_rot_norm=config["demo_collection"]["velocity"]["vel_rot_norm"]

    #位姿偏差，判定demo结束
    trans_dis_thres=config["demo_collection"]["stop_collection"]["trans_dis_thres"]
    rot_dis_thres=config["demo_collection"]["stop_collection"]["rot_dis_thres"]
    absolute_cmd=config["demo_collection"]["absolute_cmd"]

    #收集数据频率
    data_collect_freq=config["demo_collection"]["freq"]

    ac_dim=config["demo_collection"]["ac_dim"]

    replace_existed_hdf5=config["overall_setting"]["replace_existed_hdf5"]
    delete_last_demo=config["overall_setting"]["delete_last_demo"]

    dataset_type=config["dataset_type"]
    assert dataset_type in ["hdf5","npy"]
    assert dataset_type =="hdf5"          #TODO:记得修改

    base_dir = return_disc_route("One Touch")
    if dataset_type=="hdf5":
        database_dir = os.path.join(base_dir, 'AlignAnything_real', file_name, 'hdf5')
        ensure_dir(database_dir)
        dataset_dir=os.path.join(database_dir, 'mimic.hdf5')

        if replace_existed_hdf5:
            new_f_out = h5py.File(dataset_dir, "w")
        else:
            if os.path.exists(dataset_dir):
                new_f_out = h5py.File(dataset_dir, "r+")
            else:
                new_f_out = h5py.File(dataset_dir, "w")
    existed_demo_num=0
    for uu in range(cir_num):
        print("=====================collecting demo_{}=====================".format(uu))
        #preprocess
        if uu==0:
            if 'data' in new_f_out and not replace_existed_hdf5:
                existed_demo_num=len(new_f_out["data"])
                if existed_demo_num>=1 and delete_last_demo:
                    del new_f_out['data/demo_{}'.format(existed_demo_num-1)]
        #根据existed_demo_num的数量整体偏移
        if existed_demo_num>=1:
            if delete_last_demo:
                obs_path = 'data/demo_{}/obs'.format(uu+existed_demo_num-1)
                action_path = 'data/demo_{}/actions'.format(uu+existed_demo_num-1)
            else:
                obs_path = 'data/demo_{}/obs'.format(uu+existed_demo_num)
                action_path = 'data/demo_{}/actions'.format(uu+existed_demo_num)
        else:
            obs_path = 'data/demo_{}/obs'.format(uu)
            action_path = 'data/demo_{}/actions'.format(uu)

        if uu==0:
            desire_pt=in_desire_pt
            robot_ins.move_cart(desire_pt, tool=1, user=0, vel=40)
        elif uu % desire_pt_change_cycle == 0:
            desire_pt=env.place(p_0=desire_pt)

        theta, alpha, start_pt=env.generate_motion_paras(desire_pt)
        robot_ins.move_cart(start_pt, tool=1, user=0, vel=40)

        action_list = []
        img_dict = {}
        abs_rz_list = []
        tcp_list=[]

        quit = False
        flag_tr = False
        flag_rot = False

        tcp = robot_ins.get_gripper_TCP_pose()

        def robo_operator():
            #global
            global robot_ins
            global tcp
            global absolute_cmd
            global desire_pt

            # velocity
            global vel_tr_norm
            global vel_rot_norm

            #judge whether demo done
            global flag_tr
            global flag_rot
            global quit

            #distance
            global trans_dis_thres
            global rot_dis_thres
            '''
            pos_init:单次servocart点动前的初始位姿
            desc_pos:单次servocart点动后的目标位姿
            '''
            t_0 = time.time()
            pos_init = robot_ins.get_gripper_TCP_pose().copy()
            desc_pos=np.array(pos_init.copy())
            p_1 = np.array(desire_pt.copy()[0:2])
            p_0 = np.array(pos_init[0:2])
            v = vel_tr_norm * (p_1 - p_0) / np.linalg.norm(p_1 - p_0)  # *d/400   #速度大小正比于tcp和目标物体的距离
            delta_rot=desire_pt[5]-pos_init[5]

            delta_rot=make_an_angle_in_180(delta_rot)
            vrz=vel_rot_norm * (delta_rot)/abs(delta_rot)

            while not quit:
                # print('delta rot: ',delta_rot)
                # print('alpha: ',alpha)
                # print('actual_ctrl_period(ms):', 1000 * (time.time() - t_0))
                # print('actual_ctrl_freq(hz):', 1 / (time.time() - t_0))
                t_0 = time.time()
                tcp = robot_ins.get_gripper_TCP_pose()
                rela_vel_vec = np.array(desire_pt[0:2])-np.array(tcp[0:2])
                d = np.linalg.norm(rela_vel_vec)

                print("==========================distance:{}======================".format(d))
                if d > trans_dis_thres and np.dot(rela_vel_vec, v) > 0:
                    translation=[v[0],v[1],0,0,0,0]
                    if absolute_cmd:
                        desc_pos+=np.array(translation)
                        robot_ins.servo_cart(desc_pos=desc_pos, mode=0, vel=10.0)
                    else:
                        robot_ins.servo_cart(desc_pos=translation, mode=1, vel=10.0)

                if d <= trans_dis_thres or np.dot(rela_vel_vec, v) <= 0:
                    flag_tr = True

                cur_del_rot=tcp[5]-desire_pt[5]
                cur_del_rot=make_an_angle_in_180(cur_del_rot)
                if not (abs(cur_del_rot))<abs(delta_rot)+2:
                    raise RuntimeError("cur_del_rot:{},delta_rot:{}".format(cur_del_rot, delta_rot))

                if flag_tr and abs(cur_del_rot) >rot_dis_thres:
                    rotation=[0,0,0,0,0,vrz]
                    if absolute_cmd:
                        desc_pos+=np.array(rotation)
                        robot_ins.servo_cart(desc_pos=desc_pos, mode=0, vel=10.0)
                    else:
                        robot_ins.servo_cart(desc_pos=rotation, mode=1, vel=10.0)
                    print('==================delrot:{}======================'.format(cur_del_rot))

                if abs(cur_del_rot) <=rot_dis_thres:
                    flag_rot=True
                if flag_tr and flag_rot:
                    quit = True
                    break
                time.sleep(0.008)

        # robo_operator()
        operate = threading.Thread(target=robo_operator, daemon=True)
        operate.start()
        tt = time.time()

        grip_op = 0
        grip_cls = 0
        t0 = time.time()
        t_1 = time.time()
        flag_flag = False  # 表示平动刚完成
        flag_flag_flag = False
        # gkkk=0

        while True:
            frame_dict={}
            if time.time() - t0 > 1/data_collect_freq:    #此程序运行约0.01s（10hz）,因此循环频率需要低于10hz
                # print("camera circulation_time:", time.time() - t0)
                t0 = time.time()
                # 读取图像帧，包括RGB图
                tcp_high = tcp.copy()
                if flag_tr and (not flag_flag) and (not flag_flag_flag):  # 平动刚完成,flag_flag_flag用来确保此if只进入一次
                    flag_flag = True
                    flag_flag_flag = True
                if (not flag_tr) or flag_flag:  # 平动未或刚完成
                    tcp_high[5] = start_pt[5]
                    flag_flag = False  # 确保平动完成后不进入此if分支，rz会变化
                frame_dict = cam.get_frame()
                assert frame_dict is not None

                for type,img in frame_dict.items():
                    cv2.imshow(type, img)
                    cv2.waitKey(1)
                    if img_save_type=="rgb":
                        img=img[:,:,::-1]
                    img=clip_image(img,img_size)

                    if type not in img_dict.keys():
                        img_dict[type]=[img.copy()]   #很奇怪，要用到.copy()才行
                    else:
                        img_dict[type].append(img.copy())

                # 收集位姿
                tcp_high[5] = rot_angle_normalization(tcp_high[5])  # 轉換到0-360之間，-180和180之間有間斷點，不方便學習
                abs_rz_list.append(tcp_high[5])      #°
                tcp_list.append(np.array(tcp_high))  #mm,°
                # print('use time:', time.time() - t0)

            if quit:
                operate.join()
                quit = False
                tcp_list_1 = tcp_list.copy()

                cv2.destroyAllWindows()
                print('...录制结束，数据处理中...')

                # 对位姿指令进行处理
                epi_length = len(abs_rz_list)
                for i in range(epi_length):
                    action = tcp_list_1[i + 1] - tcp_list[i].copy() if i < epi_length - 1 else np.array([0, 0, 0, 0, 0, 0])
                    action=filter_action(action)
                    action_list.append(
                            np.array([action[0], action[1], 0, 0, 0, action[5]])
                            if ac_dim==6 else np.array([action[0], action[1], action[5]]))

                print("action list:", action_list)

                if not len(action_list)==len(img_dict["wrist"]):
                    print("len act:",len(action_list))
                    print("len img:", len(img_dict["wrist"]))
                assert len(action_list)==len(abs_rz_list)

                del_rot_list = np.array(action_list.copy())[:, -1]

                for gk in range(epi_length):
                    if abs(del_rot_list[gk]) > 10 or del_rot_list[
                        gk] * alpha > 0:  # start_pt=desire_pt+alpha，故正常情况下alpha与delta_rot异号
                        print('alpha(°):', alpha)
                        print('theta(rad):', theta)
                        print('action list(mm,°):', action_list)
                        print('abs_rot_list(°):', abs_rz_list)
                        print('del_rot_list(°):', np.array(del_rot_list).copy()[:][-1])
                        print("total rotation:", np.array(del_rot_list).sum())
                        raise RuntimeError('action rz error')
                if existed_demo_num>=1:
                    if delete_last_demo:
                        add_useless_things(new_f_out=new_f_out,demo_ind=uu+existed_demo_num-1,epi_len=epi_length)
                    else:
                        add_useless_things(new_f_out=new_f_out,demo_ind=uu+existed_demo_num,epi_len=epi_length)
                else:
                    add_useless_things(new_f_out=new_f_out, demo_ind=uu, epi_len=epi_length)

                if "wrist" in img_dict:
                    new_f_out.create_dataset(obs_path + '/robot0_eye_in_hand_image', data=img_dict["wrist"])
                new_f_out.create_dataset(obs_path + '/abs_rot', data=abs_rz_list)
                new_f_out.create_dataset(action_path, data=action_list)
                break

    cam.release()

    add_env_meta(new_f_out)
    add_config(new_f_out,config)
    new_f_out.close()
    compute_num_samples(dataset_dir)
    split_train_val_from_hdf5(dataset_dir, val_ratio=0.1)




