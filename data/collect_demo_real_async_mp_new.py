import multiprocessing as mp
import time
import cv2
import numpy as np
import json
import h5py
from multiprocessing import Manager
import signal
import atexit
import psutil

from utils.paths import PROJECT_ROOT_DIR
from utils.file import ensure_dir
from utils.hdf5 import add_useless_things, split_train_val_from_hdf5, add_env_meta, compute_num_samples, add_config, create_hdf5_filter_key
from utils.policy import get_expert_policy
from real.collision_detection.sdf_collision import CollisionDetector
import torch
from utils.dagger import get_policy_action, aggregate_dataset, train_policy, setup_policy_model, prepare_observation_for_policy
from utils.dagger_params import is_in_dagger_episode, should_train_policy, print_dagger_status
from utils.transform import construct_dT_from_action
import os
import math

# Global variables for inter-process communication
processes = []
queues = {}


def set_process_priority(pid=None, priority="high"):
    """
    设置进程优先级
    priority: "low", "normal", "high", "realtime"
    """
    if pid is None:
        pid = os.getpid()

    try:
        p = psutil.Process(pid)

        priority_map = {
            "low": 10,      # 低优先级
            "normal": 0,    # 普通优先级
            "high": -10,    # 高优先级
            "realtime": -20 # 实时优先级（需要root权限）
        }

        p.nice(priority_map[priority])
        print(f"Process {pid} priority set to {priority}")
        return True
    except Exception as e:
        print(f"Failed to set priority: {e}")
        return False

def post_process_episode(episode_data, config):
    from data.process_hdf5 import _portion_last_episode
    """Apply post-processing steps to episode data"""
    portion_last_episode = config["demo_collection"]['post_process']['portion_last_episode']

    if portion_last_episode["utilized"]: #only process portion_last_episode
        episode_data["expert_actions"], _ = _portion_last_episode(episode_data["expert_actions"], portion_last_episode["portion_last_num"],
                                                               ac_dim=6)

    if config["demo_collection"]['post_process']['add_end_episode']['utilized']:
        raise NotImplementedError("add_end_episode is not implemented")
    if config["demo_collection"]['post_process']['add_medium_episode']['utilized']:
        raise NotImplementedError("add_medium_episode is not implemented")
    if config["demo_collection"]['post_process']['disturb_abs_rot']['utilized']:
        raise NotImplementedError("disturb_abs_rot is not implemented")
        
    return episode_data


def frame_sample(episode_data, gap=1):
    """Sample frames from episode data with given gap, ensuring last frame is included"""
    # Calculate total number of frames
    n = len(episode_data['expert_actions'])

    # Create indices to sample: 0, gap, 2*gap, ... and always include last frame
    indices = list(range(0, n, gap))

    # Ensure last frame is included
    if n - 1 not in indices:
        indices.append(n - 1)

    # Sort indices to maintain temporal order
    indices.sort()

    # Create new sampled episode data
    sampled_data = {}
    for key in episode_data:
        if episode_data[key] is not None and len(episode_data[key]) == n:
            # Sample the list using the indices
            sampled_data[key] = [episode_data[key][i] for i in indices]
        else:
            # For non-list data or data with different length, copy as-is
            sampled_data[key] = episode_data[key]

    return sampled_data


def cleanup_processes():
    """Gracefully shut down all processes"""
    print("Shutting down all processes...")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
    print("All processes closed")

atexit.register(cleanup_processes)

class FixedFrequencyController:
    """Precise frequency controller"""
    def __init__(self, frequency):
        self.interval = 1.0 / frequency
        self.next_time = time.time() + self.interval

    def wait(self):
        current_time = time.time()
        if current_time < self.next_time:
            time.sleep(max(0, self.next_time - current_time))
        self.next_time += self.interval
        return 1.0 / (time.time() - (self.next_time - self.interval))


def control_process(config, dagger_config, state, goal_state, is_dagger_episode, collision_result, status_episode, policy_action, subprocess_permit, enable_print):
    """Control process - responsible for robot motion control and environment"""
    print("Control process started")
    from real.environment import Environment
    from utils.policy import get_expert_policy

    # Initialize environment here only
    env = Environment(
        robot_address=config["hardware"]["robot_address"],
        **config["demo_collection"]["env"],
        **config["hardware"]["camera"]
    )

    def get_goal_info():
        env.act_to_goal()
        rtn_dict = env.observation()
        img = rtn_dict['img_1']
        img2 = rtn_dict['img_2'] if 'img_2' in rtn_dict else None
        return img, img2

    # goal_pose = None
    goal_pose = [-482.7015380859375, -40.29251480102539, 181.0076141357422, -180., 0., -12.172778129577637]
    if goal_pose is None:
        goal_pose = env.robot_ins.get_gripper_TCP_pose()
    goal_pose[3] = -180.0
    goal_pose[4] = 0.0
    # Convert to meters and radians
    env.robot_ins.move_cart(goal_pose, tool=2, user=0, vel=40)

    # Initialize environment properly
    env.set_target_coordinate(use_cur=True)
    goal_img, goal_img2 = get_goal_info()

    goal_state["img_1"] = goal_img.copy()
    goal_state["img_2"] = goal_img2.copy()

    env.init()
    env.action_abs_T(env.wgT_tar @ env.g_tar_g_init_T)

    freq_controller = FixedFrequencyController(config["demo_collection"]["ctrl_freq"])
    performance_monitor = []
    print_counter = 0
    
    # DAgger variables
    if dagger_config and dagger_config["utilized"]:
        # Add termination thresholds
        pose_error_threshold = dagger_config["task_termination"]["pose_error_threshold"]
        time_upper_bound = dagger_config["task_termination"]["use_time_upperbound"]
    
    # Episode management
    current_episode = 0
    num_frames = 0
    first_in_error = False
    end_episode = False

    print(f"=============Episode {current_episode} started=============")
    episode_start_time = time.time()
    try:
        while True:
            start_time = time.time()
            num_frames += 1

            # Get observation
            rtn_dict = env.observation()
            #visualize
            img_vis = rtn_dict['img_1'].copy()
            if 'img_2' in rtn_dict:
                img2_vis = rtn_dict['img_2'].copy()
                combined_img = np.hstack((img_vis, img2_vis))
            else:
                combined_img = img_vis
            cv2.imshow("Combined Image", combined_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Send observation to main process
            reinit_res = env.reinit()
            # print(f"dist:{reinit_res['dist']},angle:{reinit_res['angle']}")

            state["img_1"] = rtn_dict.get('img_1').copy()
            state["img_2"] = rtn_dict.get('img_2').copy()
            state["wgT"] = env.wgT.copy()
            state["wgT_tar"] = env.wgT_tar.copy()
            # Check if we should use policy action

            if is_dagger_episode.value:
                dT = construct_dT_from_action(policy_action[:], dof=6)
                env.action_dT(dT)
                # ==========Check dagger termination conditions==========
                if dagger_config and dagger_config["utilized"]:
                    # Time limit check
                    if time.time() - episode_start_time > time_upper_bound:
                        print("Time limit reached, terminating episode.")
                        end_episode = True

                    # Pose error check
                    if reinit_res["dist"] > pose_error_threshold["trans"] or \
                            reinit_res["angle"] > pose_error_threshold["rot"]:
                        if first_in_error:
                            if time.time() - error_start_time > pose_error_threshold["time"]:
                                print(
                                    f"Pose error threshold exceeded for over {time.time() - error_start_time} seconds, terminating episode.")
                                end_episode = True
                        else:
                            first_in_error = True
                            error_start_time = time.time()
                    else:
                        first_in_error = False

                    if not collision_result['safe']:
                        print(f"Collision threshold reached! Distance: {collision_result['distance']:.3f}mm")
                        end_episode = True

                # ==========Check dagger termination conditions==========
                # ==========finalize episode==============
                if reinit_res["close_enough"] or end_episode:
                    print(f"Final error: trans:{reinit_res['dist']} mm,rot: {reinit_res['angle']} deg")
                    if not reinit_res["close_enough"]:
                        env.init()  # reinit manually

                    end_episode = False
                    first_in_error = False
                    print(
                        f"Episode {current_episode} completed with {num_frames} frames.")
                    current_episode += 1
                    num_frames = 0
                    # Reset for next episode
                    goal_img, goal_img2 = get_goal_info()
                    goal_state["img_1"] = goal_img.copy()
                    goal_state["img_2"] = goal_img2.copy()
                    env.action_abs_T(env.wgT_tar @ env.g_tar_g_init_T)
                    subprocess_permit.value = False
                    # rubbish_t_start = time.time()
                    status_episode.value = 1
                    # for k,v in state.items(): #clear buffer
                    #     state[k] = None
                    status_wait_t = time.time()
                    while not subprocess_permit.value:  # subprocess can start the next episode only if the main process permits in case of training
                        time.sleep(0.05)
                        # print("waiting....")
                        if time.time() - status_wait_t > 0.7:
                            status_episode.value = 0
                    if time.time() - status_wait_t > 0.7:
                        status_episode.value = 0
                    else:
                        elapsed = max(0,0.7 - (time.time() - status_wait_t)) #status_episode.value can stay 1 for about exactly 0.7 second (<1s)
                        time.sleep(elapsed)
                        status_episode.value = 0

                    # print(f"rubbish_time:{time.time() - rubbish_t_start}")
                    # print("waiting...")
                    # state["img_1"] = None
                    # state["img_2"] = None
                    # state["wgT"] = None
                    # state["wgT_tar"] = None
                    print(f"=============Episode {current_episode} started=============")
                    episode_start_time = time.time()  # reset timer
                # ==============finilize episode============
            else:
                act_dict = get_expert_policy(
                    wgT_tar=env.wgT_tar, wgT=env.wgT,
                    trans_vel=config["demo_collection"]["env"]["velocity"]['trans_vel'],
                    rot_vel=config["demo_collection"]["env"]["velocity"]['rot_vel'],
                    uniform_vel=config["demo_collection"]["env"]["velocity"]['uniform_vel'],
                    dist_eps=env.dist_eps, angle_eps=env.angle_eps,
                    motion_type="simultaneously", dof=6,
                    need_trans_unit_transform=False, fine_print=False, real=True
                )
                env.action_dT(act_dict["dT"])

                #==========finalize episode==============
                if reinit_res["close_enough"] or end_episode:
                    if not reinit_res["close_enough"]:
                        env.init() #reinit manually

                    end_episode = False
                    first_in_error = False
                    print(f"Episode {current_episode} completed with {num_frames} frames.")
                    current_episode += 1
                    num_frames = 0

                    # Reset for next episode
                    goal_img, goal_img2 = get_goal_info()
                    goal_state["img_1"] = goal_img.copy()
                    goal_state["img_2"] = goal_img2.copy()
                    env.action_abs_T(env.wgT_tar @ env.g_tar_g_init_T)
                    subprocess_permit.value = False
                    # rubbish_t_start = time.time()
                    status_episode.value = 1
                    # for k,v in state.items(): #clear buffer
                    #     state[k] = None
                    status_wait_t = time.time()
                    while not subprocess_permit.value:  #subprocess can start the next episode only if the main process permits in case of training
                        time.sleep(0.05)
                        # print("waiting....")
                        if time.time() - status_wait_t > 0.7:
                            status_episode.value = 0
                    if time.time() - status_wait_t > 0.7:
                        status_episode.value = 0
                    else:
                        elapsed = max(0,0.7 - (time.time() - status_wait_t)) #status_episode.value can stay 1 for about exactly 0.7 second (<1s)
                        time.sleep(elapsed)
                        status_episode.value = 0

                    # print(f"rubbish_time:{time.time()-rubbish_t_start}")
                        # print("waiting...")
                    # state["img_1"] = None
                    # state["img_2"] = None
                    # state["wgT"] = None
                    # state["wgT_tar"] = None
                    print(f"=============Episode {current_episode} started=============")
                    episode_start_time = time.time() #reset timer
                #==============finilize episode============

            # Frequency control
            work_time = time.time() - start_time  # Calculate work time
            sleep_time = max(0, freq_controller.interval - work_time)
            time.sleep(sleep_time)

            # Calculate actual frequency based on total cycle time
            cycle_time = time.time() - start_time
            actual_freq = 1.0 / cycle_time if cycle_time > 0 else 0
            performance_monitor.append(actual_freq)
            print_counter += 1
            # print(f"actual freq of ctrl:{actual_freq}")
            if print_counter % 100 == 0 and enable_print.value:  # Print every 100 cycles
                avg_freq = sum(performance_monitor[-100:]) / 100
                print(f"Control process frequency: {avg_freq:.2f}Hz")
                print_counter = 0

    except KeyboardInterrupt:
        pass
    finally:
        print("Control process exited")

def collision_detection_process(config,state,is_dagger_episode,collision_result,status_episode,subprocess_permit, enable_print):
    """Collision detection process"""
    from real.collision_detection.sdf_collision import Open3DVisualizer
    print("Collision detection process started")
    gripper_path = os.path.join(PROJECT_ROOT_DIR, "meshes/zhixing/crt_ctag2f120.urdf")
    object_path = os.path.join(PROJECT_ROOT_DIR, "meshes/classical_part.STL")
    
    # Calibration transform
    cali_T = np.eye(4)
    cali_T[1, 1] *= -1
    cali_T[2, 2] *= -1
    cali_T[0, 3] = -0.006
    cali_T[2, 3] = 0.085
    
    collision_detector = CollisionDetector(
        gripper_path, object_path,
        scalar_1=1.0, scalar_2=0.001,
        use_convex_hull_1=False,
        use_convex_hull_2=False,
        cali_T=cali_T
    )
    visualizer = Open3DVisualizer(collision_detector)
    
    freq_controller = FixedFrequencyController(config["demo_collection"]["dagger"]["check_freq"])
    performance_monitor = []
    print_counter = 0
    last_wgT = None

    threshold = config["demo_collection"]["dagger"]["task_termination"]["min_position_threshold"]

    last_episode_start_t = time.time()

    while state['wgT'] is None:
        time.sleep(0.1)
    try:
        while True:
            start_time = time.time()  # Add this line

            if is_dagger_episode.value:
                if status_episode.value == 1 and time.time() - last_episode_start_t > 1:
                    last_episode_start_t = time.time()
                    print("collision detection entered new episode.")
                    while not subprocess_permit.value: #subprocess can start the next episode only if the main process permits in case of training
                        time.sleep(0.1)
                    continue # to judge again is dagger episode for long waiting

                # Get state update
                wgT = state['wgT']

                if wgT is None:
                    continue

                if last_wgT is None:
                    dT = np.linalg.inv(state['wgT_tar']) @ wgT
                else:
                    # Subsequent frames
                    dT = np.linalg.inv(last_wgT) @ wgT

                dT[0:3, 3] /= 1000  # mm to meters
                collision_detector.update_pos(dT)
                last_wgT = wgT

                # Perform collision detection
                contact_flag, distance = collision_detector.check_collision(
                    num_sample_points=500,
                    threshold=threshold[0]
                )
                #update rendering
                visualizer.run_iteration()

                # Send detection result
                collision_result['timestamp'] = time.time()
                collision_result['contact_flag'] = contact_flag
                collision_result['distance'] = distance
                collision_result['safe'] = (not contact_flag) and distance >=threshold[0] and distance <= threshold[1]

           # Frequency control - measure actual work time
            work_time = time.time() - start_time  # Calculate work time
            sleep_time = max(0, freq_controller.interval - work_time)
            time.sleep(sleep_time)
            
            # Calculate actual frequency based on total cycle time
            cycle_time = time.time() - start_time
            actual_freq = 1.0 / cycle_time if cycle_time > 0 else 0
            performance_monitor.append(actual_freq)
            print_counter += 1
            # print(f"actual freq of collision:{actual_freq}")
            if print_counter % 100 == 0 and enable_print.value:  # Print every 100 cycles
                avg_freq = sum(performance_monitor[-100:]) / 100
                print(f"collision process frequency: {avg_freq:.2f}Hz")
                print_counter = 0

    except KeyboardInterrupt:
        pass
    finally:
        print("Collision detection process exited")

def data_processing_process(data_queue, processed_queue, config, enable_print):
    """Data processing process - handles image processing and storage"""
    print("Data processing process started")
    from utils.input_process import clip_image
    from utils.augmentation import AugmentationModule

    # Initialize augmentation if enabled
    augmentation_module = None
    if config["demo_collection"]["img"]["augmentation"]["utilized"]:
        augmentation_module = AugmentationModule(
            pretrained_model_pth=config["demo_collection"]["img"]["augmentation"]["pretrained_model_pth"],
            scale_range_min=config["demo_collection"]["img"]["augmentation"]["scale_range_min"],
            scale_range_max=config["demo_collection"]["img"]["augmentation"]["scale_range_max"],
            offset_range_min=config["demo_collection"]["img"]["augmentation"]["offset_range_min"],
            offset_range_max=config["demo_collection"]["img"]["augmentation"]["offset_range_max"],
            noise_std=config["demo_collection"]["img"]["augmentation"]["noise_std"],
            draw_box=False,
            box_color=(0, 255, 0),
            box_thickness=2
        )

    freq_controller = FixedFrequencyController(config["demo_collection"]["data_process_freq"])
    performance_monitor = []
    print_counter = 0

    try:
        while True:
            start_time = time.time()

            # Process data from queue
            if not data_queue.empty():
                data_packet = data_queue.get_nowait()
                img1 = data_packet['img_1']
                img2 = data_packet.get('img_2')
                timestamp = data_packet['timestamp']

                # Process image 1
                img1_processed = clip_image(img1, config["demo_collection"]["img"]["size"], keep_right=True)
                if config["demo_collection"]["img"]["save_type"] == "rgb":
                    img1_processed = img1_processed[:, :, ::-1]
                
                # Process image 2 if exists
                img2_processed = None
                if img2 is not None:
                    img2_processed = clip_image(img2, config["demo_collection"]["img"]["size"], keep_right=True)
                    if config["demo_collection"]["img"]["save_type"] == "rgb":
                        img2_processed = img2_processed[:, :, ::-1]

                # Apply augmentation if enabled
                if augmentation_module:
                    img1_light = augmentation_module.augment_image(img1_processed, True)
                    img2_light = augmentation_module.augment_image(img2_processed, True) if img2_processed is not None else None
                else:
                    img1_light = None
                    img2_light = None

                # Send processed data back to main process
                processed_data = {
                    'img_1': img1_processed,
                    'img_1_light': img1_light,
                    'img_2': img2_processed,
                    'img_2_light': img2_light,
                    'timestamp': timestamp
                }
                processed_queue.put_nowait(processed_data)

            # Frequency control
            work_time = time.time() - start_time  # Calculate work time
            sleep_time = max(0, freq_controller.interval - work_time)
            time.sleep(sleep_time)

            # Calculate actual frequency based on total cycle time
            cycle_time = time.time() - start_time
            actual_freq = 1.0 / cycle_time if cycle_time > 0 else 0
            performance_monitor.append(actual_freq)
            print_counter += 1
            # print(f"actual freq of data process:{actual_freq}")
            if print_counter % 100 == 0 and enable_print.value:  # Print every 100 cycles
                avg_freq = sum(performance_monitor[-100:]) / 100
                # print(f"Data processing process frequency: {avg_freq:.2f}Hz")
                print_counter = 0

    except KeyboardInterrupt:
        pass
    finally:
        print("Data processing process exited")

def main_process():
    """Main process - coordinates all subprocesses"""
    global state, goal_state, is_dagger_episode, collision_result, status_episode, policy_action, subprocess_permit

    manager = Manager() #create shared dict and value
    state = manager.dict({
        'img_1': None,
        'img_2': None,
        'wgT': None,
        'wgT_tar': None,
    })
    goal_state = manager.dict({
        'img_1': None,
        'img_2': None,
    })
    is_dagger_episode = manager.Value('b', False)  # 布尔值
    collision_result = manager.dict({
        'timestamp': time.time(),
        'contact_flag': False,
        'distance': 1000000,
        'safe': True
    })
    status_episode = manager.Value('i', 0)  #0 for cur_episode and 1 for new episode
    policy_action = manager.Array('d', [0.0] * 6)
    subprocess_permit = manager.Value('b', True)
    enable_print = manager.Value('b', True)

    print("Main process started")
    
    # Load configuration
    config_dir = "../configs/demo_collection_real.json"
    with open(config_dir, "r") as j:
        config = json.load(j)
    
    # DAgger configuration
    dagger_config = config["demo_collection"].get("dagger", {})
    use_dagger = dagger_config.get("utilized", False)
    if use_dagger:
        policy_model, optimizer, criterion, model_config = setup_policy_model(
            config_path="../configs/train_mlp.json",
            checkpoint_path=dagger_config.get("model_path", None)
        )
        policy_model.eval()
        save_img_size = model_config["algorithm"]["policy"]["params"]["encoder"]["params"]["img_size"]

    # Prepare for data collection
    base_dir = "/media/kiriyamagk/One Touch/AlignAnything_real"
    current_date = config['overall_setting']['file_name']
    database_dir = os.path.join(base_dir, current_date, 'hdf5')
    ensure_dir(database_dir)
    dataset_dir = os.path.join(database_dir, 'mimic.hdf5')

    # Add demo count limit
    demo_total_num = config['overall_setting']['demo_total_num']
    episode_count = 0
    epi_length = [0] * demo_total_num
    is_dagger_episode.value = use_dagger and is_in_dagger_episode(episode_count, dagger_config)

    
    # Create inter-process communication queues
    queues['data'] = mp.Queue(maxsize=500)        # Raw image data for processing,non-realtime
    queues['processed'] = mp.Queue(maxsize=500)   # Processed image data,non-realtime

    priority_list = ["high","normal","normal"]
    # Create subprocesses
    processes.append(mp.Process(
        target=control_process,
        args=(config, dagger_config, state, goal_state, is_dagger_episode, collision_result, status_episode, policy_action, subprocess_permit,enable_print),
        name="ControlProcess"
    ))
    processes.append(mp.Process(
        target=collision_detection_process,
        args=(config,state,is_dagger_episode,collision_result,status_episode,subprocess_permit,enable_print),
        name="CollisionProcess"
    ))
    processes.append(mp.Process(
        target=data_processing_process,
        args=(queues['data'], queues['processed'], config,enable_print),
        name="DataProcess"
    ))
    
    # Start all processes
    for idx,p in enumerate(processes):
        p.daemon = True
        # set_process_priority(p.pid, priority_list[idx])
        p.start()
    
    # Get initial state from control process with longer timeout
    print("Waiting for control process to initialize environment...")

    while state["img_1"] is None:
        time.sleep(0.1)
        # print("sleeping")

    local_goal_state = {
        'img_1': goal_state['img_1'].copy(),
        'img_2': goal_state['img_2'].copy(),
    }

    # Handle existing demos
    existed_demo_num = 0
    replace_existed_hdf5 = config["overall_setting"]["replace_existed_hdf5"]
    if not replace_existed_hdf5 and os.path.exists(dataset_dir):
        with h5py.File(dataset_dir, 'r') as f:
            if 'data' in f:
                existed_demo_num = len(f['data'])
        new_f_out = h5py.File(dataset_dir, "r+")
    else:
        new_f_out = h5py.File(dataset_dir, "w")
    
    print("All subprocesses started")

    main_freq = config["demo_collection"]["data_collect_freq"]
    local_new_episode = (status_episode.value == 1)
    last_episode_start_t = time.time()
    frame_sampling_gap = config["demo_collection"]["frame_sampling_gap"]
    idx_tmp = 0
    abandon_save_data = False
    try:
        # Main loop
        performance_monitor = []
        print_counter = 0
        episode_data = {
            'obs': [],
            'expert_actions': [],  # Always store expert actions
            'delta_poses': [] if config["demo_collection"]['record_pose'] else None,
            'timestamps': []
        }

        while episode_count < demo_total_num:
            start_time = time.time()

            # img_goal_vis = np.hstack((local_goal_state['img_1'], local_goal_state['img_2']))
            # cv2.imshow("Combined Goal Image", img_goal_vis)
            # cv2.waitKey(1)

            local_new_episode = status_episode.value == 1 and time.time() - last_episode_start_t > 1

            if local_new_episode:
                last_episode_start_t = time.time()

            # =============data storage(high dim)=============
            if not abandon_save_data and all(state.get(k) is not None for k in state.keys()):

                data_packet = {
                    'img_1': state.get('img_1').copy() if not local_new_episode else local_goal_state['img_1'].copy(),
                    'img_2': state.get('img_2').copy() if not local_new_episode else local_goal_state['img_2'].copy(),
                    'timestamp': time.time()
                }

                # cv2.imwrite(f"imgs_un/{episode_count}/unprocessed_{idx_tmp}.png", data_packet["img_1"])
                # idx_tmp+=1
                if not queues['data'].full():
                    queues['data'].put_nowait(data_packet)
                else:
                    print("Full")
                    raise RuntimeError("Full")

                if not local_new_episode:
                    expert_act_dict = get_expert_policy(
                        wgT_tar=state['wgT_tar'],
                        wgT=state['wgT'],
                        trans_vel=config["demo_collection"]["env"]["velocity"]['trans_vel'],
                        rot_vel=config["demo_collection"]["env"]["velocity"]['rot_vel'],
                        uniform_vel=config["demo_collection"]["env"]["velocity"]['uniform_vel'],
                        dist_eps=config["demo_collection"]["env"]["stop_policy"]["dist_eps"],
                        angle_eps=config["demo_collection"]["env"]["stop_policy"]["angle_eps"],
                        motion_type="simultaneously",
                        dof=6,
                        need_trans_unit_transform=False,
                        fine_print=False,
                        real=True
                    )
                    expert_action = np.concatenate((expert_act_dict['vel_tr'], expert_act_dict['vel_rot']))
                    episode_data['expert_actions'].append(expert_action)

                    if config["demo_collection"]['record_pose']:
                        episode_data['delta_poses'].append(expert_act_dict['cur_goal_delta_pose'])
                else:
                    episode_data['expert_actions'].append(np.zeros(6))

                    if config["demo_collection"]['record_pose']:
                        episode_data['delta_poses'].append(np.zeros(6))

                    epi_length[episode_count] = len(episode_data['expert_actions'])

                    local_goal_state['img_1'] = goal_state['img_1']
                    local_goal_state['img_2'] = goal_state['img_2']

            if local_new_episode:
                abandon_save_data = True
            # =============data storage(high dim)=============

            # ============================DAgger prediction step============================
            if is_dagger_episode.value and policy_model and not local_new_episode:
                # Prepare observation for policy
                obs_dict = prepare_observation_for_policy(
                    img_size=save_img_size,
                    hdf_img_size=config["demo_collection"]["img"]["size"],
                    img=data_packet.get('img_1').copy()[:, :, ::-1],
                    img_goal=local_goal_state["img_1"][:, :, ::-1],
                    img2=data_packet.get('img_2', None).copy()[:, :, ::-1],
                    img2_goal=local_goal_state["img_2"][:, :, ::-1],
                    keep_right=True
                )

                # Get policy action
                action_array = get_policy_action(policy_model, obs_dict)
                for i in range(6):
                    policy_action[i] = action_array[i]
            # ============================DAgger prediction step==========================


            #============================data processing================================
            if not queues['processed'].empty():
                processed_data = queues['processed'].get_nowait()
                # Store processed data for this episode
                episode_data['obs'].append({
                    'img_1': processed_data.get('img_1'),
                    'img_1_light': processed_data.get('img_1_light'),
                    'img_2': processed_data.get('img_2'),
                    'img_2_light': processed_data.get('img_2_light')
                })
                episode_data['timestamps'].append(processed_data['timestamp'])
            #============================data processing================================

            # print(f"processed img num:{len(episode_data['obs'])}")
            # print(f"data_len: {len(episode_data['obs'])}")
            # print(f"epi_length[episode_count]:{epi_length[episode_count]}")

            #===========================finish episode===========================
            if epi_length[episode_count] > 0 and len(episode_data["obs"]) == epi_length[episode_count]:

                #clear queue
                while not queues["data"].empty():
                    queues["data"].get_nowait()
                while not queues["processed"].empty():
                    queues["processed"].get_nowait()

                # Save episode data to HDF5 with original structure
                episode_data = frame_sample(episode_data,frame_sampling_gap)
                episode_data = post_process_episode(episode_data, config)
                save_episode_to_hdf5(new_f_out, episode_data, existed_demo_num + episode_count, config)
                # for i in range(len(episode_data["obs"])):
                #     cv2.imwrite(f"imgs_pro/{episode_count}/processed_{i}.png", episode_data['obs'][i]["img_1"])
                
                # DAgger: Check if we should train with proper filtering
                if use_dagger:
                    should_train, train_epochs, dagger_proportion = should_train_policy(episode_count, dagger_config)
                    if should_train:
                        print(f'[DAgger] Training policy model at episode {episode_count} with {train_epochs} epochs')
                        new_f_out.close()

                        enable_print.value = False
                        
                        # Prepare training config
                        data_cfg = model_config["dataset"].copy()
                        data_cfg["hdf5_path"] = dataset_dir
                        train_cfg = model_config["training"].copy()
                        
                        # Create model save path
                        model_path = os.path.join(base_dir, current_date, 'models')
                        ensure_dir(model_path)
                        
                        # Create filter key with proper proportion
                        filter_key = None
                        if dagger_proportion is not None:
                            # Get all demo keys
                            with h5py.File(dataset_dir, 'r') as f:
                                all_demos = sorted(list(f['data'].keys()))
                            
                            # Separate dagger and non-dagger episodes
                            dagger_ranges = dagger_config.get('dagger_episodes', {}).get('use_type', [])
                            dagger_set = set()
                            for s, e in dagger_ranges:
                                for ep in range(s, e + 1):
                                    demo_id = f"demo_{ep}"
                                    if demo_id in all_demos:
                                        dagger_set.add(demo_id)
                            
                            dagger_demos = [d for d in all_demos if d in dagger_set]
                            non_dagger_demos = [d for d in all_demos if d not in dagger_set]
                            
                            # Select demos based on proportion
                            num_total = len(all_demos)
                            num_dagger_target = int(round(dagger_proportion * num_total))
                            num_non_dagger_target = max(0, num_total - num_dagger_target)
                            
                            rng = np.random.default_rng(seed=episode_count)
                            chosen_dagger = rng.choice(
                                dagger_demos, 
                                size=min(len(dagger_demos), num_dagger_target), 
                                replace=False
                            ).tolist()
                            
                            chosen_non_dagger = rng.choice(
                                non_dagger_demos, 
                                size=min(len(non_dagger_demos), num_non_dagger_target), 
                                replace=False
                            ).tolist()
                            
                            mixed = sorted(chosen_dagger + chosen_non_dagger)
                            filter_key = f"dagger_mix_ep_{episode_count}"
                            print("created filter key:{}".format(filter_key))
                            
                            # Create filter key in HDF5
                            create_hdf5_filter_key(
                                hdf5_path=dataset_dir, 
                                demo_keys=mixed, 
                                key_name=filter_key,
                                return_length=False
                            )
                        
                        # Train policy with filter
                        policy_model = train_policy(
                            img_size=save_img_size,
                            num_train_steps=model_config["training"]["num_train_steps_per_epoch"],
                            model=policy_model,
                            optimizer=optimizer,
                            criterion=criterion,
                            num_epochs=train_epochs,
                            batch_size=model_config["training"]["batch_size"],
                            train_cfg=train_cfg,
                            data_cfg=data_cfg,
                            save_path=model_path,
                            episode_idx=episode_count,
                            filter_by_attribute=filter_key
                        )
                        # Reopen HDF5 file
                        new_f_out = h5py.File(dataset_dir, "r+")

                # Reset for next episode
                episode_count += 1
                # idx_tmp = 0
                is_dagger_episode.value = use_dagger and is_in_dagger_episode(episode_count, dagger_config)
                episode_data = {
                    'obs': [],
                    'expert_actions': [],
                    'delta_poses': [] if config["demo_collection"]['record_pose'] else None,
                    'timestamps': []
                }
                state.update({
                    'img_1': None,
                    'img_2': None,
                    'wgT': None,
                    'wgT_tar': None
                })
                abandon_save_data = False
                subprocess_permit.value = True  # subprocess can start the next episode only if the main process permits
                enable_print.value = True
            #===========================finish episode===========================

            #===========================display images===========================
            # img_vis = rtn_dict['img_1'].copy()
            # if 'img_2' in rtn_dict:
            #     img2_vis = rtn_dict['img_2'].copy()
            #     combined_img = np.hstack((img_vis, img2_vis))
            # else:
            #     combined_img = img_vis
            # cv2.imshow("Combined Image", combined_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            #===========================display images===========================
            # Frequency control
            work_time = time.time() - start_time  # Calculate work time
            sleep_time = max(0, 1/main_freq - work_time)
            time.sleep(sleep_time)

            # Calculate actual frequency based on total cycle time
            cycle_time = time.time() - start_time
            actual_freq = 1.0 / cycle_time if cycle_time > 0 else 0
            performance_monitor.append(actual_freq)
            print_counter += 1
            # print(f"actual freq of main:{actual_freq}")
            if print_counter % 100 == 0:  # Print every 100 cycles
                avg_freq = sum(performance_monitor[-100:]) / 100
                print(f"Main process frequency: {avg_freq:.2f}Hz")
                print_counter = 0

    except KeyboardInterrupt:
        print("Interrupt signal received")
    finally:
        # Graceful shutdown
        print("Shutting down...")
        
        # Wait for processes to finish
        for p in processes:
            p.join(timeout=2)
            if p.is_alive():
                print(f"Process {p.name} did not exit, terminating")
                p.terminate()
                p.join(timeout=1)
                if p.is_alive():
                    p.kill()
        
        # Finalize HDF5 file
        add_env_meta(new_f_out)
        add_config(new_f_out, config)
        new_f_out.close()
        compute_num_samples(dataset_dir)
        split_train_val_from_hdf5(dataset_dir, val_ratio=0.1)
        
        cv2.destroyAllWindows()
        print("Main process exited")


def save_episode_to_hdf5(hdf5_file, episode_data, episode_index, config):
    """Save episode data to HDF5 file with proper structure"""
    # Create groups with original naming convention
    demo_group = hdf5_file.create_group(f"data/demo_{episode_index}")
    obs_group = demo_group.create_group("obs")
    
    # Save observations with original names
    obs_group.create_dataset("robot0_eye_in_hand_image", 
                            data=np.array([obs['img_1'] for obs in episode_data['obs']]))
    
    if episode_data['obs'][0].get('img_2') is not None:
        obs_group.create_dataset("robot0_eye_in_hand_image_2", 
                                data=np.array([obs['img_2'] for obs in episode_data['obs']]))
    
    if episode_data['obs'][0].get('img_1_light') is not None:
        obs_group.create_dataset("robot0_eye_in_hand_image_light", 
                                data=np.array([obs['img_1_light'] for obs in episode_data['obs']]))
    
    if episode_data['obs'][0].get('img_2_light') is not None:
        obs_group.create_dataset("robot0_eye_in_hand_image_2_light", 
                                data=np.array([obs['img_2_light'] for obs in episode_data['obs']]))
    
    # Always save expert actions as "actions"
    demo_group.create_dataset("actions", data=np.array(episode_data['expert_actions']))
    
    # Save delta poses if configured
    if config["demo_collection"]['record_pose']:
        demo_group.create_dataset("delta_pos_curgoal", 
                                 data=np.array(episode_data['delta_poses']))
    
    print(f"==========Saved episode {episode_index} with {len(episode_data['obs'])} frames==============")

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    #Try to increase process priority
    try:
        os.nice(-10)  # Increase main process priority
        print("Main process priority set.")
    except:
        print("Error when setting main process priority.")
        pass
    
    main_process()