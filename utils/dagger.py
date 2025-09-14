import numpy as np
import torch
import os
import h5py
from algo.bc import BehaviorCloning
from utils.transform import rotation_matrix_z
from utils.policy import get_expert_policy
from utils.hdf5 import add_useless_things, compute_num_samples, split_train_val_from_hdf5
from torch.utils.data import DataLoader
from dataset.dataset import dataset_factory
import pybullet as p


def compute_position_distance(obj, grip, distance_threshold=1.0):
    closest_points = p.getClosestPoints(obj, grip[0], distance_threshold)

    if not closest_points:
        return {
            "min_distance": float('inf'),  # 无点对，距离无限大
            "is_colliding": False  # 无碰撞
        }

    min_distance = min(point[8] for point in closest_points)  # 最短距离
    is_colliding = any(point[0] == 1 for point in closest_points)  # 检查是否有 contactFlag=1

    return {
        "min_distance": min_distance,
        "is_colliding": is_colliding
    }

def load_policy_model(model_path, model, device=None):
    """
    加载预训练的策略模型
    
    Args:
        model_path: 模型权重文件路径
        model: 模型实例
        device: 运行设备
    
    Returns:
        加载了权重的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"[警告] 模型文件不存在: {model_path}")
        return model
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[成功] 从 {model_path} 加载模型")
    return model

def get_policy_action(model, observation, device=None):
    """
    使用策略模型预测动作
    
    Args:
        model: 策略模型
        observation: 观察值(图像等)，应该已经预处理为合适的格式
        device: 运行设备
        
    Returns:
        预测的动作(numpy数组)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确保模型处于评估模式
    model = model.to(device)
    model.eval()
    
    # 将观察转换为tensor并移动到正确的设备
    if isinstance(observation, dict):
        # 如果是字典形式的观察，处理每个键
        obs_tensor = {}
        for key, value in observation.items():
            if not isinstance(value, torch.Tensor):
                value = torch.FloatTensor(value)
            obs_tensor[key] = value.to(device)
    else:
        # 如果是单一的观察
        if not isinstance(observation, torch.Tensor):
            observation = torch.FloatTensor(observation)
        obs_tensor = observation.to(device)
    with torch.no_grad():
        prediction = model(obs_tensor)

    if isinstance(prediction, dict):
        action = prediction['output_tensor'].cpu().numpy().reshape(-1)
    else:
        action = prediction.cpu().numpy().reshape(-1)
    
    return action

def aggregate_dataset(new_data, f, max_size=None):
    """
    将新收集的数据聚合到现有的数据集中(DAgger的核心)
    
    Args:
        new_data: 新收集的数据，包含以下键:
            - obs: 观察数据
            - actions: 专家动作
            - delta_pos_curgoal: 相对位置信息(可选)
        dataset_path: 现有数据集的路径
        max_size: 数据集的最大大小，如果指定，则当数据集大小超过此值时会删除旧数据
    
    Returns:
        更新后的数据集路径
    """
    

    # 获取当前的demo数量
    demo_count = len([k for k in f["data"].keys() if "demo_" in k])
    print(f"[DAgger] The current dataset contains {demo_count} demonstrations.")
    
    # 处理每个新的episode
    for idx, (obs_data, actions_data, delta_pos_data) in enumerate(zip(
        new_data["obs"], new_data["actions"], 
        new_data.get("delta_pos_curgoal", [None] * len(new_data["obs"]))
    )):
        # 创建新的demo ID
        new_demo_id = f"demo_{demo_count + idx}"
        print(f"[DAgger] Adding new demonstration: {new_demo_id}")
        
        # 添加观察数据
        f.create_group(f"data/{new_demo_id}/obs")
        for key, value in obs_data.items():
            f.create_dataset(f"data/{new_demo_id}/obs/{key}", data=value)
        
        # 添加动作数据
        f.create_dataset(f"data/{new_demo_id}/actions", data=actions_data)
        
        # 添加相对位置数据(如果有)
        if delta_pos_data is not None:
            f.create_dataset(f"data/{new_demo_id}/delta_pos_curgoal", data=delta_pos_data)
        
        # 添加必要的属性和其他数据
        epi_length = actions_data.shape[0]
        add_useless_things(new_f_out=f, demo_ind=new_demo_id, epi_len=epi_length)
        
        print(f"[DAgger] Adding successfully {new_demo_id}, demo length: {epi_length}")
    
    # 如果指定了最大大小，删除早期的演示
    if max_size is not None and demo_count + len(new_data["obs"]) > max_size:
        demos_to_remove = demo_count + len(new_data["obs"]) - max_size
        print(f"[DAgger] Exceeded maximum demo buffer，deleting {demos_to_remove} earliest demos.")
        
        for i in range(demos_to_remove):
            demo_to_remove = f"demo_{i}"
            if f"data/{demo_to_remove}" in f:
                del f[f"data/{demo_to_remove}"]
                print(f"[DAgger] 已删除 {demo_to_remove}")
    
    

def train_policy(img_size, model, num_train_steps, optimizer, criterion, num_epochs=10, batch_size=16, data_cfg=None,train_cfg=None, save_path=None, episode_idx=None, filter_by_attribute=None):
    # 创建训练数据集
    train_set = dataset_factory(
        data_cfg,
        img_size = img_size,
        filter_by_attribute=filter_by_attribute,
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False
    )
    
    # 创建BC算法实例
    bc_algorithm = BehaviorCloning(model, optimizer, criterion)
    
    # 获取保存频率
    num_epochs_save = train_cfg.get("num_epochs_save", 10) if train_cfg else 10
    
    # 训练模型
    print(f"[DAgger] 开始训练，总轮数: {num_epochs}")
    for epoch in range(num_epochs):
        print(f"[DAgger] Epoch {epoch+1}/{num_epochs}")
        train_loss_dict = bc_algorithm.train(train_loader, num_train_steps=num_train_steps)
        current_loss = train_loss_dict['loss']
        print(f"[DAgger] Epoch {epoch+1} 训练损失: {current_loss:.4f}")
        
        # 按指定频率保存模型
        if (epoch + 1) % num_epochs_save == 0 and save_path is not None and episode_idx is not None:
            model_filename = f'dagger_episode_{episode_idx}_epoch_{epoch+1}_loss_{current_loss:.4f}.pth'
            model_file = os.path.join(save_path, model_filename)
            torch.save(model.state_dict(), model_file)
            print(f"[DAgger] 模型已保存到: {model_file}")
    
    print("[DAgger] 训练完成")
    return model 