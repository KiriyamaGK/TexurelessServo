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

def compute_position_distance(current_pose, target_pose):
    """
    计算当前姿态和目标姿态之间的位置距离
    
    Args:
        current_pose: 当前夹爪位姿的变换矩阵 (4x4)
        target_pose: 目标位姿的变换矩阵 (4x4)
        
    Returns:
        float: 两个位姿之间的欧氏距离
    """
    # 提取位置部分(平移向量)
    current_position = current_pose[0:3, 3]
    target_position = target_pose[0:3, 3]
    
    # 计算欧氏距离
    distance = np.linalg.norm(current_position - target_position)
    return distance

def compute_orientation_distance(current_pose, target_pose):
    """
    计算当前姿态和目标姿态之间的方向距离
    
    Args:
        current_pose: 当前夹爪位姿的变换矩阵 (4x4)
        target_pose: 目标位姿的变换矩阵 (4x4)
        
    Returns:
        float: 两个旋转矩阵之间的角度差异(弧度)
    """
    # 提取旋转部分
    current_rotation = current_pose[0:3, 0:3]
    target_rotation = target_pose[0:3, 0:3]
    
    # 计算旋转矩阵的差异
    diff_rotation = np.matmul(current_rotation, target_rotation.T)
    
    # 从差异旋转矩阵计算角度
    # 使用旋转矩阵的迹(trace)和余弦定理计算角度
    trace = np.trace(diff_rotation)
    cos_theta = (trace - 1) / 2
    
    # 防止数值误差导致的问题
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    return theta

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
    
    # 禁用梯度计算以加速推理
    with torch.no_grad():
        prediction = model(obs_tensor)
    
    # 将结果转换为numpy数组并返回
    if isinstance(prediction, dict):
        # 如果结果是字典，处理每个键
        action = {}
        for key, value in prediction.items():
            action[key] = value.cpu().numpy()
    else:
        action = prediction.cpu().numpy()
    
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
    print(f"[DAgger] 将新数据聚合到 {dataset_path}")
    

    # 获取当前的demo数量
    demo_count = len([k for k in f["data"].keys() if "demo_" in k])
    print(f"[DAgger] 当前数据集包含 {demo_count} 个演示")
    
    # 处理每个新的episode
    for idx, (obs_data, actions_data, delta_pos_data) in enumerate(zip(
        new_data["obs"], new_data["actions"], 
        new_data.get("delta_pos_curgoal", [None] * len(new_data["obs"]))
    )):
        # 创建新的demo ID
        new_demo_id = f"demo_{demo_count + idx}"
        print(f"[DAgger] 添加新演示: {new_demo_id}")
        
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
        
        print(f"[DAgger] 成功添加演示 {new_demo_id}, 长度: {epi_length}")
    
    # 如果指定了最大大小，删除早期的演示
    if max_size is not None and demo_count + len(new_data["obs"]) > max_size:
        demos_to_remove = demo_count + len(new_data["obs"]) - max_size
        print(f"[DAgger] 超出最大大小，删除 {demos_to_remove} 个早期演示")
        
        for i in range(demos_to_remove):
            demo_to_remove = f"demo_{i}"
            if f"data/{demo_to_remove}" in f:
                del f[f"data/{demo_to_remove}"]
                print(f"[DAgger] 已删除 {demo_to_remove}")
    
    

def train_policy(hdf5_file, model, optimizer, criterion, num_epochs=10, batch_size=16, device=None, config=None):
    """
    使用聚合的数据集训练策略
    
    Args:
        dataset_path: 数据集路径
        model: 策略模型
        optimizer: 优化器
        criterion: 损失函数
        num_epochs: 训练轮数
        batch_size: 批量大小
        device: 运行设备
        config: 配置字典(可选)
        
    Returns:
        训练后的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 如果没有提供config，构建一个基本配置
    if config is None:
        config = {
            "hdf5_file": hdf5_file,
            "specific_obs_keys": ["robot0_eye_in_hand_image", "robot0_eye_in_hand_image_2"],
            "label_keys": ["actions"],
            "seq_length": 1,
            "hdf5_cache_mode": "low_dim",
            "hdf5_use_swmr": True,
            "bgr2rgb": False
        }
    
    # 创建训练数据集
    train_set = dataset_factory(
        config,
        img_size=config.get("img_size", None), #train_mlp.algorithm.policy.params.encoder.params.img_size
        filter_by_attribute="train"
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    # 创建BC算法实例
    bc_algorithm = BehaviorCloning(model, optimizer, criterion)
    
    # 训练模型
    print(f"[DAgger] 开始训练，总轮数: {num_epochs}")
    for epoch in range(num_epochs):
        print(f"[DAgger] Epoch {epoch+1}/{num_epochs}")
        train_loss_dict = bc_algorithm.train(train_loader, num_train_steps=None)
        print(f"[DAgger] Epoch {epoch+1} 训练损失: {train_loss_dict['loss']:.4f}")
    
    print("[DAgger] 训练完成")
    return model 