import numpy as np
import torch
import os
from algo.bc import BehaviorCloning
from utils.transform import rotation_matrix_z
from utils.policy import get_expert_policy

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

def aggregate_dataset(new_data, dataset_path, max_size=None):
    """
    将新收集的数据聚合到现有的数据集中(DAgger的核心)
    
    Args:
        new_data: 新收集的数据
        dataset_path: 现有数据集的路径
        max_size: 数据集的最大大小，如果指定，则当数据集大小超过此值时会删除旧数据
    
    Returns:
        更新后的数据集路径
    """
    # 这里应该实现数据聚合的具体逻辑
    # 由于涉及到HDF5文件的具体操作，需要根据项目的数据结构进行定制
    print(f"[DAgger] 将新数据聚合到 {dataset_path}")
    return dataset_path

def train_policy(dataset_path, model, optimizer, criterion, num_epochs=10, batch_size=16, device=None):
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
        
    Returns:
        训练后的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 这里应该实现训练的具体逻辑
    # 需要根据项目的训练流程进行定制
    print(f"[DAgger] 使用数据集 {dataset_path} 训练模型")
    
    # 返回训练后的模型
    return model 