# kalman_filter.py - 直接复制这个文件到你的项目
import numpy as np
from typing import Optional, Tuple


class KalmanFilter:
    """纯 NumPy 实现的卡尔曼滤波器"""

    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # 状态向量
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.B = None
        self.u = np.zeros((dim_u, 1))

        # 观测相关
        self.R = np.eye(dim_z)
        self.H = np.zeros((dim_z, dim_x))

        # 身份矩阵
        self._I = np.eye(dim_x)

    def predict(self, u: Optional[np.ndarray] = None,
                F: Optional[np.ndarray] = None,
                Q: Optional[np.ndarray] = None,
                B: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:

        if F is None:
            F = np.eye(self.dim_x)  # 默认恒等矩阵
        if Q is None:
            Q = self.Q
        if B is None and self.dim_u > 0:
            B = self.B
        if u is not None:
            self.u = u.reshape(-1, 1)

        # 状态预测
        if B is not None and self.dim_u > 0:
            self.x = F @ self.x + B @ self.u
        else:
            self.x = F @ self.x

        # 协方差预测
        self.P = F @ self.P @ F.T + Q

        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray,
               R: Optional[np.ndarray] = None,
               H: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:

        if R is None:
            R = self.R
        if H is None:
            H = self.H

        z = z.reshape(-1, 1)

        # 计算残差
        y = z - H @ self.x

        # 系统不确定性
        S = H @ self.P @ H.T + R

        # 卡尔曼增益
        K = self.P @ H.T @ np.linalg.inv(S)

        # 状态更新
        self.x = self.x + K @ y

        # 协方差更新
        I_KH = self._I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x.copy(), self.P.copy()


class AdaptiveKalmanFilter(KalmanFilter):
    """自适应卡尔曼滤波器"""

    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 0,
                 forgetting_factor: float = 0.95):
        super().__init__(dim_x, dim_z, dim_u)
        self.forgetting_factor = forgetting_factor
        self.innovation_history = []
        self.window_size = 20
        self.R_estimate = self.R.copy()

    def adapt_noise_covariance(self, innovation: np.ndarray):
        """自适应调整噪声协方差"""
        d = 1 - self.forgetting_factor

        # Sage-Husa 自适应算法
        innovation_outer = innovation @ innovation.T
        HPH = self.H @ self.P @ self.H.T

        # 更新 R 的估计
        self.R_estimate = (1 - d) * self.R_estimate + d * (innovation_outer - HPH)

        # 确保 R 是对称正定的
        self.R_estimate = (self.R_estimate + self.R_estimate.T) / 2
        eigenvals, eigenvecs = np.linalg.eig(self.R_estimate)
        eigenvals = np.maximum(eigenvals, 0.001)
        self.R = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def predict_and_update(self, z: np.ndarray,
                           u: Optional[np.ndarray] = None,
                           F: Optional[np.ndarray] = None,
                           H: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:

        # 预测步骤
        self.predict(u=u, F=F)

        # 计算创新序列
        H_matrix = H if H is not None else self.H
        innovation = z.reshape(-1, 1) - H_matrix @ self.x

        # 自适应调整
        self.adapt_noise_covariance(innovation)

        # 更新步骤
        if H is not None:
            return self.update(z, R=self.R, H=H)
        else:
            return self.update(z, R=self.R)


# 使用示例
def create_visual_servo_filter():
    """创建用于视觉伺服的卡尔曼滤波器"""
    kf = AdaptiveKalmanFilter(dim_x=6, dim_z=6)
    kf.H = np.eye(6)  # 观测矩阵
    kf.Q = np.eye(6) * 0.01  # 过程噪声
    kf.R = np.eye(6) * 0.1  # 观测噪声
    return kf


class AdaptiveStrengthFilter:
    def __init__(self, dim=6):
        self.dim = dim
        self.kf = KalmanFilter(dim, dim)
        self.kf.F = np.eye(dim)
        self.kf.H = np.eye(dim)
        self.kf.x = np.zeros((dim, 1))

        # 自适应参数
        self.innovation_history = []
        self.jerk_history = []  # 加速度变化率历史
        self.max_history = 10

        # 基础噪声参数
        self.base_Q = np.eye(dim) * 0.01
        self.base_R = np.eye(dim) * 0.1

    def calculate_jerk(self, current_accel, prev_accel, dt=1.0):
        """计算加加速度（jerk）"""
        if prev_accel is None:
            return 0
        jerk = np.linalg.norm(current_accel - prev_accel) / dt
        return jerk

    def adapt_filter_strength(self, measurement, prev_state):
        """根据运动特性自适应调整滤波强度"""
        # 计算创新（观测残差）
        innovation = measurement - self.kf.x.flatten()
        self.innovation_history.append(np.linalg.norm(innovation))
        if len(self.innovation_history) > self.max_history:
            self.innovation_history.pop(0)

        # 计算加加速度
        if prev_state is not None:
            current_accel = measurement - prev_state
            if len(self.jerk_history) > 0:
                prev_accel = self.jerk_history[-1]
                jerk = self.calculate_jerk(current_accel, prev_accel)
            else:
                jerk = 0
            self.jerk_history.append(current_accel)
            if len(self.jerk_history) > self.max_history:
                self.jerk_history.pop(0)
        else:
            jerk = 0

        # 根据抖动程度调整滤波强度
        avg_innovation = np.mean(self.innovation_history) if self.innovation_history else 0
        avg_jerk = np.mean([np.linalg.norm(j) for j in self.jerk_history]) if self.jerk_history else 0

        # 动态调整参数
        innovation_scale = min(avg_innovation * 10, 5.0)  # 限制缩放范围
        jerk_scale = min(avg_jerk * 10, 3.0)

        # 更强的平滑：增大R（更不信任观测），减小Q（更信任模型）
        strength_factor = 1.0 + innovation_scale + jerk_scale

        self.kf.R = self.base_R * strength_factor
        self.kf.Q = self.base_Q / max(strength_factor, 1.0)

        return strength_factor

    def process(self, measurement):
        prev_state = self.kf.x.flatten().copy()

        # 自适应调整
        strength = self.adapt_filter_strength(measurement, prev_state)

        # 标准卡尔曼步骤
        self.kf.predict()
        smoothed, _ = self.kf.update(measurement)

        return smoothed.flatten(), strength