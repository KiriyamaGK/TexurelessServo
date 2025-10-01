import numpy as np

class AdaptiveMovingAverage:
    def __init__(self, dim=6, initial_window=5, max_window=15):
        self.dim = dim
        self.initial_window = initial_window
        self.max_window = max_window
        self.window_size = initial_window
        self.history = []

        # 震荡检测参数
        self.oscillation_threshold = 0.1
        self.stability_counter = 0

    def calculate_oscillation_level(self):
        """计算震荡程度"""
        if len(self.history) < 3:
            return 0.0

        recent_data = np.array(self.history[-3:])
        # 计算变化方向的频繁程度
        diffs = np.diff(recent_data, axis=0)
        direction_changes = 0

        for i in range(self.dim):
            signs = np.sign(diffs[:, i])
            changes = np.sum(np.abs(np.diff(signs)))
            direction_changes += changes

        oscillation_level = min(direction_changes / (self.dim * 2), 1.0)
        return oscillation_level

    def adaptive_window_size(self):
        """根据震荡程度自适应调整窗口大小"""
        oscillation_level = self.calculate_oscillation_level()

        if oscillation_level > self.oscillation_threshold:
            # 震荡严重，增大窗口
            new_window = min(self.max_window, self.window_size + 2)
            self.stability_counter = 0
        else:
            # 稳定，逐渐减小窗口
            self.stability_counter += 1
            if self.stability_counter > 5:
                new_window = max(self.initial_window, self.window_size - 1)
                self.stability_counter = 0
            else:
                new_window = self.window_size

        return new_window

    def weighted_moving_average(self):
        """加权移动平均，近期数据权重更高"""
        if not self.history:
            return np.zeros(self.dim)

        weights = np.linspace(0.5, 1.0, len(self.history))  # 线性权重
        weights = weights / np.sum(weights)  # 归一化

        weighted_sum = np.zeros(self.dim)
        for i, data in enumerate(self.history):
            weighted_sum += data * weights[i]

        return weighted_sum

    def process(self, new_data):
        new_data = np.array(new_data).flatten()

        # 更新历史数据
        self.history.append(new_data.copy())

        # 自适应调整窗口大小
        self.window_size = self.adaptive_window_size()

        # 保持窗口大小
        while len(self.history) > self.window_size:
            self.history.pop(0)

        # 使用加权移动平均
        if len(self.history) == 1:
            return new_data
        else:
            return self.weighted_moving_average()


class butter_filter():
    def __init__(self, butter_k=0.1,butter_k2=0.1) -> None:
        self.butter_k = butter_k
        self.butter_k2 = butter_k2
        self.last_output = np.zeros(6)
        self.last_input = np.zeros(6)

    def process(self, input):
        # input = (input + self.last_input) * self.butter_k - 2 * self.butter_k * self.last_output
        output = np.zeros(6)
        output[0:3] = (input[0:3] + self.last_output[0:3]) * self.butter_k + (1 - 2 * self.butter_k) * self.last_output[0:3]
        output[3:6] = (input[3:6] + self.last_output[3:6]) * self.butter_k2 + (1 - 2 * self.butter_k2) * self.last_output[3:6]
        self.last_output = output
        self.last_input = input
        return output