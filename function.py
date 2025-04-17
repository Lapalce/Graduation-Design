import torch
import numpy as np


def calculate_score(y_true, y_pred):
    loss = abs((y_pred - y_true) / y_true)

    # 定义每个特征的权重
    weights = torch.tensor([0.11, 0.50, 0.11, 0.11, 0.17])  # 权重：open, close, high, low, volume_log

    # 计算加权 MSE
    weighted_loss = loss * weights  # 每个特征的 MSE 乘以对应的权重
    total_loss = weighted_loss.sum()  # 求和得到最终损失

    return total_loss


def create_sequences(data, seq_length, step):
    xs, ys = [], []
    for i in range(0, len(data) - seq_length - step, seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length + step][[0, 1, 2, 3, 8]]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

