def calculate_score(y_true, y_pred, max_error=1.0):
    """
    计算模型评分（0-1）
    :param y_true: 实际值
    :param y_pred: 预测值
    :param max_error: 最大误差（用于归一化）
    :return: 评分（0-1）
    """
    mse = ((y_true - y_pred) ** 2).mean()  # 计算MSE
    score = 1 - (mse / max_error)  # 将MSE映射到0-1
    return max(0, min(1, score))  # 确保分数在0-1之间
