import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import Model

# 读取数据
data = pd.read_csv('./Data/000001.csv')

# 选择需要的特征
features = ['开盘', '收盘', '最高', '最低', '振幅', '涨跌幅', '涨跌额', '成交量', '成交额', '换手率']
data = data[features]

# 数据归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# 将数据转换为时间序列格式
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 30  # 选择30天的历史数据作为输入
X, y = create_sequences(data_scaled, seq_length)

# 将数据转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 划分训练集和验证集
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size = X.shape[2]  # 特征数量
hidden_size = 64
output_size = y.shape[1]  # 输出特征数量
sparse_layer_size = 128  # 稀疏层的大小
dropout_rate = 0.5  # Dropout率

# 加载模型
log_file = './logger/ori_cnn_bilstm_model_01.pth'
model = Model.CNNBiLSTM(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(log_file))
model.eval()

# 使用最后一天的数据进行验证
model.eval()
with torch.no_grad():
    last_day_data = X[-1].unsqueeze(0)  # 添加批次维度
    predicted = model(last_day_data)
    predicted = scaler.inverse_transform(predicted.numpy())
    actual = scaler.inverse_transform(y[-1].unsqueeze(0).numpy())

    print(f'Predicted: {predicted}, \nActual   : {actual}')

# 保存模型
log_file = './logger/ori_cnn_bilstm_model_01.pth'
torch.save(model.state_dict(), log_file)
