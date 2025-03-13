import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

import Model
import function as f

# 读取数据
data = pd.read_csv('./Data/000001.csv')

# 选择需要的特征
features = ['开盘', '收盘', '最高', '最低', '振幅', '涨跌幅', '涨跌额', '成交量', '成交额', '换手率']
data = data[features]

# 数据归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# 将数据转换为时间序列格式
def create_sequences(df, seq_len):
    xs, ys = [], []
    for i in range(len(df) - seq_len):
        x = df[i:i + seq_len]
        y = df[i + seq_len]
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

# model = SPre_CNNBiLSTM(input_size, hidden_size, output_size, sparse_layer_size=sparse_layer_size, dropout_rate=dropout_rate)
model = Model.CNNBiLSTM(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义最大误差（根据历史数据或经验设定）
max_error = 1.0  # 假设最大MSE为1.0

# 训练循环
num_epochs = 50
best_score = 0  # 记录最佳评分
best_model_state = None  # 保存最佳模型状态
train_history =

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 计算训练集评分
    train_score = f.calculate_score(batch_y.numpy(), outputs.detach().numpy(), max_error)

    # 验证模型
    model.eval()
    val_loss = 0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            # 收集验证集的预测值和实际值
            val_preds.append(outputs.numpy())
            val_targets.append(batch_y.numpy())

    # 计算验证集评分
    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_score = f.calculate_score(val_targets, val_preds, max_error)

    # 打印训练和验证结果
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, '
          f'Train Score: {train_score:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Score: {val_score:.4f}')



    # 保存最佳模型
    if val_score > best_score:
        best_score = val_score
        best_model_state = model.state_dict()
        print(f'New best model saved with score: {best_score:.4f}')

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
