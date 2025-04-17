import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import Model
import function as f

torch.manual_seed(12321)

start_date_str = "20150101"
end_date_str = "20250331"

start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

seq_length = 30  # 训练时间跨度
step = 7  # 预测时间跨度

# 读取数据
path = 'D:/GitHub/Graduation-Design/Data/'
stock = 'single_stock/000001.csv'
file = f'{path}{stock}'
data = pd.read_csv(file)

# 合并指数数据
a_index_list = ['sh000001', 'sz399001', 'sz399006', 'sh000905', 'sh000300', 'sz399300']
a_index_name = ['sh', 'sz', 'cy', 'zz500', 'hs300', 'sz100']

for index, code in enumerate(a_index_list):
    tmp_df = pd.read_csv(f'{path}a_index_data/{code}.csv')
    tmp_df = tmp_df.rename(columns={'close': a_index_name[index]})
    data = pd.merge(data, tmp_df[['date', a_index_name[index]]], on='date', how='left')

data['date'] = pd.to_datetime(data['date']).dt.date
data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

data.set_index('date', inplace=True)

# 选择需要的特征
features = ['open', 'close', 'high', 'low', 'amplitude', 'pct_change', 'change', 'turnover',
            'volume_log', 'volume_log_diff', 'amount', 'amount_log'
            , 'close', 'p_SMA:5', 'p_SMA:10', 'p_SMA:20', 'p_SMA:30', 'p_SMA:60'
            , 'volume_log', 'v_log_SMA:5', 'v_log_SMA:10', 'v_log_SMA:20', 'v_log_SMA:30', 'v_log_SMA:60'
            #            , 'p_WMA:5', 'p_WMA:10', 'p_WMA:20', 'p_WMA:30', 'p_WMA:60'
            #            , 'v_log_WMA:5', 'v_log_WMA:10', 'v_log_WMA:20', 'v_log_WMA:30', 'v_log_WMA:60'
            #            , 'p_EMA:5', 'p_EMA:10', 'p_EMA:20', 'p_EMA:30', 'p_EMA:60'
            #            , 'v_log_EMA:5', 'v_log_EMA:10', 'v_log_EMA:20', 'v_log_EMA:30', 'v_log_EMA:60'
            , 'close', 'sh', 'sz', 'cy', 'zz500', 'hs300', 'sz100'
            ]
data = data[features]

# 数据归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X_seq, y_seq = f.create_sequences(data_scaled, seq_length, step)

# 将数据转换为PyTorch张量
X = torch.tensor(X_seq, dtype=torch.float32)
y = torch.tensor(y_seq, dtype=torch.float32)

# 划分训练集和验证集
train_size = int(0.8 * len(X))
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 创建DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

input_size = [[12, 6, 6, 7], [24, 12, 12, 14]]  # 特征数量
hidden_size = 64
output_size = 5  # 输出特征数量
sparse_layer_size = 128  # 稀疏层的大小
dropout_rate = 0.5  # Dropout率

model = Model.PS_RegionBiLSTM(input_size, hidden_size, output_size, sparse_layer_size=sparse_layer_size,
                              dropout_rate=dropout_rate)

# 定义损失函数和优化器
criterion = f.calculate_score
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_X)
        loss = criterion(batch_y, outputs)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(batch_y, outputs)
            val_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
