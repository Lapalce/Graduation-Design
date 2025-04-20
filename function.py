import math

import numpy as np
import pandas as pd
import json
from datetime import datetime
import pickle
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler

torch.manual_seed(12321)
feature_groups = {
    'normal': ['open', 'close', 'high', 'low', 'amplitude', 'pct_change', 'change', 'turnover',
               'volume_log', 'volume_log_diff', 'amount', 'amount_log'],
    'price': ['p_WMA:5', 'p_WMA:10', 'p_WMA:20', 'p_WMA:30', 'p_WMA:60'],
    'volume': ['v_log_WMA:5', 'v_log_WMA:10', 'v_log_WMA:20', 'v_log_WMA:30', 'v_log_WMA:60'],
    'index': ['close_diff', 'sh', 'sz', 'cy', 'zz500', 'hs300', 'sz100']
}
stock_path = 'D:/GitHub/Graduation-Design/Data'


# , 'p_SMA:5', 'p_SMA:10', 'p_SMA:20', 'p_SMA:30', 'p_SMA:60'
# , 'v_log_SMA:5', 'v_log_SMA:10', 'v_log_SMA:20', 'v_log_SMA:30', 'v_log_SMA:60'
# , 'p_WMA:5', 'p_WMA:10', 'p_WMA:20', 'p_WMA:30', 'p_WMA:60'
# , 'v_log_WMA:5', 'v_log_WMA:10', 'v_log_WMA:20', 'v_log_WMA:30', 'v_log_WMA:60'
# , 'p_EMA:5', 'p_EMA:10', 'p_EMA:20', 'p_EMA:30', 'p_EMA:60'
# , 'v_log_EMA:5', 'v_log_EMA:10', 'v_log_EMA:20', 'v_log_EMA:30', 'v_log_EMA:60'
# , 'close_diff', 'sh', 'sz', 'cy', 'zz500', 'hs300', 'sz100'


def load_params(filename):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params


def load_pkl_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def calculate_score(y_true, y_pred):
    # print(y_true)
    loss = abs((y_pred - y_true) / y_true)

    # 定义每个特征的权重
    weights = torch.tensor([0.11, 0.60, 0.11, 0.11, 0.17])  # 增加close的权重

    # 使用加权平方损失
    weighted_loss = (loss ** 2) * weights  # 通过平方加大惩罚力度
    total_loss = weighted_loss.sum()

    return total_loss


def create_sequences(data, seq_length, step, date):
    xs, ys, ds = [], [], []
    for i in range(0, len(data) - seq_length - step, seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length + step][[0, 1, 2, 3, 8]]
        d = date[i + seq_length + step]
        xs.append(x)
        ys.append(y)
        ds.append(d)
    return np.array(xs), np.array(ys), np.array(ds)


def load_data(start_date_str, end_date_str, stock, seq_length, step):
    print('开始加载数据')
    start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
    end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

    # 读取数据
    file = f'{stock_path}/single_stock/{stock}.csv'
    data = pd.read_csv(file)

    # 合并指数数据
    a_index_list = ['sh000001', 'sz399001', 'sz399006', 'sh000905', 'sh000300', 'sz399300']
    a_index_name = ['sh', 'sz', 'cy', 'zz500', 'hs300', 'sz100']

    for index, code in enumerate(a_index_list):
        tmp_df = pd.read_csv(f'{stock_path}/a_index_data/{code}.csv')
        tmp_df = tmp_df.rename(columns={'close': a_index_name[index]})
        data = pd.merge(data, tmp_df[['date', a_index_name[index]]], on='date', how='left')

    # 转换部分列的值，作偏离率
    for col in feature_groups['price']:
        data[col] = (data[col] - data['close']) / data['close']

    for col in feature_groups['volume']:
        data[col] = (data[col] - data['volume_log']) / data['volume_log']

    data['close_diff'] = data['close']
    for col in feature_groups['index']:
        data[col] = (data[col] - data[col].shift(1)) / data[col].shift(1)
        # data[col].iloc[0] = 0

    # 将index设置成日期
    data['date'] = pd.to_datetime(data['date']).dt.date
    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
    date_list = data['date'].tolist()
    date_list = [date.strftime('%Y-%m-%d') for date in date_list]
    data.set_index('date', inplace=True)

    # 选择需要的特征
    features = []
    for key, value in feature_groups.items():
        features.extend(value)
    data = data[features]

    # 数据归一化
    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)

    X_seq, y_seq, date_seq = create_sequences(data_scaled, seq_length, step, date_list)

    # 将数据转换为PyTorch张量
    X = torch.tensor(X_seq, dtype=torch.float32)
    y = torch.tensor(y_seq, dtype=torch.float32)
    return X, y, date_seq


def train_model(X, y, date_seq, batch_size, model, hidden_size, output_size, sparse_layer_size, dropout_rate
                , num_epochs, model_save_path, his_save_path):
    # 划分训练集和验证集
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    _, date_val = date_seq[:train_size], date_seq[train_size:]

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    date_loader = DataLoader(date_val, batch_size=batch_size, shuffle=False)

    input_size = [[], []]  # 特征数量
    for key, value in feature_groups.items():
        input_size[0].append(len(value))
        input_size[1].append(len(value) * 2)

    model = model(input_size, hidden_size, output_size, sparse_layer_size=sparse_layer_size, dropout_rate=dropout_rate)
    # 定义损失函数和优化器
    criterion = calculate_score
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 储存输出数据
    train_loss_his = []
    val_loss_his = []
    date_his = []
    pre_y_his = []

    print('开始模型训练')
    start_time = time.time()
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
            if math.isnan(loss.item()):
                raise ValueError("loss is nan")
            elif math.isinf(loss.item()):
                raise ValueError("loss is inf")

            train_loss += loss.item()

        # 验证模型
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (batch_X, batch_y), date_batch in zip(val_loader, date_loader):
                outputs = model(batch_X)
                pre_y_his.append(outputs)
                date_his.append(date_batch)
                loss = criterion(batch_y, outputs)
                if math.isnan(loss.item()):
                    raise ValueError("loss is nan")
                elif math.isinf(loss.item()):
                    raise ValueError("loss is inf")
                val_loss += loss.item()

        train_loss_his.append(train_loss)
        val_loss_his.append(val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    end_time = time.time()
    torch.save(model.state_dict(), model_save_path)
    print(f"模型的权重已保存到 {model_save_path}")

    with open(his_save_path, 'wb') as f:
        pickle.dump({
            'train_loss_his': train_loss_his,
            'val_loss_his': val_loss_his,
            'date_his': date_his,
            'pre_y_his': pre_y_his,
            'time_spend': end_time - start_time
        }, f)

    print(f"数据已保存到{his_save_path}")
