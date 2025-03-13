import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class CNNBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CNNBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN层
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # BiLSTM层
        self.bilstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # 输入形状: (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, seq_length)

        # CNN
        x = self.cnn(x)  # 输出形状: (batch_size, 64, seq_length // 2)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_length // 2, 64)

        # BiLSTM
        h_lstm, _ = self.bilstm(x)  # 输出形状: (batch_size, seq_length // 2, hidden_size * 2)

        # 取最后一个时间步的输出
        out = self.fc(h_lstm[:, -1, :])  # 输出形状: (batch_size, output_size)
        return out


class GNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GNN层
        self.gcn1 = GCNConv(input_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)

        # LSTM层
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        # GNN
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        x = torch.relu(x)

        # LSTM
        h_lstm, _ = self.lstm(x)

        # 全连接层
        out = self.fc(h_lstm[:, -1, :])
        return out


class SPos_CNNBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, sparse_layer_size=128, dropout_rate=0.5):
        super(CNNBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN层
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        # BiLSTM层
        self.bilstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

        # 稀疏层
        self.sparse_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, sparse_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 全连接层
        self.fc = nn.Linear(sparse_layer_size, output_size)

    def forward(self, x):
        # 输入形状: (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, seq_length)

        # CNN
        x = self.cnn(x)  # 输出形状: (batch_size, 64, seq_length // 2)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_length // 2, 64)

        # BiLSTM
        h_lstm, _ = self.bilstm(x)  # 输出形状: (batch_size, seq_length // 2, hidden_size * 2)

        # 取最后一个时间步的输出
        h_lstm_last = h_lstm[:, -1, :]  # 输出形状: (batch_size, hidden_size * 2)

        # 稀疏层
        sparse_output = self.sparse_layer(h_lstm_last)  # 输出形状: (batch_size, sparse_layer_size)

        # 全连接层
        out = self.fc(sparse_output)  # 输出形状: (batch_size, output_size)
        return out


class SPre_CNNBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, sparse_layer_size=128, dropout_rate=0.5):
        super(CNNBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 稀疏层（放在最前面）
        self.sparse_layer = nn.Sequential(
            nn.Linear(input_size, sparse_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # CNN层
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=sparse_layer_size, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))

        # BiLSTM层
        self.bilstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # 输入形状: (batch_size, seq_length, input_size)
        batch_size, seq_length, input_size = x.shape

        # 稀疏层（对每个时间步的特征进行稀疏化）
        x = x.view(-1, input_size)  # 转换为 (batch_size * seq_length, input_size)
        x = self.sparse_layer(x)  # 输出形状: (batch_size * seq_length, sparse_layer_size)
        x = x.view(batch_size, seq_length, -1)  # 转换回 (batch_size, seq_length, sparse_layer_size)

        # 转换为 (batch_size, sparse_layer_size, seq_length)
        x = x.permute(0, 2, 1)

        # CNN
        x = self.cnn(x)  # 输出形状: (batch_size, 64, seq_length // 2)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, seq_length // 2, 64)

        # BiLSTM
        h_lstm, _ = self.bilstm(x)  # 输出形状: (batch_size, seq_length // 2, hidden_size * 2)

        # 取最后一个时间步的输出
        out = self.fc(h_lstm[:, -1, :])  # 输出形状: (batch_size, output_size)
        return out