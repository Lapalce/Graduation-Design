import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RegionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)  # [batch_size, out_channels, sequence_length]
        x = F.relu(x)
        x = self.conv2(x)  # [batch_size, out_channels, sequence_length]
        return x


class PSRegionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, sparse_layer_size=128, dropout_rate=0.5):
        super(PSRegionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 3 个局部区域的卷积层，每个区域提取不同特征
        self.region1 = RegionCNN(input_size[0][0], input_size[1][0])  # 第一个区域输入12个特征，输出6个特征
        self.region2 = RegionCNN(input_size[0][1], input_size[1][1])  # 第二个区域输入6个特征，输出3个特征
        self.region3 = RegionCNN(input_size[0][2], input_size[1][2])  # 第三个区域输入6个特征，输出3个特征
        self.region4 = RegionCNN(input_size[0][3], input_size[1][3])  # 第三个区域输入6个特征，输出3个特征

        # BiLSTM层
        self.bilstm = nn.LSTM(input_size=sum(input_size[1]), hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

        # 稀疏层
        self.sparse_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, sparse_layer_size),  # 双向LSTM的输出大小是 2 * hidden_size
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 全连接层
        self.fc = nn.Linear(sparse_layer_size, output_size)

    def forward(self, x):
        # 第一组特征（前 12 个特征）
        x1 = x[:, :, :12]  # [batch_size, 30, 12]
        x1 = x1.permute(0, 2, 1)  # [batch_size, 12, 30]
        x1 = self.region1(x1)  # [batch_size, 6, 30]

        # 第二组特征（第 13 到第 18 个特征）
        x2 = x[:, :, 12:17]  # [batch_size, 30, 6]
        x2 = x2.permute(0, 2, 1)  # [batch_size, 6, 30]
        x2 = self.region2(x2)  # [batch_size, 3, 30]

        # 第三组特征（第 19 到第 24 个特征）
        x3 = x[:, :, 17:22]  # [batch_size, 30, 6]
        x3 = x3.permute(0, 2, 1)  # [batch_size, 6, 30]
        x3 = self.region3(x3)  # [batch_size, 3, 30]

        # 第四组特征（第 24 到第 30 个特征）
        x4 = x[:, :, 22:]  # [batch_size, 30, 6]
        x4 = x4.permute(0, 2, 1)  # [batch_size, 6, 30]
        x4 = self.region4(x4)  # [batch_size, 3, 30]

        # 合并所有局部区域的特征，沿着特征维度（第二维）拼接
        x = torch.cat([x1, x2, x3], dim=1)  # [batch_size, 30, 12]

        # 将输入调整为 [batch_size, 30, features] 以适应 LSTM
        x = x.permute(0, 2, 1)  # [batch_size, 30, 12] -> [batch_size, 30, features]

        # 双向 LSTM
        lstm_out, _ = self.bilstm(x)

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_size * 2]

        # 稀疏层
        sparse_output = self.sparse_layer(lstm_out)  # [batch_size, sparse_layer_size]

        # 全连接层
        out = self.fc(sparse_output)  # [batch_size, output_size]
        return out


class MSRegionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, sparse_layer_size=128, dropout_rate=0.5):
        super(MSRegionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 3 个局部区域的卷积层，每个区域提取不同特征
        self.region1 = RegionCNN(input_size[0][0], input_size[1][0])  # 第一个区域输入12个特征，输出6个特征
        self.region2 = RegionCNN(input_size[0][1], input_size[1][1])  # 第二个区域输入6个特征，输出3个特征
        self.region3 = RegionCNN(input_size[0][2], input_size[1][2])  # 第三个区域输入6个特征，输出3个特征
        self.region4 = RegionCNN(input_size[0][3], input_size[1][3])  # 第四个区域输入6个特征，输出3个特征

        # 稀疏层（放在卷积层之后）
        self.sparse_layer = nn.Sequential(
            nn.Linear(sum(input_size[1]), sparse_layer_size),  # 输入特征数是卷积层输出的特征总数
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # BiLSTM层
        self.bilstm = nn.LSTM(input_size=sparse_layer_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向LSTM的输出大小是 2 * hidden_size

    def forward(self, x):
        # 第一组特征（前 12 个特征）
        x1 = x[:, :, :12]  # [batch_size, 30, 12]
        x1 = x1.permute(0, 2, 1)  # [batch_size, 12, 30]
        x1 = self.region1(x1)  # [batch_size, 6, 30]
        # print(x1.size())

        # 第二组特征（第 13 到第 18 个特征）
        x2 = x[:, :, 12:17]  # [batch_size, 30, 6]
        x2 = x2.permute(0, 2, 1)  # [batch_size, 6, 30]
        x2 = self.region2(x2)  # [batch_size, 3, 30]
        # print(x2.size())

        # 第三组特征（第 19 到第 24 个特征）
        x3 = x[:, :, 17:22]  # [batch_size, 30, 6]
        x3 = x3.permute(0, 2, 1)  # [batch_size, 6, 30]
        x3 = self.region3(x3)  # [batch_size, 3, 30]
        # print(x3.size())

        # 第四组特征（第 24 到第 30 个特征）
        x4 = x[:, :, 22:]  # [batch_size, 30, 6]
        x4 = x4.permute(0, 2, 1)  # [batch_size, 6, 30]
        x4 = self.region4(x4)  # [batch_size, 3, 30]
        # print(x4.size())

        # 合并所有局部区域的特征，沿着特征维度（第二维）拼接
        x = torch.cat([x1, x2, x3, x4], dim=1)  # [batch_size, 18, 30]
        x = x.permute(0, 2, 1)

        # print(x.size())

        # 将合并后的特征通过稀疏层
        x = self.sparse_layer(x)  # [batch_size, 18, sparse_layer_size] (x的shape是 [batch_size, 18, sparse_layer_size])

        # LSTM输入：我们将特征维度与时间维度交换，使得输入维度变为 [batch_size, sequence_length, feature_size]
        x = x.permute(0, 2, 1)  # [batch_size, sparse_layer_size, 18] -> [batch_size, 18, sparse_layer_size]

        # 双向 LSTM
        lstm_out, _ = self.bilstm(x)  # [batch_size, 18, hidden_size * 2]

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_size * 2]

        # 全连接层
        out = self.fc(lstm_out)  # [batch_size, output_size]
        return out


class RegionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.5, sparse_layer_size=0):
        super(RegionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 3 个局部区域的卷积层，每个区域提取不同特征
        self.region1 = RegionCNN(input_size[0][0], input_size[1][0])  # 第一个区域输入12个特征，输出6个特征
        self.region2 = RegionCNN(input_size[0][1], input_size[1][1])  # 第二个区域输入6个特征，输出3个特征
        self.region3 = RegionCNN(input_size[0][2], input_size[1][2])  # 第三个区域输入6个特征，输出3个特征
        self.region4 = RegionCNN(input_size[0][3], input_size[1][3])  # 第三个区域输入6个特征，输出3个特征

        # BiLSTM层
        self.bilstm = nn.LSTM(input_size=sum(input_size[1]), hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向LSTM的输出大小是 2 * hidden_size

    def forward(self, x):
        # 第一组特征（前 12 个特征）
        x1 = x[:, :, :12]  # [batch_size, 30, 12]
        x1 = x1.permute(0, 2, 1)  # [batch_size, 12, 30]
        x1 = self.region1(x1)  # [batch_size, 6, 30]

        # 第二组特征（第 13 到第 18 个特征）
        x2 = x[:, :, 12:17]  # [batch_size, 30, 6]
        x2 = x2.permute(0, 2, 1)  # [batch_size, 6, 30]
        x2 = self.region2(x2)  # [batch_size, 3, 30]

        # 第三组特征（第 19 到第 24 个特征）
        x3 = x[:, :, 17:22]  # [batch_size, 30, 6]
        x3 = x3.permute(0, 2, 1)  # [batch_size, 6, 30]
        x3 = self.region3(x3)  # [batch_size, 3, 30]

        # 第四组特征（第 24 到第 30 个特征）
        x4 = x[:, :, 22:]  # [batch_size, 30, 6]
        x4 = x4.permute(0, 2, 1)  # [batch_size, 6, 30]
        x4 = self.region4(x4)  # [batch_size, 3, 30]

        # 合并所有局部区域的特征，沿着特征维度（第二维）拼接
        x = torch.cat([x1, x2, x3, x4], dim=1)  # [batch_size, 18, 30]（假设这里的特征数为 18）

        # 将合并后的特征通过 LSTM 层
        x = x.permute(0, 2, 1)  # [batch_size, 18, 30] -> [batch_size, 30, 18]

        # 双向 LSTM
        lstm_out, _ = self.bilstm(x)  # [batch_size, 30, hidden_size * 2]

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, hidden_size * 2]

        # 全连接层
        out = self.fc(lstm_out)  # [batch_size, output_size]
        return out
