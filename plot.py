import pickle
import numpy as np
import torch

# 读取 pkl 文件
with open('./output/M_000066_step7.pkl', 'rb') as file:
    data = pickle.load(file)

train_loss_his = data['train_loss_his']

val_loss_his = data['val_loss_his']
date_his = np.array(data['date_his'])

pre_y_his = data['pre_y_his']
pre_y_his = np.array([tensor.detach().cpu().numpy() for tensor in pre_y_his])

time_spend = data['time_spend']
print(time_spend)

print(pre_y_his)

