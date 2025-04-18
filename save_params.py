import json
import pickle

path = './params/stocks.json'

span = [30, 60, 120]
step = [7, 30, 60]
batch_size = [32, 16, 8]
model_name = ['MSRegionBiLSTM', 'PSRegionBiLSTM', 'RegionBiLSTM']
model_l = ['M', 'P', 'R']
file_list = []

with open(path, 'r') as f:
    stocks_dic = json.load(f)

for key, value in stocks_dic.items():
    for stock in value:
        for i in range(3):
            for j in range(3):

                # 超参数字典
                params = {
                    'model_name': model_name[j],
                    'stock_type': key,
                    'stock': stock,
                    'seq_length': span[i],
                    'step': step[i],
                    'start_date_str': '20150101',
                    'end_date_str': '20250331',
                    'batch_size': batch_size[i],
                    'hidden_size': 64,
                    'output_size': 5,
                    'sparse_layer_size': 128,
                    'dropout_rate': 0.5,
                    'num_epochs': 120
                }

                filename = f'./params/{key}_{stock}_{model_l[j]}_{step[i]}.json'
                file_list.append(filename)

                with open(filename, 'w') as f:
                    json.dump(params, f, indent=4)
                print(f"超参数已保存到 {filename}")

with open('./params/paramsList.pkl', 'wb') as f:
    pickle.dump({
        'params_list': file_list
    }, f)

print("数据已保存到./params/paramsList.pkl")