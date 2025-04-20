import pickle

import Model
import function as f

data = f.load_pkl_file('./params/added_paramsList.pkl')

params_list = data['params_list']
train_num = len(params_list)
print(f'共有{train_num}个训练')
print('---------------训练开始---------------')

model_dic = {
    'RegionBiLSTM': Model.RegionBiLSTM,
    'MSRegionBiLSTM': Model.MSRegionBiLSTM,
    'PSRegionBiLSTM': Model.PSRegionBiLSTM
}

model_type = {
    'RegionBiLSTM': 'R',
    'MSRegionBiLSTM': 'M',
    'PSRegionBiLSTM': 'P'
}

error_index = []
error_stock = []

for index, file in enumerate(params_list):
    try:
        params = f.load_params(file)

        stock_type = params['stock_type']
        model_name = params['model_name']
        stock = params['stock']
        seq_length = params['seq_length']
        step = params['step']
        start_date_str = params['start_date_str']
        end_date_str = params['end_date_str']
        batch_size = params['batch_size']
        hidden_size = params['hidden_size']
        output_size = params['output_size']
        sparse_layer_size = params['sparse_layer_size']
        dropout_rate = params['dropout_rate']
        num_epochs = params['num_epochs']

        if stock in error_stock:
            continue

        print(f"----------加载第({index + 1} / {train_num})训练----------"
              f"\n数据类型：{stock_type}; 股票{stock}；模型{model_name}; 往后预测天数{step}")

        model_save_path = f'./logger/{model_type[model_name]}_{stock}_step{step}.pth'
        his_save_path = f'./output/{model_type[model_name]}_{stock}_step{step}.pkl'

        X, y, date_seq = f.load_data(start_date_str, end_date_str, stock, seq_length, step)
        model = model_dic[model_name]
        f.train_model(X, y, date_seq, batch_size, model, hidden_size, output_size, sparse_layer_size, dropout_rate,
                      num_epochs, model_save_path, his_save_path)

        print(f'第({index + 1} / {train_num})训练完成')
    except Exception as e:
        print(f'第({index + 1} / {train_num})训练报错')
        print(e)
        error_index.append(index)
        error_stock.append(stock)

print(f"训练已全部完成")
if len(error_index) > 0:
    print("有报错的训练如下")
    for index, i in enumerate(error_index):
        print(i, ' ', error_stock[index])

    with open('./logger/error_index.pkl', 'wb') as f:
        pickle.dump({
            'error_index': error_index,
            'error_stock': error_stock
        }, f)

    print("报错index已保存到./logger/error_index.pkl")
else:
    print("没有产生报错")