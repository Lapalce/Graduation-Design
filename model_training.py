import Model
import function as f

params_path = './params/params01.json'

params = f.load_params(params_path)

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

model_dic = {
    'RegionBiLSTM': Model.RegionBiLSTM,
    'MSRegionBiLSTM': Model.MSRegionBiLSTM,
    'PSRegionBiLSTM': Model.PSRegionBiLSTM
}

model_save_path = f'./logger/{model_name}_{stock}_step{step}.pth'
his_save_path = f'./output/{model_name}_{stock}_step{step}.pkl'

X, y = f.load_data(start_date_str, end_date_str, stock, seq_length, step)
model = model_dic[model_name]
f.train_model(X, y, batch_size, model, hidden_size, output_size, sparse_layer_size, dropout_rate,
              num_epochs, model_save_path, his_save_path)

print('训练完成')
