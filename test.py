import function as f

data = f.load_pkl_file('./logger/error_index.pkl')

error_index = data['error_index']

data = f.load_pkl_file('./params/added_paramsList.pkl')

params_list = data['params_list']
error_params = [params_list[x] for x in error_index]
for x in error_params: print(x)
