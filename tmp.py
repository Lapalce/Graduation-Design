import function as f
data = f.load_pkl_file('./params/added_paramsList.pkl')

params_list = data['params_list']
for i in params_list: print(i)