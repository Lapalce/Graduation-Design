import akshare as ak
from datetime import datetime
import time
import numpy as np
import pandas as pd

path = './Data/code_name/hk_index_info.csv'
hk_df = pd.read_csv(path)
tmp_df = hk_df[['代码', '名称']]

original_columns = ['代码', '名称']
new_columns = ['code', 'name']

# 使用 rename 方法批量重命名列名
tmp_df = tmp_df.rename(columns={original_columns[i]: new_columns[i] for i in range(len(original_columns))})
tmp_df.to_csv(path)

path = './Data/code_name/kcb_stock_info.csv'
hk_df = pd.read_csv(path)
tmp_df = hk_df[['代码', '名称']]

original_columns = ['代码', '名称']
new_columns = ['code', 'name']

# 使用 rename 方法批量重命名列名
tmp_df = tmp_df.rename(columns={original_columns[i]: new_columns[i] for i in range(len(original_columns))})
tmp_df.to_csv(path)