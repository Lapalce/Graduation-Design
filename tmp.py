import numpy as np

feature_groups = {
    'normal': ['open', 'close', 'high', 'low', 'amplitude', 'pct_change', 'change', 'turnover',
               'volume_log', 'volume_log_diff', 'amount', 'amount_log'],
    'price': ['p_WMA:5', 'p_WMA:10', 'p_WMA:20', 'p_WMA:30', 'p_WMA:60'],
    'volume': ['v_log_WMA:5', 'v_log_WMA:10', 'v_log_WMA:20', 'v_log_WMA:30', 'v_log_WMA:60'],
    'index': ['close_diff', 'sh', 'sz', 'cy', 'zz500', 'hs300', 'sz100']
}

input_size = [[], []]  # 特征数量
for key, value in feature_groups.items():
    input_size[0].append(len(value))
    input_size[1].append(len(value) * 2)

print(input_size)
print(sum(input_size[1]))

import function as f
import tensorflow

y1 = tensorflow(np.array([10, 20 ,30, 40, 50]))
y2 = tensorflow(np.array([15, 25, 35, 45 ,55]))

print(f.calculate_score(y1, y2))