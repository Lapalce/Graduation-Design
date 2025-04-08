import akshare as ak
from datetime import datetime
import time
import numpy as np


def calculate_sma(df, n, column='close'):
    """
    计算N日简单移动平均（SMA）
    :param df: 包含股票数据的DataFrame
    :param n: 移动平均的天数
    :param column: 用于计算的列名（默认为'close'）
    :return: 包含SMA的Series
    """
    sma = df[column].rolling(window=n, min_periods=1).mean()
    return sma


def calculate_ema(df, n, column='close'):
    """
    计算N日指数移动平均（EMA）
    :param df: 包含股票数据的DataFrame
    :param n: 移动平均的天数
    :param column: 用于计算的列名（默认为'close'）
    :return: 包含EMA的Series
    """
    ema = df[column].ewm(span=n, adjust=False).mean()
    return ema


def calculate_wma(df, n, column='close'):
    """
    计算N日加权移动平均（WMA）
    :param df: 包含股票数据的DataFrame
    :param n: 移动平均的天数
    :param column: 用于计算的列名（默认为'close'）
    :return: 包含WMA的Series
    """
    weights = np.arange(1, n + 1)
    wma = df[column].rolling(window=n, min_periods=1).apply(
        lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)]),
        raw=False
    )
    return wma


def data_process(df):
    df['volume_log'] = np.where((df['volume'] > 0) & (~df['volume'].isna()), np.log10(df['volume']), 0)  # 量log
    df['volume_log_diff'] = df['volume_log'].diff().fillna(0)  # 量log差

    MA_list = [5, 10, 20, 30, 60]
    for i in MA_list:
        df[f'p_SMA:{i}'] = calculate_sma(df, i)  # 价格均线
        df[f'p_WMA:{i}'] = calculate_wma(df, i)
        df[f'p_EMA:{i}'] = calculate_ema(df, i)
        df[f'v_log_SMA:{i}'] = calculate_sma(df, i, column='volume_log')  # 成交量log均线
        df[f'v_log_WMA:{i}'] = calculate_wma(df, i, column='volume_log')
        df[f'v_log_EMA:{i}'] = calculate_ema(df, i, column='volume_log')
    return df


def get_hk_index(i, code, name):
    try:
        index_his = ak.stock_hk_index_daily_sina(symbol=code)
        index_his = data_process(index_his)
        index_his['code'] = code
        index_his.to_csv(f'./Data/hk_index_data/hk_{code}_his.csv', index=False)
        print(f"{i + 1} : {name} {code} 已保存成功")
    except Exception as e:
        print(f"获取 {i}-{name}（{code}）的数据时出现错误：{str(e)}")


# 指定日期范围
start_date_str = "19910101"
end_date_str = "20250331"

start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

# 获取所有A股主板上市公司的代码和名称
stock_info_a_code_name_df = ak.stock_info_a_code_name()
tmp_stock_df = stock_info_a_code_name_df

print('-------------开始获取主板股票数据--------------')

# 循环获取每一家公司的股票数据并保存
for index, row in tmp_stock_df.iterrows():
    stock_code = row['code']
    stock_name = row['name']
    file_name = f"{stock_code}.csv"

    try:
        # 获取股票日线行情数据
        stock_data = ak.stock_zh_a_hist(symbol=stock_code,
                                        period="daily",
                                        start_date=start_date_str,
                                        end_date=end_date_str,
                                        adjust='hfq')

        # 直接获取得到的信息是中文的，更换为英文
        stock_data = stock_data.rename(columns={
            '日期': 'date',
            '股票代码': 'stock_code',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        })

        stock_data = data_process(stock_data)
        stock_data['amount_log'] = np.where((stock_data['amount'] > 0) & (~stock_data['amount'].isna()), np.log10(stock_data['amount']), 0)

        if stock_code.startswith('60'):
            stock_data['stock_code'] = f'sh{stock_code}'
        else:
            stock_data['stock_code'] = f'sz{stock_code}'

        stock_data.to_csv(f"./Data/single_stock/{file_name}", index=False)  # 保存为CSV文件
        print(f"{index}-{stock_name}（{stock_code}）的数据已成功保存到 {file_name}")
    except Exception as e:
        print(f"获取 {index}-{stock_name}（{stock_code}）的数据时出现错误：{str(e)}")

    time.sleep(3)

# 获取科创板股票数据
print('-------------开始获取科创板股票数据--------------')
stock_zh_kcb_spot_df = ak.stock_zh_kcb_spot()
stock_zh_kcb_spot_df.to_csv('./Data/a_index_data/kcb_stock_info.csv', index=False)
tmp_df = stock_zh_kcb_spot_df

for index, row in tmp_df.iterrows():
    stock_code = row['代码']
    stock_name = row['名称']
    file_name = f"{stock_code}.csv"

    try:
        # 获取股票日线行情数据
        stock_data = ak.stock_zh_kcb_daily(symbol=stock_code, adjust="hfq")
        stock_data = data_process(stock_data)
        stock_data.to_csv(f"./Data/single_stock/{file_name}", index=False)  # 保存为CSV文件
        print(f"{index}-{stock_name}（{stock_code}）的数据已成功保存到 {file_name}")
    except Exception as e:
        print(f"获取 {index}-{stock_name}（{stock_code}）的数据时出现错误：{str(e)}")

    time.sleep(3)

# 获取A股主要指数历史数据
a_stock_path = './Data/a_index_data/'
a_index_list = ["sh000001", "sz399001", "sz399006", "sh000688", "sh000905", "sh000300", "sz399300", "bj899050"]
a_index_name = ["上证指数", "深圳成指", "创业板指", "科创50", "中证500", "沪深300", "深证100", "北证50"]

print('--------------开始获取主要指数数据----------------')
for index, code in enumerate(a_index_list):
    tmp_df = ak.stock_zh_index_daily(symbol=code)
    tmp_df = data_process(tmp_df)
    tmp_df = tmp_df[(tmp_df['date'] >= start_date) & (tmp_df['date'] <= end_date)]
    tmp_df.to_csv(f"./Data/a_index_data/{code}.csv", index=False)
    print(f'{index + 1} : {a_index_name[index]}获取成功')
    time.sleep(3)

# 获取A股PE数据
market_df = ak.stock_a_all_pb()
market_df.to_csv('./Data/a_index_data/market_PE.csv', index=False)
time.sleep(3)

a_stock_fund_flow = ak.stock_market_fund_flow()
a_stock_fund_flow.to_csv('./Data/a_index_data/market_flow.csv', index=False)
time.sleep(3)

# 获取全部港股指数信息
stock_hk_index_df = ak.stock_hk_index_spot_sina()
stock_hk_index_df.to_csv('./Data/code_name/hk_index_info.csv', index=False)

print('---------------开始获取港股指数数据--------------')
index: int
for index, row in stock_hk_index_df.iterrows():
    get_hk_index(index, row['代码'], row['名称'])
    time.sleep(3)


# 获取美股指数历史日线数据
def get_us_index(i, code, name):
    try:
        index_his = ak.index_us_stock_sina(symbol=code)
        index_his['code'] = code
        index_his = data_process(index_his)
        index_his['amount_log'] = np.where((index_his['amount'] > 0) & (~index_his['amount'].isna()), np.log10(index_his['amount']), 0)
        index_his.to_csv(f'./Data/us_index_data/{code}_his.csv', index=False)
        print(f"{i + 1} : {name} {code} 已保存成功")
    except Exception as e:
        print(f"获取 {i}-{name}（{code}）的数据时出现错误：{str(e)}")


us_index_code = [".IXIC", ".DJI", ".INX", ".NDX"]
us_index_name = ["纳斯达克综合指数", "道琼斯指数", "标普500", "纳斯达克100"]

print('--------------开始获取美股指数数据-------------')
for index, code in enumerate(us_index_code):
    get_us_index(index, code, us_index_name[index])
    time.sleep(3)

print('---------------数据已全部获取完成-----------------')