import akshare as ak
import pandas as pd
from datetime import datetime
import time
import numpy as np
import os
import yfinance as yf

# 指定日期范围
start_date_str = str("20000101")
end_date_str = str("20250331")

start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
end_date = datetime.strptime(end_date_str, "%Y%m%d").date()


# # 获取所有A股上市公司的代码和名称
# stock_info_a_code_name_df = ak.stock_info_a_code_name()
#
# tmp_stock_df = stock_info_a_code_name_df[:5]
#
#
# # 循环获取每一家公司的股票数据并保存
# for index, row in tmp_stock_df.iterrows():
#     stock_code = row['code']
#     stock_name = row['name']
#     file_name = f"{stock_code}.csv"
#
#     try:
#         # 获取股票日线行情数据
#         stock_data = ak.stock_zh_a_hist(symbol=stock_code,
#                                         period="daily",
#                                         start_date=start_date_str,
#                                         end_date=end_date_str,
#                                         adjust='hfq')
#         # stock_data.to_csv(f"./Data/{file_name}", index=False)  # 保存为CSV文件
#         print(f"{index}-{stock_name}（{stock_code}）的数据已成功保存到 {file_name}")
#     except Exception as e:
#         print(f"获取 {index}-{stock_name}（{stock_code}）的数据时出现错误：{str(e)}")


# 任务3：获取主要指数历史数据
def get_main_index_history():
    # 上证指数
    sh_df = ak.stock_zh_index_daily(symbol="sh000001")
    sh_df = sh_df[(sh_df['date'] >= start_date) & (sh_df['date'] <= end_date)]
    sh_df.to_csv("./Data/index_data/sh000001.csv", index=False)
    print('上证指数获取成功')

    # 深证成指
    sz_df = ak.stock_zh_index_daily(symbol="sz399001")
    sz_df = sz_df[(sz_df['date'] >= start_date) & (sz_df['date'] <= end_date)]
    sz_df.to_csv("./Data/index_data/sz399001.csv", index=False)
    print("深圳成指获取成功")

    # 创业板指
    cy_df = ak.stock_zh_index_daily(symbol="sz399006")
    cy_df = cy_df[(cy_df['date'] >= start_date) & (cy_df['date'] <= end_date)]
    cy_df.to_csv("./Data/index_data/sz399006.csv", index=False)
    print("创业板指获取成功")


get_main_index_history()

market_df = ak.stock_a_all_pb()
market_df.to_csv('./Data/index_data/market_PE.csv', index=False)

a_stock_fund_flow = ak.stock_market_fund_flow()
a_stock_fund_flow.to_csv('./Data/index_data/market_flow.csv', index=False)



# 获取港股恒生指数历史日线数据
def get_hk_hsi_history():
    # 使用 yfinance 获取恒生指数历史日线数据
    hsi_history = yf.download("^HSI", start="2000-01-01", end="2025-03-25")
    hsi_history = hsi_history.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }
    )[["open", "high", "low", "close", "volume"]]
    hsi_history.reset_index(inplace=True)
    hsi_history.rename(columns={"Date": "date"}, inplace=True)
    return hsi_history


# 获取纳斯达克金龙指数历史日线数据
def get_nasdaq_kl_history():
    # 使用 yfinance 获取纳斯达克金龙指数历史日线数据
    nasdaq_kl_history = yf.download("KLCI.KL", start="2000-01-01", end="2025-03-25")
    nasdaq_kl_history = nasdaq_kl_history.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }
    )[["open", "high", "low", "close", "volume"]]
    nasdaq_kl_history.reset_index(inplace=True)
    nasdaq_kl_history.rename(columns={"Date": "date"}, inplace=True)
    return nasdaq_kl_history


# 保存数据到本地
def save_data_to_csv(data, filename):
    data.to_csv(filename, index=False)
    print(f"{filename}保存成功")


# 对“净额”数据进行log操作并改变列名
def log_transform_and_rename(df, columns):
    for col in columns:
        if col in df.columns:
            df[col + "-log"] = np.log(df[col])
    return df


output_dir = './Data/index_data/'

hsi_history = get_hk_hsi_history()
hsi_history = log_transform_and_rename(hsi_history, ["volume"])
save_data_to_csv(hsi_history, os.path.join(output_dir, "hk_hsi_history.csv"))

nasdaq_kl_history = get_nasdaq_kl_history()
nasdaq_kl_history = log_transform_and_rename(nasdaq_kl_history, ["volume"])
save_data_to_csv(nasdaq_kl_history, os.path.join(output_dir, "nasdaq_kl_history.csv"))
