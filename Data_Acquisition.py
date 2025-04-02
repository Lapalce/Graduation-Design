import akshare as ak
import pandas as pd

# 获取所有A股上市公司的代码和名称
stock_info_a_code_name_df = ak.stock_info_a_code_name()

tmp_stock_df = stock_info_a_code_name_df

# 指定日期范围
start_date = "20000101"
end_date = "20250331"

# 循环获取每一家公司的股票数据并保存
for index, row in tmp_stock_df.iterrows():
    stock_code = row['code']
    stock_name = row['name']
    file_name = f"{stock_code}.csv"

    try:
        # 获取股票日线行情数据
        stock_data = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date)
        stock_data.to_csv(f"./Data/{file_name}", index=False)  # 保存为CSV文件
        print(f"{index}-{stock_name}（{stock_code}）的数据已成功保存到 {file_name}")
    except Exception as e:
        print(f"获取 {index}-{stock_name}（{stock_code}）的数据时出现错误：{str(e)}")


import akshare as ak
import pandas as pd

# 获取沪指数据
sh_index_data = ak.stock_zh_index_daily(symbol="sh000001")
sh_index_data = sh_index_data.rename(
    columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
)[["date", "open", "high", "low", "close", "volume"]]

# 获取深指数据
sz_index_data = ak.stock_zh_index_daily(symbol="sz399001")
sz_index_data = sz_index_data.rename(
    columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
)[["date", "open", "high", "low", "close", "volume"]]

# 获取创指数据
cy_index_data = ak.stock_zh_index_daily(symbol="sz399006")
cy_index_data = cy_index_data.rename(
    columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
)[["date", "open", "high", "low", "close", "volume"]]

# 打印数据
print("Shanghai Composite Index (SH000001) Data:")
print(sh_index_data.head())

print("\nShenzhen Component Index (SZ399001) Data:")
print(sz_index_data.head())

print("\nChiNext Index (SZ399006) Data:")
print(cy_index_data.head())
