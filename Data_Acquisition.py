import akshare as ak
import pandas as pd

# 获取所有A股上市公司的代码和名称
stock_info_a_code_name_df = ak.stock_info_a_code_name()

tmp_stock_df = stock_info_a_code_name_df

# 指定日期范围
start_date = "20000101"
end_date = "20250228"

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

