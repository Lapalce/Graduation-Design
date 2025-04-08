import os
import pandas as pd
from datetime import datetime


# 定义一个函数，用于筛选股票数据
def filter_stocks_by_date(directory, start_date):
    # 将字符串日期转换为 datetime 对象
    start_date = datetime.strptime(start_date, "%Y%m%d")

    # 存储满足条件的股票文件名
    valid_stocks = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 构建文件路径
        file_path = os.path.join(directory, filename)

        # 确保是文件
        if os.path.isfile(file_path):
            try:
                # 读取 CSV 文件
                df = pd.read_csv(file_path)

                # 将日期列转换为 datetime 格式
                df['date'] = pd.to_datetime(df['date'])

                # 筛选日期大于等于 start_date 的数据
                filtered_df = df[df['date'] >= start_date]

                # 如果筛选后的数据不为空，说明该股票的最早时间在 start_date 之后
                if not filtered_df.empty:
                    valid_stocks.append(filename)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return valid_stocks


# 示例：筛选股票数据
directory = "./stock_data"  # 存放股票数据的文件夹路径
start_date = "20150101"  # 筛选的起始日期

# 调用函数并打印结果
valid_stocks = filter_stocks_by_date(directory, start_date)
print("Valid stocks:", valid_stocks)