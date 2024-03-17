import numpy as np
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

from numpy.random import randn

def Get_Stock_Informations(stock_code, start_date, stop_date):
    information_url = ('http://140.116.86.242:8081/stock/' +
                       'api/v1/api_get_stock_info_from_date_json/' +
                       str(stock_code) + '/' +
                       str(start_date) + '/' +
                       str(stop_date)
                       )
    result = requests.get(information_url).json()
    if(result['result'] == 'success'):
        print(len(result['data']))
        filtered_data = [{'Date':datetime.datetime.fromtimestamp(entry['date']).date(),'Open': entry['open'], 'Close': entry['close'] , 'High': entry['high'] , 'Low': entry['low']} for entry in result['data']]
        return filtered_data[::-1]
    return dict([])


def Buy_Stock(account, password, stock_code, stock_shares, stock_price):
    print('Buying stock...')
    data = {'account': account,
            'password': password,
            'stock_code': stock_code,
            'stock_shares': stock_shares,
            'stock_price': stock_price}
    buy_url = 'http://140.116.86.242:8081/stock/api/v1/buy'
    result = requests.post(buy_url, data=data).json()
    print('Result: ' + result['result'] + "\nStatus: " + result['status'])
    return result['result'] == 'success'


def write_stock_info_to_csv(stock_code, start_date, stop_date, filename):
    stock_info = Get_Stock_Informations(stock_code, start_date, stop_date)
    if stock_info:
        df = pd.DataFrame(stock_info)
        df.to_csv(filename, index=False)
        print(f"股票代碼 {stock_code} 的信息已成功寫入到 {filename} 中。")
    else:
        print("未找到相應的股票信息，無法寫入 CSV 文件。")


stock_code = '2330'  # 股票代碼
start_date = '20240101'  # 起始日期
stop_date = '20240229'   # 結束日期
filename = 'stock_prediction_info.csv'  # CSV 文件名稱
write_stock_info_to_csv(stock_code, start_date, stop_date, filename)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, :])
    return np.array(dataX), np.array(dataY)


model = load_model('transformer_model2.keras')  

df = pd.read_csv('stock_prediction_info.csv')

data = df[['Open','Close','High','Low']].values

print(data.size)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled_predict = scaler.fit_transform(data)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, :])
    return np.array(dataX), np.array(dataY)

# Parameters
time_steps = 30
last_30_days = data_scaled_predict[-time_steps:, :]
next_day_prediction = model.predict(last_30_days.reshape(1 ,30, 4))

next_day_price = scaler.inverse_transform(next_day_prediction)
print("預測的下一天股票價格：", next_day_price)


