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

account = 'P76121136'
password = "1136"


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

# 調用函數並將股票信息寫入 CSV 文件
stock_code = '2330'  # 股票代碼
start_date = '20100101'  # 起始日期
stop_date = '20231231'   # 結束日期
filename = 'stock_info.csv'  # CSV 文件名稱
write_stock_info_to_csv(stock_code, start_date, stop_date, filename)

# 读取 CSV 文件
df = pd.read_csv('stock_info.csv')



data = df[['Open','Close','High','Low']].values
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, :])
    return np.array(dataX), np.array(dataY)

# Parameters
time_step = 30
training_size = int(len(data_scaled) * 0.67)
test_size = len(data_scaled) - training_size
train_data, test_data = data_scaled[0:training_size,:], data_scaled[training_size:len(data_scaled),:]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input for the model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 4)

# Transformer Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# Model Definition
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
x = transformer_encoder(inputs, head_size=256, num_heads=4, ff_dim=4, dropout=0.1)
x = GlobalAveragePooling1D(data_format='channels_first')(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
outputs = Dense(4, activation="linear")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="mean_squared_error")

# Model Summary
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=64, verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Evaluate the model (Optional: Calculate RMSE or other metrics)
train_rmse = math.sqrt(mean_squared_error(y_train, scaler.inverse_transform(train_predict.reshape(-1, 4))))
test_rmse = math.sqrt(mean_squared_error(y_test, scaler.inverse_transform(test_predict.reshape(-1, 4))))

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# Plotting the results
# Adjust the time_step offset for plotting
trainPredictPlot = np.empty_like(data_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict)+time_step, :] = train_predict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(data_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(time_step*2)+1:len(data_scaled)-1, :] = test_predict

if os.path.exists('transformer_model2.keras'):
    os.remove('transformer_model2.keras')

model.save('transformer_model2.keras')

# Plot baseline and predictions
print(pd.to_datetime(df['Date']))
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(df['Date']), df['Open'], label='Actual Stock Open Price')
plt.plot(pd.to_datetime(df['Date']), trainPredictPlot[:,0], label='Train Stock Open Price')
plt.plot(pd.to_datetime(df['Date']), testPredictPlot[:,0], label='Test Stock Open Price')
plt.title('Stock Price Prediction using Transformer')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()




