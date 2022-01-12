# -*- coding: utf-8 -*-
"""lstm_stock.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vPsuI2rP9YWxuBFEPmZp-NMahDKmMQlc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
# 上傳csv檔案,在run此code前請先幫我載入助教提供的csv檔案(stock.csv)
uploaded = files.upload()

import io
import pandas as pd
data_df = pd.read_csv(io.BytesIO(uploaded['stock.csv']))
data_df

data_df['open'].plot()
plt.xlabel('Date')
plt.ylabel('Price')

#利用MinMaxScaler 最小最大值標準化
from sklearn.preprocessing import MinMaxScaler
data_rehsape = data_df['open'].values.reshape(-1,1).astype('float32')
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(data_rehsape)

#基於前三天的資料來預測下一天
look_back = 3
train_size = 250
#劃分成訓練與測試資料集
train, test = training_set_scaled[:-train_size], training_set_scaled[-train_size-look_back:]

print('train shape:', train.shape)
print('test shape:', test.shape)
train[1008][0]

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(dataset.shape[0]-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back-1, 0])
    return np.array(dataX), np.array(dataY)

#生成新樣本
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print('trainX:',trainX.shape)
print('trainY:',trainY.shape)

#放入LSTM的shape
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
print('trainX.shape:',trainX.shape)
print('testX.shape:',testX.shape)

trainY = np.reshape(trainY,(trainY.shape[0], 1))
testY = np.reshape(testY,(testY.shape[0], 1))
print('trainY.shape:',trainY.shape)
print('testY.shape:',testY.shape)

len_train = np.linspace(0, train.shape[0],train.shape[0]+1)
plt.plot(len_train[1:],sc.inverse_transform(train.reshape(-1,1)))
#plt.plot(len_test[1:],sc.inverse_transform(y_hat))
plt.xlabel('Date') # set a xlabel
plt.ylabel('Price')

len_test = np.linspace(train.shape[0], test.shape[0]+train.shape[0],test.shape[0]+1)
plt.plot(len_test[1:],sc.inverse_transform(test.reshape(-1,1)))
#plt.plot(len_test[1:],sc.inverse_transform(y_hat))
plt.xlabel('Date') # set a xlabel
plt.ylabel('Price')

"""Hw:create a lstm model to predict stock price"""

import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 建立及訓練 LSTM 模型
model = Sequential()
model.add(LSTM(3, input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()

#Fit model with history to check for overfitting
history = model.fit(trainX,trainY,epochs=100,validation_data=(testX,testY),shuffle=False,batch_size=5)

# save model
model.save_weights("model.hdf5")

#plot training process
fig = plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], color='#785ef0')
plt.plot(history.history['val_loss'], color='#dc267f')
plt.title('Model Loss Progress')
plt.ylabel('Brinary Cross-Entropy Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='upper right')
plt.show()

model.evaluate(testX,testY)

# 預測
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

testY = pd.DataFrame(testY)
testY

predictY = model.predict(testX)

import matplotlib.pyplot as plt 
plt.plot(testY, color = 'red', label = 'Ground truth')  # 紅線表示真實股價
plt.plot(predictY, color = 'blue', label = 'Predicted Price')  # 藍線表示預測股價
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

testY

from keras.models import load_model

model.save('model.h5')
