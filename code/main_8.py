import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import sqlite3 as lite
from os.path import basename

# 文件声明
pathDataDB = "../data/dataset_db.db"    # 数据库文件

step = 7       # 预测步长
host = ['0001', '0021', '0070', '0143', '0354', '0372']   #
host2 = ['1046']    # 9783、9605、2636、1046、

def CollectTrainData(con, host) -> pd.DataFrame:
    dataset = pd.DataFrame()
    for i in host:
        df = pd.read_sql("select Mean from datasetDB where hostname='host{}'".format(i), con)
        dataset = dataset.append(df, ignore_index=True)
    dataset = dataset.values.astype("float32")
    return dataset

def CollectData(dataset, step):
    dataX, dataY = [], []
    for i in range(len(dataset) - 1 - step):
        dataX.append(dataset[i:(i + step), 0])
        dataY.append(dataset[i + step, 0])
    return np.array(dataX), np.array(dataY)

if __name__ == '__main__':
    # 定义随机种子，以便重现结果
    np.random.seed(3)
    # 加载数据
    con = lite.connect(pathDataDB)
    trainSet = CollectTrainData(con, host)
    testSet = CollectTrainData(con, host2)
    # 收集数据
    trainX, trainY = CollectData(trainSet, step)
    testX, testY = CollectData(testSet, step)
    # 整理格式
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    # 构建网络
    model = Sequential()
    model.add(Conv1D(4, 3, activation='relu', input_shape=(step, 1)))
    model.add(Conv1D(4, 3, activation='relu'))
    model.add(MaxPool1D(2))
    model.add(Flatten())

    model.add(Reshape((1, model.output_shape[1])))
    model.add(LSTM(168, return_sequences=True))
    model.add(LSTM(168, return_sequences=True))
    model.add(LSTM(168))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, epochs=500, batch_size=64, verbose=2)
    # 对测试数据的Y进行预测
    testPre = model.predict(testX)
    # 整理
    trainY = np.reshape(trainY, (1, trainY.shape[0]))
    testY = np.reshape(testY, (1, testY.shape[0]))
    # 计算RMSE误差
    testScore = math.sqrt(mean_squared_error(testY[0], testPre[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    plt.suptitle("host{} in {}".format(host2[0], basename(__file__)))
    plt.plot(testSet)
    empty = np.empty((step, 1))
    empty[:, :] = np.nan
    plt.plot(np.append(empty, testPre, axis=0))
    plt.show()

















