import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import sqlite3 as lite

# 文件声明
pathDataDB = "../data/dataset_db.db"    # 数据库文件

step = 24       # 预测步长
host = ['0001', '0021', '0070', '0143', '0354', '0372']   #
host2 = ['0372']


def CollectTrainData(con, host):
    datasets = []
    for i in host:
        df = pd.read_sql("select Mean from datasetDB where hostname='host{}'".format(i), con)
        datasets.append(df.values.astype("float32"))
    return datasets

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
    trainSets = CollectTrainData(con, host)
    testSets = CollectTrainData(con, host2)
    # 收集数据
    trainX, trainY = [], []
    for i in trainSets:
        x, y = CollectData(i, step)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        trainX.append(x)
        trainY.append(y)
    testX, testY = [], []
    for j in testSets:
        x, y = CollectData(j, step)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        testX.append(x)
        testY.append(y)
    # 构建模型
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
    for i in range(len(trainX)):
        print("====== fitting host{} ======".format(host[i]))
        model.fit(trainX[i], trainY[0], epochs=500, batch_size=64, verbose=2)
    # 对测试数据的Y进行预测
    testPre = model.predict(testX)

    testScore = math.sqrt(mean_squared_error(testY[0], testPre[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    plt.plot(testSets[0])
    plt.plot(testPre)
    plt.show()



























