import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import pandas as pd
import sqlite3 as lite

# 文件声明
pathDataDB = "./data/dataset_db.db"    # 数据库文件

rate = 0.67     # 数据分割比例
step = 168      # 预测步长， 168 -> 1


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
    df = pd.read_sql("select Mean from datasetDB where hostname='host0001'", con)
    dataset = df.values.astype("float32")
    # 分割
    trainSize = int(len(dataset) * rate)
    testSize = len(dataset) - trainSize
    train = dataset[0:trainSize, :]
    test = dataset[trainSize:len(dataset), :]
    # 收集数据
    trainX, trainY = CollectData(train, step)
    testX, testY = CollectData(test, step)
    # 整理成LSTM需要的格式
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # 构建LSTM
    model = Sequential()
    model.add(LSTM(20, input_shape=(1, step)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=50, verbose=2)
    # 对训练数据的Y进行预测
    trainPredict = model.predict(trainX)
    print(model.output_shape)
    # 对测试数据的Y进行预测
    testPredict = model.predict(testX)
    # 整理
    trainY = np.reshape(trainY, (1, trainY.shape[0]))
    testY = np.reshape(testY, (1, testY.shape[0]))
    # 计算RMSE误差
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # 构造一个和dataset格式相同的数组，共1680行，dataset为总数据集，把预测的1125行训练数据存进去
    trainPredictPlot = np.empty_like(dataset)
    # 用nan填充数组
    trainPredictPlot[:, :] = np.nan
    # 将训练集预测的Y添加进数组，从第3位到第1125+3位，共1128行
    trainPredictPlot[step:len(trainPredict) + step, :] = trainPredict

    # 构造一个和dataset格式相同的数组，共1680行，把预测的后44行测试数据数据放进去
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    # 将测试集预测的Y添加进数组，从第1125+4位到最后，共551行
    testPredictPlot[len(trainPredict) + (step * 2) + 1:len(dataset) - 1, :] = testPredict

    # 画图
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()






















