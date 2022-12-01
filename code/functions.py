from random import random
from typing import Tuple, List
import numpy as np
import pandas as pd
import random
from enum import Enum
from sklearn.ensemble import IsolationForest

class CompleteMethod(Enum):
    OneNum = 0  # 单数字填充
    Lerp = 1  # 线性插值

class Completion:
    @staticmethod
    def DoOneNum(data: np.ndarray, num: float) -> np.ndarray:
        '''
        替换空值为指定数字

        :param data: 数据序列
        :param num: 替换成的数
        :return: 结果（时间，主机）
        '''
        data[np.where(np.isnan(data))] = num
        return data

    @staticmethod
    def DoLerp(data: np.ndarray) -> np.ndarray:
        '''
        将空值使用线性插值填充

        :param data: 数据序列
        :return: 结果（时间，主机）
        '''
        temp = pd.DataFrame(data)
        temp = temp.interpolate(limit_direction="both")
        data = temp.to_numpy()
        return data


def CollectTrainData(con, host: list, length: int) -> np.ndarray:
    '''
    从数据库收集数据

    :param con: 数据库控制器
    :param host: 主机列表
    :param length: 每台主机数据条目
    :return: 数据矩阵（时间，主机）
    '''
    print("Loading data from DB...")
    dataset = pd.read_sql("select Mean from datasetDB_new where hostname in {}".format(str(tuple(host))), con)
    dataset = dataset.values.astype("float32")
    hostNum = dataset.shape[0] / length  # 总条目/每个主机条目=主机数
    if (hostNum * 10) % 10 != 0:
        raise ValueError("dataset条数与length条数不符")

    data = []
    for i in range(int(hostNum)):
        temp = dataset[(i * length):((i + 1) * length), :]
        data.append(temp)
    data = np.concatenate(data, axis=1)

    print("Done")
    return data


def CollectData(data: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    将数据序列dataset按step划分为输入和标签

    例如：

    >>> dataset = np.arange(6).reshape(6, 1)
    >>> step = 3
    >>> X, Y = CollectData(data, step)
    >>> X
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])
    >>> Y
    array([3, 4, 5])

    :param data: 数据序列（时间，）
    :param step: 步长
    :return: 输入列表，对应标签列表
    '''
    dataX, dataY = [], []
    for i in range(len(data) - step):
        dataX.append(data[i:(i + step)])
        dataY.append(data[i + step])
    return np.array(dataX), np.array(dataY)


def CollectAllData(dataset: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    将数据矩阵中数据按主机划分为输入和标签

    :param dataset: 数据序列（时间，主机）
    :param step: 步长
    :return: 输入列表（时间，步长），对应标签列表（时间，）
    '''
    x, y = [], []
    for i in range(dataset.shape[1]):
        temp = dataset[:, i]
        tempx, tempy = CollectData(temp, step)
        x.append(tempx)
        y.append(tempy)
    dataX = np.concatenate(x)
    dataY = np.concatenate(y)
    return dataX, dataY


def SelectHosts(hosts: list, rate: float, seed: int) -> Tuple[List, List]:
    '''
    选取训练和测试的主机

    :param hosts: 全部主机列表
    :param rate: 分割比例（作为训练集）
    :param seed: 随机种子
    :return: 训练主机列表，测试主机列表
    '''
    random.seed(seed)
    random.shuffle(hosts)
    flag = int(len(hosts) * rate)
    host1 = hosts[:flag]
    host2 = hosts[flag:]
    return host1, host2


def SmoothSet(data, scope: int, d: float) -> np.ndarray:
    '''
    降低部分峰值

    对于序列内的某一个数据值，如果在[-scope, +scope]去心邻域内的最大值都要比它大d，则把该数据值设置为去心邻域内的最大值。

    :param data: 数据序列
    :param scope: 采样范围
    :param d: 差值
    :return: 处理后序列
    '''
    result = data
    i = scope
    while i <= result[:, 0].shape[0] - scope:
        if (np.max(np.hstack((result[i - scope:i, 0], result[i + 1:i + scope + 1, 0]))) + d) < result[i, 0]:
            result[i, 0] -= d
        i += 1
    return result


def HandleNoise(data: np.ndarray, estimators: int, samples: int, errorRate: float = 0.01) -> np.ndarray:
    '''
    异常值处理

    :param data: 数据序列
    :param estimators: 孤立森林分类次数
    :param samples: 每次分类样本数
    :param errorRate: 异常率
    :return: 处理结果（异常值为NaN）（时间，主机）
    '''
    result = []
    hostNum = data.shape[1]
    for i in range(hostNum):
        temp = data[:, i]
        temp = pd.DataFrame(temp)
        temp = temp.fillna(-100.0)  # 将nan设置为-100
        model = IsolationForest(n_estimators=estimators, max_samples=samples, contamination=float(errorRate))
        model.fit(temp)
        tempPre = model.predict(temp)  # 制作mask
        temp = temp.mask(tempPre.reshape(tempPre.shape[0], 1) < 0)  # 将异常值设为nan
        temp = temp.replace([-100], [np.nan])  # 将未采样到的数据替换掉
        result.append(temp)
        print("{} / {}".format(i + 1, hostNum))
    result = np.concatenate(result, axis=1)
    return result


def HandleMissing(data: np.ndarray, completeMethod: CompleteMethod, *args):
    '''
    补全缺失数据

    :param data: 数据序列
    :param completeMethod: 补全方法
    :param args: 额外参数
    :return: 补全后结果
    '''
    result = None
    if completeMethod == CompleteMethod.OneNum:
        result = Completion.DoOneNum(data, args[0])
    if completeMethod == CompleteMethod.Lerp:
        result = Completion.DoLerp(data)
    return result




















