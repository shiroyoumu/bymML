from random import random
from typing import Tuple, List
import numpy as np
import pandas as pd
from numpy import ndarray
import random
from enum import Enum


def CollectTrainData(con, host) -> pd.DataFrame:
    '''

    从数据库收集数据

    :param con: 数据库控制器
    :param host: 收集数据的主机号
    :return: host对应的数据
    '''
    print("Loading data from DB...")
    dataset = pd.read_sql("select Mean from datasetDB where hostname in {}".format(str(tuple(host))), con)
    dataset = dataset.values.astype("float32")
    # print("Done")
    return dataset


def CollectData(dataset, step: int) -> Tuple[ndarray, ndarray]:
    '''
    将数据序列dataset按step划分为输入和标签

    例如：

    >>> dataset = np.arange(6).reshape(6, 1)
    >>> step = 3
    >>> X, Y = CollectData(dataset, step)
    >>> X
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])
    >>> Y
    array([3, 4, 5])

    :param dataset:数据序列
    :param step:步长
    :return:输入列表，对应标签列表
    '''
    dataX, dataY = [], []
    for i in range(len(dataset) - step):
        dataX.append(dataset[i:(i + step), 0])
        dataY.append(dataset[i + step, 0])
    return np.array(dataX), np.array(dataY)


def CollectData2(dataset, step: int, length: int):
    hostNum = dataset.shape[0] / length # 总条目/每个主机条目=主机数
    if (hostNum * 10) % 10 != 0:
        raise ValueError("dataset条数与length条数不符")

    dataX, dataY = [], []
    for i in range(int(hostNum)):
        hostList = dataset[(i * length):((i + 1) * length), :]
        for j in range(len(hostList) - step):
            dataX.append(hostList[j:(j + step), 0])
            dataY.append(hostList[j + step, 0])
    return np.array(dataX), np.array(dataY)


def SelectHosts(hosts, rate: float, seed: int) -> Tuple[List, List]:
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


def SmoothSet(data, scope: int, d: float) -> ndarray:
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


def HandleNoise(data):

    return None


def HandleMissing(data):
    return None


class CompleteMethod(Enum):
    OneNum = 0  # 单数字填充
    Lerp = 1    # 线性插值
































