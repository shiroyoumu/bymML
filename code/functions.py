from random import random
from typing import Tuple, List
import numpy as np
import pandas as pd
from numpy import ndarray
import random


def CollectTrainData(con, host) -> ndarray:
    '''

    从数据库收集数据

    :param con: 数据库控制器
    :param host: 收集数据的主机号
    :return: host对应的数据
    '''
    print("Loading data from DB...")
    dataset = pd.read_sql("select Mean from datasetDB where hostname in {}".format(str(tuple(host))), con)
    dataset = dataset.values.astype("float32")
    print("Done")
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
    result = data
    i = scope
    while i <= result[:, 0].shape[0] - scope:
        if (np.max(np.hstack((result[i - scope:i, 0], result[i + 1:i + scope + 1, 0]))) + d) < result[i, 0]:
            result[i, 0] -= d
        i += 1
    return result















