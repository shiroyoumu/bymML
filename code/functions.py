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
    dataset = pd.DataFrame()
    n = len(host)
    for i in host:
        df = pd.read_sql("select Mean from datasetDB where hostname='{}'".format(i), con)
        dataset = dataset.append(df, ignore_index=True)
        print("\rLoading Data From DB... {} of {}".format(host.index(i), n), end="")
    dataset = dataset.values.astype("float32")
    return dataset

def CollectData(dataset, step) -> Tuple[ndarray, ndarray]:
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
    :return:分割好的输入与标签
    '''
    dataX, dataY = [], []
    for i in range(len(dataset) - step):
        dataX.append(dataset[i:(i + step), 0])
        dataY.append(dataset[i + step, 0])
    return np.array(dataX), np.array(dataY)

def SelectHosts(hosts, rate, seed) -> Tuple[List, List]:
    random.seed(seed)
    random.shuffle(hosts)
    flag = int(len(hosts) * rate)
    host1 = hosts[:flag]
    host2 = hosts[flag:]
    return host1, host2