# coding: utf-8
# Author: Ross
import datetime
import pandas as pd
import numpy as np


def date2weekday(path=None, data: pd.DataFrame = None):
    '''
    将数据中日期转化为星期
    :param path:
    :param data:
    :return:
    '''
    if path is None and data is None:
        raise AssertionError('two parameters can not be blank at the same time.')
    if path:
        data = pd.read_csv(path)
    data['Date'] = data['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').weekday(), 1)
    data.rename(columns=lambda x: 'Weekday' if x == 'Date' else x, inplace=True)  # 修改索引名字
    return data


def normalize_data(path=None, data: pd.DataFrame = None, colmns: list = None):
    '''
    标准化数据
    :param path:
    :param data:
    :return:
    '''
    if path is None and data is None:
        raise AssertionError('two parameters can not be blank at the same time.')
    if path:
        data = pd.read_csv(path)

    if colmns:
        for colmn in colmns:
            data[colmns] = data[colmns].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    else:
        data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return data


def add_increase_col(path=None, data: pd.DataFrame = None):
    if path is None and data is None:
        raise AssertionError('two parameters can not be blank at the same time.')
    if path:
        data = pd.read_csv(path)

    data['Increase'] = data['Close'] - data['Open']
    return data


if __name__ == '__main__':
    add_increase_col(data=date2weekday('../data/bitcoin-20130428-20180113.csv'))
    # date2weekday('../data/bitcoin-20130428-20180113.csv')
