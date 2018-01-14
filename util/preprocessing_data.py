# coding: utf-8
# Author: Ross
import datetime
import pandas as pd


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

if __name__ == '__main__':
    date2weekday('../data/bitcoin-20130428-20180113.csv')
