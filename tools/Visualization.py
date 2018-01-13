# coding: utf-8
# Author: Ross

import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime


def visualize(data: pd.DataFrame):
    '''
    数据可视化，将DateFrame数据可视化
    :param data:
    :return:
    '''
    start_year, end_year = get_year_range(data)
    ax = plt.subplot()
    ax.set_xticks([datetime.date(i, j, 1) for i in range(start_year, end_year) for j in range(1, 13)])
    ax.set_xticklabels(
        [datetime.date(i, j, 1).strftime('%Y-%m') for i in range(start_year, end_year) for j in range(1, 13)])
    ax.plot(data['Close'])
    plt.show()


def get_year_range(data: pd.DataFrame):
    '''

    :return: 起始时间到结束时间
    '''
    start_time = data['Date'].astype(datetime.datetime).min()
    end_time = data['Date'].astype(datetime.datetime).max()
    start_year = datetime.datetime.strptime(start_time, '%Y-%m-%d').year
    end_year = datetime.datetime.strptime(end_time, '%Y-%m-%d').year
    return (start_year, end_year)


if __name__ == '__main__':
    data = pd.read_csv('../data/bitcoin-20130428-20180113.csv')
    visualize(data)
