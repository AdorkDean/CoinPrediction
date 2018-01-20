# coding: utf-8
# Author: Ross
import pandas as pd
import numpy as np


def generate_batchs(data: np.array, window_len=10):
    '''
    每window_len天的输入作为一个输入
    :param data:
    :param window_len:
    :return:
    '''
    input_tmp = list()
    output_tmp = list()
    # 每一个数据为[window_len, features]
    for i in range(len(data) - window_len - 1):
        input_tmp.append(data[i:(i + window_len), :-1])

        # 将output_colmns 作为输出结果
        output_tmp.append(np.array(data[i + window_len + 1, -1:]))
    inputs = np.array(input_tmp)
    outputs = np.array(output_tmp)
    return inputs, outputs


def split_data(data: np.array):
    '''
    将数据分成2/3训练集和测试集1/3
    :param data:
    :return:
    '''
    split_index = np.floor(len(data) * float(2 / 3)).astype(int)
    train_set = data[:split_index]
    test_set = data[split_index:]
    generate_batchs(test_set)
    return train_set, test_set


if __name__ == '__main__':
    split_data(pd.read_csv('../data/bitcoin-20130428-20180113.csv'))
