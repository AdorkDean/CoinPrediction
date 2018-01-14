# coding: utf-8
# Author: Ross
from keras.models import Sequential
from keras import layers


def build_model(inputs, output_size, LSTM_units, activ_func='linear', dropout=0.25, loss='mae', optimizer='adam'):
    '''
    初始模型
    建立简单的LSTM的时序模型，由一层 LSTM和一层全连接层完成
    :param inputs: 输入 [a, b, c]
    :param output_size: 输出维度
    :param LSTM_units: LSTM单元数
    :param activ_func: 激活函数
    :param dropout: dropout率
    :param loss: 损失函数
    :param optimizer: 优化方法
    :return:
    '''
    model = Sequential()
    model.add(layers.LSTM(LSTM_units, input_shape=(inputs.shape[1], inputs.shape[2]), dropout=dropout))
    model.add(layers.Dense(units=output_size))
    model.add(layers.Activation(activ_func))
    model.compile(optimizer=optimizer, loss=loss)
    return model


