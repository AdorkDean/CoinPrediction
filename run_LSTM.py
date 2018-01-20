# coding: utf-8
# Author: Ross

from util import generate_batch
from util import preprocessing_data
from model import LSTM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = 'data/bitcoin-20130428-20180113.csv'
RANDOM_SEED = 188


def main():
    np.random.seed(RANDOM_SEED)  # 方便结果重现
    data = pd.read_csv(DATA_PATH)
    scaler, data = preprocessing_data.process_normalization(data=data,
                                                            useful_colmns=['Open', 'High', 'Low', 'Close', 'Volume',
                                                                           'Market Cap', 'Weekday', 'Increase'])
    training_set, test_set = generate_batch.split_data(data)
    training_inputs, train_outputs = generate_batch.generate_batchs(training_set)
    test_inputs, test_outputs = generate_batch.generate_batchs(test_set)
    model = LSTM.build_model(inputs=training_inputs, output_size=1, LSTM_units=20)
    history = model.fit(training_inputs, train_outputs, batch_size=1, epochs=30, verbose=2, shuffle=True)
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(history.epoch, history.history['loss'])
    ax1.set_title('Training loss')
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()

    h = model.predict(test_inputs)
    dat = np.concatenate((test_outputs, h), axis=1)
    dat = dat * scaler.data_range_[-1] + scaler.data_min_[-1]
    df = pd.DataFrame(dat, columns=['actual', 'prediction'])
    df.to_csv('prediction.csv', index=False)


if __name__ == '__main__':
    main()
