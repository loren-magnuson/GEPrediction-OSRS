from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


TRAIN_SPLIT = 120
matplotlib.rcParams['figure.figsize'] = (8, 6)
matplotlib.rcParams['axes.grid'] = False
tf.random.set_seed(13)


def baseline(history):
    """Show baseline prediction based on average ?"""
    return np.mean(history)


def create_time_steps(length, step=1):
    """Step from -length to 0

    :param length: start of range
    :keyword step: step size
    :return: list of steps
    """
    return [i for i in range(-length, 0, step)]


def load_univariate_data(dataset, start_index, end_index, history_size, target_size):
    """

    :param dataset:
    :param start_index:
    :param end_index:
    :param history_size:
    :param target_size:
    :return: tuple of numpy arrays, (data, labels)
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    else:
        return np.array(data), np.array(labels)


def get_univariate_timeseries_from_df(df, variable, index):
    """Get univariate timeseries from a dataframe

    :param df: dataframe
    :param variable: 'Chaos_rune'
    :param index: index for the timeseries
    """

    # Select the target
    uni_data = df[variable]

    # Set the index of the timeseries
    uni_data.index = df[index]

    # ?
    uni_data = uni_data.values

    # ?
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()

    # ?
    uni_train_std = uni_data[:TRAIN_SPLIT].std()

    # ?
    uni_data = (uni_data - uni_train_mean) / uni_train_std

    return uni_data


def show_plot(plot_data, delta, title):
    """Use pyplot to display a plot

    :param plot_data:
    :param delta:
    :param title:
    :return:
    """
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


class GEPredictor:

    def __init__(self, input_file):
        """Example predictor class for GE prices"""
        self.input = self.set_input(input_file)

    def set_input(self, input_file):
        """Load csv into a dataframe to use as input

        :param input_file: csv file to read
        """
        self.input = pd.read_csv(input_file)
        return self.input

    def run(self):
        # Set the size of history windows?
        univariate_past_history = 10
        # Set the future target to be predicted?
        univariate_future_target = 0

        uni_data = get_univariate_timeseries_from_df(self.input, 'Chaos_rune', 'timestamp')

        x_train_uni, y_train_uni = load_univariate_data(
            uni_data,
            0,
            TRAIN_SPLIT,
            univariate_past_history,
            univariate_future_target
        )

        x_val_uni, y_val_uni = load_univariate_data(
            uni_data,
            TRAIN_SPLIT,
            None,
            univariate_past_history,
            univariate_future_target
        )

        # Shows the true history of the sample
        plot = show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
        plot.show()

        # Show baseline prediction based on average
        plot = show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
                         'Baseline Prediction Example')
        plot.show()

if __name__ == "__main__":
    predictor = GEPredictor('data/Rune_Data.csv')
    predictor.run()
    # Set the size for window history ?
