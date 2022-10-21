import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from math import sqrt

# TODO file is copied -> necessary ?
from old_data_manip import load_data
from old_data_manip import data_aggregation
from old_data_manip import extract_train_test
from keras.models import Sequential
# from keras.layers import Input
from keras.layers import RNN
from keras.layers import Dense
from tfRIM import RIMCell
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

def generate_model(train_x, batch_size, units):
    rim_cell = RIMCell(units=units, nRIM=6, k=4,
                       num_input_heads=4, input_key_size=32, input_value_size=32, input_query_size=32, input_keep_prob=1,
                       num_comm_heads=4, comm_key_size=32, comm_value_size=32, comm_query_size=32, comm_keep_prob=1)

    model = Sequential()
    model.add(RNN(cell=rim_cell, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    return model

def fit_model(train_x, train_y, test_x, test_y, model, epochs, batch_size):
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_x, test_y), verbose=2, shuffle=False)
    return history.history


def predict_model(test_x, test_y, model):
    yhat = model.predict(test_x, batch_size=batch_size)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[2])
    inv_yhat = np.concatenate((test_x[:, :-1], yhat), axis=1)

    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_x[:, :-1], test_y), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]

    return inv_y, inv_yhat


def plot_val_history(history, output_path, name):
    # plot history
    pyplot.clf()
    pyplot.plot(history['loss'], label='train')
    pyplot.plot(history['val_loss'], label='test')
    pyplot.legend()
    pyplot.ylim((0.00, 0.27))
    pyplot.savefig(os.path.join(output_path, 'gcd_%s_val-loss.png' % name))


def plot_prediction(y, yhat, name):
    # line plot of observed vs predicted
    pyplot.clf()
    pyplot.plot(y[-100:], '-', color='orange', label="Raw measurements")
    pyplot.plot(yhat[-100:], '--', color='blue', label="Predictions")
    pyplot.xlabel('Steps', fontsize=20)
    pyplot.ylabel('Efficiency', fontsize=20)
    pyplot.xticks(fontsize=24)
    pyplot.yticks(fontsize=24)
    pyplot.legend(fontsize=14, frameon=False)
    pyplot.tight_layout()
    prediction_out_figure = os.path.join(figures_path, 'gcd_%s_pred.png' % name)
    pyplot.savefig(prediction_out_figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GWT-RIM predictor for job efficiency')  # TODO
    parser.add_argument('hyperparameters', metavar='H', type=int, nargs=3,
                        help='epochs neurons batch_size')
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Name of the experiment. Necessary to save results")
    # Initially use job_id = 3418339 to compare to different architectures
    parser.add_argument("--job_id", type=int, default=3418339, dest='job_id',
                        help="ID of the job considered for the model generation.")
    # TODO further useful / necessary arguments
    # cols / dropout

    with open('columns_selection.json') as f:
        columns_selection = json.load(f)

    args = parser.parse_args(sys.argv[1:])
    epochs, neurons, batch_size = args.hyperparameters
    exp_name = args.exp_name
    JOB_ID = args.job_id

    input_path = '../data/task-usage_job-ID-%i_total.csv' % JOB_ID
    figures_path = '../experiments_result/figures_GWT'
    results_path = 'results'
    model_path = '../models'

    columns_to_consider = columns_selection["GWT_efficiency_1"]  # TODO add argparse

    readings_df = load_data(input_path, columns_to_consider)
    readings_df = data_aggregation(readings_df, aggr_type='mean')  # TODO further aggregation types ?

    results = defaultdict(list)

    # TODO add repetition parameter
    train_x, train_y, test_x, test_y, scaler = extract_train_test(readings_df.values)

    model = generate_model(train_x, batch_size, neurons)

    history_loss = fit_model(train_x, train_y, test_x, test_y, model, epochs, batch_size)

    inv_y, inv_yhat = predict_model(test_x, test_y, model)

    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # rmspe =
    print('RMSE: %.3f' % rmse)

    results['experiment'].append(exp_name)
    results['units'].append(neurons)
    results['batch_size'].append(batch_size)
    results['epochs'].append(epochs)
    results['rmse'].append(rmse)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(results_path, '%s.csv' % exp_name))

    plot_val_history(history_loss, figures_path, exp_name)
    plot_prediction(inv_y, inv_yhat, exp_name)

    model.save(os.path.join(model_path, 'rim_model_%s' % exp_name), save_format='h5')