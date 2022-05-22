import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from math import sqrt

# TODO file is copied -> necessary ?
from gcd_data_manipulation import load_data
from gcd_data_manipulation import data_aggregation
from gcd_data_manipulation import extract_train_test
from keras.models import Sequential
from keras.layers import RNN
from keras.layers import Dense
from tfRIM import RIMCell
from sklearn.metrics import mean_squared_error


def generate_model(train_x, batch_size, neurons):  # TODO maybe add RIM / GWT hyperparameters ?
    # 20 hidden units, 6 RIMs, top 5 will be activated
    # TODO currently trying with (droupout = 0 <-> keep_prob = 1) -> change in further testing
    # TODO currently recommended values from RIM paper (6 cells, top 4 win, 4 heads, 32 size)
    rim_cell = RIMCell(units=neurons, nRIM=6, k=4,
                       num_input_heads=4, input_key_size=32, input_value_size=32, input_query_size=32, input_keep_prob=1,
                       num_comm_heads=4, comm_key_size=32, comm_value_size=32, comm_query_size=32, comm_keep_prob=1)

    model = Sequential()
    model.add(RNN(cell=rim_cell, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')  # TODO evaluation loss
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
    results_path = '../results'

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
    print('RMSE: %.3f' % rmse)

    results['experiment'].append(exp_name)
    results['neurons'].append(neurons)
    results['batch_size'].append(batch_size)
    results['epochs'].append(epochs)
    results['rmse'].append(rmse)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(results_path))
    model.save('models/rim_model_%s' % exp_name)