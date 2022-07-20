import torch
import torchvision.transforms
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
import math

from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler


def load_data(input_path, selected_columns):
    # Load data
    readings_df = read_csv(input_path, index_col=0)

    df = readings_df[selected_columns]

    return df


def data_aggregation(df, aggr_type="mean"):
    """
    :param aggr_type: aggregation type
    :type df: DataFrame
    """
    if aggr_type == 'mean':
        df = df.groupby('end time').mean()
    elif aggr_type == 'q95':
        df = df.groupby('end time').quantile(0.95)
    elif aggr_type == 'max':
        df = df.groupby('end time').max()

    return df


def series_to_supervised_old(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    agg.fillna(agg.mean(), inplace=True)
    if agg.isnull().sum().sum() > 0:
        agg.fillna(0., inplace=True)
    if dropnan:
        agg.dropna(inplace=True, axis='columns')
    return agg


def series_to_supervised(data, targets, sliding_window=1, dropnan=True, efficiency_as_input=True):
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(sliding_window, 0, -1):
        if efficiency_as_input:
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(data.shape[1])]
        else:
            cols.append(df.shift(i).iloc[:, :-1])
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(data.shape[1] - 1)]
    for i in targets:
        cols.append(df.shift(-i).iloc[:, -1])
        if i == 0:
            names += ['var%d(t)' % df.shape[1]]
        else:
            names += ['var%d(t+%d)' % (df.shape[1], i)]

    agg = concat(cols, axis=1)
    agg.columns = names

    # TODO double check
    # agg.drop(agg.tail(sliding_window).index, inplace=True)

    agg.fillna(agg.mean(), inplace=True)
    if agg.isnull().sum().sum() > 0:
        agg.fillna(0., inplace=True)
    if dropnan:
        agg.dropna(inplace=True, axis='columns')

    return agg


def scale_values(values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaled, scaler


def extract_final_dataframe(df, columns_to_drop: list):
    # Remove columns that don't end in the target group
    reframed_df = df.drop(df.columns[columns_to_drop], axis=1)
    return reframed_df


def extract_data_target(vals):
    # Extract data and target
    X, y = vals[:, :-1], vals[:, -1]
    # Reshape data
    reshaped_X = X.reshape(X.shape[0], 1, X.shape[1])
    return reshaped_X, y


def extract_train_test(values, n_train=(14 * 24 * 12)):
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaled, scaler = scale_values(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed_final = extract_final_dataframe(reframed, [i for i in range(values.shape[1], (2 * values.shape[1]) - 1)])

    # split into train and test sets
    values = reframed_final.values
    train = values[:n_train, :]
    test = values[n_train:, :]
    # split into input and outputs
    train_X, train_y = extract_data_target(train)
    test_X, test_y = extract_data_target(test)

    return train_X, train_y, test_X, test_y, scaler


def prepare_data(input_path, columns_to_consider, targets, sliding_window=1, aggr_type='mean'):
    readings_df = data_aggregation(load_data(input_path, columns_to_consider), aggr_type=aggr_type)
    values = readings_df.astype('float32')
    scaled, scaler = scale_values(values)
    reframed = series_to_supervised(scaled, targets=targets, sliding_window=sliding_window)
    # reframed_final = extract_final_dataframe(reframed, [i for i in range(values.shape[1], (2 * values.shape[1]) - 1)])
    # print(type(reframed_final.values))
    return reframed.values


class ClusterDataset(Dataset):
    """
    PyTorch Custom Dataset to load the cluster data

    Attributes
    ----------
    values : numpy.ndarray
        the already prepared data (via prepare_data(input_path, columns_to_consider, aggr_type))
    training : bool
        indicates if the dataset is a training set or test set
        True = Training Set, False = Test Set
    split_percentage: float
        between [0, 1], indicates the ratio of training / test split
        the given percentage is the proportion the training set
    """
    # TODO add constraints for aggr_type / split_percentage
    def __init__(self, values, num_targets, training=True, split_percentage=0.7):
        split_idx = math.floor(values.shape[0] * split_percentage)
        self.values = values[:split_idx, :] if training else values[split_idx:, :]
        self.values = torch.from_numpy(self.values)
        self.num_targets = num_targets

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        return self.values[idx, :-self.num_targets], self.values[idx, -self.num_targets:]
