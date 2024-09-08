import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_UCR(dataset):
    train_file = os.path.join('datasets/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('datasets/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y


def load_forecast_npy(name, univar=False):
    data = np.load(f'datasets/{name}.npy')
    if univar:
        data = data[: -1:]

    train_slice = slice(None, int(0.6 * len(data)))
    valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
    test_slice = slice(int(0.8 * len(data)), None)

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    data = np.expand_dims(data, 0)

    pred_lens = [24, 48, 96, 288, 672]
    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, 0


def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)


def load_forecast_csv(name, univar=False):
    """
    Loads and processes time series data for forecasting tasks, supporting both the pre-processed
    Online Retail II dataset and existing datasets like ETTh1, ETTm1, and electricity.

    Parameters:
    name (str): The name of the dataset file (without the .csv extension).
    univar (bool): Whether to load the univariate version of the data.

    Returns:
    tuple: data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols
    """
    # Try loading with 'date' as index, if it fails, try with 'InvoiceDate'
    try:
        data = pd.read_csv(f'datasets/{name}.csv', index_col='date', parse_dates=True)
    except ValueError:
        data = pd.read_csv(f'datasets/{name}.csv', index_col='InvoiceDate', parse_dates=True)

    # Ensure index is parsed as datetime
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)

    # Extract time features for the date/index column
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]

    # Handle univariate or multivariate cases
    if univar:
        if name in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'):
            data = data[['OT']]
        elif name == 'electricity':
            data = data[['MT_001']]
        else:
            data = data.iloc[:, -1:]
    else:
        # Online Retail II case
        if name == 'ts2vec_online_retail_II_data':
            data = data[['Quantity', 'Price']]

    # Convert data to numpy array
    data = data.to_numpy()

    # Define train, validation, and test splits based on dataset
    if name == 'ETTh1' or name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif name == 'ETTm1' or name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        # Default case for custom datasets
        train_slice = slice(None, int(0.6 * len(data)))
        valid_slice = slice(int(0.6 * len(data)), int(0.8 * len(data)))
        test_slice = slice(int(0.8 * len(data)), None)

    # Normalise data
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    # Reshape data based on dataset structure
    if name in 'electricity':
        data = np.expand_dims(data.T, -1)  # Each variable is an instance rather than a feature
    else:
        data = np.expand_dims(data, 0) # Single instance case

    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate([np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1)

    if name in ('ETTh1', 'ETTh2', 'electricity'):
        pred_lens = [24, 48, 168, 336, 720]
    else:
        pred_lens = [24, 48, 96, 288, 672]

    return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols


def load_anomaly(name):
    res = pkl_load(f'datasets/{name}.pkl')
    return res['all_train_data'], res['all_train_labels'], res['all_train_timestamps'], \
        res['all_test_data'],  res['all_test_labels'],  res['all_test_timestamps'], \
        res['delay']


def gen_ano_train_data(all_train_data):
    maxl = np.max([ len(all_train_data[k]) for k in all_train_data ])
    pretrain_data = []
    for k in all_train_data:
        train_data = pad_nan_to_target(all_train_data[k], maxl, axis=0)
        pretrain_data.append(train_data)
    pretrain_data = np.expand_dims(np.stack(pretrain_data), 2)
    return pretrain_data

#-----------------------------------------------------------------------------
def _get_time_features(dt):
    return np.stack([
        dt.dayofweek.to_numpy(),  # Day of the week
        dt.day.to_numpy(),        # Day of the month
        dt.dayofyear.to_numpy(),  # Day of the year
        dt.month.to_numpy(),      # Month
        dt.to_series().apply(lambda x: x.strftime("%V")).astype(int).to_numpy(), # Week of the year
    ], axis=1).astype(np.float)

#
# def load_online_retail(name='Cleaned_Online_Retail', agg_freq='D'):
#     """
#     Loads and preprocesses the Online Retail dataset for forecasting tasks.
#
#     Parameters:
#         name (str): Name of the dataset file (without extension).
#         agg_freq (str): Aggregation frequency (e.g., 'D' for daily, 'W' for weekly, '2W' for bi-weekly, 'M' for monthly).
#
#     Returns:
#         data (np.ndarray): Preprocessed time series data.
#         train_slice (slice): Slice object for training data.
#         valid_slice (slice): Slice object for validation data.
#         test_slice (slice): Slice object for testing data.
#         scaler (StandardScaler): Fitted scaler object.
#         pred_lens (list): List of prediction lengths.
#         n_covariate_cols (int): Number of covariate columns.
#     """
#
#     # Load data
#     file_path = f'datasets/UEA/{name}.csv'
#     df = pd.read_csv(file_path, parse_dates=['InvoiceDate'])
#
#     # Convert 'InvoiceDate' to datetime if not already done
#     if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
#         df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#
#     # Set 'InvoiceDate' as index
#     df.set_index('InvoiceDate', inplace=True)
#
#     # Handle different aggregation frequencies
#     if agg_freq == 'D':
#         freq = 'D'
#     elif agg_freq == 'W':
#         freq = 'W'
#     elif agg_freq == '2W':
#         freq = '2W'
#     elif agg_freq == 'M':
#         freq = 'M'
#     else:
#         raise ValueError("Invalid agg_freq value. Use 'D', 'W', '2W', or 'M'.")
#
#     # Aggregate quantity sold per specified frequency
#     df_agg = df.resample(freq).sum()['Quantity']
#
#     # Handle missing values by filling with zeros
#     df_agg = df_agg.fillna(0)
#
#     # Generate time features
#     time_features = _get_time_features(df_agg.index)
#     n_covariate_cols = time_features.shape[1]
#
#     # Convert to numpy array
#     data = df_agg.values
#     data = data.reshape(-1, 1)  # Reshape to (timesteps, features)
#
#     # Combine data with time features
#     data = np.concatenate([time_features, data], axis=1)
#
#     # Train/Validation/Test split
#     total_length = len(data)
#     train_size = int(total_length * 0.6)
#     valid_size = int(total_length * 0.2)
#
#     train_slice = slice(None, train_size)
#     valid_slice = slice(train_size, train_size + valid_size)
#     test_slice = slice(train_size + valid_size, None)
#
#     # Scaling
#     scaler = StandardScaler()
#     data[train_slice] = scaler.fit_transform(data[train_slice])
#     data[valid_slice] = scaler.transform(data[valid_slice])
#     data[test_slice] = scaler.transform(data[test_slice])
#
#     # Reshape to (1, timesteps, features) as expected by TS2Vec
#     data = data[np.newaxis, ...]
#
#     # Define prediction lengths based on aggregation frequency
#     if agg_freq == 'D':
#         pred_lens = [1, 2, 3, 4, 5, 6, 7]  # Predicting each day for the upcoming week
#     elif agg_freq == 'W':
#         pred_lens = [1, 2, 3, 4]  # Predicting each week for the upcoming month (4 weeks)
#     elif agg_freq == '2W':
#         pred_lens = [1, 2, 3]  # Predicting 2 weeks, 4 weeks, and 6 weeks ahead (in bi-weekly intervals)
#     elif agg_freq == 'M':
#         pred_lens = [1, 2, 3]  # Predicting 1 month, 2 months, and 3 months ahead (in months)
#     else:
#         raise ValueError("Invalid agg_freq value. Use 'D', 'W', '2W', or 'M'.")
#
#     return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols
#     train_slice = slice(None, train_size)
#     valid_slice = slice(train_size, train_size + valid_size)
#     test_slice = slice(train_size + valid_size, None)
#
#     # Scaling
#     scaler = StandardScaler()
#     data[train_slice] = scaler.fit_transform(data[train_slice])
#     data[valid_slice] = scaler.transform(data[valid_slice])
#     data[test_slice] = scaler.transform(data[test_slice])
#
#     # Reshape to (1, timesteps, features) as expected by TS2Vec
#     data = data[np.newaxis, ...]
#
#     # Define prediction lengths based on aggregation frequency
#     if agg_freq == 'D':
#         pred_lens = [1, 2, 3, 4, 5, 6, 7]  # Predicting each day for the upcoming week
#     elif agg_freq == 'W':
#         pred_lens = [1, 2, 3, 4]  # Predicting each week for the upcoming month (4 weeks)
#     elif agg_freq == '2W':
#         pred_lens = [1, 2, 3]  # Predicting 2 weeks, 4 weeks, and 6 weeks ahead (in bi-weekly intervals)
#     elif agg_freq == 'M':
#         pred_lens = [1, 2, 3]  # Predicting 1 month, 2 months, and 3 months ahead (in months)
#     else:
#         raise ValueError("Invalid agg_freq value. Use 'D', 'W', '2W', or 'M'.")
#
#     return data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols
