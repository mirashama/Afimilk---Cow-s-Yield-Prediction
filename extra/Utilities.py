import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import numpy.linalg as np_l
import pandas as pd
# Added to import them in model's file
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


pd.options.mode.chained_assignment = None  # default='warn'

scaler = MinMaxScaler()


def gen_data(data_set, reshape=0, out_num=1):
    k_fold = math.floor(0.1 * len(data_set))
    out = 15 * out_num

    train_data = data_set.head(len(data_set) - k_fold)
    train_data = train_data.sample(frac=1)
    train_x = train_data.iloc[:, :-out].values
    train_y = train_data[train_data.columns[-out:]].values

    val_data = data_set.tail(k_fold)
    val_x = val_data.iloc[:, :-out].values
    val_y = val_data[val_data.columns[-out:]].values

    # reshape input to be [samples, time steps, features]
    if reshape == 1:
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        val_x = np.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1]))

    return train_x, train_y, val_x, val_y


def pca(data, out_num=1):
    data = data.astype(np.float64)
    input_data = data.iloc[:, :-15 * out_num].values
    output_data = data.iloc[:, -15 * out_num:]

    eigen_value, eigen_vec = np_l.eig(np.cov(input_data.T))
    cumulative_variance = [np.sum(eigen_value[:i]) / np.sum(eigen_value) for i in range(1, len(input_data) + 1)]

    index = 0
    for i in range(len(cumulative_variance)):
        if cumulative_variance[i] > 0.99:
            index = i
            break

    projection = [np.dot(input_data, eigen_vec[:, i]) for i in range(index)]
    projection = pd.DataFrame(projection).T
    projection = pd.concat([projection, output_data], axis=1)
    projection = projection.astype(np.float64)

    return projection


def series_data_lstm(data, n_in=1, out_len=15):
    df = pd.DataFrame(data)
    df1 = df.iloc[:, :-out_len]
    df2 = df.iloc[:, -out_len:]
    input1, output1, input2, output2 = [], [], [], []

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        input1.append(df1.shift(i))
        output1.append(df2.shift(i))

    for i in range(len(df.index)):
        inp, out = [], []
        for j in range(n_in):
            inp.append(input1[j].iloc[i, :].values.tolist())
            out.append(output1[j].iloc[i, :].values.tolist())

        input2.append(inp)
        output2.append(out)

    return input2, output2


def sliding_window_lstm(num=10, mean=0):
    if mean:
        data_set = pd.read_csv('../data/mergedDataMean.csv', index_col=None, header=0)
    else:
        data_set = pd.read_csv('../data/mergedData.csv', index_col=None, header=0)

    data_set['Date'] = pd.to_datetime(data_set.Date, format='%d/%m/%Y')
    cows = data_set.groupby("CowID")

    merged_input = []
    merged_output = []

    for cow in cows.groups:
        cow_array = cows.get_group(cow)
        cow_array = cow_array.sort_values(by=['Date'])
        cow_array = cow_array.drop(columns=['Date'])

        inp, out = series_data_lstm(cow_array, num)
        merged_input.append(inp[:][num:][:])
        merged_output.append(out[:][num:][:])

    return merged_input, merged_output


def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True):
    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)
    cols, names = [], []

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, 2):
        if i == 0:
            x = df.shift(-i).iloc[:, :-15]
        else:
            x = df.shift(-n_out).iloc[:, -15:]

        cols.append(x)
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars-15)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(15)]

    # put it all together
    supervised_data = pd.concat(cols, axis=1)
    supervised_data.columns = names

    # drop rows with NaN values
    if drop_nan:
        supervised_data.dropna(inplace=True)

    return supervised_data


def sliding_window(num=10, out_num=1):
    data_set = pd.read_csv('../data/mergedDataNonScaled.csv', index_col=None, header=0)

    col = ['DateMonth', 'DIM', 'CowID', 'Gynecology_Status', 'LactationNumber', 'CurrentDryDays',
           'DailyActivity', 'WeightCalv', 'Age', 'PrevDryDays', 'DP', 'SerialNumber1',
           'SerialNumber2', 'SerialNumber3', 'Milk1', 'Milk2', 'Milk3', 'Conductivity1',
           'Conductivity2', 'Conductivity3', 'Fat1', 'Fat2', 'Fat3', 'Protein1', 'Protein2',
           'Protein3', 'Lactose1', 'Lactose2', 'Lactose3']

    data_set[col] = scaler.fit_transform(data_set[col])

    data_set['Date'] = pd.to_datetime(data_set.Date, format='%d/%m/%Y')

    cows = data_set.groupby("CowID")
    merged = []

    for cow in cows.groups:
        cow_array = cows.get_group(cow)
        cow_array = cow_array.sort_values(by=['Date'])
        cow_array = cow_array.drop(columns=['Date'])
        merged.append(series_to_supervised(cow_array, num, out_num, True))

    data_set = pd.concat(merged, axis=0, ignore_index=True)
    data_set = data_set.fillna(data_set.mean())
    data_set = pca(data_set, out_num=1)

    return data_set
