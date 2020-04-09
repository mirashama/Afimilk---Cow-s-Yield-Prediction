import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import csv


def get_distance(predict, expect):
    dist = (predict - expect) ** 2
    return np.sqrt(sum(dist))


def calc_error(pred_list, tar_list):
    diff = [abs(x1 - x2) for (x1, x2) in zip(tar_list, pred_list)]
    val = np.array(np.divide(diff, tar_list, out=np.zeros_like(diff), where=tar_list != 0))
    return 1 - val.mean(0)


def get_train_test_sets(num_history):
    data_set = pd.read_csv('../data/mergedData.csv', index_col=None, header=0)
    data_set['Date'] = pd.to_datetime(data_set.Date, format='%d/%m/%Y')
    cows = data_set.groupby("CowID")
    train = []
    test = []

    for cow in cows.groups:
        cow_array = cows.get_group(cow)
        cow_array = cow_array.sort_values(by=['Date'])
        cow_array = cow_array.fillna(cow_array.mean())
        if cow_array.shape[0] < num_history:
            continue

        train.append(cow_array.tail(num_history).head(num_history - 1))
        test.append(cow_array.tail(num_history).tail(1))

    return train, test, data_set.columns[-15:]


def write_to_file(num_history, mean_error):
    file_name = '../results/arima/' + str(num_history - 1) + '_arima.csv'
    with open(file_name, 'w', newline='') as csv_file:
        res_writer = csv.writer(csv_file)
        for out, error in mean_error:
            res_writer.writerow([str(out), str(error)])


def run_arima(num_history):
    train, test, outputs = get_train_test_sets(num_history)
    mean_error = []

    for out in outputs:
        prediction = []
        test_out = []
        j = 0
        for tr in range(len(train)):  # len(train)
            try:
                train_out = train[tr][out]
                model = ARIMA(train_out, order=(5, 1, 0))
                model_fit = model.fit(disp=0)
                predict = model_fit.forecast()
                prediction.append(predict[0].min())
                test_out.append(test[tr][out].values)

            except:
                j += 1
                continue

        error = calc_error(prediction, test_out)

        mean_error.append((out, error))
        print(str(out) + ": ", error)

    write_to_file(num_history, mean_error)


if __name__ == "__main__":
    for i in range(10, 31):
        run_arima(i)
