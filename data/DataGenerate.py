import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler


def update_non_relevant_data(data_col, non_relevant):
    list_data = []

    for serial in data_col:
        list_data.append(np.nan if serial == non_relevant else serial)

    data = pd.DataFrame(list_data)
    return data


def scale_data(data_set):
    scaler = MinMaxScaler()

    cols = ['DateMonth', 'DIM', 'CowID', 'Gynecology_Status', 'LactationNumber', 'CurrentDryDays',
            'DailyActivity', 'WeightCalv', 'Age', 'PrevDryDays', 'DP', 'SerialNumber1',
            'SerialNumber2', 'SerialNumber3', 'Milk1', 'Milk2', 'Milk3', 'Conductivity1',
            'Conductivity2', 'Conductivity3', 'Fat1', 'Fat2', 'Fat3', 'Protein1', 'Protein2',
            'Protein3', 'Lactose1', 'Lactose2', 'Lactose3']

    data_set[cols] = scaler.fit_transform(data_set[cols])

    data_set['SerialNumber1'] = update_non_relevant_data(data_set['SerialNumber1'], 0.0)
    data_set['SerialNumber2'] = update_non_relevant_data(data_set['SerialNumber2'], 0.0)
    data_set['SerialNumber3'] = update_non_relevant_data(data_set['SerialNumber3'], 0.0)

    data_set = data_set.fillna(data_set.mean())

    return data_set


if __name__ == "__main__":
    DATA_DIR1 = "./afimilk_data/AppDataNir/"
    DATA_DIR2 = "./afimilk_data/SessionDataNir/"

    file_names1 = glob.glob(DATA_DIR1 + '*.csv')
    file_names2 = glob.glob(DATA_DIR2 + '*.csv')

    AppNames = ['DateMonth', 'Date', 'DIM', 'CowID', 'Gynecology_Status', 'LactationNumber', 'IsPreviousLactation',
                'CurrentDryDays', 'DailyActivity', 'WeightCalv', 'DaysRemainingTo305', 'Age', 'CurrentRP',
                'CurrentMET', 'CurrentKET', 'CurrentMF', 'CurrentPRO', 'CurrentLDA', 'CurrentMAST',
                'CurrentEdma', 'CurrentLAME', 'Twin', 'Still', 'PrevDryDays', 'DP']

    SessNames = ['SerialNumber1', 'SerialNumber2', 'SerialNumber3', 'Date', 'AnimalId',
                 'Milk1', 'Milk2', 'Milk3', 'Conductivity1', 'Conductivity2', 'Conductivity3',
                 'Fat1', 'Fat2', 'Fat3', 'Protein1', 'Protein2', 'Protein3', 'Lactose1', 'Lactose2',
                 'Lactose3']

    dfsApp = []
    dfsSess = []

    for filename in file_names1:
        df = pd.read_csv(filename, usecols=AppNames, index_col=None, header=0)
        dfsApp.append(df)

    for filename in file_names2:
        df = pd.read_csv(filename, usecols=SessNames, index_col=None, header=0)
        dfsSess.append(df)

    allData_app = pd.concat(dfsApp, axis=0, ignore_index=True)
    allData_sess = pd.concat(dfsSess, axis=0, ignore_index=True)
    allData_sess.rename(columns={"AnimalId": "CowID"}, inplace=True)

    merged_data = pd.merge(allData_app, allData_sess, on=['Date', 'CowID'])

    # Data without scaling
    merged_data.to_csv('mergedDataNonScaled.csv', index=False)

    # Scaled Data
    merged_data = scale_data(merged_data)
    merged_data.to_csv('mergedData.csv', index=False)

    # This convert the output from 1-2-3 to the mean, result 5 features of output instead of 15
    mean_data = merged_data.iloc[:, :-15]

    Milk = pd.DataFrame(merged_data, columns=['Milk1', 'Milk2', 'Milk3']).mean(axis=1)
    Cond = pd.DataFrame(merged_data, columns=['Conductivity1', 'Conductivity2', 'Conductivity']).mean(axis=1)
    Fat = pd.DataFrame(merged_data, columns=['Fat1', 'Fat2', 'Fat3']).mean(axis=1)
    Protein = pd.DataFrame(merged_data, columns=['Protein1', 'Protein2', 'Protein3']).mean(axis=1)
    Lactose = pd.DataFrame(merged_data, columns=['Lactose1', 'Lactose2', 'Lactose3']).mean(axis=1)

    mean_data['Milk'] = Milk
    mean_data['Cond'] = Cond
    mean_data['Fat'] = Fat
    mean_data['Protein'] = Protein
    mean_data['Lactose'] = Lactose

    mean_data.to_csv('mergedDataMean.csv', index=False)
