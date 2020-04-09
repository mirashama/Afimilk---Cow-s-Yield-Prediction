from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from extra.Utilities import *


def calc_acc(predict, y_test):
    diff = abs(predict - y_test)
    divided = np.divide(diff, y_test, out=np.zeros_like(diff), where=y_test != 0)

    return 1 - divided.mean(0)


def run_knn(k_range, weights="uniform"):
    data_sat = pd.read_csv('../data/mergedData.csv', index_col=None, header=0)
    data_sat = data_sat.drop(columns=['Date'])

    x_train, y_train, x_test, y_test = gen_data(data_sat)
    errors = []
    for K in k_range:
        model = neighbors.KNeighborsRegressor(n_neighbors=K, weights=weights)

        model.fit(x_train, y_train)
        predict = model.predict(x_test)

        accuracy = calc_acc(predict, y_test)
        error = mean_squared_error(y_test, predict)

        print('MSE for k = ', K, ' is: ', np.round(error, 6))
        print(np.round(accuracy, 3))

        errors.append(error)

    return errors


def plot_error(k_range, error, title):
    plt.title(title)
    plt.scatter(k_range, error)
    plt.ylabel('Mean Squared Error')
    plt.xlabel('K value')
    plt.show()


if __name__ == "__main__":
    k_range = range(1, 50)

    print("=== Running K Nearest Neighbor ===")
    uni_err = run_knn(k_range)

    print("=== Running Weighted K Nearest Neighbor ===")
    wighted_err = run_knn(k_range, weights='distance')

    plot_error(k_range, uni_err, "Knn Error")
    plot_error(k_range, wighted_err, "Weighted Knn Error")
