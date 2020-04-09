import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_res(train, val, title):

    ranges = [0, 3, 6, 9, 12]
    labels = ["Milk1", "Milk2", "Milk3",
              "Cond1", "Cond2", "Cond3",
              "Fat1", "Fat2", "Fat3",
              "Protein1", "Protein2", "Protein3",
              "Lactose1", "Lactose2", "Lactose1"]

    y_length_train = range(1, len(train) + 1)
    y_length_val = range(1, len(val) + 1)

    plt.title(title)
    plt.subplot(5, 2, 1)
    plt.title("Training Accuracy")
    plt.subplot(5, 2, 2)
    plt.title("Validation Accuracy")

    j = 1
    for i in ranges:

        plt.subplot(5, 2, j)
        plt.plot(y_length_train, np.array(train)[:, i], label=labels[i])
        plt.plot(y_length_train, np.array(train)[:, i+1], label=labels[i+1])
        plt.plot(y_length_train, np.array(train)[:, i+2], label=labels[i+2])
        plt.legend(loc=4, prop={'size': 11})
        plt.ylabel('Accuracy', fontsize=10)

        plt.subplot(5, 2, j+1)
        plt.plot(y_length_val, np.array(val)[:, i], label=labels[i])
        plt.plot(y_length_val, np.array(val)[:, i+1], label=labels[i+1])
        plt.plot(y_length_val, np.array(val)[:, i+2], label=labels[i+2])
        plt.legend(loc=1, prop={'size': 11})

        j += 2

    plt.subplot(5, 2, 9)
    plt.xlabel('Epoch', fontsize=10)
    plt.subplot(5, 2, 10)
    plt.xlabel('Epoch', fontsize=10)
    plt.show()


def readfile(names):
    files = []
    for name in names:
        files.append(pd.read_csv(name, index_col=None, header=0))

    return files
