import matplotlib.pyplot as plt
import numpy as np
import random as rand


# -------------------------------------------------------------------------------------------------------------------- #
def get_random_data(points_qty=10) -> (np.array, np.array):
    x_values = rand.sample(range(60, 80), points_qty)
    x_values.sort()
    x = np.empty([len(x_values), 1])
    for i in range(len(x_values)):
        x[i, 0] = x_values[i] / 100

    y = np.empty([len(x_values), 1])
    for i in range(len(x_values)):
        y[i, 0] = rand.uniform(x[i, 0] - 0.15, x[i, 0] + 0.15)
    return x, y


def plot_points(x, y, plt_ax):
    plt_ax.plot(x, y, 'b o')
    # ax.set_ylim([0.4, 1])


def plot_model(x_train, y_train, degree, plt_ax):
    param, err = least_squares(x_train, y_train, degree)
    y = polynomial(x_train, param)
    plt_ax.plot(x_train, y, 'black')


# -------------------------------------------------------------------------------------------------------------------- #
def polynomial(x, w):
    dm = [w[i] * x ** i for i in range(np.shape(w)[0])]
    return np.sum(dm, axis=0)


def design_matrix(x_train, M):
    return np.array([x_train ** i for i in range(M + 1)]).transpose()[0]


def mean_squared_error(x, y, w):
    N = np.shape(x)[0]
    return 1 / N * np.sum((y - polynomial(x, w)) ** 2)


def least_squares(x_train, y_train, degree):
    design = design_matrix(x_train, degree)
    print(design.shape)
    w = np.linalg.inv(design.transpose() @ design) @ design.transpose() @ y_train
    return w, mean_squared_error(x_train, y_train, w)


# -------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

    data = get_random_data()

    plot_points(data[0], data[1], ax)
    plot_model(data[0], data[1], 1, ax)
    plt.show()
    exit(0)
