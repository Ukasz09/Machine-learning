import matplotlib.pyplot as plt
import numpy as np
import random as rand
from itertools import combinations

PLOT_ROWS = 3
PLOT_CELLS = 2
DATA_MIN_VAL = 0.5
DATA_MAX_VAL = 1
DEV_LINE_FUNC = 0.3  # deviation from line function
fig, ax = plt.subplots(nrows=PLOT_ROWS, ncols=PLOT_CELLS, figsize=(10, 10))


# -------------------------------------------------------------------------------------------------------------------- #
def get_random_data(points_qty=10) -> (np.array, np.array):
    x_values = rand.sample(range(int(DATA_MIN_VAL * 100), int(DATA_MAX_VAL * 100)), points_qty)
    x_values.sort()
    x = np.empty([len(x_values), 1])
    for i in range(len(x_values)):
        x[i, 0] = x_values[i] / 100

    y = np.empty([len(x_values), 1])
    for i in range(len(x_values)):
        y[i, 0] = rand.uniform(x[i, 0] - DEV_LINE_FUNC / 2, x[i, 0] + DEV_LINE_FUNC)
    return x, y


def plot_result_cell(started_data, degree, plot_x_vals, plot_row, plot_col):
    ax[plot_row][plot_col].set_ylim(0, DATA_MAX_VAL + DEV_LINE_FUNC)

    # plot debased model
    debased_data = data_debased_by_2_param(started_data[0], started_data[1])
    for deb_data in debased_data:
        param, _ = least_squares(deb_data[0], deb_data[1], degree)
        y = polynomial(plot_x_vals, param)
        ax[plot_row][plot_col].plot(plot_x_vals, y, "orange", linewidth=1)

    # plot correct model
    param, _ = least_squares(started_data[0], started_data[1], degree)
    y = polynomial(plot_x_vals, param)
    ax[plot_row][plot_col].plot(plot_x_vals, y, "black", linewidth=2)

    # plot points
    ax[plot_row][plot_col].plot(started_data[0], started_data[1], "b o")

    # calculate quality assurance and euclidean domain
    qa = calculate_Qa(started_data[0], started_data[1], degree)

    # plot labels
    quantity_title = "Parameters quantity: " + (degree + 1).__str__()
    quality_title = "Quality assurance (Qa): " + round(qa, 6).__str__()
    title = quantity_title + "\n" + quality_title
    ax[plot_row][plot_col].set_title(title)


def plot_compare_results(started_data):
    x_plot_data = np.arange(min(data[0]), max(data[0]), 0.001)
    row = -1
    for i in range(PLOT_CELLS * PLOT_ROWS):
        col = i % PLOT_CELLS
        if col == 0:
            row += 1
        plot_result_cell(started_data, i + 1, x_plot_data, row, col)


# -------------------------------------------------------------------------------------------------------------------- #
def calculate_Qa(data_x, data_y, degree):
    X = design_matrix(data_x, degree).transpose()
    Y = data_y.transpose()
    a, _ = least_squares(data_x, data_y, degree)
    return ((Y - a.transpose() @ X) @ (Y - a.transpose() @ X).transpose())[0][0]


def polynomial(x, w):
    dm = [w[i] * x ** i for i in range(np.shape(w)[0])]
    return np.sum(dm, axis=0)


def design_matrix(x_train, degree):
    return np.array([x_train ** i for i in range(degree + 1)]).transpose()[0]


def mean_squared_error(x, y, w):
    N = np.shape(x)[0]
    return 1 / N * np.sum((y - polynomial(x, w)) ** 2)


def least_squares(x_train, y_train, degree):
    design = design_matrix(x_train, degree)
    w = np.linalg.inv(design.transpose() @ design) @ design.transpose() @ y_train
    return w, mean_squared_error(x_train, y_train, w)


def data_debased_by_2_param(x_data, y_data):
    comb = combinations(np.arange(0, x_data.shape[0]).tolist(), 2)
    x_data_list = []
    for i in list(comb):
        x = np.delete(x_data, i[0], axis=0)
        y = np.delete(y_data, i[0], axis=0)
        x = np.delete(x, i[1] - 1, axis=0)
        y = np.delete(y, i[1] - 1, axis=0)
        x_data_list.append((x, y))
    return x_data_list


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    data = get_random_data()
    plot_compare_results(data)

    plt.show()
    exit(0)
