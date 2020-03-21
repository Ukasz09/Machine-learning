import random
import matplotlib.pyplot as plt

x = []
y = []
max_a_coefficient = 5


def draw_plot(x_list, y_list, visual_representation):
    plt.plot(x_list, y_list, visual_representation)


def make_test_data(points_qty, y_min, y_max, rand_step):
    global x, y
    x = [x for x in range(points_qty)]
    y = [random.randrange(y_min, y_max, step=rand_step) + x for x in range(points_qty)]


def calculate_approximation_error(a, b, x_list, y_list):
    err_sum = 0
    for i in range(len(x_list)):
        err_sum += abs(y_list[i] - (a * x_list[i] + b))
    return 1 / (len(x_list)) * err_sum


def calculate_best_approximation_coefficient(x_list, y_list):
    max_a = max_a_coefficient
    max_b = max(y_list)
    min_error = calculate_approximation_error(0, 0, x_list, y_list)
    best_a = 0
    best_b = 0
    actual_a = -max_a
    actual_b = 0.1
    while actual_a < max_a:
        while actual_b < max_b:
            new_error = calculate_approximation_error(actual_a, actual_b, x_list, y)
            if new_error < min_error:
                min_error = new_error
                best_a = actual_a
                best_b = actual_b
            actual_b += 0.1
        actual_a += 0.1
        actual_b = 0.1
    return best_a, best_b


def print_result(points_qty):
    make_test_data(points_qty, 0, 50, 1)
    coef_tup = calculate_best_approximation_coefficient(x, y)
    y_apr = [coef_tup[0] * x[i] + coef_tup[1] for i in range(points_qty)]

    plt.ylim(min(y) - 1, max(y) + 1)
    draw_plot(x, y, 'b x')
    draw_plot(x, y_apr, 'r')
    plt.title("Q=1/N sume(|e|) - numerical")
    plt.show()

print_result(100)
exit(0)
