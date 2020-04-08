import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

T = 200
h = 1e-2
t = np.arange(start=0, stop=T + h, step=h)

bet, gam = 0.15, 1 / 50
# todo: zmienic poziej na randoma
# S_pocz = np.random.uniform(0.7, 1)
S_start = 0.8
I_start = 1 - S_start
R_start = 0
N = S_start + I_start + R_start  # is const


# using odeint
# ---------------------------------------------------------------------------------------------------------------------#
def two_diff_ode_equation(state, t, bet, gam):
    S, I = state
    return [- bet * I * S / N, bet * I * S / N - gam * I]


def one_diff_equation_ode(state, t, bet, gam):
    S = state[0]
    C = I_start - gam / bet * np.log(S_start) + S_start  # C - const
    return [(-bet / N * S * (gam / bet * np.log(S) - S + C))]


def calc_R(S_arr, I_arr):
    R_arr = np.zeros(len(t))
    for i in range(len(R_arr)):
        R_arr[i] = N - S_arr[i] - I_arr[i]
    return R_arr


def calc_I(S_arr):
    C = I_start - gam / bet * np.log(S_start) + S_start  # C - const
    I_arr = np.zeros(len(t))
    for i in range(len(I_arr)):
        I_arr[i] = gam / bet * np.log(S_arr[i]) - S_arr[i] + C
    return I_arr


def two_equation_ode_plot(t, sym, labelt='$t$', labels=['S', 'I', 'R']):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

    # plot drawing (S, I)
    for i in range(len(labels) - 1):
        ax.plot(t, sym[:, i], label=labels[i])
    # plot drawing (R)
    ax.plot(t, calc_R(sym[:, 0], sym[:, 1]), label=labels[2])

    ax.set_xlabel(labelt, fontsize=14)
    ax.set_ylabel('stan', fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend()
    plt.show()


def one_equation_ode_plot(t, sym, labelt='$t$', labels=['S', 'I', 'R']):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

    # plot drawing (S)
    ax.plot(t, sym[:, 0], label=labels[0])
    # plot drawing (I)
    I_arr = calc_I(sym[:, 0])
    ax.plot(t, I_arr, label=labels[2])
    # plot drawing (R)
    ax.plot(t, calc_R(sym[:, 0], I_arr), label=labels[2])

    ax.set_xlabel(labelt, fontsize=14)
    ax.set_ylabel('stan', fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend()
    plt.show()


def two_equation_ode_main():
    start_state = S_start, I_start
    sym = odeint(two_diff_ode_equation, start_state, t, args=(bet, gam))
    two_equation_ode_plot(t, sym, labels=['S', 'I', 'R'])


def one_equation_ode_main():
    start_state = S_start
    sym = odeint(one_diff_equation_ode, start_state, t, args=(bet, gam))
    one_equation_ode_plot(t, sym, labels=['S', 'I', 'R'])


# using manual
# ---------------------------------------------------------------------------------------------------------------------#
S = np.zeros(len(t))
S[0] = S_start
I = np.zeros(len(t))
I[0] = I_start
R = np.zeros(len(t))
R[0] = R_start


def two_diff_equation_manual():
    for i in range(t.size - 1):
        S[i + 1] = S[i] + h * (- bet * I[i] * S[i] / N)
        I[i + 1] = I[i] + h * (bet * I[i] * S[i + 1] / N - gam * I[i])
        R[i + 1] = N - S[i + 1] - I[i + 1]


def one_diff_equation_manual():
    C = I_start - gam / bet * np.log(S_start) + S_start  # C - const
    for i in range(t.size - 1):
        S[i + 1] = S[i] + h * (-bet / N * S[i] * (gam / bet * np.log(S[i]) - S[i] + C))
        I[i + 1] = gam / bet * np.log(S[i + 1]) - S[i + 1] + C
        R[i + 1] = N - S[i + 1] - I[i + 1]


def equation_man_plot(t, sirList, labelt='$t$', labels=['S', 'I', 'R']):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    # plot drawing (R, S, I)
    for i in range(len(sirList)):
        ax.plot(t, sirList[i], label=labels[i])
    ax.set_xlabel(labelt, fontsize=14)
    ax.set_ylabel('stan', fontsize=14)
    ax.set_ylim([0, 1])
    ax.legend()
    plt.show()


def two_equation_man_main():
    two_diff_equation_manual()
    equation_man_plot(t, [S, I, R], labels=['S', 'I', 'R'])


def one_equation_man_main():
    one_diff_equation_manual()
    equation_man_plot(t, [S, I, R], labels=['S', 'I', 'R'])


if __name__ == "__main__":
    # one_equation_ode_main()
    # one_equation_man_main()
    # two_equation_ode_main()
    two_equation_man_main()
    exit(0)
