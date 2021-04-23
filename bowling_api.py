import matplotlib.pyplot as plt
import numpy as np


def create_video(X_values, Y_values):
    plt.scatter()


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


@memoize
def simulate_hits(x, y, vx, vy, wx, wy):
    pass


def main():
    error_rates = get_error_rates()
    scores = np.zeros((error_rates, throw_num_per_error_rate))
    for i, error_rate in enumerate(error_rates):
        for j in range(throw_num_per_error_rate):
            x, vx, vy, wx, wy = get_throwing_parameters(error_rate)  # y default is 0
            x, y, vx, vy, wx, wy = calc_first_hit_position(x, 0, vx, vy, wx, wy)
            score = simulate_hits(x, y, vx, vy, wx, wy)
            scores[i, j] = score
    avg_hits = np.average(scores, axis=1)
    plot_graph(error_rates, avg_hits)


if __name__ == '__main__':
    main()
