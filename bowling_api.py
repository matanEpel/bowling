from math import dist

import matplotlib.pyplot as plt
import numpy as np

R_BALL = 20
R_PIN = 5
BALL_M = 10
PIN_M = 0.5
BALL_I = 2 / 5 * BALL_M * R_BALL * R_BALL
OILED_MU = 0.25  # TODO
G = 9.81


def create_video(all_obj_locs):
    plt.scatter()  # TODO


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


def still_going(ball_stats):
    if ball_stats[2] < 0 or ball_stats[3] < 0:
        return False
    pins_loc = [(720, 20.5), (730.375, 26.5), (740.75, 32.5), (751.125, 38.5)
        , (730.375, 14.5), (740.75, 8.5), (751.125, 2.5)]
    pins_loc = [(2.54 * p[0], 2.54 * p[1]) for p in pins_loc]
    if ball_stats[0] > 41 * 2.54 or ball_stats[0] < 0:
        return False
    for p in pins_loc:
        if dist((ball_stats[0], ball_stats[1]), p) < R_BALL + R_PIN:
            return False
    return True


def calc_throw_dt(ball_stats):
    # ball stats: [0] x=x1, [1] y=x2, [2] vx=x3, [3] vy=x4, [4] wx=x5, [5] wy=x6
    Fx1, Fx2 = -OILED_MU * BALL_M * G * np.sign(ball_stats[2:4])
    ball_stats = ball_stats + np.array(
        #    x3              x4           Fx1/m         Fx2/m              Fx2*R/I             -Fx1*R/I
        [ball_stats[2], ball_stats[3], Fx1 / BALL_M, Fx2 / BALL_M, Fx2 * R_BALL / BALL_I, -Fx1 * R_BALL / BALL_I]
    )
    return ball_stats


@memoize
def simulate_throw(x, vx, vy, wx, wy, show_video=False):
    ball_locs = []
    ball_stats = np.array([x, 0, vx, vy, wx, wy])
    while not still_going(ball_stats):
        ball_stats = calc_throw_dt(ball_stats)
        ball_locs.append(ball_stats[:2])

    if show_video:
        create_video(ball_locs)
    return ball_stats


def init_pins():
    return np.zeros((10, 12))  # TODO


def get_locs(pins_stats, ball_stats):
    return pins_stats[:, :6], ball_stats[:2]


@memoize
def simulate_hits(x, y, vx, vy, wx, wy, show_video=False):
    all_obj_locs = []
    ball_stats = np.array([x, y, vx, vy, wx, wy])
    pins_stats = init_pins()
    while not throw_ended(ball_stats, pins_stats):
        calc_hits_dt(ball_stats, pins_stats)  # TODO HARD
        all_obj_locs.append(get_locs(pins_stats, ball_stats))

    if show_video:
        create_video(all_obj_locs)
    return calc_score(pins_stats)


def plot_graph(error_rates, avg_hits):
    plt.xlabel("Error rates (σ)")
    plt.ylabel("Average pins hit")
    plt.plot(error_rates, avg_hits)
    plt.show()


def main():
    error_rates = get_error_rates()
    scores = np.zeros((error_rates, throw_num_per_error_rate))
    for i, error_rate in enumerate(error_rates):
        for j in range(throw_num_per_error_rate):
            x, vx, vy, wx, wy = get_random_throwing_parameters(error_rate)  # y default is 0
            x, y, vx, vy, wx, wy = simulate_throw(x, 0, vx, vy, wx, wy)
            score = simulate_hits(x, y, vx, vy, wx, wy)
            scores[i, j] = score
    avg_hits = np.average(scores, axis=1)
    plot_graph(error_rates, avg_hits)


if __name__ == '__main__':
    main()
