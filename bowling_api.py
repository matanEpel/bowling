from math import dist

import matplotlib.pyplot as plt
import numpy as np

R_BALL = 20
R_PIN = 5
BALL_W = 10
PIN_W = 0.5

def create_video(all_obj_locs):
    plt.scatter()


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper


def still_going(ball_stats):
    pins_loc = [(720,20.5), (730.375, 26.5), (740.75, 32.5), (751.125, 38.5)
                ,(730.375, 14.5), (740.75, 8.5), (751.125, 2.5)]
    pins_loc = [(2.54*p[0], 2.54*p[1]) for p in pins_loc]
    if ball_stats[0] > 41*2.54 or ball_stats[0] < 0:
        return False
    for p in pins_loc:
        if dist((ball_stats[0], ball_stats[1]), p) < R_BAll + R_PIN:
            return False
    return True

@memoize
def simulate_throw(x, vx, vy, wx, wy, show_video=False):
    ball_locs = []
    ball_stats = np.array([x, 0, vx, vy, wx, wy])
    while not still_going(ball_stats):
        calc_throw_dt(ball_stats)  # TODO MED
        ball_locs.append(ball_stats[:2])

    if show_video:
        create_video(ball_locs)
    return ball_stats


def init_pins():
    return np.zeros((10, 12))


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
