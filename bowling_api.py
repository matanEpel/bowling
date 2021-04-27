from math import dist
import matplotlib.pyplot as plt
import numpy as np

"""
BALL: (x, y, vx, vy, wx, wy)
pin: (x1, y1, z1, x2, y2, z2, vx, vy, vz, wx, wy, wz
"""

# TODO:
# 1. finding mu of oiled lane
# 2. finding the best throwing parameters
# 3. deciding on max variance we wnt in our throws (error rate will be in percentage out of this variance)
# 4. joining frames to video
# 5. finding the correct locations of the pins
# 6. creating the init_pins function (agter 5)
# 7. correcting the throw_dt function
# 8. writing the hit_dt function


R_BALL = 10.795
R_PIN = 5
BALL_M = 10
PIN_M = 0.5
BALL_I = 2 / 5 * BALL_M * R_BALL * R_BALL
OILED_MU = 0.25  # TODO
G = 9.81
LANE_LENGTH = 1828.8
LANE_WIDTH = 105.41
SIZE = 6
BOWLING_SIZE = np.pi * R_BALL ** 2
PIN_HEIGHT = 40
ERR_RT = 0.01
REPEATITIONS = 100
# TODO: change values:
BEST_VY = 8
BEST_VX = 4
BEST_WX = 5
BEST_WY = 3
DEFAULT_VAR = 0.1


def create_video(all_obj_locs):
    """
    creating all the frames for the video by the locations of the ball and pins
    :param all_obj_locs: locations (x,y) of the ball and the pins
    :return: none
    """
    i = 0
    for f in all_obj_locs:
        plt.figure(figsize=(SIZE * 2, SIZE), dpi=80)
        plt.xlim([-5, LANE_LENGTH])
        plt.ylim([-5, LANE_WIDTH])
        x_s = [y for (x, y, size) in f]
        y_s = [x for (x, y, size) in f]
        s = [size * 40 * LANE_WIDTH / LANE_LENGTH for (x, y, size) in f]
        plt.scatter(x_s, y_s, s=s)
        plt.savefig("data/frame" + str(i) + ".png")
        plt.show()
        i += 1


def memoize(f):
    """
    memoizing function we use twce to save running time
    :param f: the function
    :return: the saving function
    """
    memo = {}

    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]

    return helper


def still_going(ball_stats):
    """
    checking if the simulation of the ball throwing should continue
    we stop in case were the ball does not move or he hitted one of the pins
    :param ball_stats: velocity and location of the ball
    :return: if the ball should continue or not
    """
    if ball_stats[3] == 0 and ball_stats[2] == 0:  # if vy = vx = 0 we should stop
        return False
    pins_loc = [(720, 20.5), (730.375, 26.5), (740.75, 32.5), (751.125, 38.5)
        , (730.375, 14.5), (740.75, 8.5),
                (751.125, 2.5)]  # the locations of the pins in inch's. TODO: update to correct
    pins_loc = [(2.54 * p[0], 2.54 * p[1]) for p in pins_loc]  # converting to cm
    if ball_stats[0] > 41 * 2.54 or ball_stats[0] < 0:  # checking if we are out of the lane
        return False
    for p in pins_loc:
        if dist((ball_stats[0], ball_stats[1]), p) < R_BALL + R_PIN:  # checking if we hit one of the balls
            return False
    return True


def calc_throw_dt(ball_stats):
    """
    calculating the next location of the ball
    :param ball_stats: the ball stats (x,y,vx,vy,wx,wy)
    :return: the new ball stats
    """
    # ball stats: [0] x=x1, [1] y=x2, [2] vx=x3, [3] vy=x4, [4] wx=x5, [5] wy=x6
    Fx1, Fx2 = -OILED_MU * BALL_M * G * np.sign(ball_stats[2:4])
    ball_stats = ball_stats + np.array(
        #    x3              x4           Fx1/m         Fx2/m              Fx2*R/I             -Fx1*R/I
        [ball_stats[2], ball_stats[3], Fx1 / BALL_M, Fx2 / BALL_M, Fx2 * R_BALL / BALL_I, -Fx1 * R_BALL / BALL_I]
    )
    return ball_stats


def get_locs(pins_stats, ball_stats):
    """
    getting all the locations of the ball and pins
    :param pins_stats: the pins stats
    :param ball_stats: the ball stats
    :return: the ball and pins locations
    """
    return pins_stats[:, :6], ball_stats[:2]


def calc_score(pins_stats):
    """
    calculating the score we have after the throw
    :param pins_stats:
    :return:
    """
    count = 0
    for p in pins_stats:
        if p[5] < PIN_HEIGHT / 2:  # p[5] = z of top pin position
            count += 1  # if z < 0.5 of the top position then the pin fell
    return count


def throw_ended(ball_stats, pins_stats):
    """
    checking if we need to stop (ball out of lane/stopped or all the pins stopped)
    :param ball_stats:
    :param pins_stats:
    :return:
    """
    ball_stopped = (ball_stats[2] == 0 and ball_stats[3] == 0) or ball_stats[0] < 0 or ball_stats[0] > LANE_WIDTH or \
                   ball_stats[1] > LANE_LENGTH
    pins_stopped = all(p[6] == 0 and p[7] == 0 and p[8] == 0 for p in pins_stats)  # p[6:8] = vx, vy, vz
    return ball_stopped and pins_stopped


@memoize
def simulate_throw(x, vx, vy, wx, wy, show_video=False, ball_locs_return=False):
    ball_locs = []
    ball_stats = np.array([x, 0, vx, vy, wx, wy])
    while still_going(ball_stats):
        ball_stats = calc_throw_dt(ball_stats)
        ball_locs.append(ball_stats[:2])

    if show_video:
        create_video(ball_locs)
    if ball_locs_return:
        return ball_locs, ball_stats
    return ball_stats


def init_pins():
    return np.zeros((10, 12))  # TODO


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
    """
    plotting the graph of average hits per error rate
    :param error_rates: the error rate list
    :param avg_hits: the hits list per error rate
    :return: none
    """
    plt.xlabel("Error rates (σ)")
    plt.ylabel("Average pins hit")
    plt.plot(error_rates, avg_hits)
    plt.show()


def get_error_rates():
    """
    getting a list of error rates to check
    :return: the list
    """
    return [(ERR_RT) * i for i in range(int((1 / ERR_RT) / 4))]  # error up to 25%


def get_random_throwing_parameters(error_rate):
    """
    getting the trowing parameters by error rate (based on the best throw)
    :param error_rate: the error rate
    :return: the throwing parameters
    """
    return np.random.normal(LANE_WIDTH / 2, error_rate * DEFAULT_VAR), np.random.normal(BEST_VX,
                                                                                        error_rate * DEFAULT_VAR), \
           np.random.normal(BEST_VY, error_rate * DEFAULT_VAR), np.random.normal(BEST_WX, error_rate * DEFAULT_VAR), \
           np.random.normal(BEST_WY, error_rate * DEFAULT_VAR)


def main():
    error_rates = get_error_rates()
    throw_num_per_error_rate = REPEATITIONS
    scores = np.zeros((error_rates, throw_num_per_error_rate))
    for i, error_rate in enumerate(error_rates):
        for j in range(throw_num_per_error_rate):
            x, vx, vy, wx, wy = get_random_throwing_parameters(error_rate)  # y default is 0
            x, y, vx, vy, wx, wy = simulate_throw(x, vx, vy, wx, wy)
            score = simulate_hits(x, y, vx, vy, wx, wy)
            scores[i, j] = score
    avg_hits = np.average(scores, axis=1)
    plot_graph(error_rates, avg_hits)


if __name__ == '__main__':
    main()
