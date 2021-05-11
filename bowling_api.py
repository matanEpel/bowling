from math import dist

import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
BALL: (x, y, vx, vy, wx, wy)
pin: (x1, y1, z1, x2, y2, z2, vx, vy, vz, wx, wy, wz
"""

# TODO:
# 1. finding mu of oiled lane
# 2. finding the best throwing parameters
# 3. deciding on max variance we want in our throws (error rate will be in percentage out of this variance)
# 4. joining frames to video
# 5. finding the correct locations of the pins
# 6. creating the init_pins function (after 5)
# 7. correcting the throw_dt function
# 8. writing the hit_dt function


R_BALL = 5
R_PIN = 9.5
BALL_M = 10
PIN_M = 0.5
BALL_I = 2 / 5 * BALL_M * R_BALL * R_BALL
OILED_MU = 0.04  # TODO
NO_OIL_MU = 0.2  # TODO
G = 9.81
LANE_LENGTH = 1910
LANE_WIDTH = 105.41
SIZE = 6
BOWLING_SIZE = np.pi * R_BALL ** 2
PIN_HEIGHT = 40
ERR_RT = 0.01
REPEATITIONS = 100
DT = 0.001
ORIG_PINS_LOC = [(720, 20.5), (730.375, 26.5), (740.75, 32.5), (751.125, 38.5), (730.375, 14.5), (740.75, 8.5),
                 (751.125, 2.5), (740.75, 20.5), (751.125, 26.5),
                 (751.125, 14.5)]  # the locations of the pins in inch's.
ORIG_PINS_LOC = [(2.54 * p[1], 2.54 * p[0]) for p in ORIG_PINS_LOC]  # converting to cm
STEP = 10
# TODO: change values:
BEST_VY = 900
BEST_VX = 40
BEST_WX = -5
BEST_WY = 0
DEFAULT_VAR = 0.1


def create_video_from_frames(amount_of_frames, fps):
    img_array = []
    for i in range(amount_of_frames):
        img = cv2.imread('data/frame' + str(i) + '.png')
        img_array.append(img)

    out = cv2.VideoWriter('throw.mp4',0x7634706d, fps/10, (960, 480))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def create_video_from_frames_hit(amount_of_frames, fps):
    img_array = []
    for i in range(amount_of_frames):
        img = cv2.imread('data/frame' + str(i) + '.png')
        img_array.append(img)

    out = cv2.VideoWriter('hit.mp4',0x7634706d, fps/10, (960, 480))

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def create_video_hit(all_obj_locs, fps=30):
    """
    creating all the frames for the video by the locations of the ball and pins
    :param all_obj_locs: locations (x,y) of the ball and the pins
    :param fps: frames per second of video
    :return: none
    """
    i = 0
    print(len(all_obj_locs[::STEP]))
    for f in all_obj_locs[::STEP]:
        plt.figure(figsize=(SIZE * 2, SIZE), dpi=80)
        plt.ylim([-20, 120])
        plt.xlim([1800-40, LANE_LENGTH + 50+40])
        x_s = [p[1] for p in f]
        y_s = [p[0] for p in f]
        s = 1000
        plt.scatter(x_s, y_s, s=s)
        plt.savefig("data/frame" + str(i) + ".png")
        plt.close()
        # plt.show()
        i += 1
    create_video_from_frames_hit(len(all_obj_locs[::STEP]), fps/STEP/DT/10)

def create_video(all_obj_locs, fps=30):
    """
    creating all the frames for the video by the locations of the ball and pins
    :param all_obj_locs: locations (x,y) of the ball and the pins
    :param fps: frames per second of video
    :return: none
    """
    i = 0
    print(len(all_obj_locs[::STEP]))
    for f in all_obj_locs[::STEP]:
        plt.figure(figsize=(SIZE * 2, SIZE), dpi=80)
        plt.ylim([-LANE_LENGTH/4+25, LANE_LENGTH/4+75])
        plt.xlim([-50, LANE_LENGTH + 50])
        x_s = [p[1] for p in f]
        y_s = [p[0] for p in f]
        s = 10
        plt.plot([0, 0], [0,LANE_WIDTH], color="red")
        plt.plot([LANE_LENGTH, LANE_LENGTH], [0,LANE_WIDTH], color="red")
        plt.plot([0, LANE_LENGTH], [0, 0], color="red")
        plt.plot([0, LANE_LENGTH], [LANE_WIDTH, LANE_WIDTH], color="red")
        plt.scatter(x_s, y_s, s=s)
        plt.savefig("data/frame" + str(i) + ".png")
        plt.close()
        # plt.show()
        i += 1
    create_video_from_frames(len(all_obj_locs[::STEP]), fps/STEP/DT)


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
    if ball_stats[3] <= 0:  # if vy = vx = 0 we should stop
        return False

    if ball_stats[0] > 41 * 2.54 or ball_stats[0] < 0:  # checking if we are out of the lane
        return False
    pins_loc = ORIG_PINS_LOC.copy()
    for p in pins_loc:
        if dist((ball_stats[0], ball_stats[1]), p) < R_BALL + R_PIN:  # checking if we hit one of the balls
            return False
    return True


def calc_throw_dt(ball_stats, sliding_change_x, sliding_change_y):
    """
    calculating the next location of the ball
    :param ball_stats: the ball stats (x,y,vx,vy,wx,wy)
    :return: the new ball stats
    """
    # ball stats: [0] x=x1, [1] y=x2, [2] vx=x3, [3] vy=x4, [4] wx=x5, [5] wy=x6
    Fx1, Fx2 = -OILED_MU * BALL_M * G * np.sign(ball_stats[2:4])
    ball_stats = ball_stats + np.array(
        #    x3              x4           Fx1/m         Fx2/m              Fx2*R/I             -Fx1*R/I
        [ball_stats[2] * DT, ball_stats[3] * DT, Fx1 / BALL_M * DT, Fx2 / BALL_M * DT, (Fx2 * R_BALL/100) / BALL_I * DT,
         -Fx1 * (R_BALL/100) / BALL_I * DT]
    )
    if sliding_change_x:
        ball_stats[2] = ball_stats[4]*R_BALL
    if sliding_change_y:
        ball_stats[3] = ball_stats[5]*R_BALL
    return ball_stats



def get_locs(pins_stats, ball_stats):
    """
    getting all the locations of the ball and pins
    :param pins_stats: the pins stats
    :param ball_stats: the ball stats
    :return: the ball and pins locations
    """
    return np.vstack((pins_stats[:, :2], ball_stats[:2]))


def calc_score(pins_stats):
    """
    calculating the score we have after the throw
    :param pins_stats:
    :return:
    """
    count = 0
    new = pins_stats[:,:2]-ORIG_PINS_LOC
    for p in new:
        if np.linalg.norm(p) > R_PIN/2:
            count+=1
    return count


def throw_ended(ball_stats, pins_stats):
    """
    checking if we need to stop (ball out of lane/stopped or all the pins stopped)
    :param ball_stats:
    :param pins_stats:
    :return:
    """
    # ball stats: [0] x, [1] y, [2] vx, [3] vy, [4] wx, [5] wy
    # pin stats: [0] x, [1] y, [2] z [3] vx, [4] vy, [5] vz
    ball_stopped = (np.isclose(ball_stats[2], 0) and np.isclose(ball_stats[3], 0)) or ball_stats[0] < 0 or ball_stats[0] > LANE_WIDTH or \
                   ball_stats[1] > LANE_LENGTH or ball_stats[1] < 0
    pins_stopped = all(np.isclose(p[3], 0) and np.isclose(p[4], 0) and np.isclose(p[5], 0) or p[0] < 0 or p[0] > LANE_WIDTH or \
                   p[1] > LANE_LENGTH or p[1] < 0 for p in pins_stats)  # p[3:6] = vx, vy, vz
    return ball_stopped and pins_stopped


@memoize
def simulate_throw(x, vx, vy, wx, wy, show_video=False, ball_locs_return=False):
    v0x = vx
    v0y = vy
    w0x = wx
    w0y = wy
    ball_locs = []
    ball_stats = np.array([x, 0, vx, vy, wx, wy])
    sliding_change_x = False
    sliding_change_y = False
    count = 0
    print((BALL_I*(v0x/100) + (w0x/100)*R_BALL/100))
    print((OILED_MU*G*BALL_I+BALL_M*G*OILED_MU*(R_BALL/100)**2))
    print((BALL_I*(v0x/100) + (w0x/100)*R_BALL/100)/(OILED_MU*G*BALL_I+BALL_M*G*OILED_MU*(R_BALL/100)**2))
    while still_going(ball_stats):
        count += 1
        if (BALL_I*(v0x/100) + (w0x/100)*R_BALL/100)/(OILED_MU*G*BALL_I+BALL_M*G*OILED_MU*(R_BALL/100)**2) <= count*DT:
            sliding_change_x = True
        if (BALL_I*v0y + w0y*R_BALL)/(OILED_MU*G*BALL_I+BALL_M*G*OILED_MU*(R_BALL/10)**2) <= count*DT:
            sliding_change_y = True
        ball_stats = calc_throw_dt(ball_stats, sliding_change_x, sliding_change_y)
        ball_locs.append(ball_stats[:2])
    print("time:" + str(count*DT))
    if show_video:
        create_video([[p] for p in ball_locs], 15)
    if ball_locs_return:
        return ball_locs, ball_stats
    return ball_stats


def init_pins():
    pins_loc = ORIG_PINS_LOC.copy()
    pins = np.hstack((pins_loc, R_PIN * np.ones((10, 1)), np.zeros((10, 3))))
    return pins


def calc_change_pinpin_velocity(pin1, pin2):
    # pin stats: [0] x, [1] y, [2] z [3] vx, [4] vy, [5] vz

    x_tilde = pin1[:3] - pin2[:3]
    y_tilde = np.array([x_tilde[2], 0, -x_tilde[0]])
    z_tilde = np.array([-x_tilde[0] * x_tilde[1], x_tilde[2] ** 2 + x_tilde[0] ** 2, -x_tilde[2] * x_tilde[1]])

    x_tilde = x_tilde * 1 / np.linalg.norm(x_tilde)
    y_tilde = y_tilde * 1 / np.linalg.norm(y_tilde)
    z_tilde = z_tilde * 1 / np.linalg.norm(z_tilde)

    A = np.vstack((x_tilde, y_tilde, z_tilde)).T
    A = np.vstack((A, np.zeros(3)))
    A = np.hstack((A, np.array([0, 0, 0, 1]).reshape(-1, 1)))

    v_tilde_pin1 = A @ np.hstack((pin1[3:], 1))
    v_tilde_pin2 = A @ np.hstack((pin2[3:], 1))

    v_tilde_pin1[0], v_tilde_pin2[0] = v_tilde_pin2[0], v_tilde_pin1[0]

    inv_A = np.linalg.inv(A)

    v_pin1 = inv_A @ v_tilde_pin1
    v_pin2 = inv_A @ v_tilde_pin2

    pin1[3:] = v_pin1[:3]
    pin2[3:] = v_pin2[:3]

    return pin1, pin2


def calc_change_ballpin_velocity(ball, pin):
    # ball stats: [0] x, [1] y, [2] vx, [3] vy, [4] wx, [5] wy
    # pin stats: [0] x, [1] y, [2] z [3] vx, [4] vy, [5] vz

    x_tilde = np.hstack((ball[:2], R_BALL)) - pin[:3]
    y_tilde = np.array([x_tilde[2], 0, -x_tilde[0]])
    z_tilde = np.array([-x_tilde[0] * x_tilde[1], x_tilde[2] ** 2 + x_tilde[0] ** 2, -x_tilde[2] * x_tilde[1]])

    x_tilde = x_tilde * 1 / np.linalg.norm(x_tilde)
    y_tilde = y_tilde * 1 / np.linalg.norm(y_tilde)
    z_tilde = z_tilde * 1 / np.linalg.norm(z_tilde)

    A = np.vstack((x_tilde, y_tilde, z_tilde)).T
    A = np.vstack((A, np.zeros(3)))
    A = np.hstack((A, np.array([0, 0, 0, 1]).reshape(-1, 1)))

    v_tilde_ball = A @ np.hstack((ball[2:4], 0, 1))
    v_tilde_pin = A @ np.hstack((pin[3:], 1))

    v_tilde_ball[0] = (v_tilde_ball[0]*(BALL_M - PIN_M) + 2 * v_tilde_pin[0] * PIN_M) / (BALL_M + PIN_M)
    v_tilde_pin[0] = (v_tilde_pin[0]*(PIN_M - BALL_M) + 2 * v_tilde_ball[0] * BALL_M) / (BALL_M + PIN_M)

    inv_A = np.linalg.inv(A)

    v_ball = inv_A @ v_tilde_ball
    v_pin = inv_A @ v_tilde_pin

    ball[2:4] = v_ball[:2]
    pin[3:] = v_pin[:3]

    return ball, pin


def calc_hits_dt(ball_stats, pins_stats):
    ball_stats = calc_throw_dt(ball_stats, False, False)

    # ball stats: [0] x, [1] y, [2] vx, [3] vy, [4] wx, [5] wy
    # pin stats: [0] x, [1] y, [2] z [3] vx, [4] vy, [5] vz
    for i in range(pins_stats.shape[0]):
        if pins_stats[i, 2] > R_PIN:
            pins_stats[i, :] = pins_stats[i, :] + np.array(
                [pins_stats[i, 3] * DT, pins_stats[i, 4] * DT, -pins_stats[i, 5] * DT, 0, 0, G * DT]
            )
        else:
            Fx1, Fx2 = -OILED_MU * PIN_M * G * np.sign(pins_stats[i, 3:5])
            pins_stats[i, :] = pins_stats[i, :] + np.array(
                [pins_stats[i, 3] * DT, pins_stats[i, 4] * DT, -pins_stats[i, 5] * DT,
                 0, 0, 0]
            )
    pins_stats[pins_stats[:, 2] < R_PIN][:, 5] = 0
    pins_stats[pins_stats[:, 2] < R_PIN][:, 2] = R_PIN

    for i in range(pins_stats.shape[0]):
        for j in range(i + 1, pins_stats.shape[0]):
            d = np.linalg.norm(pins_stats[i, :3] - pins_stats[j, :3])
            if d > 2 * R_PIN:
                continue
            pins_stats[i], pins_stats[j] = calc_change_pinpin_velocity(pins_stats[i], pins_stats[j])
        if np.linalg.norm(pins_stats[i, :3] - np.array([ball_stats[0], ball_stats[1], R_BALL])) > R_PIN + R_BALL:
            continue
        ball_stats, pins_stats[i] = calc_change_ballpin_velocity(ball_stats, pins_stats[i])

    return ball_stats, pins_stats


@memoize
def simulate_hits(x, y, vx, vy, wx, wy, show_video=False):
    all_obj_locs = []
    ball_stats = np.array([x, y, vx, vy, wx, wy])
    pins_stats = init_pins()
    while not throw_ended(ball_stats, pins_stats):
        ball_stats, pins_stats = calc_hits_dt(ball_stats, pins_stats)
        all_obj_locs.append(get_locs(pins_stats, ball_stats))

    if show_video:
        create_video_hit(all_obj_locs)
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
    return [ERR_RT * i for i in range(int((1 / ERR_RT) / 4))]  # error up to 25%


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
    scores = np.zeros((len(error_rates), throw_num_per_error_rate))
    # for i, error_rate in enumerate(error_rates):
    #     for j in range(throw_num_per_error_rate):
    x, vx, vy, wx, wy = get_random_throwing_parameters(error_rates[0])  # y default is 0
    x, y, vx, vy, wx, wy = simulate_throw(x, vx, vy, wx, wy, True)
    score = simulate_hits(x, y, vx, vy, wx, wy, True)
    print(score)
    # scores[i, j] = score
    # avg_hits = np.average(scores, axis=1)
    # plot_graph(error_rates, avg_hits)


if __name__ == '__main__':
    main()
