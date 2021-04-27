from math import dist

import matplotlib.pyplot as plt
import numpy as np
from bowling_api import *
import cv2

LANE_LENGTH = 1828.8
LANE_WIDTH = 105.41
SIZE = 6
BOWLING_RADIUS = 10.795
BOWLING_SIZE = np.pi*BOWLING_RADIUS**2


def create_video(all_obj_locs):
    i = 0
    for f in all_obj_locs:
        plt.figure(figsize=(SIZE*2, SIZE), dpi=80)
        plt.xlim([-5, LANE_LENGTH])
        plt.ylim([-5, LANE_WIDTH])
        x_s = [p[1] for p in f]
        y_s = [p[0] for p in f]
        plt.scatter(x_s, y_s,s=100)
        plt.savefig("data/frame" + str(i)+".png")
        plt.show()
        i += 1

locs, s = simulate_throw(20.5*2.54, 0, 8, 0, 100, False, True)
print(len(locs[::10]))
# create_video([[p] for p in locs[::10]])
