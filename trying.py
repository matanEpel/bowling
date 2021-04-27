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


# def create_video(all_obj_locs):
#     i = 0
#     for f in all_obj_locs:
#         plt.figure(figsize=(SIZE*2, SIZE), dpi=80)
#         plt.xlim([-5, LANE_LENGTH])
#         plt.ylim([-5, LANE_WIDTH])
#         x_s = [p[1] for p in f]
#         y_s = [p[0] for p in f]
#         plt.scatter(x_s, y_s,s=100)
#         plt.savefig("data/frame" + str(i)+".png")
#         plt.show()
#         i += 1

locs, s = simulate_throw(20.5*2.54, 0, 800, 0, 100, False, True)

frames_to_delete = 100
print(len(locs[::frames_to_delete]))
create_video([[p] for p in locs[::frames_to_delete]], int(1/(DT*frames_to_delete)))

import numpy as np
import glob

# def create_video_from_frames(amount_of_frames):
#     img_array = []
#     for i in range(amount_of_frames):
#         img = cv2.imread('data/frame' + str(i) + '.png')
#         height, width, layers = img.shape
#         size = (width, height)
#         img_array.append(img)
#
#     out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (960, 480))
#
#     for i in range(len(img_array)):
#         out.write(img_array[i])
#     out.release()
