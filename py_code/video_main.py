"""
This is a program to recognize lane from video.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import numpy as np
from find_lane import FindLane
from time import sleep
import argparse

__author__ = 'ryutaShitomi'
__version__ = '1.0'
__date__ = '2018/10'



parser = argparse.ArgumentParser()
parser.add_argument('-p',"--path",
                    default=os.path.join(os.pardir, 'test_videos', 'harder_challenge_video.mp4'),
                    help="video path")
parser.add_argument('-o', '--output',
                    default=os.path.join(os.pardir, 'test_videos_output', 'output.mp4'),
                    help='output path')
args = parser.parse_args()

ym_per_pix = 30/720
xm_per_pix = 3.7/700
# Create the class pipeline.FindLane.
find_lane = FindLane(ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)
video_path = args.path
cap = cv2.VideoCapture(video_path)
# get the image width, height and fps.
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
# Specify extension of video to save.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
print(fps, width, height)
# output_path = '../test_videos_output/analyze_' + os.path.basename(video_path)
output_path = args.output
out = cv2.VideoWriter(output_path, fourcc, int(fps), (int(width), int(height)))
end_flag, frame = cap.read()
ESC_KEY = 27
count = 0

src = np.float32([[685, 450], [1120, 720], [190, 720], [595, 450]])
dst = np.float32([[900, 0], [900, 720], [320, 720], [320, 0]])
vertices = np.array([[(300,height) ,(460, 450), (750, 450), (1000, height)]], dtype=np.int32)
vertices = np.array([[(0,height) ,(0, 0), (1200, 0), (1200, height)]], dtype=np.int32)
## for challenge video  ##
# src = np.float32([[730, 480], [1050, 730], [265, 720], [630, 480]])
# dst = np.float32([[900, 100], [900, 720], [320, 720], [320, 100]])
# rows = height
# vertices = np.array([[(200,rows) ,(650, 430), (750, 430), (1130, rows)]], dtype=np.int32)
##########################

## for harder_challenge ##
src = np.float32([[730, 500], [1030, 720], [190, 720], [525, 500]])
dst = np.float32([[900, 0], [900, 720], [320, 720], [320, 0]])
vertices = np.array([[(300,height) ,(460, 450), (750, 450), (1000, height)]], dtype=np.int32)
vertices = np.array([[(0,height) ,(0, 0), (1200, 0), (1200, height)]], dtype=np.int32)
##########################



warped_size = (int(width), int(height))
find_lane.createColorGrad('BGR')
find_lane.createWarp(src, dst, warped_size, vertices)


while end_flag == True:
    out_img = find_lane.pipeline(frame, True)
    ## If you apply sliding window each frame, restore below comment ##
    # find_lane.left_fit = None

    cv2.imshow('A', out_img)
    out.write(out_img)
    key = cv2.waitKey(1)
    if key == ESC_KEY:
        break
    # sleep(0.1)
    end_flag, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
