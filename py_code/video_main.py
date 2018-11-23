import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import numpy as np
from pipeline import FindLane
from time import sleep


ym_per_pix = 30/720
xm_per_pix = 3.7/700
find_lane = FindLane(ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix)
cap = cv2.VideoCapture('../test_videos/project_video.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
print(fps, width, height)
out = cv2.VideoWriter('../test_videos_output/project_video.mp4', fourcc, int(fps), (int(width), int(height)))
end_flag, frame = cap.read()
ESC_KEY = 27
count = 0

src = np.float32([[685, 450], [1120, 720], [190, 720], [595, 450]])
dst = np.float32([[900, 0], [900, 720], [320, 720], [320, 0]])
vertices = np.array([[(300,height) ,(460, 450), (750, 450), (1000, height)]], dtype=np.int32)
vertices = np.array([[(0,height) ,(0, 0), (1200, 0), (1200, height)]], dtype=np.int32)
# # challenge video exculsive use
# src = np.float32([[730, 480], [1050, 730], [265, 720], [630, 480]])
# dst = np.float32([[900, 100], [900, 720], [320, 720], [320, 100]])
# rows = height
# vertices = np.array([[(200,rows) ,(650, 430), (750, 430), (1130, rows)]], dtype=np.int32)


warped_size = (int(width), int(height))
find_lane.createColorGrad('BGR')
find_lane.createWarp(src, dst, warped_size, vertices)
while end_flag == True:
    out_img = find_lane.pipeline(frame)
    #find_lane.left_fit = None
    # cv2.imshow('A', out_img)
    out.write(out_img)
    key = cv2.waitKey(3)
    if key == ESC_KEY:
        break
    # sleep(0.1)
    end_flag, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
