import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('/home/karl/DTU/Perception/Stereo_conveyor_without_occlusions.mp4')

if (cap.isOpened()== False):
    print("Error opening video stream or file")


while 1:
    ret, frame = cap.read()
    if ret == 0:
        print("no frame")
        break


    cv2.imshow('frame', frame)
    cv2.waitKey(20)

cap.release()

