import numpy as np
import cv2
import glob
import imutils
import math
import random

#cap = cv2.VideoCapture('/home/karl/DTU/Perception/Stereo_conveyor_without_occlusions.mp4')

images_left = glob.glob(r'C:\Users\kosta\PycharmProjects\PerceptionProject\31392-Percepetion\Stereo_conveyor_without_occlusions/left/*.png')
images_right = glob.glob(r'C:\Users\kosta\PycharmProjects\PerceptionProject\31392-Percepetion\Stereo_conveyor_without_occlusions/right/*.png')


assert images_right, images_left
assert (len(images_right) == len(images_left))
n_images = len(images_right)
images_right.sort()
images_left.sort()


def getBG(images):
    frame_indices=random.sample(range(1, len(images)), 20)
    #frame_indices = len(images) * np.random.uniform(size=50)
    frames = []
    for i in frame_indices:
        # set the frame id to read that particular frame
        frames.append(cv2.imread(images_left[i]))
    # calculate the median
    print(type(images[0]))
    median_frame = np.median(frames, axis=0).astype(np.uint8)

    return median_frame

def getBlueMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([105,40,95]), np.array([120,255,255]))
    kernel = np.ones((5,5),np.uint8)
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_morph = cv2.bitwise_not(mask_morph)
    return mask_morph

def getRectangleCenter(p1,p2):
	x = int(p1[0] + (p2[0] - p1[0])/2)
	y = int(p1[1] + (p2[1] - p1[1])/2)
	return (x,y)

def getPointsFromCenter(c,w,h):
	p1 = (int(c[0]-w/2),int(c[1]-h/2))
	p2 = (int(c[0]+w/2),int(c[1]+h/2))
	return (p1,p2)

background=getBG(images_left)
cv2.imshow('Detected Objects', getBG(images_left))
# convert the background model to grayscale format
backgroundgray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

h, w = cv2.imread(images_left[0]).shape[:2] # size of the images (pixels)
print(h)
out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (2560,720))

mask_belt_x = np.zeros((h,w),dtype='uint8')
mask_belt_x[:,400:1150] = 255

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

for i in range(100, n_images):

    # grab current frame
    frame = cv2.imread(images_left[i])

    orig_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find the difference between current frame and base frame
    frame_diff = cv2.absdiff(gray, backgroundgray)
    np.multiply(frame_diff, 5)
    frame_diff = cv2.bitwise_and(frame_diff, mask_belt_x)
    # thresholding to convert the frame to binary
    ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
    # dilate the frame a bit to get some more white area...
    # ... makes the detection of contours a bit easier
    dilate_frame = cv2.dilate(thres, None, iterations=2)
    dilate_frame = cv2.morphologyEx(dilate_frame, cv2.MORPH_CLOSE, kernel)
    #dilate_frame = cv2.Canny(dilate_frame, h, w)
    # find the contours around the white segmented areas
    contours, hierarchy = cv2.findContours(dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw the contours, not strictly necessary
    for i, cnt in enumerate(contours):
        cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
    for contour in contours:
        # continue through the loop if contour area is less than 2500...
        # ... helps in removing noise detection
        if cv2.contourArea(contour) < 2500:
            continue
        # get the xmin, ymin, width, and height coordinates from the contours
        (x, y, w, h) = cv2.boundingRect(contour)
        # draw the bounding boxes
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detected Objects', dilate_frame)
    out.write(orig_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()










# if (cap.isOpened()== False):
#     print("Error opening video stream or file")
#
#
# while 1:
#     ret, frame = cap.read()
#     if ret == 0:
#         print("no frame")
#         break
#
#
#     cv2.imshow('frame', frame)
#     cv2.waitKey(20)
#
# cap.release()

