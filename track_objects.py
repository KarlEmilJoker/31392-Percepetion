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
    frame_indices=random.sample(range(1, len(images)), 50)
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
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

h, w = cv2.imread(images_left[0]).shape[:2] # size of the images (pixels)

out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (2560,720))

mask_belt_x = np.zeros((h,w),dtype='uint8')
mask_belt_x[:,400:1200] = 255

for i in range(n_images):

    # grab current frame
    frame = cv2.imread(images_left[i])

    orig_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find the difference between current frame and base frame
    frame_diff = cv2.absdiff(gray, background)
    frame_diff = cv2.bitwise_and(frame_diff, mask_belt_x)
    # thresholding to convert the frame to binary
    ret, thres = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)
    # dilate the frame a bit to get some more white area...
    # ... makes the detection of contours a bit easier
    dilate_frame = cv2.dilate(thres, None, iterations=2)
    # find the contours around the white segmented areas
    contours, hierarchy = cv2.findContours(dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw the contours, not strictly necessary
    for i, cnt in enumerate(contours):
        cv2.drawContours(frame, contours, i, (0, 0, 255), 3)
    for contour in contours:
        # continue through the loop if contour area is less than 500...
        # ... helps in removing noise detection
        if cv2.contourArea(contour) < 2500:
            continue
        # get the xmin, ymin, width, and height coordinates from the contours
        (x, y, w, h) = cv2.boundingRect(contour)
        # draw the bounding boxes
        cv2.rectangle(orig_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Detected Objects', orig_frame)
    out.write(orig_frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()

# dir_calib = "../31392-Percepetion/calibration_matrix/"
# mtx_P_l = np.load(dir_calib + "projection_matrix_l.npy")
# mtx_P_r = np.load(dir_calib + "projection_matrix_r.npy")
# rect_map_l_x = np.load(dir_calib + "map_l_x.npy")
# rect_map_l_y = np.load(dir_calib + "map_l_y.npy")
# rect_map_r_x = np.load(dir_calib + "map_r_x.npy")
# rect_map_r_y = np.load(dir_calib + "map_r_y.npy")
#
# backSub = cv2.createBackgroundSubtractorKNN()
#
# h, w = cv2.imread(images_left[0]).shape[:2]
# # define previous image
# prev_img = cv2.imread(images_left[0])
# prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
#
# # initialize frame_prev and features_prev
# feat_prev = np.empty((0,1,2),dtype='float32')
# gray_prev = np.zeros((h,w),dtype='uint8')
#
# # define region of the belt
# belt_contour = np.array([[[387,476]],[[464,696]],[[1217,359]],[[1131,260]]])
# belt_x0 = 400 # x start of the conveyor (pixels)
# belt_x1 = 1240 # x end of the conveyor (pixels)
#
# # define kernels
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#
# # initialize object count and status
# obj_count = 0
# object_on_conveyor = None
# obj_present = False # (is there an object on the scene?)
# obj_found = False # (was it possible to localize the object on the scene?)
#
# # initialize object classification counter
# obj_type_hist = {"cup":0,"book":0,"box":0}
#
# # triangulation constants
# template_h = 10
# template_w = 60
#
# # roi within tamplate is being matched
# roi_h = 10
# roi_left_off = -230
# roi_right_off = -30
#
# # %% TRACKING AND CLASSIFICATION
# out = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (2560,720))
#
#
#
# for i in range(n_images):
#
#     # grab current frame
#     frame = cv2.imread(images_left[i])
#     frame_right = cv2.imread(images_right[i])
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
#
#     # undistort and rectify left and right image
#     frame = cv2.remap(frame, rect_map_l_x, rect_map_l_y, cv2.INTER_LINEAR)
#     frame_right = cv2.remap(frame_right, rect_map_r_x, rect_map_r_y, cv2.INTER_LINEAR)
#     gray = cv2.remap(gray, rect_map_l_x, rect_map_l_y, cv2.INTER_LINEAR)
#     gray_right = cv2.remap(gray_right, rect_map_r_x, rect_map_r_y, cv2.INTER_LINEAR)
#
#     # get blue mask that characterizes the conveyor belt
#     mask_blue = getBlueMask(frame)
#
#     mask_fg = cv2.bitwise_and(mask_fg, mask_blue)








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

