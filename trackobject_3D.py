import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from get_depth import *
from numpy.linalg import inv, pinv
import imutils
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# initial max
mtx_P_l = np.load("/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/projection_matrix_l.npy")
mtx_P_r = np.load("/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/projection_matrix_r.npy")
rect_map_left_x = np.load(r'/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/map_l_x.npy')
rect_map_left_y = np.load(r'/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/map_l_y.npy')
rect_map_right_x = np.load(r'/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/map_r_x.npy')
rect_map_right_y = np.load(r'/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/map_r_y.npy')
mtx_l = np.load(r'/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/mtx_l.npy')
mtx_r = np.load(r'/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/mtx_r.npy')
mtx_Q = np.load(r'/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/mtx_Q.npy')
mtx_T = np.load(r'/home/karl/DTU/Perception/git/31392-Percepetion/Matrix/mtx_T.npy')

# LOAD IMAGES
dataset = '/home/karl/DTU/Perception/images'
images_left = glob.glob(dataset + '/left/*.png')
images_right = glob.glob(dataset + '/right/*.png')
assert images_right, images_left
assert (len(images_right) == len(images_left))
images_right.sort()
images_left.sort()

# The initial karman_filter (6x1).
def karman_init(start_point):

    # initiale state of the point
    x = np.array([[start_point[0]],
                  [-0.15],
                  [start_point[1]],
                  [0.04],
                  [start_point[2]],
                  [-0.1]])

    # The measurement uncertainty.
    R = 10
    # The initial uncertainty (6x6).

    P = np.diagflat([[R],
                     [R],
                     [R],
                     [R],
                     [R],
                     [R]])

    # The external motion (6x1).
    u = np.zeros((6, 1))

    # The transition matrix (6x6).
    F = np.array([[1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1]])

    # The observation matrix (2x6).
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])

    # The identity matrix. Simply a matrix with 1 in the diagonal and 0 elsewhere.
    I = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    return x, P, u, F, H, R


def update(x, P, Z, H, R):
    # Insert update function
    y = Z - np.dot(H, x)
    S = np.dot(np.dot(H, P), np.transpose(H)) + R
    K = np.dot(np.dot(P, np.transpose(H)), np.linalg.pinv(S))
    return [x + np.dot(K, y), np.dot((np.eye(x.shape[0]) - np.dot(K, H)), P)]


def predict(x, P, F, u):
    # insert predict function
    N_X = np.dot(F, x) + u
    N_P = np.dot(np.dot(F, P), np.transpose(F))

    return [N_X, N_P]


def getbeltmask(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_blue = np.array([105, 40, 95])
    up_blue = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, low_blue, up_blue)
    kernel = np.ones((5, 5), np.uint8)
    beltmask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    beltmask = cv2.bitwise_not(beltmask)

    return beltmask


def getobjectmask(frame, bgs):
    frontMask = bgs.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    frontMask = cv2.dilate(frontMask, kernel, iterations=1)  # threshold
    # out_mask = np.zeros_like(frontMask)
    return frontMask


def findobjectcontours(frame, belt_mask, bgmog):
    result = cv2.bitwise_and(frame, frame, mask=belt_mask)
    obj_mask = bgmog.apply(result)
    kernel = np.ones((3, 3), np.uint8)
    obj_mask = cv2.dilate(obj_mask, kernel, iterations=1)
    obj_mask = cv2.erode(obj_mask, kernel, iterations=1)
    # obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(obj_mask, 2, 1)

    return contours


def triangulate(p1, p2, mtx1, mtx2, T):
    # project the feature points to 3D with triangulation

    # projection matrix for Left and Right Image
    M_left = mtx1.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    M_rght = mtx2.dot(np.hstack((np.eye(3), T)))

    p1_flip = np.vstack((p1.T, np.ones((1, p1.shape[0]))))
    p2_flip = np.vstack((p2.T, np.ones((1, p2.shape[0]))))

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])
    # P = cv2.triangulatePoints(mtx_P_l, mtx_P_r, p1_flip[:2], p2_flip[:2])

    # Normalize homogeneous coordinates (P->Nx4  [N,4] is the normalizer/scale)
    P = P / P[3]
    land_points = P[:3]

    return land_points.T


def init_triangulate(mtx_P_left, mtx_P_right, p1, p2):
    center3D_homogeneous = cv2.triangulatePoints(mtx_P_left, mtx_P_right, p1, p2)
    center3D = cv2.transpose(center3D_homogeneous)
    center3D = cv2.convertPointsFromHomogeneous(center3D).squeeze()

    return center3D


def detect_obj(frame, bgMOG, flag=False):
    # get belt mask
    belt_mask = getbeltmask(frame)
    contours = findobjectcontours(frame, belt_mask, bgMOG)

    pts = np.array([[[387, 476]], [[464, 696]], [[1217, 359]], [[1131, 260]]])
    proj_point = []

    if contours:
        c = max(contours, key=cv2.contourArea)
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)  # find the minicircle around the object
        maxx = -1
        for point in c:
            if point[0][0] > maxx:
                maxx = point[0][0]
                proj_point = point[0]
        if radius > 30 and radius < 200:
            if flag:
                return True, center_x, center_y, radius, proj_point
            if cv2.pointPolygonTest(pts, (center_x, center_y), False) > 0:
                return True, center_x, center_y, radius, proj_point
    return False, 0, 0, 0, proj_point

def detect_obj2(frame, bgKNN):
    # apply the background subtractor to the current frame
    mask_fg = bgKNN.apply(frame)

    # get blue mask that characterizes the conveyor belt
    mask_blue = getbeltmask(frame)

    # mask to remove hands
    mask_belt_x = np.zeros((720, 1280), dtype='uint8')
    mask_belt_x[:, 400:1230] = 255

    # combine blue mask, background subtractor and hands mask
    mask_fg = cv2.bitwise_and(mask_fg, mask_belt_x)
    mask_fg = cv2.bitwise_and(mask_fg, mask_blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=2)

    # find the contours
    cnts = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        # get contour with the highest area
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 2000:
            ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)
            # print(center_x)
            # print(center_y)
            return True, center_x, center_y, radius, mask_fg
    return False, 0, 0, 0, mask_fg


# initialize background subtractor
back_frame_l = cv2.imread(images_left[0])
back_frame_l = cv2.remap(back_frame_l, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
back_frame_r = cv2.imread(images_right[0])
back_frame_r = cv2.remap(back_frame_r, rect_map_right_x, rect_map_right_y, cv2.INTER_LINEAR)

# bgMOG = cv2.createBackgroundSubtractorMOG2(30, 16,
#                                            False)  # createBackgroundSubtractorMOG2 dectect the moving object
# _ = bgMOG.apply(back_frame_l)

bgMOG2 = cv2.createBackgroundSubtractorMOG2(70, 16,
                                            False)  # createBackgroundSubtractorMOG2 dectect the moving object
_ = bgMOG2.apply(back_frame_r)
fgbg = cv2.createBackgroundSubtractorKNN(history=600, dist2Threshold=800, detectShadows=False)

# initialize object count and status
# initialize setting
initialized = False
obj_count = 0
obj_track = False  # (was it possible to localize the object on the scene?)
obj_dectect_l = False
obj_dectect_r = False
obj_picture = np.zeros((1, 1, 3), dtype="uint8")
pre_center = []
start, end = 10, -1
disp = 0

model = tf.keras.models.load_model('my_model')
clasificationtracker =[0, 0, 0]
# %% TRACKING AND CLASSIFICATION
out = cv2.VideoWriter('Track_3d_final.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (1280, 720))

for i, (imgL, imgR) in enumerate(zip(images_left[start:end], images_right[start:end])):

    # grab current frame
    frame = cv2.imread(imgL)
    frame = cv2.remap(frame, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
    left_img = frame.copy()
    # undistort and rectify left and right image
    frame_r = cv2.imread(imgR)
    right_img = cv2.remap(frame_r, rect_map_right_x, rect_map_right_y, cv2.INTER_LINEAR)

    # detect the object
    obj_dectect_l, center_x_l, center_y_l, radius_l, mask_fg = detect_obj2(left_img, fgbg)
    obj_dectect_r, center_x_r, center_y_r, radius_r, rep_point_r = detect_obj(right_img, bgMOG2)

    # obj_dectect_l, center_x_l, center_y_l, radius_l, rep_point = detect_obj(left_img, bgMOG)
    # obj_dectect_r, center_x_r, center_y_r, radius_r = detect_obj2(right_img, bgs)
    #
    # print(center_x_l)
    # print(obj_dectect_r)

    if obj_dectect_l and center_x_l < 1210:


        # print(pre_center)
        # center_x_r = center_x_l - 100
        # center_y_r = center_y_l
        currect_center = np.array([[center_x_l], [center_y_l]])
        # triangulate point
        P = triangulate(currect_center,
                        np.array([[center_x_r], [center_y_r]]), mtx_l, mtx_r, mtx_T)
        point3D = P[0]  # choose the positive z


        text = "Position: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(point3D[0] / 10), float(point3D[1] / 10),
                                                                        float(point3D[2]) / 10)
        cv2.putText(frame, 'Detected', (10, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, 5)
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, 5)

        if center_x_l > 1000 and center_x_l < 1170 and not initialized:
            # """
            print('start track')
            disparity = get_disparity(left_img, right_img, int(center_x_l), int(center_y_l))
            disp = (disparity if disparity > 85 and disparity < 150 else 96)
            # print('disp')
            # print(disp)
            center_3D_init = init_triangulate(mtx_P_l, mtx_P_r, np.array([[center_x_l], [center_y_l]]),
                                             np.array([[center_x_l - disp], [center_y_l]]))
            # print(center_3D_init)
            k_x, k_P, k_u, k_F, k_H, k_R = karman_init(center_3D_init)
            initialized = True
            cv2.putText(frame, 'First dectect', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, 5)

        if initialized and center_x_l != 0 and center_y_l != 0:
            # update/predict kalman
            obj_track = True
            # k_measurement = np.array([[point3D[0]], [point3D[1]], [point3D[2]]])
            # [km_x, km_P] = update(km_x, km_P, km_measurement, km_F, km_u)
            # cv2.circle(frame_r, (int(center_x_l - disp), int(center_y_l)), 50, (0, 0, 255), 2)  # cirle
            # cv2.circle(frame_r, (int(center_x_l - disp), int(center_y_l)), 5, (0, 0, 255), -1)  # center
            [k_x, k_P] = predict(k_x, k_P, k_F, k_u)

            # __ CLASSIFICATION __
            # extract object from current frame
            h, w = cv2.imread(images_left[0]).shape[:2]  # size of the images (pixels)
            mask_roi = np.zeros((h, w), dtype='uint8')
            p1 = (int(center_x_l - 300 / 2), int(center_y_l - 200 / 2))
            p2 = (int(center_x_l + 300 / 2), int(center_y_l + 200 / 2))
            # print(p1, p2)
            mask_roi[p1[1]:p2[1], p1[0]:p2[0]] = 255
            mask_obj = cv2.bitwise_and(mask_roi, mask_fg)
            obj_picture = cv2.bitwise_or(frame, frame)
            #obj_picture = cv2.bitwise_or(frame, frame, mask=mask_obj)
            obj_picture = obj_picture[p1[1]:p2[1], p1[0]:p2[0]]

            obj_picture = cv2.resize(obj_picture, (100, 100))
            obj_pictureGray = cv2.cvtColor(obj_picture, cv2.COLOR_BGR2GRAY)
            #plt.imshow(obj_picture)
            #plt.show()
            #cv2.imshow('obj', obj_pictureGray)
            obj_pictureGray = obj_pictureGray[np.newaxis, :, :,np.newaxis]
            classification = model.predict(obj_pictureGray)
            if point3D[0] / 10 < 0:
                if classification[0][2] == 1.0:
                    clasificationtracker[2] = clasificationtracker[2]+1
                if classification[0][1] == 1.0:
                    clasificationtracker[1] = clasificationtracker[1] + 1
                if classification[0][0] == 1.0:
                    clasificationtracker[0] = clasificationtracker[0] + 1
            # classify object
            # obj_type = myClassifier.detectAndClassify(obj_picture)

            # vis_obj_height = 100
            # vis_object_y = 40 + 25
            # vis_object_x = 20
            # vis_obj_picture = imutils.resize(obj_picture, height=vis_obj_height)
            # frame[vis_object_y:vis_object_y + vis_obj_height,
            # vis_object_x:(vis_object_x + vis_obj_picture.shape[1])] = vis_obj_picture
        cv2.circle(frame, (int(center_x_l), int(center_y_l)), int(radius_l), (0, 0, 255), 2)  # cirle
        cv2.circle(frame, (int(center_x_l), int(center_y_l)), 5, (0, 0, 255), -1)  # center
    # if object reaches the end of the conveyor:
    if obj_track and center_x_l <= 450 and center_y_l > 580:
        # prepare for next object (reset status and roi)
        obj_track = False
        initialized = False
        obj_dectect_l = False
        obj_dectect_r = False
        disp = 0
        cv2.putText(frame, 'Object out', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2, 5)
        clasificationtracker = [0, 0, 0]
        # k_x, k_P, k_u, k_F, k_H, k_R = karman_init([0, 0, 0])

    if obj_track and not obj_dectect_l:

        [k_x, k_P] = predict(k_x, k_P, k_F, k_u)
        center_repr = np.dot(mtx_P_l, np.vstack((np.array([k_x[0], k_x[2], k_x[4]]), 1)))
        center_repr = center_repr / center_repr[2]
        center_pred = np.array([int(center_repr[0]), int(center_repr[1])])
        pre_center = center_pred

        normal_disp = disp
        pred_center_r = np.array([center_pred[0] - normal_disp, center_pred[1]])
        cv2.circle(frame, center_pred, 10, (0, 255, 255), -1)
        cv2.circle(frame, center_pred, 50, (0, 255, 255), 2)  # cirle
        # cv2.circle(frame_r, (int(center_pred[0] - normal_disp/2), center_pred[1]), 10, (0, 255, 255), -1)
        # cv2.circle(frame_r, (int(center_pred[0] - normal_disp/2), center_pred[1]), 50, (0, 255, 255), 2)  # cirle

        P = triangulate(center_pred, pred_center_r, mtx_l, mtx_r, mtx_T)
        point3D = P[0]  # choose the positive z

        text2 = "pred_Position: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(point3D[0] / 10),
                                                                              float(point3D[1] / 10),
                                                                              float(point3D[2]) / 10)
        cv2.putText(frame, 'Behind the occlusion', (10, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, 5)
        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, 5)


    # take max classification and print it
    best_guess = np.argmax(clasificationtracker)
    count = clasificationtracker.count(0)
    if count < 3:
        if best_guess == 2:
            text3 = "Classification: Coffee Cup"
        if best_guess == 1:
            text3 = "Classification: Box"
        if best_guess == 0:
            text3 = "Classification: Book"
        cv2.putText(frame, text3, (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, 5)

    # Show the frame
    Final_frame = frame
    # cv2.imshow('Final_frame', cv2.resize(Final_frame, None, fx=0.9, fy=0.9))
    cv2.imshow('Final_video', Final_frame)
    # cv2.imshow('obj_picture', obj_picture)
    # frame_left_right = np.hstack((frame, frame_r))
    # show final result
    # scale = 0.6
    # cv2.imshow('Final Project', cv2.resize(frame_left_right, None, fx=scale, fy=scale))
    out.write(Final_frame)

    cv2.waitKey(50)
cv2.destroyAllWindows()
out.release()
