import cv2
import numpy as np
import glob
from get_depth import *
from numpy.linalg import inv, pinv
import imutils
from kalman_filter_class import *

# initial max
mtx_P_l = np.load("Matrix/projection_matrix_l.npy")
mtx_P_r = np.load("Matrix/projection_matrix_r.npy")
rect_map_left_x = np.load(r'Matrix\map_l_x.npy')
rect_map_left_y = np.load(r'Matrix\map_l_y.npy')
rect_map_right_x = np.load(r'Matrix\map_r_x.npy')
rect_map_right_y = np.load(r'Matrix\map_r_y.npy')
mtx_l = np.load(r'Matrix\mtx_l.npy')
mtx_r = np.load(r'Matrix\mtx_r.npy')
mtx_Q = np.load(r'Matrix\mtx_Q.npy')
mtx_T = np.load(r'Matrix\mtx_T.npy')

# %% LOAD IMAGES
dataset = 'Stereo_conveyor_with_occlusions'
images_left = glob.glob(dataset + '/left/*.png')
images_right = glob.glob(dataset + '/right/*.png')

assert images_right, images_left
assert (len(images_right) == len(images_left))
# n_images = len(images_right)
images_right.sort()
images_left.sort()

### Initialize Kalman filter ###
# The initial state (6x1).
def karman_init(start_point):
    # initiale state of the point
    x = np.array([[start_point[0]],
                  [-0.15],
                  [start_point[1]],
                  [0.04],
                  [start_point[2]],
                  [-0.1]])
    # x = np.array([[0],
    #               [0],
    #               [0],
    #               [0],
    #               [0],
    #               [0]])

    # The measurement uncertainty.
    R = 10
    # The initial uncertainty (6x6).
    # P = np.array([[10, 0, 0, 0, 0, 0],
    #               [0, 10, 0, 0, 0, 0],
    #               [0, 0, 10, 0, 0, 0],
    #               [0, 0, 0, 10, 0, 0],
    #               [0, 0, 0, 0, 10, 0],
    #               [0, 0, 0, 0, 0, 10]])
    # initial uncertainty
    P = np.diagflat([[R],
                     [R],
                     [R],
                     [R],
                     [R],
                     [R]])

    # The external motion (6x1).
    # u = np.array([[0],
    #               [0],
    #               [0],
    #               [0],
    #               [0],
    #               [0]])
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
    ### Insert update function
    y = Z - np.dot(H, x)
    S = np.dot(np.dot(H, P), np.transpose(H)) + R
    K = np.dot(np.dot(P, np.transpose(H)), np.linalg.pinv(S))
    return [x + np.dot(K, y), np.dot((np.eye(x.shape[0]) - np.dot(K, H)), P)]


def predict(x, P, F, u):
    ### insert predict function
    # X = np.dot(F, x) + u
    # P = np.dot(np.dot(F, P), np.transpose(F))
    # return [X, P]
    # return [np.dot(F, x) + u, np.dot(np.dot(F, P), np.transpose(F))]
    x_p = np.dot(F, x) + u
    P_p = np.dot(np.dot(F, P), np.transpose(F))
    return [x_p, P_p]

def getbeltmask(img):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.bitwise_not(cv2.inRange(hsv, np.array([105, 40, 95]), np.array([120, 255, 255])))

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # low_blue = np.array([105, 40, 95])
    # up_blue = np.array([120, 255, 255])
    # mask = cv2.inRange(hsv, low_blue, up_blue)
    # kernel = np.ones((5, 5), np.uint8)
    # mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # mask_morph = cv2.bitwise_not(mask_morph)
    # return mask_morph

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

    # P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])
    P = cv2.triangulatePoints(mtx_P_l, mtx_P_r, p1_flip[:2], p2_flip[:2])

    # Normalize homogeneous coordinates (P->Nx4  [N,4] is the normalizer/scale)
    P = P / P[3]
    land_points = P[:3]

    return land_points.T
    # return P


def detect_obj(frame, bgMOG, flag=False):
    # get belt mask
    belt_mask = getbeltmask(frame)
    contours = findobjectcontours(frame, belt_mask, bgMOG)

    # belt_area
    # pts = np.array([[380, 485], [1050, 320], [1240, 365], [450, 670]], np.int32)
    # pts = pts.reshape((-1, 1, 2))
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

def getBlueMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_blue = np.array([105, 40, 95])
    up_blue = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, low_blue, up_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask_morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_morph = cv2.bitwise_not(mask_morph)
    return mask_morph

def getRectangleCenter(p1, p2):
    x = int(p1[0] + (p2[0] - p1[0]) / 2)
    y = int(p1[1] + (p2[1] - p1[1]) / 2)
    return (x, y)

def detect_obj2(frame, bgKNN):
    # apply the background subtractor to the current frame
    mask_fg = fgbg.apply(frame)

    # get blue mask that characterizes the conveyor belt
    mask_blue = getBlueMask(frame)
    # mask to remove hands
    mask_belt_x = np.zeros((720,1280), dtype='uint8')
    mask_belt_x[:, 400:1240] = 255

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
            # compute the bounding box for the contour
            (x, y, width, height) = cv2.boundingRect(c)
            point1 = (x, y)
            point2 = (x + width, y + height)
            # compute center of the bounding box
            center_rectangle = getRectangleCenter(point1, point2)
            print(center_rectangle)
            ((center_x, center_y), radius) = cv2.minEnclosingCircle(c)
            # print(center_x)
            # print(center_y)
            return True, float(center_rectangle[0]), float(center_rectangle[1]), radius
    return False, 0, 0, 0

# initialize background subtractor
back_frame_l = cv2.imread(images_left[0])
back_frame_l = cv2.remap(back_frame_l, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
back_frame_r = cv2.imread(images_right[0])
back_frame_r = cv2.remap(back_frame_r, rect_map_right_x, rect_map_right_y, cv2.INTER_LINEAR)
bgMOG = cv2.createBackgroundSubtractorMOG2(30, 16,
                                           False)  # createBackgroundSubtractorMOG2 dectect the moving object
_ = bgMOG.apply(back_frame_l)

bgMOG2 = cv2.createBackgroundSubtractorMOG2(30, 16,
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
pre_center = []
start, end = 80, -1
counter = 0
kalman1 = Kalman()

# %% TRACKING AND CLASSIFICATION
# out = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (2560, 720))

for i, (imgL, imgR) in enumerate(zip(images_left[start:end], images_right[start:end])):

    # grab current frame
    counter += 1
    frame = cv2.imread(imgL)
    frame = cv2.remap(frame, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
    left_img = frame.copy()
    # undistort and rectify left and right image
    frame_r = cv2.imread(imgR)
    right_img = cv2.remap(frame_r, rect_map_right_x, rect_map_right_y, cv2.INTER_LINEAR)

    # detect the object
    obj_dectect_l, center_x_l, center_y_l, radius_l = detect_obj2(left_img, fgbg)
    obj_dectect_r, center_x_r, center_y_r, radius_r, rep_point_r = detect_obj(right_img, bgMOG2)

    # obj_dectect_l, center_x_l, center_y_l, radius_l, rep_point = detect_obj(left_img, bgMOG)
    # obj_dectect_r, center_x_r, center_y_r, radius_r = detect_obj2(right_img, bgs)

    print(center_x_l)
    # print(center_x_r)
    print(obj_dectect_r)
    disp, depth = get_Depth(left_img, right_img, int(center_x_l), int(center_y_l))
    # print(disp)

    if obj_dectect_l:
        cv2.circle(frame, (int(center_x_l), int(center_y_l)), int(radius_l), (0, 0, 255), 2)  # cirle
        cv2.circle(frame, (int(center_x_l), int(center_y_l)), 5, (0, 0, 255), -1)  # center

        # print('pre_center1:')
        # print(pre_center)
        currect_center = np.array([[center_x_l], [center_y_l]])
        P = triangulate(np.array([[center_x_l], [center_y_l]]),
                        np.array([[center_x_r], [center_y_r]]), mtx_l, mtx_r, mtx_T)

        point3D = P[0]
        pre_center = currect_center
        # print(pre_center)
        # triangulate point
        mp_3d_homogeneous = cv2.triangulatePoints(mtx_P_l, mtx_P_r, np.array([[center_x_l], [center_y_l]]), np.array([[center_x_r], [center_y_r]]))
        mp_3d = cv2.transpose(mp_3d_homogeneous)
        mp_3d = cv2.convertPointsFromHomogeneous(mp_3d).squeeze()
        print(mp_3d)

        text = "Position: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(point3D[0]/10), float(point3D[1]/10),
                                                                        float(point3D[2])/10)
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 5)

        if center_x_l > 1000 and center_x_l < 1175 and not initialized:
            # """
            print('start track')
            print(currect_center)
            print(center_x_r, center_y_r)
            km_x, km_P, km_u, km_F, km_H, km_R = karman_init(mp_3d)
            initialized = True
            kalman1.reset()
            cv2.putText(frame, 'First dectect', (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 15)

        if initialized and center_x_l != 0 and center_y_l != 0:
            # update/predict kalman
            obj_track = True  # (was it possible to track the object)
            # km_measurement = np.array([[point3D[0]], [point3D[1]], [point3D[2]]])
            # [km_x, km_P] = update(km_x, km_P, km_measurement, km_F, km_u)
            [km_x, km_P] = predict(km_x, km_P, km_F, km_u)

    # if object reaches the end of the conveyor:
    if obj_track and center_x_l <= 450 and center_x_l != 0:
        # prepare for next object (reset status and roi)
        obj_track = False
        initialized = False
        obj_dectect_l = False
        obj_dectect_r = False
        km_x, km_P, km_u, km_F, km_H, km_R = karman_init([0, 0, 0])

    if obj_track and not obj_dectect_l:

        # X1 = kalman1.predict()
        # pre_pos1 = np.array([X1[0], X1[3]]).T[0]
        # print(pre_pos1)
        # cv2.circle(frame, (int(pre_pos1[0]), int(pre_pos1[1])), 10, (255, 255, 255), -1)

        [km_x, km_P] = predict(km_x, km_P, km_F, km_u)
        km_x_repr = np.dot(mtx_P_l, np.vstack((np.array([km_x[0], km_x[2], km_x[4]]), 1)))
        km_x_repr = km_x_repr / km_x_repr[2]
        center_pred = np.array([int(km_x_repr[0]), int(km_x_repr[1])])
        pre_center = center_pred
        print('pre_center2:')
        print(pre_center)

        pred_center_r = np.array([center_pred[0]-disp-50, center_pred[1]])
        cv2.circle(frame, center_pred, 10, (0, 255, 255), -1)
        cv2.circle(frame, center_pred, 50, (0, 255, 255), 2)  # cirle

        P = triangulate(center_pred, pred_center_r, mtx_l, mtx_r, mtx_T)
        point3D = P[0]  # choose the positive z
        text2 = "pred_Position: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(point3D[0]/10), float(point3D[1]/10),
                                                                         float(point3D[2])/10)
        cv2.putText(frame, text2, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 5)

    # Show the frame
    cv2.imshow('Frame', frame)
    # cv2.imshow('Frame_r', frame_r)
    cv2.waitKey(50)

cv2.destroyAllWindows()
