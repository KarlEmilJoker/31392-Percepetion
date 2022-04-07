import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

## rs images
nb_vertical = 6
nb_horizontal = 9

def getcorners(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_horizontal * nb_vertical, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nb_vertical, 0:nb_horizontal].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)
    return objpoints, imgpoints

def getCamMatrix(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_horizontal * nb_vertical, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nb_vertical, 0:nb_horizontal].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    # get the camera matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    img = cv2.imread('/home/algora/Desktop/PFAS_DTU/FinalProject/31392-Percepetion/Stereo_calibration_images/right-0000.png')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    return newcameramtx, dist, mtx, roi, rvecs, tvecs

images_right = glob.glob('/home/algora/Desktop/PFAS_DTU/FinalProject/31392-Percepetion/Stereo_calibration_images/right*.png')
assert images_right

images_left = glob.glob('/home/algora/Desktop/PFAS_DTU/FinalProject/31392-Percepetion/Stereo_calibration_images/left*.png')
assert images_left

cameramtx_right, dist_right, mtx_right, roi_right, rvecs_right, tvecs_right = getCamMatrix(images_right)
cameramtx_left, dist_left, mtx_left, roi_left, rvecs_left, tvecs_left = getCamMatrix(images_left)

print(cameramtx_right)

print(cameramtx_left)

# undistort all the images / ok it works, but for what purpose ??

# for fname in images:
#     img = cv2.imread(fname)
#     dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#
#     cv2.imshow('original', img)
#     cv2.imshow('undistorted', dst)
#     cv2.waitKey(500)

# work only on one img
imgleft = cv2.imread('/home/algora/Desktop/PFAS_DTU/FinalProject/31392-Percepetion/Stereo_calibration_images/right-0027.png')
imgright = cv2.imread('/home/algora/Desktop/PFAS_DTU/FinalProject/31392-Percepetion/Stereo_calibration_images/left-0027.png')

dstright = cv2.undistort(imgright, mtx_right, dist_right, None, cameramtx_right)
dstleft = cv2.undistort(imgleft, mtx_left, dist_left, None, cameramtx_left)

# crop the image
xr,yr,wr,hr = roi_right
xl,yl,wl,hl = roi_left
dstright = dstright[yr:yr+hr, xr:xr+wr]
dstleft = dstleft[yl:yl+hl, xl:xl+wl]

(h, w, d) = dstright.shape

cv2.imshow('original right ', imgright)
cv2.imshow('original left', imgleft)
cv2.imshow('undistorted right', dstright)
cv2.imshow('undistorted left', dstleft)




cv2.waitKey(0)