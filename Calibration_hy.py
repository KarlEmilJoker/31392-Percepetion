import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

#Implement the number of vertical and horizontal corners
nb_vertical = 6
nb_horizontal = 9


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp_l = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp_l[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_l = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.

left_rs = glob.glob('Stereo_calibration_images/left-*.png')
assert left_rs


for fname in left_rs:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #Implement findChessboardCorners here
    ret, corners = cv2.findChessboardCorners(img, (nb_vertical, nb_horizontal))


    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_l.append(objp_l)

        imgpoints_l.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
#         cv2.imshow('img',img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints_l, imgpoints_l, gray.shape[::-1], None, None)
img = cv2.imread('Stereo_calibration_images/left-0000.png')
h,  w = img.shape[:2]
newcameramtx_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(w,h),1,(w,h))
print(newcameramtx_l)

# undistort
img_l = cv2.imread('Stereo_calibration_images/left-0002.png')
dst_l = cv2.undistort(img_l, mtx_l, dist_l, None, newcameramtx_l)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,18))
ax[0].imshow(img_l[...,[2,1,0]])
ax[0].set_title('Original image')
ax[1].imshow(dst_l[...,[2,1,0]])
ax[1].set_title('Undistorted image')

# crop the image
x,y,w,h = roi_l
dst_l = dst_l[y:y+h, x:x+w]
plt.figure(figsize=(10,10))
plt.imshow(dst_l[...,[2,1,0]])

plt.show()