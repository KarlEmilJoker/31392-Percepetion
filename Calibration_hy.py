import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    print(img1.shape)
    r,c, joe = img1.shape
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,2)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

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


ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints_l, imgpoints_l, gray.shape[::-1], None, None)
img = cv2.imread('Stereo_calibration_images/left-0000.png')
h,  w = img.shape[:2]
newcameramtx_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(w,h),1,(w,h))
print(newcameramtx_l)

# FOR RIGHT IMAGES ONLY!!!
#Implement the number of vertical and horizontal corners
nb_vertical = 6
nb_horizontal = 9


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp_r = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp_r[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_r = [] # 3d point in real world space
imgpoints_r = [] # 2d points in image plane.

right_rs = glob.glob('Stereo_calibration_images/right-*.png')
assert right_rs


for fname in right_rs:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #Implement findChessboardCorners here
    ret, corners = cv2.findChessboardCorners(img, (nb_vertical, nb_horizontal))

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_r.append(objp_r)
        imgpoints_r.append(corners)

# FOR RIGHT IMAGES ONLY!!!
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints_r, imgpoints_r, gray.shape[::-1], None, None)
img = cv2.imread('Stereo_calibration_images/right-0000.png')
h,  w = img.shape[:2]
newcameramtx_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(w,h),1,(w,h))
print(mtx_r)


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

cameramtx_left = [[112.28690338, 0., 982.05689904],
                  [0, 111.80508423, 582.24763194],
                  [0., 0., 1.]]

cameramtx_right = [[4.12920609e+01, 0.0, 1.20742963e+03],
 [0.00000000e+00, 5.25937042e+01, 6.71338991e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
plt.figure(figsize=(10,10))
plt.imshow(dst_l[...,[2,1,0]])

# undistort
img_r = cv2.imread('Stereo_calibration_images/right-0002.png')
dst_r = cv2.undistort(img_r, mtx_r, dist_r, None, newcameramtx_r)

# crop the image
x,y,w,h = roi_r
dst_r = dst_r[y:y+h, x:x+w]
# plt.figure(figsize=(10,10))
# plt.imshow(dst_r[...,[2,1,0]])

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,18))
# ax[0].imshow(img_r[...,[2,1,0]])
# ax[0].set_title('Original image')
# ax[1].imshow(dst_r[...,[2,1,0]])
# ax[1].set_title('Undistorted image')

plt.show()

img1 = dst_l
img2 = dst_r

sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

kp_img = cv2.drawKeypoints(img2, kp2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize = (10,10))
plt.imshow(kp_img)

# create BFMatcher object
bf = cv2.BFMatcher()
# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance (i.e. best matches first).
matches = sorted(matches, key = lambda x:x.distance)

nb_matches = 200

good = []
pts1 = []
pts2 = []

for m in matches[:nb_matches]:
    good.append(m)
    pts1.append(kp1[m.queryIdx].pt)
    pts2.append(kp2[m.trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
    

#Implement findFundamentalMat here:
F, mask = cv2.findFundamentalMat(pts1, pts2)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

print(img1.shape)

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2 ,F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(10,10))
axs[0, 0].imshow(img4)
axs[0, 0].set_title('left keypoints')
axs[0, 1].imshow(img6)
axs[0, 1].set_title('right keypoints')
axs[1, 0].imshow(img5)
axs[1, 0].set_title('left epipolar lines')
axs[1, 1].imshow(img3)
axs[1, 1].set_title('right epipolar lines')
plt.show()