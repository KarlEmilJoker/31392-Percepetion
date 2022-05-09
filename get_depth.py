import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_disparity(left_img, right_img, px_c, py_c):  # left and right undistoreted and rectified images (3 channels)
    focal_lenght = 685
    baseline = 120
    roi_gap = 5
    stereo = cv2.StereoSGBM_create(minDisparity=5,
                                   numDisparities=160,  # 160
                                   blockSize=5,
                                   P1=8 * 3 * 5 ** 2,  # 600
                                   P2=32 * 3 * 5 ** 2,  # 2400
                                   disp12MaxDiff=100,
                                   preFilterCap=32,
                                   uniquenessRatio=10,
                                   speckleWindowSize=0,
                                   speckleRange=32)

    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity = cv2.medianBlur(disparity, 5)
    depth = focal_lenght*baseline/(disparity[py_c,px_c]/16.0) #in mm
    # print('disparity')
    print(disparity[py_c, px_c])
    result = disparity[py_c, px_c]
    return result

if __name__ == "__main__":
    # map the matrix file with images
    imgL = cv2.imread(r'Stereo_conveyor_with_occlusions\left\1585434753_889844656_Left.png')
    imgR = cv2.imread(r'Stereo_conveyor_with_occlusions\right\1585434753_889844656_Right.png')
    rect_map_left_x = np.load(r'Matrix\map_l_x.npy')
    rect_map_left_y = np.load(r'Matrix\map_l_y.npy')
    rect_map_right_x = np.load(r'Matrix\map_r_x.npy')
    rect_map_right_y = np.load(r'Matrix\map_r_y.npy')
    imgL = cv2.remap(imgL, rect_map_left_x, rect_map_left_y, cv2.INTER_LINEAR)
    imgR = cv2.remap(imgR, rect_map_right_x, rect_map_right_y, cv2.INTER_LINEAR)

    # plt.imshow(imgL, cmap='gray')
    # plt.show()
    print(imgR.shape)
    disp, depth = get_disparity(imgL, imgR, 1160, 318)

    h, w = disp.shape[:2]
    f = .8 * w
    Q1 = np.float32([[1,0,0,-0.5*w],
                    [0,-1,0,0.5*h],
                    [0,0,0,-f],
                    [0,0,1,0]])
    depth_estimate = cv2.reprojectImageTo3D(disp, Q1)

    print(disp)
    depth_estimate_roi = depth_estimate[318-5:318+5, 1160-5:1160+5:]
    depth_estimate_roi_mult = np.multiply(depth_estimate_roi, depth_estimate_roi)
    depth_estimate_roi_sum = np.sum(depth_estimate_roi_mult, axis=2)
    depth_estimate_roi_sqrt = np.sqrt(depth_estimate_roi_sum)
    distance = depth_estimate_roi_sqrt.min() / 100

    print(np.median(depth_estimate_roi[:, :, 0])/10)
    print(np.median(depth_estimate_roi[:, :, 1])/10)
    print(np.median(depth_estimate_roi[:, :, 2])/10)

    print(distance)
    # 0.08335267066955566
    # -0.011843969821929931
    # 0.105145263671875
    # 0.020101053714752196

    plt.imshow(depth, cmap='plasma')
    plt.colorbar()
    plt.show()
    # (1173, 310)
    # homogeneous
    # [[-0.62066471]
    #  [0.08828016]
    #  [-0.7784707]
    #  [-0.03106937]]
    # mp_3d1
    # [[-0.62066471  0.08828016 - 0.7784707 - 0.03106937]]
    # mp_3d2
    # [19.97674184 - 2.84138899 25.05589228]
    # find
    # it