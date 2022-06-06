#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import statistics as stat
import matplotlib.pyplot as plt
import cv2
import json

class pnp_solution():
    def __init__(self, NUM_KEYPOINTS, mtx, dist, im_path, cord_path):
        self.img_path = im_path
        self.cord_path = cord_path

        self.NUM_KEYPOINTS = NUM_KEYPOINTS
        self.mtx = mtx
        self.dist = dist

#         try:
#             self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#         except Exception as e:
#             print(e, 'device: cpu')

#         self.precise_model = get_ResNet50(num_keypoints=self.NUM_KEYPOINTS, weights_path='/home/viacheslav/PycharmProjects/ResNetSocketPred/outputs/keypointsrcnn_weights_10_epoch50.pth', load_device=self.device)
#         self.precise_model.to(self.device)

    def predict_socket_points(self, points_path):
        points = np.load(points_path)
        key_points = points["points"]
        return key_points

    def real_coordinate(self):
        pi = np.pi
        cos = np.cos
        sin = np.sin

        d_between_holes = 1.4
        d_on_side = 2.3
        d_up = 11.1
        phi_bighole = 14.5
        phi_smallhole = 10
        d_undersocket = 36.6
        d_arroundsocket = 6.3

        d_center2center_bigbig = phi_bighole + d_between_holes
        d_center2center_bigsmall = phi_bighole / 2 + phi_smallhole / 2 + d_between_holes
        d_center_smallhole2x_axis = np.sqrt(d_center2center_bigsmall ** 2 - (d_center2center_bigbig / 2) ** 2)
        d_center2endright = 3 / 2 * phi_bighole + d_between_holes + d_on_side / 2
        d_center2up = phi_bighole / 2 + d_up - d_on_side / 2
        d_center2point1718 = 3 / 2 * phi_bighole + d_between_holes + d_on_side + d_arroundsocket

        angle_isoscale_triangle = np.arcsin(d_center_smallhole2x_axis / d_center2center_bigsmall)
        ###################
        ###################
        objectPoints = np.zeros((19, 3))

        objectPoints[0] = np.array([[-d_center2center_bigbig / 2, d_center_smallhole2x_axis + phi_smallhole / 2 + d_on_side / 2, 0]])
        objectPoints[1] = objectPoints[0] * [-1, 1, 1]
        objectPoints[4] = np.array([[(phi_bighole / 2 + d_between_holes / 2) * np.cos(angle_isoscale_triangle),
                                     (phi_bighole / 2 + d_between_holes / 2) * np.sin(angle_isoscale_triangle),
                                     0]])
        objectPoints[5] = np.array([[d_center2center_bigbig - (phi_bighole / 2 + d_between_holes / 2) * np.cos(angle_isoscale_triangle),
                                     (phi_bighole / 2 + d_between_holes / 2) * np.sin(angle_isoscale_triangle),
                                     0]])
        objectPoints[2] = objectPoints[5] * [-1, 1, 1]
        objectPoints[3] = objectPoints[4] * [-1, 1, 1]

        objectPoints[6] = np.array([[-d_center2endright, 0, 0]])
        objectPoints[7] = np.array([[-phi_bighole / 2 - d_between_holes / 2, 0, 0]])
        objectPoints[8] = np.array([[phi_bighole / 2 + d_between_holes / 2, 0, 0]])
        objectPoints[9] = np.array([[d_center2endright, 0, 0]])

        objectPoints[13] = np.array([[d_center2center_bigbig * (1 + cos(pi / 3)) / 2, -d_center2center_bigbig * sin(pi / 3) / 2, 0]])
        objectPoints[12] = np.array([[d_center2center_bigbig / 2 * cos(pi / 3), -d_center2center_bigbig * sin(pi / 3) / 2, 0]])
        objectPoints[10] = objectPoints[13] * [-1, 1, 1]
        objectPoints[11] = objectPoints[12] * [-1, 1, 1]

        objectPoints[14] = np.array([[0, -d_center2center_bigbig * sin(pi / 3), 0]])

        objectPoints[15] = np.array([[-d_center2endright * cos(pi / 3), -d_center2endright * sin(pi / 3), 0]])
        objectPoints[16] = np.array([[d_center2endright * cos(pi / 3), -d_center2endright * sin(pi / 3), 0]])

        objectPoints[17] = np.array([[-d_center2point1718 * cos(pi / 3), -d_center2point1718 * sin(pi / 3), 0]])
        objectPoints[18] = objectPoints[17] * [-1, 1, 1]
        # print(objectPoints)
        return objectPoints#*[1,-1,1]

    def draw_points(self, img, points):
        image = img
        for kp in points:
            # for idx, kp in enumerate(kps):
            image = cv2.circle(image.copy(), (int(kp[0]), int(kp[1])), 2, (210, 32, 13), 2)
            # cv2.imwrite(f'data/reprojection_{id}.jpg', image)
        cv2.imshow("image", image)
        cv2.waitKey()

    def solvepnp_problem(self, objectPoints, imagePoints):
        retval, rot_vec, trans_vec = cv2.solvePnP(objectPoints, imagePoints, self.dist, self.mtx)
        return retval, rot_vec, trans_vec
                 
                 