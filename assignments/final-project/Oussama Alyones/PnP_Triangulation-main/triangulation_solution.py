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

class triangulation_solution():
    def __init__(self, cameraMatrix):
        self.mtx = cameraMatrix
        
    def get_keypoints(self, points_path):
        points = np.load(points_path)
        key_points = points["points"]
        return key_points
    
    def triangulate_two_pairs(self, frame1_points, frame2_points, frame1_tcp0, frame2_tcp0, cam2gripper):
        datas = [frame1_tcp0, frame2_tcp0]
        corn = []

        corn.append(frame1_points)
        corn.append(frame2_points)

        grip2base = []
        for dname in datas:
            pose_vector = dname

            ori = R.from_rotvec(np.array(pose_vector[3:]))
            ori_m = ori.as_matrix()

            mm_data = np.array(pose_vector[:3]) * 1000
            proxy = np.concatenate([ori_m, mm_data.reshape((3, 1))], axis=-1)
            grip2bas = np.concatenate([proxy, np.array([[0, 0, 0, 1]])], axis=0)

            grip2base.append(grip2bas)

        # evaluation of cameras projection matrices
        cam2base1 = np.matmul(grip2base[0], cam2gripper)
        base2cam1 = np.linalg.inv(cam2base1)
        base2cam1 = np.delete(base2cam1, (3), axis=0)

        cam2base2 = np.matmul(grip2base[1], cam2gripper)
        base2cam2 = np.linalg.inv(cam2base2)
        base2cam2 = np.delete(base2cam2, (3), axis=0)
        # these proj matrices
        proj1 = np.matmul(self.mtx, base2cam1)
        proj2 = np.matmul(self.mtx, base2cam2)

        # method to get 3D coord of points column-wise [x,y,z,1]
        points = cv2.triangulatePoints(proj1, proj2, np.array(corn[0]), np.array(corn[1]))
    #     print(points)
        pointsT = points / points[3]
    #     print(points[3])
        pointsT = pointsT[:-1].T
#         print(pointsT[0])
        return (pointsT)


    def triangulate_all_pairs(self, nums_im, num_keypoits, cord_path1, cord_path2, keypoints_path1, keypoints_path2, T_cam2gripper):
        points = np.zeros((num_keypoits, 3))
        keypoints_all1 = self.get_keypoints(keypoints_path1)
        keypoints_all2 = self.get_keypoints(keypoints_path2)
        
        for num in range(nums_im):
            data = []
            with open(cord_path1) as json_file:
                json_data = json.load(json_file)
                data.append(json_data['l'])
            with open(cord_path2) as json_file:
                json_data = json.load(json_file)
                data.append(json_data['l'])

            keypoints1 = np.concatenate((keypoints_all1[num][2:6,:], keypoints_all1[num][7:9,:],keypoints_all1[num][10:15,:]), axis=0)
            keypoints2 = np.concatenate((keypoints_all2[num][2:6,:], keypoints_all2[num][7:9,:],keypoints_all2[num][10:15,:]), axis=0)

            points += self.triangulate_two_pairs(keypoints1.T, keypoints1.T,
                                             data[0], data[1],
                                             T_cam2gripper)/nums_im

        return points

    def equation_plane(self, p0, p1, p2):
        u = p1 - p0
        v = p2 - p1
        u_cross_v = np.cross(u, v)
        normal = u_cross_v
        if normal[0]>0:
            normal = -normal

        d = -p0.dot(normal)
        equation = np.concatenate((normal, np.array([d])), axis = 0)
        return equation

