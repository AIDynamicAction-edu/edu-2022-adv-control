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

class pos_rotation_statistics():
    def __init__(self, values, mean =[]):
        self.values = values
        self.values_x = values[:, 0]
        self.values_y = values[:, 1]
        self.values_z = values[:, 2]
        if (mean == []):
            self.mean_std_samples()
        else:
            self.mean_std_ground_truth(mean)
               
    def mean_std_samples(self):

        self.x_mean = stat.mean(self.values_x)
        self.y_mean = stat.mean(self.values_y)
        self.z_mean = stat.mean(self.values_z)
        
        self.x_std = stat.stdev(self.values_x)
        self.y_std = stat.stdev(self.values_y)
        self.z_std = stat.stdev(self.values_z)
        
    def mean_std_ground_truth(mean):
        
        self.x_mean = mean[0]
        self.y_mean = mean[1]
        self.z_mean = mean[2]
        
        self.x_std = np.sqrt(stat.mean((self.values_x - self.x_mean)**2))
        self.y_std = np.sqrt(stat.mean((self.values_y - self.y_mean)**2))
        self.z_std = np.sqrt(stat.mean((self.values_z - self.z_mean)**2))

    def draw_stat(self, name, degrees = False):
        num = len(self.values)
        
        x_std_mat = 3*self.x_std*np.ones((num))
        y_std_mat = 3*self.y_std*np.ones((num))
        z_std_mat = 3*self.z_std*np.ones((num))

        x_mean_mat = self.x_mean*np.ones((num))
        y_mean_mat = self.y_mean*np.ones((num))
        z_mean_mat = self.z_mean*np.ones((num))

        fig, axs = plt.subplots(1, 3, figsize = (15,5))
        
        if(degrees == False):
            axs[0].scatter(range(num), self.values_x)
            axs[0].plot(range(num), x_mean_mat, c='g', label = "mean")
            axs[0].plot(range(num), x_std_mat + self.x_mean, range(num), -x_std_mat + self.x_mean, c='r', label = "3|$\sigma$| boundries")
            axs[0].set_title(name[0])
            axs[0].legend()
            
            axs[1].scatter(range(num), self.values_y)
            axs[1].plot(range(num), y_mean_mat, c='g', label = "mean")
            axs[1].plot(range(num), y_std_mat + self.y_mean, range(num), -y_std_mat + self.y_mean, c='r', label = "3|$\sigma$| boundries")
            axs[1].set_title(name[1])
            axs[1].legend()

            axs[2].scatter(range(num), self.values_z)
            axs[2].plot(range(num), z_mean_mat, c='g', label = "mean")
            axs[2].plot(range(num), z_std_mat + self.z_mean, range(num), -z_std_mat + self.z_mean, c='r', label = "3|$\sigma$| boundries")
            axs[2].set_title(name[2])
            axs[2].set_ylabel
            axs[2].legend()
        
        else:
            axs[0].scatter(range(num), np.rad2deg(self.values_x))
            axs[0].plot(range(num), np.rad2deg(x_mean_mat), c='g', label = "mean")
            axs[0].plot(range(num), np.rad2deg(x_std_mat + self.x_mean), range(num), np.rad2deg(-x_std_mat + self.x_mean), c='r', label = "3|$\sigma$| boundries")
            axs[0].set_title(name[0])
            axs[0].legend()

            axs[1].scatter(range(num), np.rad2deg(self.values_y))
            axs[1].plot(range(num), np.rad2deg(y_mean_mat), c='g', label = "mean")
            axs[1].plot(range(num), np.rad2deg(y_std_mat + self.y_mean), range(num), np.rad2deg(-y_std_mat + self.y_mean), c='r', label = "3|$\sigma$| boundries")
            axs[1].set_title(name[1])
            axs[1].legend()

            axs[2].scatter(range(num), np.rad2deg(self.values_z))
            axs[2].plot(range(num), np.rad2deg(z_mean_mat), c='g', label = "mean")
            axs[2].plot(range(num), np.rad2deg(z_std_mat + self.z_mean), range(num), np.rad2deg(-z_std_mat + self.z_mean), c='r', label = "3|$\sigma$| boundries")
            axs[2].set_title(name[2])
            axs[2].legend()
            
class rotation_statistics(pos_rotation_statistics):
    def __init__(self, quat, rot_mean = []):
        self.rot = R.from_quat(quat)
        self.values = self.rot.as_euler('xyz')
        self.values_x = self.values[:, 0]
        self.values_y = self.values[:, 1]
        self.values_z = self.values[:, 2]
        if (rot_mean == []):
            self.mean_std_samples()
            self.rot_mean = R.from_euler('xyz', np.array([self.x_mean, self.y_mean, self.z_mean]))
        else:
            self.rot_mean = rot_mean
        
    def error_between_two_rotation(self, rot1, rot2):
        mat1 = rot1.as_matrix()
        mat2 = rot2.as_matrix()
        mat_error = mat1@(mat2.T)
        rotvec_error = R.from_matrix(mat_error).as_rotvec()
        angle_error = np.linalg.norm(rotvec_error)
        return angle_error

    def angle_error(self):
        angle_error = []
        for rot in self.rot:
            angle = self.error_between_two_rotation(self.rot_mean, rot)
            angle_error += [angle]
        angle_error = np.array(angle_error)
        return np.mean(angle_error)
    
class trans_statistics(pos_rotation_statistics):
    def translation_error(self):
        trans_mean = np.array([self.x_mean, self.y_mean, self.z_mean])
        trans_error = self.values - trans_mean
        trans_error_std = 3 * np.sqrt(np.mean(trans_error**2, axis = 0))
        return np.linalg.norm(trans_error_std)

