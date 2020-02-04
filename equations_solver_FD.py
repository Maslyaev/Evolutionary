#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:25:28 2020

@author: mike_ubuntu
"""

import numpy as np


def Interval_6_impl(data):
    
    D_t = 1; D_x = 1; D_y = 1
    
    solution = np.empty(data.shape)
    
    solution[0:2, :, :] = data[0:2, :, :]
    
    #solution[:, 0, :] = data[:, 0, :]; solution[:, -1, :] = data[:, -1, :]
    solution[:, :, 0] = data[:, :, 0]; solution[:, :, -1] = data[:, :, -1]    

    alpha_3 = 1/(2*D_x) - 8.508/D_x**2
    alpha_2 = 0.05153/D_t**2 + 2 * 8.508/D_y**2 
    alpha_1 = (- 1/(2*D_y) - 8.508/D_y**2)
    alpha_4 = 0.05153*2/D_t**2
    alpha_5 = -0.05153/D_t**2
    
    for k in np.arange(2, data.shape[0]):
        for i in np.arange(0, data.shape[1]):
            
            A = np.eye(data.shape[2])
            B = np.empty(data.shape[2])
            B[0] = solution[k, i, 0]; B[-1] = solution[k, i, -1]
            for j in np.arange(1, data.shape[2]-1):
                A[j, j] = alpha_2; A[j, j-1] = alpha_1; A[j, j+1] = alpha_3
                B[j] = alpha_4*solution[k-1, i, j] + alpha_5*solution[k-2, i, j]
                
            row = np.linalg.solve(A, B)
            solution[k, i, :] = row
        print(np.max(solution[k, :, :]))
    return solution

data = np.load('ssh_sept_min_sample.npy')
data = data[6*24:7*24, :, :]
solution = Interval_6_impl(data)