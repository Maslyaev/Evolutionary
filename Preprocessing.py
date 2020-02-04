#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:44:49 2019

@author: mike_ubuntu
"""

import math
from scipy.special import comb
import numpy as np
import sys
import time
import datetime
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
        

def Get_Polynomials_for_point(matrix, axis, idx, grid, poly_power = 5):
    power = poly_power + 1
    I = np.array([np.int(-(power-1)/2 + i) for i in np.arange(power)]) + idx[axis]
    F = matrix.take(I, axis = axis)
    x_raw = grid[axis].take(I, axis = axis)
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis = 0)
            x_raw = x_raw.take(idx[i], axis = 0)            
        elif i > axis:
            F = F.take(idx[i], axis = 1)
            x_raw = x_raw.take(idx[i], axis = 1)         
            
    X = np.array([np.power(x_raw[0], power - i) for i in np.arange(1, power)] + [1])
    for j in np.arange(1, power):
        X = np.vstack((X, np.array([np.power(x_raw[j], power - i) for i in np.arange(1, power)] + [1])))
    return x_raw[int(x_raw.size/2.)], np.flip(np.linalg.solve(X, F)[:-1])
    

def Get_LSQ_for_point(matrix, axis, idx, grid, max_der_order = 3, points = 9):
    max_power = max_der_order + 1
    I = np.array([np.int(-(points-1)/2 + i) for i in np.arange(points)]) + idx[axis]
    F = matrix.take(I , axis = axis)
    x_raw = grid[axis].take(I, axis = axis)
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis = 0)
            x_raw = x_raw.take(idx[i], axis = 0)            
        elif i > axis:
            F = F.take(idx[i], axis = 1)
            x_raw = x_raw.take(idx[i], axis = 1)     
            
    X = np.array([np.power(x_raw[0], max_power - i) for i in np.arange(1, max_power)] + [1])
    for j in np.arange(1, points):
        X = np.vstack((X, np.array([np.power(x_raw[j], max_power - i) for i in np.arange(1, max_power)] + [1])))
    estimator = LinearRegression()
    estimator.fit(X, F)
    return x_raw[int(x_raw.size/2.)], np.flip(np.array(estimator.coef_)[:-1])

def Get_cheb_for_point(matrix, axis, idx, grid, max_der_order = 3, points = 9):
    max_power = max_der_order + 1
    I = np.array([np.int(-(points-1)/2 + i) for i in np.arange(points)]) + idx[axis]
    F = matrix.take(I , axis = axis)
    x_raw = grid[axis].take(I, axis = axis)
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis = 0)
            x_raw = x_raw.take(idx[i], axis = 0)            
        elif i > axis:
            F = F.take(idx[i], axis = 1)
            x_raw = x_raw.take(idx[i], axis = 1)     
            

    poly = np.polynomial.chebyshev.Chebyshev.fit(x_raw, F, max_power)
    return x_raw[int(x_raw.size/2.)], poly

def FD_derivatives(matrix, axis, idx, grid, max_order):
    assert idx[axis] < PolyBoundary or idx[axis] > matrix.shape[axis] - PolyBoundary
    if idx[axis] < PolyBoundary:
        I = idx[axis] + np.arange(6) 
    else:
        I = idx[axis] - np.arange(6)
    
    x = grid[axis].take(I, axis = axis)
    F = matrix.take(I , axis = axis)    
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis = 0)
            x = x.take(idx[i], axis = 0)            
        elif i > axis:
#            print(i, idx, F.shape, x_raw.shape)
            F = F.take(idx[i], axis = 1)
            x = x.take(idx[i], axis = 1)     

    derivatives = np.empty(3)            
    derivatives[0] = (F[1] - F[0]) / (x[1] - x[0])
    derivatives[1] = (2*F[0] - 5*F[1] + 4*F[2] - F[3]) / (x[1] - x[0]) ** 2
    derivatives[2] = (-2.5*F[0] + 9*F[1] - 12*F[2] + 7*F[3] - 1.5*F[4]) / (x[1] - x[0]) ** 3
    return derivatives[:max_order]        
        

def Process_Point_Poly(args):
    idx = np.array(args[0]); matrix = args[1]; grid = args[2]; poly_power = args[3]; n_der = args[4]
    assert poly_power >= n_der
    print(args[0])
    poly_mask = [idx[dim] >= PolyBoundary and idx[dim] <= matrix.shape[dim] - PolyBoundary for dim in np.arange(matrix.ndim)]
    coeffs = np.empty((matrix.ndim, poly_power))
    x = np.empty(idx.shape)
    for i in range(coeffs.shape[0]): #  [1]: #
        if poly_mask[i]:
            x_temp, coeffs_temp = Get_Polynomials_for_point(matrix, i, idx, grid, poly_power = poly_power)
            #print(x_temp, coeffs_temp.shape)
            x[i] = x_temp
            coeffs[i, :] = coeffs_temp     
    
    derivatives = np.zeros(coeffs.shape[0] * (n_der))
    for var_idx in np.arange(coeffs.shape[0]):
        if poly_mask[var_idx]:
            for der_idx in np.arange(1, n_der+1):
                derivatives[var_idx*(n_der) + (der_idx-1)] = np.sum([coeffs[var_idx, j-1] * np.math.factorial(j)/
                                     np.math.factorial(j - der_idx) * x[var_idx] ** (j - der_idx) for j in range(der_idx, n_der+1)])          
        else:
#            print('derivatives shape:', FD_derivatives(matrix, axis = var_idx, idx = idx, grid = grid, max_order = n_der).shape) #derivatives[var_idx*(n_der) : var_idx+1*(n_der)] = 
#            print('matrix space shape', derivatives[var_idx*(n_der) : (var_idx+1)*(n_der)].shape, 'for idx =', idx, 'axis', var_idx)
            derivatives[var_idx*(n_der) : (var_idx+1)*(n_der)] = FD_derivatives(matrix, axis = var_idx, idx = idx, grid = grid, max_order = n_der)
    return(derivatives)


def Process_Point_Cheb(args):
    global PolyBoundary
    idx = np.array(args[0]); matrix = args[1]; grid = args[2]; points = args[3]; n_der = args[4]
    print(args[0])
    poly_mask = [idx[dim] >= PolyBoundary and idx[dim] <= matrix.shape[dim] - PolyBoundary for dim in np.arange(matrix.ndim)]
    polynomials = np.empty(matrix.ndim, dtype = np.polynomial.chebyshev.Chebyshev)
    x = np.empty(idx.shape)
    for i in range(matrix.ndim):
        if poly_mask[i]:
            x_temp, poly_temp = Get_cheb_for_point(matrix, i, idx, grid, max_der_order=n_der, points = points)
            x[i] = x_temp
            polynomials[i] = poly_temp 

    derivatives = np.empty(matrix.ndim * (n_der))
    for var_idx in np.arange(matrix.ndim):
        if poly_mask[var_idx]:
            for der_idx in np.arange(1, n_der+1):
                derivatives[var_idx*(n_der) + (der_idx-1)] = polynomials[var_idx].deriv(m=der_idx)(x[var_idx])
        else:
            #print('shapes:', FD_derivatives(matrix, axis = var_idx, idx = idx, grid = grid, max_order = n_der).shape, derivatives[var_idx*(n_der) : var_idx+1*(n_der)].shape)
            derivatives[var_idx*(n_der) : (var_idx+1)*(n_der)] = FD_derivatives(matrix, axis = var_idx, idx = idx, grid = grid, max_order = n_der)
    return(derivatives)


def Process_Point_LSQ(args):
    global PolyBoundary
    idx = np.array(args[0]); matrix = args[1]; grid = args[2]; points = args[3]; n_der = args[4]
    print(args[0])
    poly_mask = [idx[dim] >= PolyBoundary and idx[dim] <= matrix.shape[dim] - PolyBoundary for dim in np.arange(matrix.ndim)]
    coeffs = np.empty((matrix.ndim, n_der))
    x = np.empty(idx.shape)
    for i in range(coeffs.shape[0]):
        if poly_mask[i]:
            x_temp, coeffs_temp = Get_LSQ_for_point(matrix, i, idx, grid, max_der_order=n_der, points = points)
            x[i] = x_temp
            coeffs[i, :] = coeffs_temp 

    derivatives = np.empty(coeffs.shape[0] * (n_der))
    for var_idx in np.arange(coeffs.shape[0]):
        if poly_mask[var_idx]:
            for der_idx in np.arange(1, n_der+1):
                derivatives[var_idx*(n_der) + (der_idx-1)] = np.sum([coeffs[var_idx, j-1] * np.math.factorial(j)/
                                     np.math.factorial(j - der_idx) * x[var_idx] ** (j - der_idx) for j in range(der_idx, n_der+1)])          
        else:
            #print('shapes:', FD_derivatives(matrix, axis = var_idx, idx = idx, grid = grid, max_order = n_der).shape, derivatives[var_idx*(n_der) : var_idx+1*(n_der)].shape)
            derivatives[var_idx*(n_der) : (var_idx+1)*(n_der)] = FD_derivatives(matrix, axis = var_idx, idx = idx, grid = grid, max_order = n_der)
    return(derivatives)

def Chebyshev_grid(a, b, n):   
    ''' 
    Calculation of grid by roots of Chebyshev polynominals for 1D - case
    '''
    nodes = np.zeros(n)
    nodes = list(map(lambda x: (b+a)/2. + (b-a)/2.*math.cos(math.pi*(2*x - 1)/(2*n)), range(1, n+1)))
    nodes = np.fliplr([nodes])[0]
    return nodes

         
def Add_noise_percentile(V_matrix, part, points_prop = 0.1):
    #
    V_noised = np.copy(V_matrix)
    for idx1 in range(V_matrix.shape[0]):
        mean_value = V_matrix[idx1, :].mean()
        selection_size = int(points_prop*V_matrix[0].size)
        x_rand = np.random.choice(V_matrix.shape[1], size = selection_size, replace = True)        
        y_rand = np.random.choice(V_matrix.shape[2], size = selection_size, replace = True)       
        #print('Timestep:', idx1)
        for point_idx in np.arange(selection_size):
            #print('Noise added to point ', x_rand[point_idx], y_rand[point_idx])
            V_noised[idx1, x_rand[point_idx], y_rand[point_idx]] = np.random.normal(V_noised[idx1, x_rand[point_idx], y_rand[point_idx]], 
                    math.sqrt(abs(part * mean_value)))
    print(selection_size)
    time.sleep(5)
    noise_lvl = np.linalg.norm(V_noised - V_matrix) / np.linalg.norm(V_matrix) * 100
    return V_noised, noise_lvl
    
        
def Add_noise(V_matrix, part): # Addition of noise for the stability test: designed for 1D - case
    V_noised = np.copy(V_matrix)
    for idx1 in range(V_matrix.shape[0]):
        max_value = V_matrix[idx1][:].max()
        for idx2 in range(V_matrix.shape[1]):
            for idx3 in range(V_matrix.shape[2]):
                V_noised[idx1, idx2, idx3] = np.random.normal(V_noised[idx1, idx2, idx3], math.sqrt(abs(part * max_value))) 
    noise_lvl = np.linalg.norm(V_noised - V_matrix) / np.linalg.norm(V_matrix) * 100
    return V_noised, noise_lvl

def Smoothing(data, kernel_fun, **params):
    smoothed = np.empty(data.shape)
    if kernel_fun == gaussian_filter:
        for time_idx in np.arange(data.shape[0]):
            smoothed[time_idx, :, :] = gaussian_filter(data[time_idx, :, :], sigma = params['sigma'])
    else:
        raise Exception('Wrong kernel passed into function')
    
    return smoothed

def Smoothing_simple(data, kernel_fun, **params):
    smoothed = np.empty(data.shape)
    if kernel_fun == gaussian_filter:
        smoothed[:, :, :] = kernel_fun(data[:, :, :], sigma = params['sigma'])
    else:
        raise Exception('Wrong kernel passed into function')
    
    return smoothed

        

if __name__ == "__main__":
    n_der = int(sys.argv[1]); op_file_name = sys.argv[2]; filename = sys.argv[3]
    t1 = datetime.datetime.now()

    u_all = np.load(filename)

    print('Executing on grid with uniform nodes:')
    t_array = np.linspace(0, 695, 696)
    x_array = np.linspace(0, 49, 50)
    y_array = np.linspace(0, 49, 50)
        
    grid = np.meshgrid(t_array, x_array, y_array, indexing = 'ij')

    i_idxs = np.arange(u_all.shape[0])
    j_idxs = np.arange(u_all.shape[1])
    k_idxs = np.arange(u_all.shape[2])
    global PolyBoundary
    PolyBoundary = 5
    u_all = Smoothing(u_all, gaussian_filter, sigma = 9)
    index_array = [((i, j, k), u_all, grid, 8, 2) for i in i_idxs for j in j_idxs for k in k_idxs]

    poolsize = 24
    pool = mp.Pool(poolsize)
    derivatives = pool.map_async(Process_Point_Cheb, index_array)
    pool.close()
    pool.join()
    derivatives = derivatives.get()
    t2 = datetime.datetime.now()

    print('Start:', t1, '; Finish:', t2)
    print('Preprocessing runtime:', t2 - t1)

    if not '.npy' in op_file_name:
        op_file_name += '.npy'
        
    np.save('ssh_field.npy', u_all)        
    np.save(op_file_name, derivatives)    
