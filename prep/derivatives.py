#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:41:16 2020

@author: mike_ubuntu
"""

import numpy as np
import datetime
import multiprocessing as mp

from prep.cheb import Process_Point_Cheb
from prep.smoothing import Smoothing

def Preprocess_derivatives(field, output_file_name = None, mp_poolsize = 4, max_order = 2, polynomial_window = 8, polynomial_boundary = 5):
    t1 = datetime.datetime.now()

    dim_coords = []
    for dim in np.arange(np.ndim(field)):
        dim_coords.append(np.linspace(0, field.shape[dim]-1, field.shape[dim]))

        
    grid = np.meshgrid(*dim_coords, indexing = 'ij')

#    idx_list = [np.arange(field.shape[dim]) for dim in np.arange(np.ndim(field))]

    field = Smoothing(field, 'gaussian', sigma = 9)
    index_array = []
    
    for idx, _ in np.ndenumerate(field):
        index_array.append((idx, field, grid, polynomial_window, max_order, polynomial_boundary))
    

    pool = mp.Pool(mp_poolsize)
    derivatives = pool.map_async(Process_Point_Cheb, index_array)
    pool.close()
    pool.join()
    derivatives = derivatives.get()
    t2 = datetime.datetime.now()

    print('Start:', t1, '; Finish:', t2)
    print('Preprocessing runtime:', t2 - t1)
        
    #np.save('ssh_field.npy', field)   
    if output_file_name:
        if not '.npy' in output_file_name:
            output_file_name += '.npy'        
        np.save(output_file_name, derivatives)
    else:
        return derivatives        
