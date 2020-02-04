#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Apr  3 15:50:51 2019

@author: mike_ubuntu
"""

#'Wave2D_100x100x301_uniform.txt'

import numpy as np
import GenPDE_01_10_2 as GenPDE
import EPDE
import datetime
import multiprocessing


def Process_Cell(data, alpha, timeslice = None):   

    u = data[0]; i = data[1]; j = data[2]
    t1 = datetime.datetime.now()
    print('Part size:', u.shape, '; Cell', i, j)
    variables, variables_names = EPDE.Create_Var_Matrices(u, steps = (1, 1, 1), max_order = 2)   

    var_1 = np.load('Preprocessing/ssh_field.npy')
    variables[1] = var_1
    derivatives = np.load('Preprocessing/Derivatives.npy')
    for i_outer in range(0, derivatives.shape[1]):
        variables[i_outer+2] = derivatives[:, i_outer].reshape(variables[i_outer+2].shape)
        

    skipped_elems = 15
    
    if not timeslice:
        timeslice = (skipped_elems, -skipped_elems)
    
    variables = variables[:, timeslice[0]:timeslice[1], skipped_elems:-skipped_elems, skipped_elems:-skipped_elems]
    

    part_with_offsprings = 0.2
    crossover_probability = 0.4
    mutation_probability = 0.5
    iter_number = 100
    
    population_size = 8             # population size
    max_terms_number = 6            # genes number per individual (max number of possible terms in equation)


    population = GenPDE.Population(variables, variables_names, 
                                   population_size = population_size, part_with_offsprings = part_with_offsprings, 
                                   crossover_probability = crossover_probability,
                                   mutation_probability=mutation_probability, alpha = alpha,
                                   terms_number = max_terms_number)

    population.Initiate_Evolution(iter_number = iter_number, estimator_type='Lasso', log_file = None, test_indicators = True)
    
    population.Calculate_True_Weights()
                    
    t2 = datetime.datetime.now()
    res = ((t1, t2), (i, j), (population.target_term, population.zipped_list)) 
    print('result:', res)            
    return res    
    

def Main(part = (1, 1), exec_type = 'serial', poolsize = 10, alpha = 0.05, timeslice = None):  
    
    t1 = datetime.datetime.now()
    u_all = np.load('Preprocessing/ssh_field.npy')

    data_generator = EPDE.Slice_Data_3D(u_all, part_tuple = part)
    
    data = []
    for idx in range(part[0]*part[1]):
        temp_data = next(data_generator)
        print(temp_data[0].shape)
        data.append(temp_data)

    print('List of data matrixes:', type(data), len(data))
        
    if exec_type == 'map':
        with multiprocessing.Pool(poolsize) as pool:
            res = pool.map(Process_Cell, data)
        for result in res:
            print(result)

    if exec_type == 'spawn':
        with multiprocessing.Pool(poolsize) as pool:
            res = pool.map(Process_Cell, data)
        for result in res:
            print(result)
        
    elif exec_type == 'serial':
        res = []
        for x in data:
            result = Process_Cell(x, alpha, timeslice)
            res.append(result)

    t2 = datetime.datetime.now()
    print('Start:', t1, '; Finish:', t2)
    print('Runtime:', t2 - t1)
    return res
    
if __name__ == "__main__":
    result = []
    
    timeslices = np.linspace(0, 696, 30, dtype=int)
    for i in np.arange(25):
        alpha = 0.037
        result.append((i, Main(part = (1, 1), poolsize = 1, exec_type = 'serial', alpha=alpha, timeslice = (timeslices[i], timeslices[i+1]))))
    
    for res in result:
        print('Day:', res[0])
        print(res[1])    

     