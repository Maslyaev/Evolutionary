#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:50:51 2019

@author: mike_ubuntu
"""

import numpy as np
import EPDE
import datetime
import multiprocessing
import sys

def Preprocessing(u, part = (1, 1), steps = (1, 1, 1), max_order = 2):
    global prefix

    var_generator = EPDE.Create_Var_Matrices_gen(u, method = 'FDM', steps = steps, max_order = max_order)
    var_names = []
    if prefix == '':
        prefix += '_'
    elif not prefix[-1] == '_':
        prefix += '_'     
    for var_idx in range(2 + u.ndim * max_order):
        var, var_name = next(var_generator)
        var_names.append(var_name)
        data_generator = EPDE.Slice_Data_3D(var, part_tp = part)
        for i in range(part[0]):
            for j in range(part[1]):
                data = next(data_generator)[0]    
                data = np.array(data)
                filename = 'Preprocessing/' + (prefix + str(var_name)).replace('/', '') + '_' +  str(i) + '_' + str(j)
                print('Writing cell i: %2d , j: %2d' %(i, j))
                np.save(filename, data)
    var_names = np.array(var_names)
    filename = 'Preprocessing/' + prefix + 'variables.npy'
    np.save(filename, var_names)

        
def Process_Cell(index):    
    i = index[0]; j = index[1]#, Logger = index[2]
    t1 = datetime.datetime.now()
    print('Cell', i, j)
    global prefix, Logger
    if prefix == '':
        prefix += '_'
    elif not prefix[-1] == '_':
        prefix += '_'
    variables_names = np.load('Preprocessing/' + prefix + 'variables.npy')
    variables_names = list(variables_names)
    variables = [np.load('Preprocessing/' + (prefix + str(var)).replace('/', '') + '_' + str(i) + '_' + str(j) + '.npy') for var in variables_names]
    print('Matrix shape:', variables[2].shape)

    """

    Declaration of evolutionary metaparameters

    """
    
    part_with_offsprings = 0.2      # crossover part ratio
    crossover_probability = 0.4     # crossover probability
    mutation_probability = 0.5      # mutation probability
    iter_number = 150               # number of evolutionary algorithm iterations
    alpha = 0.0005                  # sparsity constant
    population_size = 8             # population size
    max_terms_number = 7            # genes number per individual (max number of possible terms in equation)

    population = []    
    for idx in range(population_size):
        temp_chromosome = EPDE.chromosome(variables, variables_names, max_terms_number = 7)
        temp_chromosome.Split_data()
        population.append(temp_chromosome)
    
    for idx in range(iter_number):
        EPDE.Genetic_iteration(i, population, part_with_offsprings, crossover_probability, mutation_probability,
                                              variables, variables_names, estimator_type = 'Lasso', alpha = alpha)
            
    for chromo in population:
        chromo.Apply_ML(estimator_type = 'Lasso', alpha = alpha)
    map(lambda x: x.Calculate_Fitness(), population)            
    population = EPDE.Population_Sort(population)
    print(population[0].fitness_value, population[0].target_key)
    
    for Idx in range(len(population[0].weights)):
        if population[0].weights[Idx] != 0.0:
            print('feature', Idx, ':', population[0].features_keys[Idx], population[0].weights[Idx])
            
    target_term, zipped_list = EPDE.Get_true_coeffs(variables, variables_names, population[0])
    t2 = datetime.datetime.now()
    Logger.Write_logs((t1, t2), (i, j), (target_term, zipped_list))        
        
    
if __name__ == "__main__":
    process_part = bool(int(sys.argv[1]))
    part = (4, 12); poolsize = 10
    global prefix
    prefix = 'partial_grid'
    if prefix == '':
        prefix += '_'
    elif not prefix[-1] == '_':
        prefix += '_'         
    try:
        check = open('Preprocessing/' + prefix + '1_0_0.npy')
        check.close()
    except FileNotFoundError:
        u_all = np.load('ssh_sept.npy')
        u_all = u_all[:,0:int(0.6*u_all.shape[1]), int(0.2*u_all.shape[2]):int(0.8*u_all.shape[2]) ]
        Preprocessing(u_all, part, steps = (1, 5e6, 5e6), max_order = 3)
        
    global Logger    
    Logger = EPDE.Logger()
    index_array = [(i, j) for i in range(part[0]) for j in range(part[1])]
    if process_part:
        print(str(sys.argv))
        i_idxs = range(2, part[0]); j_idxs = range(0, part[1]); 
        index_array = [(i, j) for i in i_idxs for j in j_idxs]

    with multiprocessing.Pool(poolsize) as pool:
        pool.map(Process_Cell, index_array)
    del Logger
