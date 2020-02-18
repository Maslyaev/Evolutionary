#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:46 2020

@author: mike_ubuntu
"""

import numpy as np
from src.supplementary import Define_Derivatives
from src.term import normalize_ts, Term
from src.trainer import Equation_Trainer

#    skipped_elems = params['skipped'] if 'skipped' in params.keys() else skipped_elems = 10
#    
#    timeslice = (skipped_elems, -skipped_elems)
#
#    variables = variables[:, timeslice[0]:timeslice[1], skipped_elems:-skipped_elems, skipped_elems:-skipped_elems]


def derivative_evaluator(term, eval_params):
    # term - gene, encoding the term: протетсить на синтезированном гене
    assert 'token_matrices' in eval_params and 'max_power' in eval_params
    if type(term) == Term:
        term = term.gene
    token_matrices = eval_params['token_matrices']
    value = np.copy(token_matrices[0])
    for var_idx in np.arange(eval_params['max_power'], term.shape[0], eval_params['max_power']):
        power = (np.sum(term[var_idx : var_idx + eval_params['max_power']]))
        value *= eval_params['token_matrices'][int(var_idx / float(eval_params['max_power']))] ** int(power)
    value = normalize_ts(value)
    value = value.reshape(np.prod(value.shape))
    return value    
   

if __name__ == '__main__':
    u_initial = np.load('Preprocessing/ssh_field.npy')
    derivatives = np.load('Preprocessing/Derivatives.npy')
    variables = np.ones((2 + derivatives.shape[1], ) + u_initial.shape)
    variables[1, :] = u_initial
    for i_outer in range(0, derivatives.shape[1]):
        variables[i_outer+2] = derivatives[:, i_outer].reshape(variables[i_outer+2].shape) 
        
    skipped_elems = 15
    
    timeslice = (skipped_elems, -skipped_elems)
    
    token_names = Define_Derivatives(u_initial.ndim, max_order = 2)
    print(token_names)
    variables = variables[:, timeslice[0]:timeslice[1], skipped_elems:-skipped_elems, skipped_elems:-skipped_elems]
        
    Trainer = Equation_Trainer(token_list=token_names, evaluator = derivative_evaluator, evaluator_params={'token_matrices':variables})
    Trainer.Parameters_grid(('alpha', 'a_proc', 'r_crossover', 'r_mutation', 'mut_chance', 'pop_size', 'eq_len'), (1., 10., 5), 0.2, 0.6, 0.5, 1.0, 10, 6)
    Trainer.Train(epochs = 200)
#def Process_Zones(token_list, evaluator, eval_params, part = (1, 1), exec_type = 'serial', poolsize = 10, alpha = 0.05, timeslice = None):
#    
#
#        
#def Main(part = (1, 1), exec_type = 'serial', poolsize = 10, alpha = 0.05, timeslice = None):  
#    t1 = datetime.datetime.now()
#    u_all = np.load('Preprocessing/ssh_field.npy')
#
#    data_generator = EPDE.Slice_Data_3D(u_all, part_tuple = part)
#    
#    data = []
#    for idx in range(part[0]*part[1]):
#        temp_data = next(data_generator)
#        print(temp_data[0].shape)
#        data.append(temp_data)
#
#    print('List of data matrixes:', type(data), len(data))
#        
#    if exec_type == 'map':
#        with multiprocessing.Pool(poolsize) as pool:
#            res = pool.map(Process_Cell, data)
#        for result in res:
#            print(result)
#
#    if exec_type == 'spawn':
#        with multiprocessing.Pool(poolsize) as pool:
#            res = pool.map(Process_Cell, data)
#        for result in res:
#            print(result)
#        
#    elif exec_type == 'serial':
#        res = []
#        for x in data:
#            result = Process_Cell(x, alpha, timeslice)
#            res.append(result)
#
#    t2 = datetime.datetime.now()
#    print('Start:', t1, '; Finish:', t2)
#    print('Runtime:', t2 - t1)
#    return res