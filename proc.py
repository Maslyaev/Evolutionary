#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb 14 13:11:46 2020

@author: mike_ubuntu
"""

import numpy as np
import collections

from src.supplementary import Define_Derivatives
from src.term import normalize_ts, Term
from src.trainer import Equation_Trainer

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
    token_parameters = collections.OrderedDict([('power', (0, 3))])
    variables = variables[:, timeslice[0]:timeslice[1], skipped_elems:-skipped_elems, skipped_elems:-skipped_elems]
        
    Trainer = Equation_Trainer(token_list=token_names, evaluator = derivative_evaluator, evaluator_params={'token_matrices':variables})
    Trainer.Parameters_grid(('alpha', 'a_proc', 'r_crossover', 'r_param_mutation', 'r_mutation', 'mut_chance', 'pop_size', 'eq_len'), (1., 10., 5), 0.2, 0.6, 0.8, 0.5, 0.8, 10, 6)
    Trainer.Train(epochs = 200)
    
