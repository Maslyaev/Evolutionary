#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:11:46 2020

@author: mike_ubuntu
"""

import numpy as np
import datetime
from src.population import Population
from src.supplementary import Define_Derivatives
from src.term import normalize_ts, Term

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


def Process_Cell(token_names, **kwargs):   
    t1 = datetime.datetime.now()

    assert 'evaluator' in kwargs.keys() and 'eval_params' in kwargs.keys()
    
    alpha = kwargs['alpha'] if 'alpha' in kwargs.keys() else 0.2
    a_proc = kwargs['a_proc'] if 'a_proc' in kwargs.keys() else 0.2
    r_crossover = kwargs['r_crossover'] if 'r_crossover' in kwargs.keys() else 0.3 
    r_mutation = kwargs['r_mutation'] if 'r_mutation' in kwargs.keys() else 0.3 
    mut_chance = kwargs['mut_chance'] if 'mut_chance' in kwargs.keys() else 0.6
    iter_number = kwargs['iter_number'] if 'iter_number' in kwargs.keys() else 100
    pop_size = kwargs['pop_size'] if 'pop_size' in kwargs.keys() else 8
    eq_len = kwargs['eq_len'] if 'eq_len' in kwargs.keys() else 6
    print('in "process cell" function: ', type(token_names), token_names)
    population = Population(kwargs['evaluator'], kwargs['eval_params'], token_names, 
                                   pop_size = pop_size, a_proc = a_proc, 
                                   r_crossover = r_crossover,
                                   r_mutation=r_mutation, mut_chance = mut_chance, 
                                   alpha = alpha, eq_len = eq_len)

    best_fitnesses = population.Initiate_Evolution(iter_number = iter_number, estimator_type='Lasso', log_file = None, test_indicators = True)
    
    print('Achieved best fitness:', best_fitnesses[-1])
    
    population.Calculate_True_Weights(kwargs['evaluator'], kwargs['eval_params'])
                    
    t2 = datetime.datetime.now()
    res = ((t1, t2), (population.target_term, population.zipped_list), best_fitnesses) 
    print('result:', res[:-1])            
#    return res    


class Equation_Trainer:
    def __init__(self, token_list, evaluator, evaluator_params):
        self.tokens = token_list
        self.evaluator = evaluator
        self.evaluator_params = evaluator_params
        self.tuning_grid = None
    
    
    def Parameters_grid(self, parameters_order, *params):
        self.parameters_order = parameters_order
        parameter_arrays = []
        for parameter in params:
            parameter_arrays.append(parameter if (isinstance(parameter, int) or isinstance(parameter, float)) else np.linspace(parameter[0], parameter[1], parameter[2]))
        self.tuning_grid = np.meshgrid(*parameter_arrays, indexing = 'ij')
      
    def Delete_grid(self):
        self.tuning_grid = None
    
    def Train(self, epochs, parameters_order = None, parameters = None):
        if self.tuning_grid: #.any()
    
            use_params = np.vectorize(Process_Cell, excluded = ['token_names', 'evaluator', 'eval_params', 'iter_number'])
            use_params(token_names = self.tokens, evaluator = self.evaluator, eval_params = self.evaluator_params, iter_number = epochs, 
                       alpha = self.tuning_grid[self.parameters_order.index('alpha')], 
                       a_proc = self.tuning_grid[self.parameters_order.index('a_proc')], 
                       r_crossover = self.tuning_grid[self.parameters_order.index('r_crossover')], 
                       r_mutation = self.tuning_grid[self.parameters_order.index('r_mutation')],
                       mut_chance = self.tuning_grid[self.parameters_order.index('mut_chance')],
                       pop_size = self.tuning_grid[self.parameters_order.index('pop_size')],
                       eq_len = self.tuning_grid[self.parameters_order.index('eq_len')])
        elif parameters: # .any()
            Process_Cell(token_names = self.tokens, evaluator = self.evaluator, eval_params = self.evaluator_params, iter_number = epochs, 
                       alpha = parameters[parameters_order.index('alpha')], 
                       a_proc = parameters[parameters_order.index('a_proc')], 
                       r_crossover = parameters[parameters_order.index('r_crossover')], 
                       r_mutation = parameters[parameters_order.index('r_mutation')],
                       mut_chance = self.tuning_grid[self.parameters_order.index('mut_chance')],
                       pop_size = parameters[parameters_order.index('pop_size')],
                       eq_len = parameters[parameters_order.index('eq_len')])
        else:
            raise ValueError('The EA hyperparameters are not defined')
   

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