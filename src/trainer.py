#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:33:54 2020

@author: mike_ubuntu
"""

import numpy as np
import datetime
from src.population import Population


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