#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:33:34 2020

@author: mike_ubuntu
"""

import numpy as np

def Slice_Data_3D(matrix, part = 4, part_tuple = None):     # Input matrix slicing for separate domain calculation
    if part_tuple:
        for i in range(part_tuple[0]):
            for j in range(part_tuple[1]):
                yield matrix[:, i*int(matrix.shape[1]/float(part_tuple[0])):(i+1)*int(matrix.shape[1]/float(part_tuple[0])), 
                             j*int(matrix.shape[2]/float(part_tuple[1])):(j+1)*int(matrix.shape[2]/float(part_tuple[1]))], i, j   
    part_dim = int(np.sqrt(part))
    for i in range(part_dim):
        for j in range(part_dim):
            yield matrix[:, i*int(matrix.shape[1]/float(part_dim)):(i+1)*int(matrix.shape[1]/float(part_dim)), 
                         j*int(matrix.shape[2]/float(part_dim)):(j+1)*int(matrix.shape[2]/float(part_dim))], i, j

def Define_Derivatives(dimensionality, max_order = 2):
    var_names = ['1', 'u']
    for var_idx in range(dimensionality):
        for order in range(max_order):
            if order == 0:
                var_names.append('du/dx'+str(var_idx+1))
            else:
                var_names.append('d^'+str(order+1)+'u/dx'+str(var_idx+1)+'^'+str(order+1))
    return tuple(var_names)    

def Create_Var_Matrices(U_input, max_order = 3):
    var_names = ['1', 'u']

    for var_idx in range(U_input.ndim):
        for order in range(max_order):
            if order == 0:
                var_names.append('du/dx'+str(var_idx+1))
            else:
                var_names.append('d^'+str(order+1)+'u/dx'+str(var_idx+1)+'^'+str(order+1))

    variables = np.ones((len(var_names),) + U_input.shape)      
    return variables, tuple(var_names)


def Prepare_Data_matrixes(raw_matrix, dim_info):
    resulting_matrix = np.reshape(raw_matrix, dim_info)
    return resulting_matrix 


def Decode_Gene(gene, variables_names, max_power = 2):
    term_dict = {}
    for i in range(0, gene.shape[0], max_power):
        term_dict[variables_names[int(i/max_power)]] = int(np.sum(gene[i:i+max_power]))
    return term_dict


def Encode_Gene(label_dict, variables_names, max_power = 2):
#    print(type(variables_names), variables_names)
    gene = np.zeros(shape = len(variables_names) * max_power)

    for i in range(len(variables_names)):
        if variables_names[i] in label_dict:
            for power in range(label_dict[variables_names[i]]):
                gene[i*max_power + power] = 1
    return gene

def Population_Sort(input_population):
    pop_sorted = [x for x, _ in sorted(zip(input_population, [individual.fitness_value for individual in input_population]), key=lambda pair: pair[1])]
    return list(reversed(pop_sorted))