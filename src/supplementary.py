#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:33:34 2020

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def Decode_Gene(gene, variables_names, max_power = 2):
    term_dict = {}
    for i in range(0, gene.shape[0], max_power):
        term_dict[variables_names[int(i/max_power)]] = int(np.sum(gene[i:i+max_power]))
    return term_dict


def Encode_Gene(label_dict, variables_names, max_power = 2):
    gene = np.zeros(shape = len(variables_names) * max_power)

    for i in range(len(variables_names)):
        if variables_names[i] in label_dict:
            for power in range(label_dict[variables_names[i]]):
                gene[i*max_power + power] = 1
                
    return gene


def Population_Sort(input_popuation):
    output_population = input_popuation
    
    for j in range(1, len(output_population)):
        key_chromosome = output_population[j]
        i = j - 1        
        while i >= 0 and output_population[i].fitness_value > key_chromosome.fitness_value:
            output_population[i+1] = output_population[i]
            i = i - 1
        output_population[i+1] = key_chromosome
        
    return list(reversed(output_population))


def Get_true_coeffs(variables, variables_names, equation, max_power = 2):
    target = equation.terms[equation.target_idx]
    print('Target key:', Decode_Gene(target.gene, variables_names, max_power))
    
    target_vals = np.copy(variables[0])
    for idx in range(0, target.gene.size, target.max_power):
        target_vals *= variables[int(idx/target.max_power)] ** np.sum(target.gene[idx : idx + target.max_power])
    target_vals = np.reshape(target_vals, np.prod(target_vals.shape))

    features_list = []
    features_list_labels = []
    for i in range(len(equation.terms)):
        if i == equation.target_idx:
            continue
        idx = i if i < equation.target_idx else i-1
        if equation.weights[idx] != 0:
            features_list_labels.append(Decode_Gene(equation.terms[i].gene, variables_names, max_power))
            feature_vals = np.copy(variables[0])
            
            for gene_idx in range(0, equation.terms[i].gene.size, equation.terms[i].max_power):
                feature_vals *= (variables[int(gene_idx/equation.terms[i].max_power)] ** 
                                          np.sum(equation.terms[i].gene[gene_idx : gene_idx + equation.terms[i].max_power]))
            
            feature_vals = np.reshape(feature_vals, np.prod(feature_vals.shape))
            features_list.append(feature_vals)

    if len(features_list) == 0:
        return Decode_Gene(target.gene, variables_names, max_power), [('0', 1)]
    features = features_list[0]
    if len(features_list) > 1:
        for i in range(1, len(features_list)):
            features = np.vstack([features, features_list[i]])
    features = np.transpose(features)    
    estimator = LinearRegression()
    try:
        estimator.fit(features, target_vals)
    except ValueError:
        features = features.reshape(-1, 1)
        estimator.fit(features, target_vals)
    weights = estimator.coef_
    return Decode_Gene(target.gene, variables_names, max_power), list(zip(features_list_labels, weights))    