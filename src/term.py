#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:39:55 2020

@author: mike_ubuntu
"""

import numpy as np

from src.supplementary import Encode_Gene


def Check_Unqueness(term, equation):
    if type(term) == Term:
        return not any([all(term.gene == equation_term.gene) for equation_term in equation])
    else:
        return not any([all(term == equation_term.gene) for equation_term in equation])

def normalize_ts(Input):    # Normalization of data time-frame
    Matrix = np.copy(Input)
    for i in np.arange(Matrix.shape[0]):
        norm  = np.abs(np.max(np.abs(Matrix[i, :])))
        if norm != 0:
            Matrix[i] = Matrix[i] / norm
        else:
            Matrix[i] = 1
    return Matrix


class Term:
    def __init__(self, gene = None, tokens_list = ['1', 'u'], init_random = False, 
                 label_dict = None, max_power = 2, max_factors_in_term = 2):

        """
        Class for the possible terms of the PDE, contating both packed symbolic form, and values on the grid;
        
        Attributes:
            gene : 1d - array of ints \r\n
            An array of 0 and 1, contating packed symbolic form of the equation. Shape: number_of_functions * max_power. Each subarray of max_power length 
            contains information about power of the corresponding function (power = sum of ones in the substring). Can be passed as the parameter into the 
            class during initiation;
            
            value : matrix of floats \r\n
            An array, containing value of the term in the grid in the studied space. It can be acquired from the self.Calculate_Value() method;
            
        Parameters:
            
            gene : 1d - array of integers \r\n
            Initiation of the gene for term. For further details look into the attributes;
            
            tokens_list : list of strings \r\n
            List of symbolic forms of the functions, that can be in the resulting equation;
            
            init_random : boolean, base value of False \r\n
            False, if the gene is created by passed label_dict, or gene. If True, than the gene is randomly generated, according to the parameters of max_power
            and max_factors_in_term;
            
            label_dict : dictionary \r\n
            Dictionary, containing information about the term: key - string of function symbolic form, value - power; 
            
            max_power : int, base value of 2 \r\n
            Maximum power of one function, that can exist in the term of the equation;
            
            max_factors_in_term : int, base value of 2 \r\n
            Maximum number of factors, that can exist in the equation; 
            
        """
        
        #assert init_random or ((gene or label_dict) and (not gene or not label_dict)), 'Gene initialization done incorrect'
        self.max_factors_in_term = max_factors_in_term            
        self.max_power = min(max_power, max_factors_in_term); self.tokens = tokens_list

        if init_random:
            self.Randomize_Gene() 
        else:    
            if type(gene) == np.ndarray:
                self.gene = gene
            else:
                self.gene = Encode_Gene(label_dict, tokens_list, self.max_power)  #np.empty(shape = len(variable_list) * max_power)
    
    
    def Randomize_Gene(self):
        
        factor_num = np.random.randint(low = 0, high = self.max_factors_in_term + 1)
        self.gene = np.zeros(shape = len(self.tokens) * self.max_power)
        self.gene[0] = 1
        
        for factor_idx in range(factor_num):
            #print(factor_idx)
            while True:
                factor_choice_idx = self.max_power * np.random.randint(low = 1, high = len(self.tokens))
                if self.gene[factor_choice_idx + self.max_power - 1] == 0:
                    break
                
            addendum = 0
            while self.gene[factor_choice_idx + addendum] == 1:
                addendum += 1
            self.gene[factor_choice_idx + addendum] = 1
       
#    def Calculate_Value(self, variables, normalize = True):
#        self.value = np.copy(variables[0])
#        for var_idx in np.arange(self.max_power, self.gene.shape[0], self.max_power):
#            power = (np.sum(self.gene[var_idx : var_idx + self.max_power]))
#            self.value *= variables[int(var_idx / float(self.max_power))] ** int(power)
#        if normalize: self.value = normalize_ts(self.value)
#        self.value = self.value.reshape(np.prod(self.value.shape))


    def Remove_Dublicated_Factors(self, allowed_factors, background_terms):
        gene_cleared = np.copy(self.gene)
        factors_cleared = 0
        allowed = list(np.nonzero(allowed_factors)[0])
        
        for idx in range(allowed_factors.size):
            if np.sum(self.gene[idx*self.max_power : (idx+1)*self.max_power]) > 0 and not allowed_factors[idx]:
                factors_cleared += np.sum(self.gene[idx*self.max_power : (idx+1)*self.max_power], dtype=int)
                gene_cleared[idx*self.max_power : (idx+1)*self.max_power] = 0
        
        max_power_elements = [idx for idx in range(len(self.tokens)) if self.gene[idx*self.max_power + self.max_power - 1] == 1]
        allowed = [factor for factor in allowed if not factor in max_power_elements]
        if self.max_power != 1:
            allowed.remove(0)
        
        while True:
            gene_filled = np.copy(gene_cleared)             
            for i in range(factors_cleared):
                selected_idx = np.random.choice(allowed)
                addendum = 0
                while gene_filled[selected_idx*self.max_power + addendum] == 1:
                    addendum += 1
                gene_filled[selected_idx*self.max_power + addendum] = 1
                if addendum == self.max_power - 1:
                    allowed.remove(selected_idx)
            if Check_Unqueness(gene_filled, background_terms):
                self.gene = gene_filled
                break
        
        
    def Mutate(self, background_terms, allowed_factors, reverse_mutation_prob = 0.1):

        allowed = list(np.nonzero(allowed_factors)[0])
        allowed.remove(0)

        if int(np.sum(self.gene[1:])) == 0:
            iteration_idx = 0; max_attempts = 15
            while iteration_idx < max_attempts:
                mutated_gene = np.copy(self.gene)
                new_factor_idx = np.random.choice(allowed)
                mutated_gene[new_factor_idx*self.max_power] = 1
                if Check_Unqueness(mutated_gene, background_terms):
                    self.gene = mutated_gene
                    return
                iteration_idx += 1
                
            while True:
                mutated_gene = np.copy(self.gene)
                new_factor_idx = np.random.choice(allowed, size = 2)
                mutated_gene[new_factor_idx[0] * self.max_power] = 1; mutated_gene[new_factor_idx[1] * self.max_power] = 1
                if Check_Unqueness(mutated_gene, background_terms):
                    self.gene = mutated_gene
                    return
        
        max_power_elements = [idx for idx in range(len(self.tokens)) if self.gene[idx*self.max_power + self.max_power - 1] == 1]
        zero_power_elements = [idx for idx in range(len(self.tokens)) if self.gene[idx*self.max_power] == 0]
        
        iteration_idx = 0; max_attempts = 15
        total_power = int(np.sum(self.gene[1:]))
        
        while True:
            mutated_gene = np.copy(self.gene)
            if np.random.uniform(0, 1) <= reverse_mutation_prob or iteration_idx > 15:
                mutation_type = np.random.choice(['Reduction', 'Increasing'])
                if mutation_type == 'Reduction' or total_power >= self.max_factors_in_term and not iteration_idx > 15:
                    red_factor_idx = np.random.choice([i for i in allowed if i not in zero_power_elements])
                    addendum = self.max_power - 1
                    while mutated_gene[red_factor_idx*self.max_power + addendum] == 0:
                        addendum -= 1
                    mutated_gene[red_factor_idx*self.max_power + addendum] = 0
                else:
                    incr_factor_idx = np.random.choice([i for i in allowed if i not in max_power_elements])
                    addendum = 0
                    while mutated_gene[incr_factor_idx*self.max_power + addendum] == 1:
                        addendum += 1
                    mutated_gene[incr_factor_idx*self.max_power + addendum] = 1 
            else:
                red_factor_idx = np.random.choice([i for i in allowed if i not in zero_power_elements])
                addendum = self.max_power - 1
                while mutated_gene[red_factor_idx*self.max_power + addendum] == 0:
                    addendum -= 1
                mutated_gene[red_factor_idx*self.max_power + addendum] = 0                
                incr_factor_idx = np.random.choice([i for i in allowed if i not in max_power_elements])
                addendum = 0
                while mutated_gene[incr_factor_idx*self.max_power + addendum] == 1:
                    addendum += 1
                mutated_gene[incr_factor_idx*self.max_power + addendum] = 1
            if Check_Unqueness(mutated_gene, background_terms):
                self.gene = mutated_gene
                return            