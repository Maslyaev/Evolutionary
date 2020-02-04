#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:22:06 2019

@author: mike_ubuntu
"""

import warnings
import datetime
import time
import copy

import numpy as np
import math

from sklearn.linear_model import LinearRegression, Lasso, Ridge
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numba as nb

def Slice_Data_3D(matrix, part = 4, part_tuple = None):     # Input matrix slicing for separate domain calculation
    if part_tuple:
        for i in range(part_tuple[0]):
            for j in range(part_tuple[1]):
                yield matrix[:, i*int(matrix.shape[1]/float(part_tuple[0])):(i+1)*int(matrix.shape[1]/float(part_tuple[0])), 
                             j*int(matrix.shape[2]/float(part_tuple[1])):(j+1)*int(matrix.shape[2]/float(part_tuple[1]))], i, j   
    part_dim = int(math.sqrt(part))
    for i in range(part_dim):
        for j in range(part_dim):
            yield matrix[:, i*int(matrix.shape[1]/float(part_dim)):(i+1)*int(matrix.shape[1]/float(part_dim)), 
                         j*int(matrix.shape[2]/float(part_dim)):(j+1)*int(matrix.shape[2]/float(part_dim))], i, j


def Prepare_Data_matrixes(raw_matrix, dim_info):
    resulting_matrix = np.reshape(raw_matrix, dim_info)
    return resulting_matrix 

def Differentiate_by_Matrix(U_input, var_index, step, order): # Differentiation of data: order =< 3
    if order == 1:
        left = tf.Variable((-1/(2*float(step)))*np.roll(U_input,shift=1,axis=var_index), name="matrix_left")
        right = tf.Variable((1/(2*float(step)))*np.roll(U_input,shift=-1,axis=var_index), name="matrix_right")
        init = tf.variables_initializer([left, right], name="init")
        u1 = tf.add(left, right)

    if order == 2:
        left = tf.Variable((1/(pow(float(step), 2)))*np.roll(U_input,shift=1,axis=var_index), name="matrix_left")        
        center = tf.Variable((-2/(pow(float(step), 2)))*U_input, name="matrix_center")
        right = tf.Variable((1/(pow(float(step), 2)))*np.roll(U_input,shift=-1,axis=var_index), name="matrix_right")
        init = tf.variables_initializer([left, center, right], name="init")        
        u1 = tf.add_n([left, center, right])

    if order == 3:
        leftmost = tf.Variable((-1/(2*pow(float(step), 3)))*np.roll(U_input,shift=2,axis=var_index), name="matrix_leftmost")
        left = tf.Variable((2/(2*pow(float(step), 3)))*np.roll(U_input,shift=1,axis=var_index), name="matrix_left")
        right = tf.Variable((-2/(2*pow(float(step), 3)))*np.roll(U_input,shift=-1,axis=var_index), name="matrix_right")        
        rightmost = tf.Variable((1/(2*pow(float(step), 3)))*np.roll(U_input,shift=-2,axis=var_index), name="matrix_rightmost")
        init = tf.variables_initializer([leftmost, left, right, rightmost], name="init")
        u1 = tf.add_n([leftmost, left, right, rightmost])

    with tf.Session() as s:
        s.run(init)
        der = (s.run(u1))        
    return der

def Create_Var_Matrices(U_input, method = 'FDM', steps = (1, 1), max_order = 3):
    var_names = ['1', 'u']
    
    for var_idx in range(U_input.ndim):
        for order in range(max_order):
            if order == 0:
                var_names.append('du/dx'+str(var_idx+1))
            else:
                #print(order+1)
                var_names.append('d^'+str(order+1)+'u/dx'+str(var_idx+1)+'^'+str(order+1))
                
    yield np.ones(U_input.shape), var_names[0]
    yield U_input, var_names[1]
    for var_idx in range(U_input.ndim):
        for var_order in range(max_order):
            print(2 + var_order + max_order * var_idx)
            temp = Differentiate_by_Matrix(U_input, var_index = var_idx, step = steps[var_idx], order = var_order+1)
            print(type(temp), temp.shape)
            yield temp, var_names[2 + var_order + max_order * var_idx]

#@nb.jit((nb.float64[:, :, :])(nb.float64[:, :, :]), nopython=True)
def norm_time_series(Input):    # Normalization of data time-frame
    Matrix = np.copy(Input)
    for i in np.arange(Matrix.shape[0]):
        norm  = np.abs(np.max(np.abs(Matrix[i, :])))
        if norm != 0:
            Matrix[i] = Matrix[i] / norm
        else:
            Matrix[i] = 1
    return Matrix

# Defining term class
# --------------------------------------------------------------------------------------------------------------------

class Term:
    def __init__(self, gene = None, var_label_list = ['1', 'u'], init_random = False, 
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
            
            var_label_list : list of strings \r\n
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
        self.max_power = min(max_power, max_factors_in_term); self.var_labels = var_label_list

        if init_random:
            self.Randomize_Gene() 
        else:    
            if type(gene) == np.ndarray:
                self.gene = gene
            else:
                self.gene = Encode_Gene(label_dict, var_label_list, self.max_power)  #np.empty(shape = len(variable_list) * max_power)
    
    
    def Randomize_Gene(self):
        
        factor_num = np.random.randint(low = 0, high = self.max_factors_in_term + 1)
        self.gene = np.zeros(shape = len(self.var_labels) * self.max_power)
        self.gene[0] = 1
        
        for factor_idx in range(factor_num):
            while True:
                factor_choice_idx = self.max_power * np.random.randint(low = 1, high = len(self.var_labels))
                if self.gene[factor_choice_idx + self.max_power - 1] == 0:
                    break
                
            addendum = 0
            while self.gene[factor_choice_idx + addendum] == 1:
                addendum += 1
            self.gene[factor_choice_idx + addendum] = 1
       
    def Calculate_Value(self, variables, normalize = True):
        self.value = np.copy(variables[0])
        for var_idx in np.arange(self.max_power, self.gene.shape[0], self.max_power):
            power = (np.sum(self.gene[var_idx : var_idx + self.max_power]))
            self.value *= variables[int(var_idx / float(self.max_power))] ** int(power)
        if normalize: self.value = norm_time_series(self.value)
        self.value = self.value.reshape(np.prod(self.value.shape))


    def Remove_Dublicated_Factors(self, allowed_factors, background_terms):
        gene_cleared = np.copy(self.gene)
        factors_cleared = 0
        allowed = list(np.nonzero(allowed_factors)[0])
        
        for idx in range(allowed_factors.size):
            if np.sum(self.gene[idx*self.max_power : (idx+1)*self.max_power]) > 0 and not allowed_factors[idx]:
                factors_cleared += np.sum(self.gene[idx*self.max_power : (idx+1)*self.max_power], dtype=int)
                gene_cleared[idx*self.max_power : (idx+1)*self.max_power] = 0
        
        max_power_elements = [idx for idx in range(len(self.var_labels)) if self.gene[idx*self.max_power + self.max_power - 1] == 1]
        allowed = [factor for factor in allowed if not factor in max_power_elements]
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
        
        max_power_elements = [idx for idx in range(len(self.var_labels)) if self.gene[idx*self.max_power + self.max_power - 1] == 1]
        zero_power_elements = [idx for idx in range(len(self.var_labels)) if self.gene[idx*self.max_power] == 0]
        
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
                

# --------------------------------------------------------------------------------------------------------------------

# Setting equation class
# --------------------------------------------------------------------------------------------------------------------
 
class Equation:
    def __init__(self, variables, variables_names, terms_number = 6, max_factors_in_term = 2): 

        """

        Class for the single equation for the dynamic system.
            
        attributes:
            terms : list of Term objects \r\n
            List, containing all terms of the equation; first 2 terms are reserved for constant value and the input function;
        
            target_idx : int \r\n
            Index of the target term, selected in the Split phase;
        
            target : 1-d array of float \r\n
            values of the Term object, reshaped into 1-d array, designated as target for application in sparse regression;
            
            features : matrix of float \r\n
            matrix, composed of terms, not included in target, value columns, designated as features for application in sparse regression;
        
            fitness_value : float \r\n
            Inverse value of squared error for the selected target function and features and discovered weights; 
        
            estimator : sklearn estimator of selected type \r\n
        
        parameters:
            variables : matrix of floats \r\n 
            Matrix of derivatives: first axis through various orders/coordinates in order: ['1', 'f', all derivatives by one coordinate axis
            in increasing order, ...]; second axis: time, further - spatial coordinates;

            variables_names : list of strings \r\n
            Symbolic forms of functions, including derivatives;

            terms_number : int, base value of 6 \r\n
            Maximum number of terms in the discovered equation; 

            max_factors_in_term : int, base value of 2\r\n
            Maximum number of factors, that can form a term (e.g. with 2: df/dx_1 * df/dx_2)

        """
        
        self.variables = variables; self.variables_names = variables_names
        self.terms = []
        self.terms_number = terms_number; self.max_factors_in_term = max_factors_in_term
        
        if (terms_number <= 5): 
            raise Exception('Number of terms ({}) is too low to contain all required ones'.format(terms_number))        
            
        basic_terms = [{'1':1}, {'1':1, 'u':1}] #, {'1':1, 'du/dx1':1}, {'1':1, 'du/dx2':1}
        self.terms.extend([Term(var_label_list=variables_names, label_dict = label) for label in basic_terms])
        
        for i in range(2, terms_number):
            print('creating term number', i)
            new_term = Term(var_label_list=variables_names, init_random = True, max_factors_in_term = self.max_factors_in_term)
            #print('Term:', new_term.gene)
            while not Check_Unqueness(new_term, self.terms):
                print('Generationg random term for idx:', i)
                new_term = Term(var_label_list=variables_names, init_random = True, max_factors_in_term = self.max_factors_in_term)
                print(Check_Unqueness(new_term, self.terms), new_term.gene)
            self.terms.append(new_term)


    def Apply_ML(self, estimator_type = 'Lasso', alpha = 0.001): # Apply estimator to get weights of the equation
        self.Fit_estimator(estimator_type = estimator_type, alpha = alpha)

    
    def Calculate_Fitness(self): # Calculation of fitness function as the inverse value of L2 norm of error
        #print('weights:', self.weights)
        self.fitness_value = 1 / (np.linalg.norm(np.dot(self.features, self.weights) - self.target, ord = 2)) 
        return self.fitness_value

        
    def Split_data(self): 
        
        '''
        
        Separation of target term from features & removal of factors, that are in target, from features
        
        '''
        
        self.target_idx = np.random.randint(low = 1, high = len(self.terms)-1)
        self.terms[self.target_idx].Calculate_Value(self.variables)
        self.target = self.terms[self.target_idx].value
        self.allowed_derivs = np.ones(len(self.variables_names))
        

        for idx in range(1, self.allowed_derivs.size):
            if self.terms[self.target_idx].gene[idx * self.terms[self.target_idx].max_power] == 1: self.allowed_derivs[idx] = 0
        
        for feat_idx in range(len(self.terms)): # \
            if feat_idx == 0:
                self.terms[feat_idx].Calculate_Value(self.variables)
                self.features = self.terms[feat_idx].value
            elif feat_idx != 0 and self.target_idx != feat_idx:

                self.terms[feat_idx].Remove_Dublicated_Factors(self.allowed_derivs, self.terms[:feat_idx]+self.terms[feat_idx+1:])
                self.terms[feat_idx].Calculate_Value(self.variables) 
                self.features = np.vstack([self.features, self.terms[feat_idx].value])
            else:
                continue
        self.features = np.transpose(self.features)

            
    def Fit_estimator(self, estimator_type = 'Ridge', alpha = 0.001): # Fitting selected estimator
        if estimator_type == 'Lasso':
            self.estimator = Lasso(alpha = alpha)
            self.estimator.fit(self.features, self.target) 
        elif estimator_type == 'Ridge':
            self.estimator = Ridge(alpha = alpha)
            self.estimator.fit(self.features, self.target) 
        else:
            self.estimator = LinearRegression()
            self.estimator.fit(self.features, self.target) 
        self.weights = self.estimator.coef_

      
    def Mutate(self, mutation_probability = 0.4):
        for i in range(4, len(self.terms)):
            if np.random.uniform(0, 1) <= mutation_probability and i != self.target_idx:
                self.terms[i].Mutate(self.terms[:i] + self.terms[i+1:], self.allowed_derivs)
                self.terms[i].Calculate_Value(self.variables)
    
                
                
# -------------------------------------------------------------------------------------------------------------------- 

# Setting equation class
# --------------------------------------------------------------------------------------------------------------------

class Population:
    def __init__(self, variables, var_names, population_size, part_with_offsprings,
                 crossover_probability, mutation_probability, alpha, terms_number = 8, max_factors_in_terms = 2, max_power = 2): 
        
        self.variables = variables; self.variables_names = var_names
        
        self.part_with_offsprings = part_with_offsprings
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.alpha = alpha
        self.max_power = max_power

        self.pop_size = population_size
        self.population = [Equation(self.variables, self.variables_names, terms_number, max_factors_in_terms) for i in range(population_size)]
        print(self.population[0].terms[0].gene, self.population[0].terms[1].gene, self.population[0].terms[5].gene)
        #time.sleep(15)
        for eq in self.population:
            eq.Split_data()

    def Genetic_Iteration(self, estimator_type):
        for eq in self.population:
            eq.Apply_ML(estimator_type = estimator_type, alpha = self.alpha)
            
        for eq in self.population:
            eq.Calculate_Fitness()

        self.population = Population_Sort(self.population)
                
        children = Tournament_crossover(self.population, self.part_with_offsprings, self.variables, self.variables_names, 
                                        crossover_probability = self.crossover_probability)

        for i in range(len(children)):
            self.population[len(self.population)-1-len(children)+i] = children[i]

        for i in range(int(len(self.population)*self.part_with_offsprings), len(self.population)):
            self.population[i].Mutate(mutation_probability = self.mutation_probability)
            self.population[i].Split_data()        


    def Initiate_Evolution(self, iter_number, estimator_type, log_file = None, test_indicators = False):
        for idx in range(iter_number):
            print('iteration %3d' % idx)
            self.Genetic_Iteration(estimator_type = estimator_type)
            if log_file: log_file.Write_apex(self.population[0], idx)
            if test_indicators: 
                self.population = Population_Sort(self.population)
                print(self.population[0].fitness_value, self.population[1].fitness_value)
                print(self.population[0].terms[self.population[0].target_idx].gene)


    def Calculate_True_Weights(self):
        for equation in self.population:
            equation.Apply_ML(estimator_type = 'Lasso', alpha = self.alpha)
        map(lambda x: x.Calculate_Fitness(), self.population)            
        self.population = Population_Sort(self.population)
        print('Final gene:', self.population[0].terms[self.population[0].target_idx].gene)
        print(self.population[0].fitness_value, Decode_Gene(self.population[0].terms[self.population[0].target_idx].gene,
              self.variables_names, self.max_power))
        print('weights:', self.population[0].weights)
          
        self.target_term, self.zipped_list = Get_true_coeffs(self.variables, self.variables_names, self.population[0], self.max_power)      
        
# --------------------------------------------------------------------------------------------------------------------         
        
def Decode_Gene(gene, variables_names, max_power = 2):
    term_dict = {}
    for i in range(0, gene.shape[0], max_power):    # Select data type for gene: first approach - np.array
        term_dict[variables_names[int(i/max_power)]] = int(np.sum(gene[i:i+max_power]))
    return term_dict


def Encode_Gene(label_dict, variables_names, max_power = 2):
    gene = np.zeros(shape = len(variables_names) * max_power)

    for i in range(len(variables_names)):
        if variables_names[i] in label_dict:
            for power in range(label_dict[variables_names[i]]):
                gene[i*max_power + power] = 1
                
    return gene


def Check_Unqueness(term, equation):
    if type(term) == Term:
        return not any([all(term.gene == equation_term.gene) for equation_term in equation])
    else:
        return not any([all(term == equation_term.gene) for equation_term in equation])


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


def Crossover(equation_1, equation_2, variables, variables_names, crossover_probability = 0.1):

    if len(equation_1.terms) != len(equation_2.terms):
        raise IndexError('Equations have diffferent number of terms')

    result_equation_1 = copy.deepcopy(equation_1)
    result_equation_2 = copy.deepcopy(equation_1) 

  

    for i in range(2, len(result_equation_1.terms)):
        if np.random.uniform(0, 1) <= crossover_probability and Check_Unqueness(result_equation_1.terms[i], result_equation_2.terms) and Check_Unqueness(result_equation_2.terms_label[i], result_equation_1.terms_label):
            internal_term = result_equation_1.terms[i]
            result_equation_1.terms[i] = result_equation_2.terms[i]
            result_equation_2.terms[i] = internal_term

    return result_equation_1, result_equation_2


def Parent_selection_for_crossover(population, k_parameter = 0.75):
    selection_indexes = np.random.choice(len(population), 2)
    selection = list(map(lambda x: population[x], selection_indexes)) 

    if selection[1].fitness_value > selection[0].fitness_value:
        temp = selection[1]; selection[1] = selection[0]; selection[0] = temp
        temp_idx = selection_indexes[1]; selection_indexes[1] = selection_indexes[0]; selection_indexes[0] = temp_idx
    
    if (np.random.uniform(0, 1) <= k_parameter):
        parent = selection[0]; parent_idx = selection_indexes[0]
    else:
        parent = selection[1]; parent_idx = selection_indexes[1]
        
    return parent, parent_idx


def Tournament_crossover(population, part_with_offsprings, variables, variables_names, 
                         k_parameter = 0.75, crossover_probability = 0.1):
    children = []
    for i in range(int(len(population)*part_with_offsprings)):
        parent_1, parent_1_idx = Parent_selection_for_crossover(population, k_parameter)
        parent_2, parent_2_idx = Parent_selection_for_crossover(population, k_parameter)
        child_1, child_2 =  Crossover(parent_1, parent_2, variables, variables_names,
                                                       crossover_probability = crossover_probability)
        child_1.Split_data(); child_2.Split_data()
        children.append(child_1); children.append(child_2)
    return children

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

class Predetermined_Equation:
    def __init__(self, feature_terms, target_term, variables, var_names):
        self.terms = [target_term]
        self.terms.extend(feature_terms)
        self.target_idx = 0
        self.weights = np.ones(len(self.terms))
        

def Evaluate_Fitness(feature_terms, target_term, variables, var_names, max_power = 2):
    equation = Predetermined_Equation(feature_terms, target_term, variables, var_names)
 
    for term in equation.terms:
        term.Calculate_Value(variables)

    target_matrix = equation.terms[0].value        
    feature_matrix = equation.terms[1].value
    for idx in range(2, len(feature_terms)+1):
        feature_matrix = np.vstack([feature_matrix, equation.terms[idx].value])    
    print(feature_matrix.shape)
    estimator = Lasso(alpha = 0.006)
    estimator.fit(np.transpose(feature_matrix), target_matrix)
    weights = estimator.coef_
    print(weights)

    return 1 / (np.linalg.norm(np.dot(np.transpose(feature_matrix), weights) - target_matrix, ord = 2))
    
# --------------------------------------------------------------------------------------------------------------------         


class Logger:
    def __init__(self, baseline = True, filename = None):
        if baseline:
            self.logfile_name = 'Logs/' + str(datetime.datetime.now()).replace(' ', '_') + '.txt'
            self.logfile = open(self.logfile_name, 'w')
        else:
            self.logfile_name = filename
            self.logfile = open(self.logfile_name, 'w')


    def Write_string(self, string):
        self.logfile.write(string + '\n')
        
    def Write_logs(self, runtime, cell, equation):
        self.logfile.write('Cell '+ str(cell[0]) + ' ' + str(cell[1]) + ' : ')
        time_delta = runtime[1] - runtime[0]
        self.logfile.write('time:' + str(time_delta)+ '\n')
        self.logfile.write('-1 * '+ str(equation[0]))
        print('Result:', 'Cell '+ str(cell[0]) + ' ' + str(cell[1]) + ' : ', equation)
        for term in equation[1]:    #Оттестить
            try:
                if term[1] >= 0:
                    self.logfile.write(' + '+ str(term[1]) + ' * ' + str(term[0]))
                else:
                    self.logfile.write(' '+ str(term[1]) + ' * ' + str(term[0]))
            except IndexError:
                pass
        self.logfile.write('\n')

    def Write_apex(self, equation, step_idx):
        self.logfile.write('Step ' + str(step_idx) + '-1 * '+ str(equation[0]))
        for term in equation[1]:    
            try:
                if term[1] >= 0:
                    self.logfile.write(' + '+ str(term[1]) + ' * ' + str(term[0]))
                else:
                    self.logfile.write(' '+ str(term[1]) + ' * ' + str(term[0]))
            except IndexError:
                pass
        self.logfile.write('\n')
        
    
    def General_Log(self, runtime):
        time_delta = runtime[1] - runtime[0]
        self.logfile.write('Total runtime: '+ str(time_delta))

        
    def __del__(self):
        self.logfile.close()
        
