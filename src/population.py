#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:26:03 2020

@author: mike_ubuntu
"""

import numpy as np
import copy 

from src.term import Check_Unqueness
from src.equation import Equation
from src.supplementary import *

class Population:
    def __init__(self, tokens, population_size, part_with_offsprings,
                 crossover_probability, mutation_probability, alpha, terms_number = 8, max_factors_in_terms = 2, max_power = 2): 
        
        self.tokens = tokens
        
        self.part_with_offsprings = part_with_offsprings
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.alpha = alpha
        self.max_power = max_power

        self.pop_size = population_size
        self.population = [Equation(self.tokens, terms_number, max_factors_in_terms) for i in range(population_size)]
        print(self.population[0].terms[0].gene, self.population[0].terms[1].gene, self.population[0].terms[5].gene)
        #time.sleep(15)
        for eq in self.population:
            eq.Split_data()

    def Genetic_Iteration(self, estimator_type, elitism = 1):
        self.population = Population_Sort(self.population)
        self.population = self.population[:self.pop_size]
#        for idx in range(len(self.population[0].terms)):
#            if idx < self.population[0].target_idx:
#                print(self.population[0].terms[idx].gene, self.population[0].weights[idx])
#            elif idx == self.population[0].target_idx:
#                print(self.population[0].terms[idx].gene, 1)
#            else:
#                print(self.population[0].terms[idx].gene, self.population[0].weights[idx-1])
                
        children = Tournament_crossover(self.population, self.part_with_offsprings, self.tokens, 
                                        crossover_probability = self.crossover_probability)

        map(lambda x: x.Split_data, children)
        map(lambda x: x.Calculate_Fitness, children)
        self.population = self.population + children

        for i in range(elitism, len(self.population)):
            self.population[i].Mutate(mutation_probability = self.mutation_probability)


    def Initiate_Evolution(self, iter_number, estimator_type, log_file = None, test_indicators = False):
        self.fitness_values = np.empty(iter_number)
        for idx in range(iter_number):
            print('iteration %3d' % idx)
            self.Genetic_Iteration(estimator_type = estimator_type)
            self.population = Population_Sort(self.population)
            self.fitness_values[idx]= self.population[0].fitness_value
            if log_file: log_file.Write_apex(self.population[0], idx)
            if test_indicators: 
                print(self.population[0].fitness_value, self.population[1].fitness_value)
                print(self.population[0].terms[self.population[0].target_idx].gene)
        return self.fitness_values

    def Calculate_True_Weights(self):
        for equation in self.population:
            equation.Apply_ML(estimator_type = 'Lasso', alpha = self.alpha)
        map(lambda x: x.Calculate_Fitness(), self.population)            
        self.population = Population_Sort(self.population)
        print('Final gene:', self.population[0].terms[self.population[0].target_idx].gene)
        print(self.population[0].fitness_value, Decode_Gene(self.population[0].terms[self.population[0].target_idx].gene,
              self.tokens, self.max_power))
        print('weights:', self.population[0].weights)       
        self.target_term, self.zipped_list = Get_true_coeffs(self.variables, self.tokens, self.population[0], self.max_power)  
        
def Crossover(equation_1, equation_2, variables, tokens, crossover_probability = 0.1):

    if len(equation_1.terms) != len(equation_2.terms):
        raise IndexError('Equations have diffferent number of terms')

    result_equation_1 = copy.deepcopy(equation_1) #Equation(variables, variables_names, terms_number = len(equation_1.terms))
    result_equation_2 = copy.deepcopy(equation_1) #Equation(variables, variables_names, terms_number = len(equation_2.terms))      

    for i in range(2, len(result_equation_1.terms)):
        if np.random.uniform(0, 1) <= crossover_probability and Check_Unqueness(result_equation_1.terms[i], result_equation_2.terms) and Check_Unqueness(result_equation_2.terms_label[i], result_equation_1.terms_label):
            internal_term = result_equation_1.terms[i]
            result_equation_1.terms[i] = result_equation_2.terms[i]
            result_equation_2.terms[i] = internal_term

    return result_equation_1, result_equation_2


def Parent_selection_for_crossover(population, tournament_groups = 2):
    selection_indexes = np.random.choice(len(population), tournament_groups, replace = False)
    candidates = [population[idx] for idx in selection_indexes]
    parent_idx = [idx for _, idx in sorted(zip(candidates, selection_indexes), key=lambda pair: pair[0].fitness_value)][-1] #np.argmax([population[y].fitness_value for y in selection_indexes])
    parent = population[parent_idx]
    return parent, parent_idx


def Tournament_crossover(population, part_with_offsprings, variables, tokens, 
                         tournament_groups = 2, crossover_probability = 0.1):
    children = []
    for i in range(int(len(population)*part_with_offsprings)):
        parent_1, parent_1_idx = Parent_selection_for_crossover(population, tournament_groups)
        parent_2, parent_2_idx = Parent_selection_for_crossover(population, tournament_groups)
        child_1, child_2 =  Crossover(parent_1, parent_2, variables, tokens,
                                                       crossover_probability = crossover_probability)
        child_1.Split_data(); child_2.Split_data()
        children.append(child_1); children.append(child_2)
    return children        