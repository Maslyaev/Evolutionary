#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:43:10 2020

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from src.term import Term

from src.term import Check_Unqueness

class Equation:
    def __init__(self, tokens, token_params, evaluator, eval_args, terms_number = 6, max_factors_in_term = 2, max_power = 2): 

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
            Inverse value of squared error for the selected target 2function and features and discovered weights; 
        
            estimator : sklearn estimator of selected type \r\n
        
        parameters:

            Matrix of derivatives: first axis through various orders/coordinates in order: ['1', 'f', all derivatives by one coordinate axis
            in increasing order, ...]; second axis: time, further - spatial coordinates;

            tokens : list of strings \r\n
            Symbolic forms of functions, including derivatives;

            terms_number : int, base value of 6 \r\n
            Maximum number of terms in the discovered equation; 

            max_factors_in_term : int, base value of 2\r\n
            Maximum number of factors, that can form a term (e.g. with 2: df/dx_1 * df/dx_2)

        """

        self.tokens = tokens; self.token_params = token_params
        self.evaluator = evaluator; self.eval_args = eval_args
        self.terms = []
        self.terms_number = terms_number; self.max_factors_in_term = max_factors_in_term; self.max_power = max_power
        
        if (terms_number <= 5): 
            raise Exception('Number of terms ({}) is too low to contain all required ones'.format(terms_number))        
            
        basic_terms = [{'1':1}, {'1':1, 'u':1}] #, {'1':1, 'du/dx1':1}, {'1':1, 'du/dx2':1}
        self.terms.extend([Term(tokens_list=tokens, token_params = self.token_params, label_dict = label, 
                                max_factors_in_term = self.max_factors_in_term) for label in basic_terms])
        
        for i in range(2, terms_number):
            print('creating term number', i)
            new_term = Term(tokens_list = self.tokens, token_params = self.token_params, init_random = True, 
                            max_factors_in_term = self.max_factors_in_term)

            while not Check_Unqueness(new_term, self.terms):
                print('Generationg random term for idx:', i)
                new_term = Term(tokens_list = self.tokens, token_params = self.token_params, init_random = True, 
                                max_factors_in_term = self.max_factors_in_term)
                print(Check_Unqueness(new_term, self.terms), new_term.gene)
            self.terms.append(new_term)

    def Evaluate_equation(self):
        if not 'max_power' in self.eval_args:
            self.eval_args['max_power'] = self.max_power
        self.target = self.terms[self.target_idx].Evaluate(self.evaluator, self.eval_args)
        
        for feat_idx in range(len(self.terms)):
            if feat_idx == 0:
                self.features = self.terms[feat_idx].Evaluate(self.evaluator, self.eval_args)
            elif feat_idx != 0 and self.target_idx != feat_idx:
                temp = self.terms[feat_idx].Evaluate(self.evaluator, self.eval_args)
                self.features = np.vstack([self.features, temp])
            else:
                continue
        self.features = np.transpose(self.features)
        #print(self.features.shape)

    def Apply_ML(self, estimator_type = 'Lasso', alpha = 0.001): # Apply estimator to get weights of the equation
        self.Fit_estimator(estimator_type = estimator_type, alpha = alpha)
            
        
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
    
    
    def Calculate_Fitness(self, penalty_coeff = 0.4): # Calculation of fitness function as the inverse value of L2 norm of error
        # Evaluate target & features
        self.Evaluate_equation()
        self.Apply_ML()
        
        #print('Norm of error:', np.linalg.norm(np.dot(self.features, self.weights) - self.target, ord = 2))
        self.fitness_value = 1 / (np.linalg.norm(np.dot(self.features, self.weights) - self.target, ord = 2)) 
        if np.sum(self.weights) == 0:
            self.fitness_value = self.fitness_value * penalty_coeff
        return self.fitness_value

        
    def Split_data(self): 
        
        '''
        
        Separation of target term from features & removal of factors, that are in target, from features
        
        '''
        
        self.target_idx = np.random.randint(low = 1, high = len(self.terms)-1)
        self.allowed_derivs = np.ones(len(self.tokens))

        for idx in range(1, self.allowed_derivs.size):
            if self.terms[self.target_idx].gene[idx * self.terms[self.target_idx].max_power] == 1: self.allowed_derivs[idx] = 0
        
        for feat_idx in range(len(self.terms)): # \
            if feat_idx == 0:
                continue
            elif feat_idx != 0 and self.target_idx != feat_idx:
                self.terms[feat_idx].Remove_Dublicated_Factors(self.allowed_derivs, self.terms[:feat_idx]+self.terms[feat_idx+1:])
            else:
                continue

      
    def Mutate(self, r_mutation = 0.5, r_param_mutation = 0.5, strict_restrictions):
        for i in range(4, len(self.terms)):
            if np.random.uniform(0, 1) <= r_mutation and i != self.target_idx:
                if np.random.random() < 1/pow(len(self.token_params), 2): # Сомнительная эвристика
                    self.terms[i] = Term(tokens_list = self.tokens, token_params = self.token_params, init_random = True,
                              max_factors_in_term = self.max_factors_in_term)
                else:
                    self.terms[i].Mutate_parameters(r_param_mutation = r_param_mutation, strict_restrictions = strict_restrictions)
                
        self.Calculate_Fitness()