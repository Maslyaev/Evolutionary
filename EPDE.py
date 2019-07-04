import warnings
import datetime

import numpy as np
import math
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


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


def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1))):
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots()
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=-abs(np.max(Matrix)),
                          vmax=abs(np.max(Matrix)))
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.show()


def Visualize(V_vis, angle = (40, 60), labels = ('$X$', '$T$', '$U$'), colormap = 'viridis'):
    X, Y = np.meshgrid(np.linspace(0, 100, V_vis.shape[1]), np.linspace(0, 100, V_vis.shape[0]))
    fig = plt.figure(figsize=(16,13))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(labels[0], fontsize=20)
    ax.set_ylabel(labels[1])
    ax.yaxis._axinfo['label']['space_factor'] = 3.0
    ax.plot_surface(X, Y, V_vis, rstride=1, cstride=1,
                    cmap=colormap, edgecolor='none')
    ax.set_zlabel(labels[2], fontsize=20, rotation = 0)
    ax.view_init(angle[0], angle[1])
    plt.show()


def Add_noise(V_matrix, part): # Addition of noise for the stability test: designed for 1D - case
    V_noised = np.copy(V_matrix)
    for idx1 in range(V_matrix.shape[0]):
        max_value = V_matrix[idx1][:].max()
        for idx2 in range(V_matrix.shape[1]):
            V_noised[idx1][idx2] = np.random.normal(V_matrix[idx1][idx2], math.sqrt(abs(part * max_value))) 
    noise_lvl = np.linalg.norm(V_noised - V_matrix) / np.linalg.norm(V_matrix) * 100
    return V_noised, noise_lvl


def Chebyshev_grid(a, b, n):    # Calculation of grid by roots of Chebyshev polynominals for 1D - case
    nodes = np.zeros(n)
    nodes = list(map(lambda x: (b+a)/2. + (b-a)/2.*math.cos(math.pi*(2*x - 1)/(2*n)), range(1, n+1)))
    nodes = np.fliplr([nodes])[0]
    return nodes


def PolyDiff(u, x, deg = 3, diff = 1, width = 5): # Polynomial differentiation, adapted only for 1D - process
    
    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)+1
    du = np.zeros((n,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        points = np.arange(j - width, j + width)

        # Fit to a Chebyshev polynomial
        # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            for boundary_point in range(0,width):
                if d==1: du[boundary_point, d-1]=(u[boundary_point+1]-u[boundary_point])/(x[boundary_point+1]-x[boundary_point])
                if d==2: du[boundary_point, d-1]= (2*u[boundary_point] - 5*u[boundary_point+1] + 4*u[boundary_point+2] - u[boundary_point+3]) / (x[boundary_point+1]-x[boundary_point])**2
                if d==3: du[boundary_point, d-1]= (-2.5*u[boundary_point]+9*u[boundary_point+1]-12*u[boundary_point+2]+7*u[boundary_point+3]-1.5*u[boundary_point+4]) / (x[boundary_point+1]-x[boundary_point])**3
            for boundary_point in range(n-width,n-1):
                if d==1: du[boundary_point, d-1]=(u[boundary_point]-u[boundary_point-1])/(x[boundary_point]-x[boundary_point-1])
                if d==2: du[boundary_point, d-1]= (2*u[boundary_point] - 5*u[boundary_point-1] + 4*u[boundary_point-2] - u[boundary_point-3]) / (x[boundary_point]-x[boundary_point-1])**2
                if d==3: du[boundary_point, d-1]= (2.5*u[boundary_point]-9*u[boundary_point-1]+12*u[boundary_point-2]-7*u[boundary_point-3]+1.5*u[boundary_point-4]) / (x[boundary_point]-x[boundary_point-1])**3
            du[j, d-1] = poly.deriv(m=d)(x[j])

    return np.transpose(du)


def matrix_derivative_poly(U,a,b,dx,deg = 5, diff = 1, width = 6, cheb_grid = False): # Polynomial differentiation, adapted only for 1D - process
    if cheb_grid:
        x=np.array(Chebyshev_grid(a, b, int((b-a)/float(dx)))) 
        #np.loadtxt(open("filename.csv", "rb"), delimiter=",", skiprows=0)
    else:
        x=np.linspace(a,b,num=int((b-a)/float(dx)))
    mat_temp=[]
    mat_der=[]
    
    for u in U:
        mat_temp.append(PolyDiff(u, x,deg=deg,diff=diff,width=width))
    for d in range(len(mat_temp[0])):
        der_temp=list()
        for string in mat_temp:
            der_temp.append(string[d])
        mat_der.append(np.asarray(der_temp))
    return mat_der


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


def Create_Var_Matrices_gen(U_input, method = 'FDM', steps = (1, 1), max_order = 3): # Generator of derivatives matrixes
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
            yield Differentiate_by_Matrix(U_input, var_index = var_idx, step = steps[var_idx], order = var_order+1), var_names[2 + var_order + max_order * var_idx] # Использовать остаток


def norm_time_series(Input):    # Normalization of data time-frame
    Matrix = np.copy(Input)
    for i in range(Matrix.shape[0]):
        norm  = abs(np.max(abs(Matrix[i, :])))
        if norm != 0:
            Matrix[i] = Matrix[i] / norm
        else:
            Matrix[i] = 1
    return Matrix


def Create_term_by_dict(variables, variables_names, term_label): # Get matrix of term values from symbolic form
    term = np.copy(variables[0]) 
    for key, value in term_label.items():
        term *= variables[variables_names.index(key)] ** value
    return term  


def Create_term(variables, variables_names, max_factors_in_term = 2,
                term_type = 'Random', target_term = None):  # Create term: matrix & symbolic form 
    if term_type == 'Random':
        factors_in_term = np.random.randint(low = 0, high = max_factors_in_term)
        term = np.copy(variables[0])
        term_label = {'1': 1}
        for factor_idx in range(factors_in_term + 1):
            factor_choice_idx = np.random.randint(low = 1, high = len(variables))
            if target_term != None:
                while variables_names[factor_choice_idx] in target_term:
                    factor_choice_idx = np.random.randint(low = 1, high = len(variables))
            if variables_names[factor_choice_idx] in term_label:
                term_label[variables_names[factor_choice_idx]] += 1
            else:
                term_label[variables_names[factor_choice_idx]] = 1
        term = Create_term_by_dict(variables, variables_names, term_label)
        #print term_label
                
    elif term_type == 'u': 
        term = np.copy(variables[variables_names.index('u')]); term_label = {'1':1, 'u':1}
    elif term_type == 'Ones':
        term = np.copy(variables[variables_names.index('1')]); term_label = {'1':1}
    elif term_type == 'du/dx2':
        term = np.copy(variables[variables_names.index('du/dx2')]); term_label = {'1':1, 'du/dx2':1}
    elif term_type == 'du/dx1':
        term = np.copy(variables[variables_names.index('du/dx1')]); term_label = {'1':1, 'du/dx1':1}
    else:
        raise Exception('Incorrect query for pre-set term {}'.format(term_type))
    return term, term_label

# Setting chromosome class
# --------------------------------------------------------------------------------------------------------------------

class chromosome:
    def __init__(self, variables, variables_names, terms_number = 6, max_factors_in_term = 2): 

        """

        Initiation of individual for evolutionary algorithm:
            
        variables = list of derivatives values of various orders;
        variables_names = list of symbolic forms of derivatives;
        terms_number = max number of terms in the discovered equation
        max_factors_in_term = max number of factors, that can form a term (e.g. with 2: df/dx_1 * df/dx_2)

        """
        self.variables = variables; self.variables_names = variables_names
        self.terms = []
        self.terms_label = []
        self.terms_number = terms_number; self.max_factors_in_term = max_factors_in_term
        
        if (terms_number <= 5): 
            raise Exception('Number of terms ({}) is too low to contain all required ones'.format(terms_number))        
            
        ones_term, ones_label = Create_term(self.variables, self.variables_names, term_type = 'Ones')
        self.terms.append(norm_time_series(ones_term)); self.terms_label.append(ones_label) 
        u_term, u_label = Create_term(self.variables, self.variables_names, term_type = 'u')
        self.terms.append(norm_time_series(u_term)); self.terms_label.append(u_label) 
        dudx1_term, dudx1_label = Create_term(self.variables, self.variables_names, term_type = 'du/dx1')
        self.terms.append(norm_time_series(dudx1_term)); self.terms_label.append(dudx1_label) 
        dudx2_term, dudx2_label = Create_term(self.variables, self.variables_names, term_type = 'du/dx2')
        self.terms.append(norm_time_series(dudx2_term)); self.terms_label.append(dudx2_label) 
        
        for i in range(4, terms_number):
            term, term_label = Create_term(self.variables, self.variables_names,
                                          max_factors_in_term = self.max_factors_in_term)
            while not Check_Unqueness(term_label, self.terms_label):
                term, term_label = Create_term(self.variables, self.variables_names,
                                               max_factors_in_term = self.max_factors_in_term)
            self.terms_label.append(term_label)
            self.terms.append(norm_time_series(term))
        
        for i in range(len(self.terms)):
            self.terms[i] = np.reshape(self.terms[i], np.prod(self.terms[i].shape))
        

    def Apply_ML(self, estimator_type = 'Lasso', alpha = 0.001): # Apply estimator to get weights of the equation
        self.Fit_estimator(estimator_type = estimator_type, alpha = alpha)

    
    def Calculate_Fitness(self): # Calculation of fitness function as the inverse value of L2 norm of error
        self.fitness_value = 1 / (np.linalg.norm(np.dot(self.features, self.weights) - self.target, ord = 2)) 
        return self.fitness_value

        
    def Split_data(self): 
        
        '''
        
        Separation of target term from features & removal of factors, that are in target, from features
        
        '''
        
        self.features_keys = []
        self.target_idx = np.random.randint(low = 1, high = len(self.terms)-1)
        self.target = self.terms[self.target_idx]
        self.target_key = self.terms_label[self.target_idx]

        self.keys_in_target = self.target_key.keys()
        self.free_keys = [var_name for var_name in self.variables_names if not var_name in self.keys_in_target]

        for feat_idx in range(len(self.terms_label)): # _dict
            if feat_idx == 0:
                self.features = self.terms[feat_idx]
                self.features_keys.append(self.terms_label[feat_idx]) 
            elif feat_idx != 0 and self.target_idx != feat_idx:
                temp_term, self.terms_label[feat_idx] = self.Remove_Dublicated_Factors(self.terms[feat_idx]
                                                                                                  , self.terms_label[feat_idx])
                temp_term = norm_time_series(temp_term)
                temp_term = np.reshape(temp_term, np.prod(temp_term.shape))
                self.terms[feat_idx] = temp_term
                self.features = np.vstack([self.features, self.terms[feat_idx]])
                self.features_keys.append(self.terms_label[feat_idx])
            else:
                continue
        self.features = np.transpose(self.features)

    
    def Remove_Dublicated_Factors(self, term, term_label):
        
        '''
        
        Replace factors, present in target term; 
        if can not create unique terms - addition of one more random factor
        
        '''
        
        list_copy = [label for label in self.terms_label if label != term_label]
        try_index = 0

        while True:
            resulting_term_label = {}
            for key, value in term_label.items():
                if key in self.free_keys or key == '1':
                    resulting_term_label[key] = value
                else:
                    new_key = np.random.choice(self.free_keys)
                    resulting_term_label[new_key] = value

            if try_index > 10:
                new_key = np.random.choice(self.free_keys)
                if new_key in resulting_term_label:
                    resulting_term_label[new_key] += 1
                else:
                    resulting_term_label[new_key] = 1                   

            if Check_Unqueness(resulting_term_label, list_copy):
                break
            try_index += 1
        return Create_term_by_dict(self.variables, self.variables_names, resulting_term_label), resulting_term_label

            
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
    
    def Term_Mutation(self, term, term_label, reverse_mutation_probability = 0.1): # Mutation of population individual           
        if len(term_label) == 1:
            new_key = np.random.choice(self.free_keys)
            term_label[new_key] = 1
            term_new = Create_term_by_dict(self.variables, self.variables_names, term_label)
            return term_new, term_label
        key = np.random.choice(list(term_label))
        while key == '1':
            key = np.random.choice(list(term_label))
        value = term_label[key]
        
        mutation_try = 0
        while True:
            term_label_temp = dict(term_label)
            total_power = 0
            for key, value in term_label_temp.items():
                total_power += value
            if np.random.uniform(0, 1) <= reverse_mutation_probability or mutation_try > 15:
                mutation_type = np.random.choice(['Reduction', 'Increasing'])
                if mutation_type == 'Reduction' or total_power >= self.max_factors_in_term and not mutation_try > 15:
                    if value == 1:
                        del term_label_temp[key]
                    else: 
                        term_label_temp[key] -= 1
                else:
                    new_key = np.random.choice(self.free_keys)
                    if new_key in term_label_temp:
                        term_label_temp[new_key] += 1
                    else:
                        term_label_temp[new_key] = 1                    
            else:
                if value == 1:
                    del term_label_temp[key]
                else:
                    term_label_temp[key] -= 1               
                new_key = np.random.choice(self.free_keys)
                if new_key in term_label_temp:
                    term_label_temp[new_key] += 1
                else:
                    term_label_temp[new_key] = 1
            if Check_Unqueness(term_label_temp, self.terms_label):
                break
            mutation_try += 1
        term_new = Create_term_by_dict(self.variables, self.variables_names, term_label_temp)
        return term_new, term_label_temp
      
    def Mutate(self, mutation_probability = 0.4):
        for i in range(4, len(self.terms_label)):
            if np.random.uniform(0, 1) <= mutation_probability and i != self.target_idx:
                term_new, self.terms_label[i] = self.Term_Mutation(self.terms[i], self.terms_label[i])
                term_new = norm_time_series(term_new)
                self.terms[i] = np.reshape(term_new, np.prod(term_new.shape))     
                
# --------------------------------------------------------------------------------------------------------------------
    
def Check_Unqueness(term, prev_terms):  # Check if term is unique in the chromosome
    for prev in prev_terms:
        if term ==prev:
            return False 
    return True

    
def Population_Sort(input_popuation): # Sort population in decreasing order by fitness function value
    output_population = input_popuation
    
    for j in range(1, len(output_population)):
        key_chromosome = output_population[j]
        i = j - 1        
        while i >= 0 and output_population[i].fitness_value > key_chromosome.fitness_value:
            output_population[i+1] = output_population[i]
            i = i - 1
        output_population[i+1] = key_chromosome
        
    return list(reversed(output_population))


def Crossover(chromosome_1, chromosome_2, variables, variables_names, crossover_probability = 0.1):
    
    '''
    
    Crossover between two individuals of population: returns 2 new individuals, that are recombination of 
    their parents' genes
    
    '''
    
    if len(chromosome_1.terms_label) != len(chromosome_2.terms_label):
        raise IndexError('Chromosomes have different number of genes')
    
    result_chromosome_1 = chromosome(variables, variables_names, terms_number = len(chromosome_1.terms_label))
    result_chromosome_2 = chromosome(variables, variables_names, terms_number = len(chromosome_2.terms_label))    
    
    for i in range(0, len(chromosome_1.terms_label)):
        result_chromosome_1.terms_label[i] = chromosome_1.terms_label[i]
        result_chromosome_2.terms_label[i] = chromosome_2.terms_label[i]
        result_chromosome_1.terms[i] = chromosome_1.terms[i]        
        result_chromosome_2.terms[i] = chromosome_2.terms[i]        
    
    for i in range(4, len(result_chromosome_1.terms_label)):
        if np.random.uniform(0, 1) <= crossover_probability and Check_Unqueness(result_chromosome_1.terms_label[i], result_chromosome_2.terms_label) and Check_Unqueness(result_chromosome_2.terms_label[i], result_chromosome_1.terms_label):
            internal_label = result_chromosome_1.terms_label[i]; internal_value = result_chromosome_1.terms[i]
            result_chromosome_1.terms_label[i] = result_chromosome_2.terms_label[i]
            result_chromosome_1.terms[i] = result_chromosome_2.terms[i]
            result_chromosome_2.terms_label[i] = internal_label
            result_chromosome_2.terms[i] = internal_value
    
    return result_chromosome_1, result_chromosome_2


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

    
def Genetic_iteration(iter_num, population, part_with_offsprings, crossover_probability, mutation_probability, 
                      variables, variables_names, estimator_type = 'Ridge', alpha = 0.001):
    
    for chromo in population:
        chromo.Apply_ML(estimator_type = estimator_type, alpha = alpha)
        
    for chromo in population:
        chromo.Calculate_Fitness()
    population = Population_Sort(population)    
    children = Tournament_crossover(population, part_with_offsprings, variables, variables_names, 
                                    crossover_probability = crossover_probability)
    
    for i in range(len(children)):
        population[len(population)-1-len(children)+i] = children[i]

    for i in range(int(len(population)*part_with_offsprings), len(population)):
        population[i].Mutate(mutation_probability = mutation_probability)
        population[i].Split_data()


def Get_true_coeffs(variables, variables_names, eq_final_form):
    var_names = dict(zip(variables_names, variables))
    target_key = eq_final_form.target_key
    target_term = np.copy(variables[0])
    for factor, power in target_key.items():
        target_term *= var_names[factor] ** power
    target_term = np.reshape(target_term, np.prod(target_term.shape))

    features_list = []
    features_list_labels = []
    for i in range(len(eq_final_form.features_keys)):
        if eq_final_form.weights[i] != 0:
            features_list_labels.append(eq_final_form.features_keys[i])
            feature_term = np.copy(variables[0])
            for factor, power in eq_final_form.features_keys[i].items():
                feature_term *= var_names[factor] ** power
            feature_term = np.reshape(feature_term, np.prod(feature_term.shape))
            features_list.append(feature_term)

    if len(features_list) == 0:
        return eq_final_form.target_key, [('0', 1)]
    features = features_list[0]
    if len(features_list) > 1:
        for i in range(1, len(features_list)):
            features = np.vstack([features, features_list[i]])
    features = np.transpose(features)    
    estimator = LinearRegression()
    try:
        estimator.fit(features, target_term)
    except ValueError:
        features = features.reshape(-1, 1)
        estimator.fit(features, target_term)
    weights = estimator.coef_
    return eq_final_form.target_key, list(zip(features_list_labels, weights))    


# Logger
# --------------------------------------------------------------------------------------------------------------------

class Logger:
    def __init__(self):
        self.logfile_name = 'Logs/' + str(datetime.datetime.now()).replace(' ', '_') + '.txt'
        self.logfile = open(self.logfile_name, 'w')

    def Write_string(self, string):
        self.logfile.write(string + '\n')
        
    def Write_logs(self, time, cell, equation):
        self.logfile.write('Cell '+ str(cell[0]) + ' ' + str(cell[1]) + ' : ')
        time_delta = time[1] - time[0]
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

    
    def General_Log(self, time):
        time_delta = time[1] - time[0]
        self.logfile.write('Total runtime: '+ str(time_delta))

        
    def __del__(self):
        self.logfile.close()
        