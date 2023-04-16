'''Neural Network, Question A:
Write a program that calculates the mean-squared error of the iris data for simple
one-layer neural network using a sigmoid non-linearity (linear classification with sig-
moid non-linearity). The function should take three arguments: the data vectors, the
parameters defining the neural network, and the pattern classes. (10 marks)
'''
import numpy as np
from iris.data import iris
from iris.subsets import *

def mean_sq_error(weights: list,
                  bias: float,
                  class_0: int, 
                  class_1: int,
                  two_dimensions_subset: tuple = None) -> float: 
    '''
    Calculates the mean squared error over the iris dataset for Setosa and Versicolor (the classes which are most difficult to distinguish)\n
    Given weight parameters and pattern class(es) for the one-layer neural network. \n
    If a tuple is given for `two_dimensions_subset`, only those two dimensions will be calculated.

    Mean squared error is calculated as the sum of (y - z)^2 over the entire iris dataset (for Setosa and Versicolor), /n
    where y is the correct label (0 for Setosa and 1 for Versicolor) and z is the output of the neural network, from 0 to 1. 
    '''
    data = iris['data']
    class_0_start, class_0_end = get_class_start_end(class_0)
    class_1_start, class_1_end = get_class_start_end(class_1)
    
    class_0_data = data[class_0_start:class_0_end]
    class_1_data = data[class_1_start:class_1_end]
    sum = 0

    if two_dimensions_subset != None:
        one_layer_network = one_layer_network_2D
    else:
        one_layer_network = one_layer_network_4D

    for sample in class_0_data:
        prediction = one_layer_network(sample, weights, bias, two_dimensions_subset)
        sum += (0 - prediction)**2

    for sample in class_1_data:
        prediction = one_layer_network(sample, weights, bias, two_dimensions_subset)
        sum += (1 - prediction)**2

    MSE = sum / ((len(class_0_data) + len(class_1_data)))
    return MSE

def one_layer_network_4D( query: list, weights: list, bias: float, subset_ignore = None) -> float:
    '''
    Calculates the output of a neural network with given weight parameters:\n
    Given one class, it will output the certainty (0 to 1) that the query belongs to that class\n
    Given two classes, it will output which class the query seems to belong to, class 1 (0) to class 2 (1) \n
    '''
    if len(weights) != 4 and len(query) != 4:
        raise Exception('Should have four weights and four dimensions to the query')
    dot_product = 0
    for i in range(4):
        dot_product += weights[i] * query[i]
    dot_product += bias
    return sig(dot_product)

def one_layer_network_2D( query: list,  weights: list, bias: float, two_dimensions_subset: tuple) -> float:
    '''
    Calculates the output of a neural network with given weight parameters:\n
    Given one class, it will output the certainty (0 to 1) that the query belongs to that class\n
    Given two classes, it will output which class the query seems to belong to, class 1 (0) to class 2 (1) \n
    '''
    if len(two_dimensions_subset) != 2 and len(query) != 2 and len(query) != 4:
        raise Exception('Should have two weights, two dimensions to the desired subset, and a 4D query')
    dot_product = weights[0]*query[two_dimensions_subset[0]] + weights[1]*query[two_dimensions_subset[1]]
    dot_product += bias
    return sig(dot_product)

def sig(z):
    return 1 / (1 + np.exp(-z))

def get_class_start_end(class_index):
    if class_index == VIRGINICA:
        return VIRGINICA_START, VIRGINICA_END
    elif class_index == SETOSA:
        return SETOSA_START, SETOSA_END
    elif class_index == VERSICOLOR:
        return VERSICOLOR_START, VERSICOLOR_END
def class_to_string(class_index):
    if class_index == VIRGINICA:
        return 'VIRGINICA'
    elif class_index == SETOSA:
        return 'SETOSA'
    elif class_index == VERSICOLOR:
        return 'VERSICOLOR'


#TODO delete
#Plotting good weights. Approximate dividing line is: y = -.2x + 2.6
#Decision boundary is .5 = sig(x1*w1 + x2*w2 + w0). Replacing x1 with x and x2 with y, 
#We can find an equivalent line with weights w1 = .2, w2 = 1, w0 = -2.6 
def test():
    w1 = 1
    w2 = 1
    w0 = -6.5
    lin_fun = lambda x: -x + 6.5
    print(f'From versicolor, just below: 5 < {lin_fun(1.4)}?')
    print(f'From virginica, just above: 5 > {lin_fun(1.8)}?')

    print(f'From versicolor, just below: {one_layer_network_2D([0, 0, 5, 1.4], [w1, w2], w0, (PETAL_WIDTH, PETAL_LENGTH))}')
    print(f'From virginica, just above: {one_layer_network_2D([0, 0, 5, 1.8], [w1, w2], w0, (PETAL_WIDTH, PETAL_LENGTH))}')
    print(f'So versicolor = class 0, virgnica = class 1 here')

