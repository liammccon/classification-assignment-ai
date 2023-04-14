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
                  class_0_name: str, 
                  class_1_name: str,
                  two_dimensions_subset: tuple = None) -> float: 
    '''
    Calculates the mean squared error over the iris dataset for Setosa and Versicolor (the classes which are most difficult to distinguish)\n
    Given weight parameters and pattern class(es) for the one-layer neural network. \n
    If a tuple is given for `two_dimensions_subset`, only those two dimensions will be calculated.

    Mean squared error is calculated as the sum of (y - z)^2 over the entire iris dataset (for Setosa and Versicolor), /n
    where y is the correct label (0 for Setosa and 1 for Versicolor) and z is the output of the neural network, from 0 to 1. 
    '''
    data = iris['data']
    class_0_start, class_0_end = get_class_start_end(class_0_name)
    class_1_start, class_1_end = get_class_start_end(class_1_name)
    
    class_0_data = data[class_0_start:class_0_end]
    class_1_data = data[class_1_start:class_1_end]
    sum = 0
    
    if two_dimensions_subset != None:
        one_layer_network = one_layer_network_2D
    else:
        one_layer_network = one_layer_network_4D

    for sample in class_0_data:
        prediction = one_layer_network(sample, weights, two_dimensions_subset)
        sum += (0 - prediction)**2
    
    for sample in class_1_data:
        prediction = one_layer_network(sample, weights, two_dimensions_subset)
        sum += (1 - prediction)**2

    MSE = sum / (len(class_0_data) + len(class_1_data))
    return MSE

def one_layer_network_4D( query: list, weights: list, subset_ignore) -> float:
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
    return sig(dot_product)

def one_layer_network_2D( query: list, weights: list, two_dimensions_subset: tuple) -> float:
    '''
    Calculates the output of a neural network with given weight parameters:\n
    Given one class, it will output the certainty (0 to 1) that the query belongs to that class\n
    Given two classes, it will output which class the query seems to belong to, class 1 (0) to class 2 (1) \n
    '''
    if len(two_dimensions_subset) != 2 and len(query) != 2:
        raise Exception('Should have two weights and two dimensions to the desired subset')
    weights_2D = [weights[two_dimensions_subset[0]], weights[two_dimensions_subset[1]]]
    dot_product = 0
    for i in range(2):
        dot_product += weights[i] * query[i]
    return sig(dot_product)

def sig(z):
    return 1 / (1 + np.exp(-z))

def get_class_start_end(class_name):
    if class_name.lower() == 'virginica':
        return VIRGINICA_START, VIRGINICA_END
    elif class_name.lower() == 'setosa':
        return SETOSA_START, SETOSA_END
    elif class_name.lower() == 'versicolor':
        return VERSICOLOR_START, VERSICOLOR_END
